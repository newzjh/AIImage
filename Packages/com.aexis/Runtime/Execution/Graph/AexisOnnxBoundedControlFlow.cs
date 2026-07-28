using System;
using System.Collections.Generic;
using System.Globalization;
using Aexis.Onnx;

namespace Aexis.Execution
{
    // ONNX control-flow values are not a second execution backend. This pass proves
    // their bounds while importing and emits only ordinary static ONNX nodes. The
    // normal Pack4 CommandBuffer planner therefore owns every resulting activation.
    internal sealed class AexisOnnxBoundedControlFlowResult
    {
        internal OnnxModel model;
        internal readonly List<AexisOnnxBoundedControlFlowDiagnostic> diagnostics = new List<AexisOnnxBoundedControlFlowDiagnostic>();
    }

    internal sealed class AexisOnnxBoundedControlFlowDiagnostic
    {
        internal int nodeIndex;
        internal string node;
        internal string opType;
        internal string code;
        internal string message;
        internal string recommendedAction;
    }

    internal static class AexisOnnxBoundedControlFlow
    {
        private sealed class SequenceContract
        {
            internal int count;
            internal long[] elementShape;
        }

        internal static AexisOnnxBoundedControlFlowResult Flatten(
            OnnxModel source,
            AexisOnnxGraphLoweringOptions options)
        {
            var result = new AexisOnnxBoundedControlFlowResult
            {
                model = CloneModel(source)
            };
            if (options == null || !options.enableBoundedControlFlowLowering)
                return result;

            FlattenGraph(result.model.graph, options, result, "root");
            return result;
        }

        private static void FlattenGraph(
            OnnxGraph graph,
            AexisOnnxGraphLoweringOptions options,
            AexisOnnxBoundedControlFlowResult result,
            string scope)
        {
            if (graph == null)
                return;

            var source = new List<OnnxNode>(graph.nodes);
            var expanded = new List<OnnxNode>();
            var sequences = new Dictionary<string, SequenceContract>(StringComparer.Ordinal);
            var optionals = new HashSet<string>(StringComparer.Ordinal);
            for (var index = 0; index < source.Count; index++)
            {
                var node = source[index];
                if (node == null)
                    continue;

                var nodeScope = scope + "_" + index.ToString(CultureInfo.InvariantCulture) + "_" + Sanitize(NodeName(node, index));
                if (string.Equals(node.opType, "If", StringComparison.Ordinal))
                {
                    if (TryInlineIf(graph, node, index, nodeScope, options, result, expanded))
                        continue;
                }
                else if (string.Equals(node.opType, "Loop", StringComparison.Ordinal))
                {
                    if (TryUnrollLoop(graph, node, index, nodeScope, options, result, expanded))
                        continue;
                }
                else if (string.Equals(node.opType, "Scan", StringComparison.Ordinal))
                {
                    if (TryUnrollScan(graph, node, index, nodeScope, options, result, expanded))
                        continue;
                }
                else if (IsSequenceOperator(node.opType))
                {
                    if (TryLowerSequence(graph, node, index, nodeScope, sequences, result, expanded))
                        continue;
                }
                else if (IsOptionalOperator(node.opType))
                {
                    if (TryLowerOptional(graph, node, index, optionals, result, expanded))
                        continue;
                }

                expanded.Add(node);
            }

            graph.nodes.Clear();
            graph.nodes.AddRange(expanded);
        }

        private static bool TryInlineIf(
            OnnxGraph parent,
            OnnxNode node,
            int index,
            string scope,
            AexisOnnxGraphLoweringOptions options,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output)
        {
            if (node.inputs.Count != 1 || node.outputs.Count == 0
                || !TryReadStaticBool(parent, node.inputs[0], out var condition)
                || !TryGetGraphAttribute(node, condition ? "then_branch" : "else_branch", out var branch)
                || branch == null)
            {
                AddDiagnostic(result, index, node, "bounded-if-proof-required",
                    "If requires one initializer/Constant scalar Bool or Int condition and a selected graph-valued branch.",
                    "Fold the branch condition before import and retain a static then_branch/else_branch GraphProto.");
                return false;
            }
            if (branch.inputs.Count != 0 || branch.outputs.Count != node.outputs.Count)
            {
                AddDiagnostic(result, index, node, "unsupported-bounded-if-signature",
                    "The bounded If profile accepts capture-only branches with exactly one output per If output.",
                    "Remove explicit branch inputs and make both branch output arities match the parent If node.");
                return false;
            }

            var aliases = new Dictionary<string, string>(StringComparer.Ordinal);
            if (!InlineGraph(parent, branch, scope + (condition ? "_then" : "_else"), aliases, options, result, output, out var branchOutputs))
                return false;
            for (var outputIndex = 0; outputIndex < node.outputs.Count; outputIndex++)
                output.Add(Identity(scope + "_out_" + outputIndex.ToString(CultureInfo.InvariantCulture), branchOutputs[outputIndex], node.outputs[outputIndex]));
            return true;
        }

        private static bool TryUnrollLoop(
            OnnxGraph parent,
            OnnxNode node,
            int index,
            string scope,
            AexisOnnxGraphLoweringOptions options,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output)
        {
            if (node.inputs.Count < 2 || !TryGetGraphAttribute(node, "body", out var body) || body == null
                || !TryReadStaticInt(parent, node.inputs[0], out var tripCount)
                || !TryReadStaticBool(parent, node.inputs[1], out var initialCondition))
            {
                AddDiagnostic(result, index, node, "bounded-loop-proof-required",
                    "Loop requires immutable scalar trip-count and condition inputs plus a graph-valued body.",
                    "Export a fixed positive trip count and a true constant loop condition; data-dependent loop control is not a Pack4 arena contract.");
                return false;
            }
            if (tripCount < 0 || tripCount > Math.Max(1, options.maximumStaticLoopIterations) || !initialCondition)
            {
                AddDiagnostic(result, index, node, "bounded-loop-capacity-rejected",
                    "Loop trip count or initial condition is outside the declared bounded profile.",
                    "Set a true initial condition and a trip count from zero through maximumStaticLoopIterations.");
                return false;
            }

            var carriedCount = node.inputs.Count - 2;
            if (body.inputs.Count != carriedCount + 2 || body.outputs.Count < carriedCount + 1 || node.outputs.Count != body.outputs.Count - 1)
            {
                AddDiagnostic(result, index, node, "unsupported-bounded-loop-signature",
                    "Loop body must declare [iteration,condition,carried...] inputs and [condition,carried...,scan...] outputs.",
                    "Use the canonical ONNX Loop body signature and expose every carried/scan output on the parent Loop node.");
                return false;
            }
            if (!TryReadStaticBool(body, body.outputs[0].name, out var bodyCondition) || !bodyCondition)
            {
                AddDiagnostic(result, index, node, "data-dependent-loop-condition",
                    "The bounded Loop profile requires a body condition output proven to be a constant true scalar.",
                    "Use fixed-trip unrolling for GPU execution or lower data-dependent control flow outside this strict profile.");
                return false;
            }
            var scanCount = body.outputs.Count - carriedCount - 1;
            if (tripCount == 0 && scanCount != 0)
            {
                AddDiagnostic(result, index, node, "empty-loop-scan-output",
                    "A zero-trip Loop cannot produce a non-empty texture-backed scan output.",
                    "Use at least one static iteration or remove scan outputs before strict import.");
                return false;
            }
            if (!ValidateScanOutputShapes(body, carriedCount + 1, scanCount, result, index, node))
                return false;

            var carries = new string[carriedCount];
            for (var carryIndex = 0; carryIndex < carriedCount; carryIndex++)
                carries[carryIndex] = node.inputs[carryIndex + 2];
            var scans = new List<string>[scanCount];
            for (var scanIndex = 0; scanIndex < scanCount; scanIndex++) scans[scanIndex] = new List<string>();

            for (var iteration = 0; iteration < tripCount; iteration++)
            {
                var aliases = new Dictionary<string, string>(StringComparer.Ordinal)
                {
                    [body.inputs[0].name] = AddScalarInitializer(parent, scope + "_iter_" + iteration.ToString(CultureInfo.InvariantCulture), iteration, 6),
                    [body.inputs[1].name] = AddScalarInitializer(parent, scope + "_cond_" + iteration.ToString(CultureInfo.InvariantCulture), 1, 9)
                };
                for (var carryIndex = 0; carryIndex < carriedCount; carryIndex++)
                    aliases[body.inputs[carryIndex + 2].name] = carries[carryIndex];
                if (!InlineGraph(parent, body, scope + "_iter_" + iteration.ToString(CultureInfo.InvariantCulture), aliases, options, result, output, out var bodyOutputs))
                    return false;
                for (var carryIndex = 0; carryIndex < carriedCount; carryIndex++)
                    carries[carryIndex] = bodyOutputs[carryIndex + 1];
                for (var scanIndex = 0; scanIndex < scanCount; scanIndex++)
                    scans[scanIndex].Add(bodyOutputs[carriedCount + 1 + scanIndex]);
            }

            for (var carryIndex = 0; carryIndex < carriedCount; carryIndex++)
                output.Add(Identity(scope + "_carry_" + carryIndex.ToString(CultureInfo.InvariantCulture), carries[carryIndex], node.outputs[carryIndex]));
            for (var scanIndex = 0; scanIndex < scanCount; scanIndex++)
            {
                var outputIndex = carriedCount + scanIndex;
                output.Add(Concat(scope + "_scan_" + scanIndex.ToString(CultureInfo.InvariantCulture), scans[scanIndex], node.outputs[outputIndex]));
            }
            return true;
        }

        private static bool TryUnrollScan(
            OnnxGraph parent,
            OnnxNode node,
            int index,
            string scope,
            AexisOnnxGraphLoweringOptions options,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output)
        {
            if (!TryGetGraphAttribute(node, "body", out var body) || body == null)
            {
                AddDiagnostic(result, index, node, "bounded-scan-body-required",
                    "Scan requires a graph-valued body.", "Export the canonical ONNX Scan body GraphProto.");
                return false;
            }
            var scanInputs = GetIntAttribute(node, "num_scan_inputs", 0);
            var stateCount = node.inputs.Count - scanInputs;
            if (scanInputs <= 0 || stateCount < 0 || body.inputs.Count != node.inputs.Count || body.outputs.Count != node.outputs.Count)
            {
                AddDiagnostic(result, index, node, "unsupported-bounded-scan-signature",
                    "Scan requires a positive num_scan_inputs and matching parent/body state plus scan arities.",
                    "Use the canonical Scan signature with explicit static state and scan tensors.");
                return false;
            }
            if (!HasOnlyZeroAxes(node, "scan_input_axes", scanInputs) || !HasOnlyZeroAxes(node, "scan_output_axes", body.outputs.Count - stateCount)
                || !HasOnlyForwardDirections(node, "scan_input_directions", scanInputs) || !HasOnlyForwardDirections(node, "scan_output_directions", body.outputs.Count - stateCount))
            {
                AddDiagnostic(result, index, node, "unsupported-bounded-scan-axis",
                    "The bounded Scan profile accepts only forward axis-0 scan inputs and outputs.",
                    "Transpose before/after Scan and use axis=0, direction=0 in the strict Pack4 profile.");
                return false;
            }

            var steps = -1;
            for (var scanIndex = 0; scanIndex < scanInputs; scanIndex++)
            {
                if (!TryFindStaticShape(parent, node.inputs[stateCount + scanIndex], out var shape)
                    || shape.Length < 1 || shape.Length > 4 || shape[0] <= 0 || shape[0] > int.MaxValue)
                {
                    AddDiagnostic(result, index, node, "bounded-scan-shape-proof-required",
                        "Each Scan input requires a declared static rank-1 through rank-4 axis-0 extent.",
                        "Provide static value_info for every scan input before strict import.");
                    return false;
                }
                var count = (int)shape[0];
                if (steps < 0) steps = count;
                if (steps != count)
                {
                    AddDiagnostic(result, index, node, "inconsistent-bounded-scan-length",
                        "All Scan input axis-0 extents must match.", "Export equal static scan lengths.");
                    return false;
                }
            }
            if (steps <= 0 || steps > Math.Max(1, options.maximumStaticScanSteps))
            {
                AddDiagnostic(result, index, node, "bounded-scan-capacity-rejected",
                    "Scan length is outside the declared maximumStaticScanSteps capacity.",
                    "Use a non-empty static sequence within the configured Scan capacity.");
                return false;
            }
            var scanOutputCount = body.outputs.Count - stateCount;
            if (!ValidateScanOutputShapes(body, stateCount, scanOutputCount, result, index, node))
                return false;

            var states = new string[stateCount];
            for (var stateIndex = 0; stateIndex < stateCount; stateIndex++) states[stateIndex] = node.inputs[stateIndex];
            var scans = new List<string>[scanOutputCount];
            for (var scanIndex = 0; scanIndex < scanOutputCount; scanIndex++) scans[scanIndex] = new List<string>();
            for (var step = 0; step < steps; step++)
            {
                var aliases = new Dictionary<string, string>(StringComparer.Ordinal);
                for (var stateIndex = 0; stateIndex < stateCount; stateIndex++) aliases[body.inputs[stateIndex].name] = states[stateIndex];
                for (var scanIndex = 0; scanIndex < scanInputs; scanIndex++)
                {
                    var slice = scope + "_input_" + scanIndex.ToString(CultureInfo.InvariantCulture) + "_step_" + step.ToString(CultureInfo.InvariantCulture);
                    AddAxisZeroSlice(parent, output, slice, node.inputs[stateCount + scanIndex], step);
                    aliases[body.inputs[stateCount + scanIndex].name] = slice;
                }
                if (!InlineGraph(parent, body, scope + "_step_" + step.ToString(CultureInfo.InvariantCulture), aliases, options, result, output, out var bodyOutputs))
                    return false;
                for (var stateIndex = 0; stateIndex < stateCount; stateIndex++) states[stateIndex] = bodyOutputs[stateIndex];
                for (var scanIndex = 0; scanIndex < scanOutputCount; scanIndex++) scans[scanIndex].Add(bodyOutputs[stateCount + scanIndex]);
            }
            for (var stateIndex = 0; stateIndex < stateCount; stateIndex++)
                output.Add(Identity(scope + "_state_" + stateIndex.ToString(CultureInfo.InvariantCulture), states[stateIndex], node.outputs[stateIndex]));
            for (var scanIndex = 0; scanIndex < scanOutputCount; scanIndex++)
                output.Add(Concat(scope + "_output_" + scanIndex.ToString(CultureInfo.InvariantCulture), scans[scanIndex], node.outputs[stateCount + scanIndex]));
            return true;
        }

        private static bool TryLowerSequence(
            OnnxGraph parent,
            OnnxNode node,
            int index,
            string scope,
            Dictionary<string, SequenceContract> sequences,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output)
        {
            if (string.Equals(node.opType, "SequenceConstruct", StringComparison.Ordinal))
            {
                if (node.outputs.Count != 1 || node.inputs.Count == 0 || !TryFindStaticShape(parent, node.inputs[0], out var shape)
                    || !IsSequenceElementShape(shape))
                {
                    AddDiagnostic(result, index, node, "bounded-sequence-profile-required",
                        "SequenceConstruct requires one or more static rank-1 through rank-4 inputs whose leading extent is one.",
                        "Normalize sequence elements to [1,...] tensors before strict import.");
                    return false;
                }
                for (var inputIndex = 1; inputIndex < node.inputs.Count; inputIndex++)
                {
                    if (!TryFindStaticShape(parent, node.inputs[inputIndex], out var candidate) || !ShapesEqual(shape, candidate))
                    {
                        AddDiagnostic(result, index, node, "inconsistent-sequence-element-shape",
                            "SequenceConstruct element shapes must be static and equal.", "Use a fixed element shape for every sequence member.");
                        return false;
                    }
                }
                output.Add(Concat(scope + "_construct", node.inputs, node.outputs[0]));
                sequences[node.outputs[0]] = new SequenceContract { count = node.inputs.Count, elementShape = Clone(shape) };
                return true;
            }
            if (string.Equals(node.opType, "SplitToSequence", StringComparison.Ordinal))
            {
                if (node.inputs.Count != 1 || node.outputs.Count != 1 || GetIntAttribute(node, "axis", 0) != 0
                    || !TryFindStaticShape(parent, node.inputs[0], out var shape) || !IsSequenceTensorShape(shape))
                {
                    AddDiagnostic(result, index, node, "unsupported-split-to-sequence-profile",
                        "SplitToSequence requires a static axis-0 tensor with a positive leading extent and element shape [1,...].",
                        "Use axis=0 and static rank-1 through rank-4 input shape.");
                    return false;
                }
                output.Add(Identity(scope + "_split", node.inputs[0], node.outputs[0]));
                var elementShape = Clone(shape); elementShape[0] = 1;
                sequences[node.outputs[0]] = new SequenceContract { count = (int)shape[0], elementShape = elementShape };
                return true;
            }
            if (node.inputs.Count < 1 || !sequences.TryGetValue(node.inputs[0], out var sequence))
            {
                AddDiagnostic(result, index, node, "sequence-texture-contract-missing",
                    "Sequence consumer requires a preceding bounded SequenceConstruct/SplitToSequence texture contract.",
                    "Keep the sequence in the same static graph and use the supported bounded sequence operators.");
                return false;
            }
            if (string.Equals(node.opType, "SequenceAt", StringComparison.Ordinal))
            {
                if (node.inputs.Count != 2 || node.outputs.Count != 1 || !TryReadStaticInt(parent, node.inputs[1], out var item) || item < 0 || item >= sequence.count)
                {
                    AddDiagnostic(result, index, node, "bounded-sequence-index-proof-required",
                        "SequenceAt requires an immutable in-range scalar index.", "Fold the sequence index to an Int32/Int64 initializer or Constant.");
                    return false;
                }
                AddAxisZeroSlice(parent, output, node.outputs[0], node.inputs[0], item);
                return true;
            }
            if (string.Equals(node.opType, "SequenceLength", StringComparison.Ordinal))
            {
                if (node.outputs.Count != 1)
                    return false;
                output.Add(ConstantInt(scope + "_length", node.outputs[0], sequence.count));
                return true;
            }
            if (string.Equals(node.opType, "ConcatFromSequence", StringComparison.Ordinal))
            {
                if (node.outputs.Count != 1 || GetIntAttribute(node, "axis", 0) != 0 || GetIntAttribute(node, "new_axis", 0) != 0)
                {
                    AddDiagnostic(result, index, node, "unsupported-concat-from-sequence-profile",
                        "ConcatFromSequence requires axis=0 and new_axis=0 in the bounded texture profile.",
                        "Use the existing axis-0 sequence layout or lower non-zero axis concatenation before import.");
                    return false;
                }
                output.Add(Identity(scope + "_concat", node.inputs[0], node.outputs[0]));
                return true;
            }
            if (string.Equals(node.opType, "SequenceInsert", StringComparison.Ordinal))
            {
                if (node.inputs.Count != 2 || node.outputs.Count != 1 || !TryFindStaticShape(parent, node.inputs[1], out var insertShape)
                    || !ShapesEqual(insertShape, sequence.elementShape))
                {
                    AddDiagnostic(result, index, node, "unsupported-sequence-insert-profile",
                        "SequenceInsert requires one static [1,...] element and no dynamic position input.",
                        "Append a shape-matched element to the bounded axis-0 sequence.");
                    return false;
                }
                output.Add(Concat(scope + "_insert", new[] { node.inputs[0], node.inputs[1] }, node.outputs[0]));
                sequences[node.outputs[0]] = new SequenceContract { count = sequence.count + 1, elementShape = Clone(sequence.elementShape) };
                return true;
            }
            AddDiagnostic(result, index, node, "unsupported-bounded-sequence-operator",
                "This sequence operator has no fixed Pack4 texture expansion.", "Use SequenceConstruct, SplitToSequence, SequenceAt, SequenceLength, SequenceInsert, or ConcatFromSequence.");
            return false;
        }

        private static bool TryLowerOptional(
            OnnxGraph parent,
            OnnxNode node,
            int index,
            HashSet<string> optionals,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output)
        {
            if (string.Equals(node.opType, "Optional", StringComparison.Ordinal))
            {
                if (node.inputs.Count != 1 || node.outputs.Count != 1)
                {
                    AddDiagnostic(result, index, node, "optional-none-not-supported",
                        "The texture arena cannot represent OptionalNone because it has no immutable texture descriptor.",
                        "Provide one concrete static tensor to Optional or lower the absence case outside strict import.");
                    return false;
                }
                output.Add(Identity("optional_" + NodeName(node, index), node.inputs[0], node.outputs[0]));
                optionals.Add(node.outputs[0]);
                return true;
            }
            if (node.inputs.Count != 1 || node.outputs.Count != 1 || !optionals.Contains(node.inputs[0]))
            {
                AddDiagnostic(result, index, node, "optional-texture-contract-missing",
                    "Optional consumer requires a concrete Optional tensor created in the same static graph.",
                    "Keep Optional and OptionalGetElement/OptionalHasElement in the bounded texture graph.");
                return false;
            }
            if (string.Equals(node.opType, "OptionalGetElement", StringComparison.Ordinal))
            {
                output.Add(Identity("optional_get_" + NodeName(node, index), node.inputs[0], node.outputs[0]));
                return true;
            }
            if (string.Equals(node.opType, "OptionalHasElement", StringComparison.Ordinal))
            {
                output.Add(ConstantInt("optional_has_" + NodeName(node, index), node.outputs[0], 1));
                return true;
            }
            return false;
        }

        private static bool InlineGraph(
            OnnxGraph parent,
            OnnxGraph source,
            string scope,
            Dictionary<string, string> aliases,
            AexisOnnxGraphLoweringOptions options,
            AexisOnnxBoundedControlFlowResult result,
            List<OnnxNode> output,
            out string[] outputs)
        {
            outputs = Array.Empty<string>();
            if (source == null)
                return false;
            var graph = CloneGraph(source);
            FlattenGraph(graph, options, result, scope);
            var map = new Dictionary<string, string>(aliases, StringComparer.Ordinal);
            foreach (var initializer in graph.initializers)
            {
                var promoted = scope + "_const_" + Sanitize(initializer.Key);
                if (parent.initializers.ContainsKey(promoted))
                    promoted += "_" + parent.initializers.Count.ToString(CultureInfo.InvariantCulture);
                parent.initializers[promoted] = CloneTensor(initializer.Value, promoted);
                map[initializer.Key] = promoted;
            }
            foreach (var graphNode in graph.nodes)
                foreach (var name in graphNode.outputs)
                    if (!string.IsNullOrEmpty(name) && !map.ContainsKey(name))
                        map[name] = scope + "_value_" + Sanitize(name);

            foreach (var graphNode in graph.nodes)
            {
                var clone = CloneNode(graphNode);
                clone.name = scope + "_" + Sanitize(NodeName(graphNode, output.Count));
                for (var inputIndex = 0; inputIndex < clone.inputs.Count; inputIndex++)
                    if (!string.IsNullOrEmpty(clone.inputs[inputIndex]) && map.TryGetValue(clone.inputs[inputIndex], out var mappedInput))
                        clone.inputs[inputIndex] = mappedInput;
                for (var outputIndex = 0; outputIndex < clone.outputs.Count; outputIndex++)
                    if (!string.IsNullOrEmpty(clone.outputs[outputIndex]) && map.TryGetValue(clone.outputs[outputIndex], out var mappedOutput))
                        clone.outputs[outputIndex] = mappedOutput;
                output.Add(clone);
            }
            outputs = new string[graph.outputs.Count];
            for (var outputIndex = 0; outputIndex < outputs.Length; outputIndex++)
            {
                var name = graph.outputs[outputIndex]?.name;
                if (string.IsNullOrWhiteSpace(name) || !map.TryGetValue(name, out outputs[outputIndex]))
                    return false;
            }
            return true;
        }

        private static bool ValidateScanOutputShapes(OnnxGraph body, int offset, int count, AexisOnnxBoundedControlFlowResult result, int index, OnnxNode node)
        {
            for (var scanIndex = 0; scanIndex < count; scanIndex++)
            {
                var name = body.outputs[offset + scanIndex]?.name;
                if (!TryFindStaticShape(body, name, out var shape) || !IsSequenceElementShape(shape))
                {
                    AddDiagnostic(result, index, node, "bounded-scan-output-shape-proof-required",
                        "Every Loop/Scan scan output requires static value_info with rank 1..4 and leading extent one.",
                        "Declare the body scan output as [1,...] so its fixed texture capacity can be planned.");
                    return false;
                }
            }
            return true;
        }

        private static void AddAxisZeroSlice(OnnxGraph graph, List<OnnxNode> output, string resultName, string source, int index)
        {
            var prefix = "__aexis_bounded_slice_" + Sanitize(resultName);
            var starts = AddScalarInitializer(graph, prefix + "_start", index, 7);
            var ends = AddScalarInitializer(graph, prefix + "_end", index + 1, 7);
            var axes = AddScalarInitializer(graph, prefix + "_axis", 0, 7);
            var steps = AddScalarInitializer(graph, prefix + "_step", 1, 7);
            var slice = new OnnxNode { name = prefix, opType = "Slice" };
            slice.inputs.Add(source); slice.inputs.Add(starts); slice.inputs.Add(ends); slice.inputs.Add(axes); slice.inputs.Add(steps);
            slice.outputs.Add(resultName);
            output.Add(slice);
        }

        private static string AddScalarInitializer(OnnxGraph graph, string name, int value, int onnxType)
        {
            var unique = name;
            var suffix = 0;
            while (graph.initializers.ContainsKey(unique)) unique = name + "_" + (++suffix).ToString(CultureInfo.InvariantCulture);
            graph.initializers[unique] = new OnnxTensor
            {
                name = unique,
                dataType = TensorDataType.Int32,
                onnxDataType = onnxType,
                dims = Array.Empty<long>(),
                int32Data = new[] { value }
            };
            return unique;
        }

        private static OnnxNode Identity(string name, string input, string output)
        {
            var node = new OnnxNode { name = name, opType = "Identity" };
            node.inputs.Add(input); node.outputs.Add(output);
            return node;
        }

        private static OnnxNode Concat(string name, IList<string> inputs, string output)
        {
            var node = new OnnxNode { name = name, opType = "Concat" };
            for (var index = 0; index < inputs.Count; index++) node.inputs.Add(inputs[index]);
            node.outputs.Add(output);
            node.attributes["axis"] = new OnnxAttribute { name = "axis", type = 2, i = 0 };
            return node;
        }

        private static OnnxNode ConstantInt(string name, string output, int value)
        {
            var node = new OnnxNode { name = name, opType = "Constant" };
            node.outputs.Add(output);
            node.attributes["value"] = new OnnxAttribute
            {
                name = "value",
                type = 4,
                tensor = new OnnxTensor { dataType = TensorDataType.Int32, onnxDataType = 6, dims = Array.Empty<long>(), int32Data = new[] { value } }
            };
            return node;
        }

        private static bool TryReadStaticBool(OnnxGraph graph, string name, out bool value)
        {
            value = false;
            if (!TryReadStaticInt(graph, name, out var integer)) return false;
            if (integer != 0 && integer != 1) return false;
            value = integer != 0;
            return true;
        }

        private static bool TryReadStaticInt(OnnxGraph graph, string name, out int value)
        {
            value = 0;
            if (string.IsNullOrEmpty(name)) return false;
            if (graph.initializers.TryGetValue(name, out var initializer)) return TryReadScalar(initializer, out value);
            foreach (var node in graph.nodes)
            {
                if (node == null || !string.Equals(node.opType, "Constant", StringComparison.Ordinal) || node.outputs.Count != 1 || !string.Equals(node.outputs[0], name, StringComparison.Ordinal))
                    continue;
                if (node.attributes.TryGetValue("value", out var valueAttribute) && TryReadScalar(valueAttribute.tensor, out value)) return true;
                if (node.attributes.TryGetValue("value_int", out var integerAttribute) && integerAttribute.i >= int.MinValue && integerAttribute.i <= int.MaxValue)
                {
                    value = (int)integerAttribute.i;
                    return true;
                }
            }
            return false;
        }

        private static bool TryReadScalar(OnnxTensor tensor, out int value)
        {
            value = 0;
            if (tensor == null || tensor.ElementCount != 1) return false;
            if (tensor.int32Data != null && tensor.int32Data.Length == 1) { value = tensor.int32Data[0]; return true; }
            if (tensor.int64Data != null && tensor.int64Data.Length == 1 && tensor.int64Data[0] >= int.MinValue && tensor.int64Data[0] <= int.MaxValue)
            {
                value = (int)tensor.int64Data[0];
                return true;
            }
            if (tensor.rawData != null && tensor.rawData.Length == sizeof(int) && (tensor.onnxDataType == 6 || tensor.onnxDataType == 9))
            {
                value = BitConverter.ToInt32(tensor.rawData, 0);
                return true;
            }
            if (tensor.rawData != null && tensor.rawData.Length == sizeof(long) && tensor.onnxDataType == 7)
            {
                var wide = BitConverter.ToInt64(tensor.rawData, 0);
                if (wide < int.MinValue || wide > int.MaxValue) return false;
                value = (int)wide;
                return true;
            }
            if (tensor.rawData != null && tensor.rawData.Length == 1 && tensor.onnxDataType == 9)
            {
                value = tensor.rawData[0] == 0 ? 0 : 1;
                return true;
            }
            return false;
        }

        private static bool TryFindStaticShape(OnnxGraph graph, string name, out long[] shape)
        {
            shape = null;
            if (graph == null || string.IsNullOrEmpty(name)) return false;
            if (graph.initializers.TryGetValue(name, out var tensor) && IsStaticShape(tensor.dims)) { shape = Clone(tensor.dims); return true; }
            foreach (var value in graph.inputs)
                if (value != null && string.Equals(value.name, name, StringComparison.Ordinal) && IsStaticShape(value.dims)) { shape = Clone(value.dims); return true; }
            foreach (var value in graph.valueInfos)
                if (value != null && string.Equals(value.name, name, StringComparison.Ordinal) && IsStaticShape(value.dims)) { shape = Clone(value.dims); return true; }
            foreach (var value in graph.outputs)
                if (value != null && string.Equals(value.name, name, StringComparison.Ordinal) && IsStaticShape(value.dims)) { shape = Clone(value.dims); return true; }
            return false;
        }

        private static bool TryGetGraphAttribute(OnnxNode node, string name, out OnnxGraph graph)
        {
            graph = null;
            return node != null && node.attributes.TryGetValue(name, out var attribute) && (graph = attribute?.graph) != null;
        }

        private static int GetIntAttribute(OnnxNode node, string name, int defaultValue)
        {
            return node != null && node.attributes.TryGetValue(name, out var attribute)
                && attribute.i >= int.MinValue && attribute.i <= int.MaxValue ? (int)attribute.i : defaultValue;
        }

        private static bool HasOnlyZeroAxes(OnnxNode node, string name, int count)
        {
            if (!node.attributes.TryGetValue(name, out var attribute) || attribute.ints.Count == 0) return true;
            if (attribute.ints.Count != count) return false;
            for (var index = 0; index < attribute.ints.Count; index++) if (attribute.ints[index] != 0) return false;
            return true;
        }

        private static bool HasOnlyForwardDirections(OnnxNode node, string name, int count)
        {
            return HasOnlyZeroAxes(node, name, count);
        }

        private static bool IsStaticShape(long[] shape)
        {
            if (shape == null) return false;
            for (var index = 0; index < shape.Length; index++) if (shape[index] <= 0) return false;
            return true;
        }

        private static bool IsSequenceTensorShape(long[] shape)
        {
            return IsStaticShape(shape) && shape.Length >= 1 && shape.Length <= 4 && shape[0] > 0 && shape[0] <= int.MaxValue;
        }

        private static bool IsSequenceElementShape(long[] shape)
        {
            return IsSequenceTensorShape(shape) && shape[0] == 1;
        }

        private static bool ShapesEqual(long[] left, long[] right)
        {
            if (left == null || right == null || left.Length != right.Length) return false;
            for (var index = 0; index < left.Length; index++) if (left[index] != right[index]) return false;
            return true;
        }

        private static bool IsSequenceOperator(string op)
        {
            return op == "SequenceConstruct" || op == "SequenceAt" || op == "SequenceLength" || op == "SequenceInsert"
                || op == "ConcatFromSequence" || op == "SplitToSequence" || op == "SequenceErase";
        }

        private static bool IsOptionalOperator(string op)
        {
            return op == "Optional" || op == "OptionalGetElement" || op == "OptionalHasElement";
        }

        private static void AddDiagnostic(AexisOnnxBoundedControlFlowResult result, int index, OnnxNode node, string code, string message, string action)
        {
            result.diagnostics.Add(new AexisOnnxBoundedControlFlowDiagnostic
            {
                nodeIndex = index,
                node = NodeName(node, index),
                opType = node?.opType ?? string.Empty,
                code = code,
                message = message,
                recommendedAction = action
            });
        }

        private static OnnxModel CloneModel(OnnxModel source)
        {
            return new OnnxModel { opset = source.opset, graph = CloneGraph(source.graph) };
        }

        private static OnnxGraph CloneGraph(OnnxGraph source)
        {
            var result = new OnnxGraph { name = source?.name ?? string.Empty };
            if (source == null) return result;
            foreach (var node in source.nodes) result.nodes.Add(CloneNode(node));
            foreach (var input in source.inputs) result.inputs.Add(CloneValueInfo(input));
            foreach (var output in source.outputs) result.outputs.Add(CloneValueInfo(output));
            foreach (var value in source.valueInfos) result.valueInfos.Add(CloneValueInfo(value));
            foreach (var initializer in source.initializers) result.initializers[initializer.Key] = CloneTensor(initializer.Value, initializer.Key);
            return result;
        }

        private static OnnxNode CloneNode(OnnxNode source)
        {
            var result = new OnnxNode { name = source?.name ?? string.Empty, opType = source?.opType ?? string.Empty, domain = source?.domain ?? string.Empty };
            if (source == null) return result;
            result.inputs.AddRange(source.inputs); result.outputs.AddRange(source.outputs);
            foreach (var attribute in source.attributes) result.attributes[attribute.Key] = CloneAttribute(attribute.Value);
            return result;
        }

        private static OnnxAttribute CloneAttribute(OnnxAttribute source)
        {
            var result = new OnnxAttribute { name = source?.name ?? string.Empty, type = source?.type ?? 0, f = source?.f ?? 0f, i = source?.i ?? 0,
                s = source?.s == null ? Array.Empty<byte>() : (byte[])source.s.Clone(), tensor = source?.tensor == null ? null : CloneTensor(source.tensor, source.tensor.name), graph = source?.graph == null ? null : CloneGraph(source.graph) };
            if (source == null) return result;
            result.floats.AddRange(source.floats); result.ints.AddRange(source.ints); result.strings.AddRange(source.strings);
            foreach (var graph in source.graphs) result.graphs.Add(CloneGraph(graph));
            return result;
        }

        private static OnnxValueInfo CloneValueInfo(OnnxValueInfo source)
        {
            return new OnnxValueInfo { name = source?.name ?? string.Empty, dataType = source?.dataType ?? TensorDataType.Unknown, onnxDataType = source?.onnxDataType ?? 0, dims = source?.dims == null ? Array.Empty<long>() : Clone(source.dims) };
        }

        private static OnnxTensor CloneTensor(OnnxTensor source, string name)
        {
            var result = new OnnxTensor
            {
                name = name ?? string.Empty,
                dataType = source?.dataType ?? TensorDataType.Unknown,
                onnxDataType = source?.onnxDataType ?? 0,
                dims = source?.dims == null ? Array.Empty<long>() : Clone(source.dims),
                rawData = source?.rawData == null ? Array.Empty<byte>() : (byte[])source.rawData.Clone(),
                floatData = source?.floatData == null ? Array.Empty<float>() : (float[])source.floatData.Clone(),
                int32Data = source?.int32Data == null ? Array.Empty<int>() : (int[])source.int32Data.Clone(),
                int64Data = source?.int64Data == null ? Array.Empty<long>() : (long[])source.int64Data.Clone(),
                dataLocation = source?.dataLocation ?? 0
            };
            if (source != null) foreach (var pair in source.externalData) result.externalData[pair.Key] = pair.Value;
            return result;
        }

        private static long[] Clone(long[] values)
        {
            return values == null ? Array.Empty<long>() : (long[])values.Clone();
        }

        private static string NodeName(OnnxNode node, int index)
        {
            return !string.IsNullOrWhiteSpace(node?.name) ? node.name : (node?.opType ?? "node") + "_" + index.ToString(CultureInfo.InvariantCulture);
        }

        private static string Sanitize(string value)
        {
            if (string.IsNullOrEmpty(value)) return "node";
            var chars = value.ToCharArray();
            for (var index = 0; index < chars.Length; index++)
                if (!(char.IsLetterOrDigit(chars[index]) || chars[index] == '_')) chars[index] = '_';
            return new string(chars);
        }
    }
}
