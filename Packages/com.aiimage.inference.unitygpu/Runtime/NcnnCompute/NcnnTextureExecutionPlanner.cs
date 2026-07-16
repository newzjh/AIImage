using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace NcnnCompute
{
    public static class NcnnTexturePlanLayout
    {
        public const string Packed4 = "Packed4";
    }

    [Serializable]
    public sealed class NcnnTexturePlanTensorDescriptor
    {
        public string blob;
        // Shape encoding is [dims, w, h, d, c], matching NcnnRepro.BufferShape.
        public int[] logicalShape;
        public int[] storageShape;
        // MaxPoolingInd indices retain the source activation shape so MaxUnPooling can
        // reconstruct its exact output dimensions without a CPU-side readback.
        public int[] sourceLogicalShape;
        public string layout = NcnnTexturePlanLayout.Packed4;
        public string dtype = "FP16";
        public string aliasGroup;
        public bool textureBacked = true;
    }

    // Partial matrix entries need a loaded-runtime proof for a concrete node. The planner
    // accepts only a real CommandBuffer Pack4 path returned by this callback.
    public delegate NcnnTextureExecutionPlanNodeVerification NcnnTextureExecutionPlanNodeVerifier(
        NcnnParamModel.Layer layer,
        IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
        NcnnTextureExecutionPlanRequest request);

    public sealed class NcnnTextureExecutionPlanNodeVerification
    {
        public bool accepted;
        public bool usesDescriptorAlias;
        public string executionPath;
        public string reason;
        public NcnnTexturePlanTensorDescriptor[] outputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
    }

    [Serializable]
    public sealed class NcnnTextureExecutionPlanRequest
    {
        public string modelName;
        public string targetBackend = NcnnOperatorCapabilityBackend.CommandBuffer;
        public string targetDtype = "FP16";
        public string targetLayout = NcnnTexturePlanLayout.Packed4;
        public bool strict = true;
        public bool debugOracleRelaxed;
        // Activations still use targetDtype. When true, Conv/Gemm/InnerProduct must
        // additionally prove the D2 immutable INT8 weight kernel contract.
        public bool int8WeightOnly;
        // Empty means the legacy all-weight D2 plan.  A non-empty list selects the
        // only operators allowed to consume immutable packed INT8 weights.
        public string[] int8WeightOnlyOperators = Array.Empty<string>();
        // Optional layer whitelist for explicit selective plans. Empty means operator
        // selection decides coverage.
        public string[] int8WeightOnlyLayerNames = Array.Empty<string>();
        public bool int8WeightOnlyLayerSelectionExplicit;
        public NcnnTexturePlanTensorDescriptor[] inputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
        [NonSerialized] public NcnnTextureExecutionPlanNodeVerifier nodeVerifier;
    }

    [Serializable]
    public sealed class NcnnTextureExecutionPlanNode
    {
        public int layerIndex;
        public string layer;
        public string operatorName;
        public string canonicalOperator;
        public string capabilityStatus;
        public bool accepted;
        public bool usesDescriptorAlias;
        public bool acceptedByDebugOracle;
        public string executionPath;
        public NcnnTexturePlanTensorDescriptor[] inputs;
        public NcnnTexturePlanTensorDescriptor[] outputs;
    }

    [Serializable]
    public sealed class NcnnTextureExecutionPlanDiagnostic
    {
        public int layerIndex;
        public string layer;
        public string operatorName;
        public string canonicalOperator;
        public string capabilityStatus;
        public string code;
        public string reason;
        public string targetBackend;
        public string targetDtype;
        public string targetLayout;
        public bool int8WeightOnly;
        public NcnnTexturePlanTensorDescriptor[] inputs;
        public string[] rejectedPaths;
        public string recommendedAction;
        public bool blocking;
    }

    [Serializable]
    public sealed class NcnnTextureExecutionPlan
    {
        public int schemaVersion;
        public string contract;
        public string modelName;
        public string targetBackend;
        public string targetDtype;
        public string targetLayout;
        public bool int8WeightOnly;
        public bool strict;
        public bool debugOracleRelaxed;
        public NcnnTextureExecutionPlanNode[] nodes;
        public NcnnTextureExecutionPlanDiagnostic[] diagnostics;
        public bool strictEligible;
        public bool dispatchAllowed;
        public string summary;
    }

    public sealed class StrictTextureInferencePlanException : InvalidOperationException
    {
        public StrictTextureInferencePlanException(NcnnTextureExecutionPlan plan)
            : base(FormatMessage(plan))
        {
            Plan = plan;
            Diagnostics = plan?.diagnostics ?? Array.Empty<NcnnTextureExecutionPlanDiagnostic>();
        }

        public NcnnTextureExecutionPlan Plan { get; }
        public IReadOnlyList<NcnnTextureExecutionPlanDiagnostic> Diagnostics { get; }

        private static string FormatMessage(NcnnTextureExecutionPlan plan)
        {
            var first = plan?.diagnostics != null && plan.diagnostics.Length > 0 ? plan.diagnostics[0] : null;
            if (first == null)
                return "StrictTextureInference rejected the CommandBuffer Pack4 execution plan.";

            var input = first.inputs != null && first.inputs.Length > 0 ? first.inputs[0] : null;

            return "StrictTextureInference rejected the CommandBuffer Pack4 execution plan"
                + " | layer_index=" + first.layerIndex
                + " | layer=" + (first.layer ?? string.Empty)
                + " | operator=" + (first.operatorName ?? string.Empty)
                + " | status=" + (first.capabilityStatus ?? string.Empty)
                + " | code=" + (first.code ?? string.Empty)
                + " | target_backend=" + (first.targetBackend ?? string.Empty)
                + " | target_dtype=" + (first.targetDtype ?? string.Empty)
                + " | target_layout=" + (first.targetLayout ?? string.Empty)
                + " | blob=" + (input?.blob ?? string.Empty)
                + " | logical_shape=" + FormatShape(input?.logicalShape)
                + " | storage_shape=" + FormatShape(input?.storageShape)
                + " | layout=" + (input?.layout ?? string.Empty)
                + " | dtype=" + (input?.dtype ?? string.Empty)
                + " | rejected_paths=" + string.Join(",", first.rejectedPaths ?? Array.Empty<string>())
                + " | recommendation=" + (first.recommendedAction ?? string.Empty);
        }

        private static string FormatShape(int[] shape)
        {
            return shape == null ? string.Empty : "[" + string.Join(",", shape) + "]";
        }
    }

    // Metadata-only planner. It does not allocate textures, materialize buffers, or record dispatches.
    public static class NcnnTextureExecutionPlanner
    {
        public const int SchemaVersion = 1;
        public const string Contract = "aiimage.strict-texture-execution-plan/v1";

        private static readonly string[] RejectedComputationPaths =
        {
            "alias-only", "placeholder", "materialize-from-buffer", "legacy-path"
        };

        public static NcnnTextureExecutionPlan Compile(
            NcnnParamModel model,
            NcnnTextureExecutionPlanRequest request)
        {
            var plan = Analyze(model, request);
            ThrowIfDispatchRejected(plan);
            return plan;
        }

        public static NcnnTextureExecutionPlan Analyze(
            NcnnParamModel model,
            NcnnTextureExecutionPlanRequest request)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            request ??= new NcnnTextureExecutionPlanRequest();
            var descriptors = new Dictionary<string, NcnnTexturePlanTensorDescriptor>(StringComparer.Ordinal);
            foreach (var input in request.inputs ?? Array.Empty<NcnnTexturePlanTensorDescriptor>())
            {
                if (input == null || string.IsNullOrWhiteSpace(input.blob))
                    continue;
                descriptors[input.blob] = CloneDescriptor(input, input.blob);
            }

            var nodes = new List<NcnnTextureExecutionPlanNode>();
            var diagnostics = new List<NcnnTextureExecutionPlanDiagnostic>();
            var layers = model.layers ?? new List<NcnnParamModel.Layer>();
            for (var index = 0; index < layers.Count; index++)
            {
                var layer = layers[index];
                if (layer == null)
                {
                    nodes.Add(new NcnnTextureExecutionPlanNode { layerIndex = index, accepted = false, executionPath = "invalid" });
                    diagnostics.Add(CreateDiagnostic(request, index, null, null, null, "null-layer", "The model contains a null layer.", Array.Empty<NcnnTexturePlanTensorDescriptor>(), true, "Re-export the model graph."));
                    continue;
                }

                var operatorName = string.IsNullOrWhiteSpace(layer.typeName) ? layer.type.ToString() : layer.typeName;
                NcnnOperatorCapabilities.TryGet(operatorName, out var capability);
                var inputs = ResolveInputs(layer, descriptors);
                var node = new NcnnTextureExecutionPlanNode
                {
                    layerIndex = index,
                    layer = layer.name ?? string.Empty,
                    operatorName = operatorName,
                    canonicalOperator = capability?.canonicalOperator ?? operatorName,
                    capabilityStatus = capability?.status ?? NcnnOperatorCapabilityStatus.Unsupported,
                    inputs = inputs.Select(input => input == null ? null : CloneDescriptor(input, input.blob)).ToArray(),
                    outputs = Array.Empty<NcnnTexturePlanTensorDescriptor>(),
                    executionPath = "rejected"
                };

                if (string.Equals(operatorName, "Input", StringComparison.Ordinal))
                {
                    PlanInputNode(request, layer, node, descriptors, diagnostics);
                    nodes.Add(node);
                    continue;
                }

                if (inputs.Any(input => input == null))
                {
                    diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, "missing-input-descriptor", "A required input has no declared texture descriptor.", inputs, true, "Declare every model input with logical/storage shape, dtype, layout, and alias group."));
                    nodes.Add(node);
                    continue;
                }

                if (IsViewOperator(operatorName))
                {
                    if (TryPlanViewAlias(request, layer, inputs, out var viewOutputs, out var viewReason))
                    {
                        node.accepted = true;
                        node.usesDescriptorAlias = true;
                        node.executionPath = "descriptor-alias";
                        node.outputs = viewOutputs;
                        RegisterOutputs(descriptors, viewOutputs);
                        nodes.Add(node);
                        continue;
                    }

                    if (!CanUseTextureTransform(operatorName))
                    {
                        diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, "missing-descriptor-alias-evidence", viewReason, inputs, true, "Use an alias-compatible view or implement a real Pack4 texture transform."));
                        nodes.Add(node);
                        continue;
                    }
                }

                var inputsMatchTarget = inputs.Select((input, inputIndex) =>
                    MatchesTarget(input, request) || IsMaxPoolingIndexInput(operatorName, inputIndex, input, request)).All(value => value);
                var quantizesOperator = IsInt8WeightOnlyOperator(request, layer.name, operatorName);
                if (quantizesOperator && HasImmutableWeightsWithoutInt8WeightOnlyKernel(operatorName))
                {
                    diagnostics.Add(CreateDiagnostic(
                        request,
                        index,
                        layer,
                        capability,
                        operatorName,
                        "missing-int8-weight-only-kernel",
                        "INT8 selective quantization has no verified immutable packed-weight CommandBuffer kernel for this operator; strict quant planning refuses an FP32 parameter or Buffer fallback.",
                        inputs,
                        true,
                        "Implement and verify a packed INT8 CommandBuffer kernel before enabling this model quantization plan."));
                    nodes.Add(node);
                    continue;
                }
                var requiresInt8WeightKernel = quantizesOperator && RequiresInt8WeightOnlyKernel(operatorName);
                var strictCapability = NcnnOperatorCapabilities.IsStrictlySupported(
                    capability,
                    request.targetBackend,
                    request.targetDtype,
                    request.targetLayout)
                    && (!requiresInt8WeightKernel || capability.int8);
                var verifiedOutputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
                var verifiedPath = string.Empty;
                var verificationReason = string.Empty;
                var verifiedUsesDescriptorAlias = false;
                var profileVerified = !strictCapability
                    && inputsMatchTarget
                    && string.Equals(capability?.status, NcnnOperatorCapabilityStatus.Partial, StringComparison.Ordinal)
                    && IsProfileTargetCompatible(capability, request, requiresInt8WeightKernel)
                    && TryAcceptRuntimeVerifiedNode(layer, inputs, request, out verifiedOutputs, out verifiedPath, out verifiedUsesDescriptorAlias, out verificationReason);
                if (strictCapability && inputsMatchTarget)
                {
                    var outputs = CreateComputedOutputs(layer, inputs[0], request);
                    node.accepted = true;
                    node.executionPath = "command-buffer-pack4";
                    node.outputs = outputs;
                    RegisterOutputs(descriptors, outputs);
                }
                else if (profileVerified)
                {
                    node.accepted = true;
                    node.usesDescriptorAlias = verifiedUsesDescriptorAlias;
                    node.executionPath = verifiedPath;
                    node.outputs = verifiedOutputs;
                    RegisterOutputs(descriptors, verifiedOutputs);
                }
                else if (request.debugOracleRelaxed && capability != null && inputs.Count > 0)
                {
                    var outputs = CreateComputedOutputs(layer, inputs[0], request);
                    node.accepted = true;
                    node.acceptedByDebugOracle = true;
                    node.executionPath = "debug-oracle-relaxed";
                    node.outputs = outputs;
                    RegisterOutputs(descriptors, outputs);
                    diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, "debug-oracle-relaxed", "DebugOracle explicitly accepts a node that is not strict Pack4 eligible.", inputs, false, "Replace this branch with a verified CommandBuffer Pack4 implementation before production use."));
                }
                else
                {
                    var reason = capability == null
                        ? "No capability entry exists for this operator."
                        : !inputsMatchTarget
                            ? "Input descriptor dtype/layout does not match the requested CommandBuffer Pack4 target."
                            : string.Equals(capability.status, NcnnOperatorCapabilityStatus.Partial, StringComparison.Ordinal)
                                && IsProfileTargetCompatible(capability, request, requiresInt8WeightKernel)
                                ? "The loaded runtime profile cannot prove that this node reaches a real CommandBuffer Pack4 path"
                                    + (string.IsNullOrWhiteSpace(verificationReason) ? "." : ": " + verificationReason)
                            : "The capability matrix does not record a verified CommandBuffer Pack4 implementation for this dtype/layout.";
                    var code = string.Equals(capability?.status, NcnnOperatorCapabilityStatus.Partial, StringComparison.Ordinal)
                        && IsProfileTargetCompatible(capability, request, requiresInt8WeightKernel)
                        ? "command-buffer-pack4-profile-rejected"
                        : "missing-command-buffer-pack4-capability";
                    diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, code, reason, inputs, true, "Implement and verify a real CommandBuffer Pack4 path, then update the capability matrix."));
                }

                nodes.Add(node);
            }

            var blocking = diagnostics.Any(diagnostic => diagnostic.blocking);
            var strictEligible = !blocking && nodes.All(node => node.accepted && !node.acceptedByDebugOracle);
            var dispatchAllowed = !blocking && (!request.strict || strictEligible || request.debugOracleRelaxed);
            return new NcnnTextureExecutionPlan
            {
                schemaVersion = SchemaVersion,
                contract = Contract,
                modelName = request.modelName ?? string.Empty,
                targetBackend = request.targetBackend ?? string.Empty,
                targetDtype = request.targetDtype ?? string.Empty,
                targetLayout = request.targetLayout ?? string.Empty,
                int8WeightOnly = request.int8WeightOnly,
                strict = request.strict,
                debugOracleRelaxed = request.debugOracleRelaxed,
                nodes = nodes.ToArray(),
                diagnostics = diagnostics.ToArray(),
                strictEligible = strictEligible,
                dispatchAllowed = dispatchAllowed,
                summary = "nodes=" + nodes.Count + " | diagnostics=" + diagnostics.Count + " | int8_weight_only=" + request.int8WeightOnly + " | strict_eligible=" + strictEligible + " | dispatch_allowed=" + dispatchAllowed
            };
        }

        public static void ThrowIfDispatchRejected(NcnnTextureExecutionPlan plan)
        {
            if (plan == null)
                throw new ArgumentNullException(nameof(plan));
            if (!plan.dispatchAllowed)
                throw new StrictTextureInferencePlanException(plan);
        }

        public static string ToStableJson(NcnnTextureExecutionPlan plan, bool prettyPrint = true)
        {
            if (plan == null)
                throw new ArgumentNullException(nameof(plan));
            return JsonUtility.ToJson(plan, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, NcnnTextureExecutionPlan plan)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));
            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(plan), new System.Text.UTF8Encoding(false));
        }

        private static void PlanInputNode(
            NcnnTextureExecutionPlanRequest request,
            NcnnParamModel.Layer layer,
            NcnnTextureExecutionPlanNode node,
            Dictionary<string, NcnnTexturePlanTensorDescriptor> descriptors,
            List<NcnnTextureExecutionPlanDiagnostic> diagnostics)
        {
            var topNames = layer.topNames ?? Array.Empty<string>();
            var sourceName = topNames.FirstOrDefault(name => descriptors.ContainsKey(name)) ?? layer.name;
            if (string.IsNullOrWhiteSpace(sourceName) || !descriptors.TryGetValue(sourceName, out var source) || !MatchesTarget(source, request))
            {
                diagnostics.Add(CreateDiagnostic(request, node.layerIndex, layer, null, node.operatorName, "missing-pack4-input-descriptor", "Input requires a texture-backed descriptor matching the requested dtype/layout.", Array.Empty<NcnnTexturePlanTensorDescriptor>(), true, "Supply a CommandBuffer Pack4 input descriptor for this blob."));
                return;
            }

            var outputs = new List<NcnnTexturePlanTensorDescriptor>();
            foreach (var topName in topNames)
                outputs.Add(CloneDescriptor(source, topName));
            if (outputs.Count == 0)
            {
                diagnostics.Add(CreateDiagnostic(request, node.layerIndex, layer, null, node.operatorName, "missing-output-blob", "Input has no output blob to bind to its descriptor.", new[] { source }, true, "Re-export the input layer."));
                return;
            }

            node.accepted = true;
            node.executionPath = "external-pack4-input";
            node.outputs = outputs.ToArray();
            RegisterOutputs(descriptors, node.outputs);
        }

        private static List<NcnnTexturePlanTensorDescriptor> ResolveInputs(
            NcnnParamModel.Layer layer,
            Dictionary<string, NcnnTexturePlanTensorDescriptor> descriptors)
        {
            var inputs = new List<NcnnTexturePlanTensorDescriptor>();
            var bottomNames = layer?.bottomNames ?? Array.Empty<string>();
            // aten::to in the exported SD graph carries dtype/device/non-blocking metadata
            // after its data input.  Its runtime layer aliases only the first texture and
            // consumes the rest as scalar metadata, so strict descriptor planning must
            // validate precisely that same single data dependency.
            var count = layer != null && layer.type == NcnnLayerTypes.AtenTo
                ? Math.Min(1, bottomNames.Length)
                : bottomNames.Length;
            for (var index = 0; index < count; index++)
            {
                var bottomName = bottomNames[index];
                descriptors.TryGetValue(bottomName, out var descriptor);
                inputs.Add(descriptor == null ? null : CloneDescriptor(descriptor, bottomName));
            }
            return inputs;
        }

        private static bool IsViewOperator(string operatorName)
        {
            return string.Equals(operatorName, "Noop", StringComparison.Ordinal)
                || string.Equals(operatorName, "aten::to", StringComparison.Ordinal)
                || string.Equals(operatorName, "Split", StringComparison.Ordinal)
                || string.Equals(operatorName, "Reshape", StringComparison.Ordinal)
                || string.Equals(operatorName, "Flatten", StringComparison.Ordinal)
                || string.Equals(operatorName, "Squeeze", StringComparison.Ordinal)
                || string.Equals(operatorName, "ExpandDims", StringComparison.Ordinal)
                || string.Equals(operatorName, "Permute", StringComparison.Ordinal)
                || string.Equals(operatorName, "Tile", StringComparison.Ordinal)
                || string.Equals(operatorName, "Packing", StringComparison.Ordinal)
                || string.Equals(operatorName, "Cast", StringComparison.Ordinal);
        }

        private static bool CanUseTextureTransform(string operatorName)
        {
            return string.Equals(operatorName, "Reshape", StringComparison.Ordinal)
                || string.Equals(operatorName, "Flatten", StringComparison.Ordinal)
                || string.Equals(operatorName, "Permute", StringComparison.Ordinal)
                || string.Equals(operatorName, "Tile", StringComparison.Ordinal)
                || string.Equals(operatorName, "Packing", StringComparison.Ordinal)
                || string.Equals(operatorName, "Cast", StringComparison.Ordinal);
        }

        private static bool TryPlanViewAlias(
            NcnnTextureExecutionPlanRequest request,
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            out NcnnTexturePlanTensorDescriptor[] outputs,
            out string reason)
        {
            outputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
            reason = null;
            if (inputs.Count != 1 || inputs[0] == null)
            {
                reason = "A view node must have one descriptor-backed source tensor.";
                return false;
            }

            var source = inputs[0];
            if (!MatchesTarget(source, request) || string.IsNullOrWhiteSpace(source.aliasGroup))
            {
                reason = "The source lacks a matching Pack4 descriptor or alias group.";
                return false;
            }

            var outputLogicalShape = source.logicalShape;
            var operatorName = string.IsNullOrWhiteSpace(layer.typeName) ? layer.type.ToString() : layer.typeName;
            if (string.Equals(operatorName, "Reshape", StringComparison.Ordinal))
            {
                if (!TryResolveReshape(source.logicalShape, layer, out outputLogicalShape, out reason))
                    return false;
            }
            else if (string.Equals(operatorName, "Flatten", StringComparison.Ordinal))
            {
                if (!TryToBufferShape(source.logicalShape, out var input))
                {
                    reason = "Flatten source logical shape is missing or invalid.";
                    return false;
                }
                outputLogicalShape = ToShapeArray(new NcnnRepro.BufferShape(1, ElementCount(input), 1, 1, 1));
            }
            else if (string.Equals(operatorName, "Squeeze", StringComparison.Ordinal))
            {
                if (!TryToBufferShape(source.logicalShape, out var input))
                {
                    reason = "Squeeze source logical shape is missing or invalid.";
                    return false;
                }
                try
                {
                    outputLogicalShape = ToShapeArray(NcnnRepro.ResolveSqueezeShape(input, layer));
                }
                catch (Exception exception)
                {
                    reason = "Squeeze shape resolution failed: " + exception.Message;
                    return false;
                }
            }
            else if (string.Equals(operatorName, "ExpandDims", StringComparison.Ordinal))
            {
                if (!TryResolveExpandDims(source.logicalShape, layer, out outputLogicalShape, out reason))
                    return false;
            }
            else if (string.Equals(operatorName, "Permute", StringComparison.Ordinal))
            {
                if (layer.GetInt(0, 0) != 0)
                {
                    reason = "Non-identity Permute requires a real Pack4 texture transform.";
                    return false;
                }
            }
            else if (string.Equals(operatorName, "Tile", StringComparison.Ordinal))
            {
                if (!HasIdentityTileParameters(layer))
                {
                    reason = "Non-identity Tile requires a real Pack4 texture transform.";
                    return false;
                }
            }
            else if (string.Equals(operatorName, "Packing", StringComparison.Ordinal))
            {
                if (layer.GetInt(0, 1) != 4 || (layer.GetInt(3, 0) != 0 && layer.GetInt(2, 0) != layer.GetInt(3, 0)))
                {
                    reason = "Non-identity Packing requires a real Pack4 texture transform.";
                    return false;
                }
            }
            else if (string.Equals(operatorName, "Cast", StringComparison.Ordinal))
            {
                if (layer.GetInt(0, 0) != layer.GetInt(1, 0))
                {
                    reason = "Non-identity Cast requires a real Pack4 texture transform.";
                    return false;
                }
            }

            if (!HasPack4AliasEvidence(source, outputLogicalShape, operatorName))
            {
                reason = "Logical/storage descriptor evidence does not prove that this view preserves the Pack4 physical mapping.";
                return false;
            }

            var topNames = layer.topNames ?? Array.Empty<string>();
            if (topNames.Length == 0)
            {
                reason = "The view node has no output blob.";
                return false;
            }

            outputs = topNames.Select(topName => new NcnnTexturePlanTensorDescriptor
            {
                blob = topName,
                logicalShape = CopyShape(outputLogicalShape),
                storageShape = CopyShape(source.storageShape),
                layout = source.layout,
                dtype = source.dtype,
                aliasGroup = source.aliasGroup,
                textureBacked = source.textureBacked
            }).ToArray();
            return true;
        }

        private static bool TryResolveReshape(int[] sourceShape, NcnnParamModel.Layer layer, out int[] outputShape, out string reason)
        {
            outputShape = null;
            reason = null;
            if (!TryToBufferShape(sourceShape, out var source))
            {
                reason = "Reshape source logical shape is missing or invalid.";
                return false;
            }

            try
            {
                var output = NcnnRepro.ResolveReshapeShape(source, layer);
                outputShape = ToShapeArray(output);
                return true;
            }
            catch (Exception exception)
            {
                reason = "Reshape shape resolution failed: " + exception.Message;
                return false;
            }
        }

        private static bool HasPack4AliasEvidence(NcnnTexturePlanTensorDescriptor source, int[] outputLogicalShape, string operatorName)
        {
            if (!source.textureBacked || !TryToBufferShape(source.logicalShape, out var input) || !TryToBufferShape(outputLogicalShape, out var output))
                return false;
            if (!TryToBufferShape(source.storageShape, out _))
                return false;
            if (ElementCount(input) != ElementCount(output))
                return false;
            if (string.Equals(operatorName, "Flatten", StringComparison.Ordinal))
            {
                var storage = source.storageShape;
                return output.dims == 1
                    && storage[0] == 3
                    && storage[1] == output.w
                    && storage[2] == 1
                    && storage[3] == 1
                    && storage[4] == 1;
            }
            if (ShapesEqual(input, output))
                return true;
            return string.Equals(operatorName, "Squeeze", StringComparison.Ordinal)
                || string.Equals(operatorName, "ExpandDims", StringComparison.Ordinal);
        }

        private static NcnnTexturePlanTensorDescriptor[] CreateComputedOutputs(
            NcnnParamModel.Layer layer,
            NcnnTexturePlanTensorDescriptor source,
            NcnnTextureExecutionPlanRequest request)
        {
            return (layer.topNames ?? Array.Empty<string>()).Select((topName, index) => new NcnnTexturePlanTensorDescriptor
            {
                blob = topName,
                logicalShape = CopyShape(source.logicalShape),
                storageShape = CopyShape(source.storageShape),
                layout = request.targetLayout,
                dtype = request.targetDtype,
                aliasGroup = "computed:" + (layer.name ?? layer.typeName ?? "layer") + ":" + index,
                textureBacked = true
            }).ToArray();
        }

        private static bool IsProfileTargetCompatible(
            NcnnOperatorCapability capability,
            NcnnTextureExecutionPlanRequest request,
            bool requiresInt8WeightKernel)
        {
            if (capability == null || request == null)
                return false;
            if (!string.Equals(request.targetBackend, NcnnOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                || !capability.commandBuffer)
                return false;

            var dtypeSupported = string.Equals(request.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase) ? capability.fp32
                : string.Equals(request.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase) ? capability.fp16
                : string.Equals(request.targetDtype, "INT8", StringComparison.OrdinalIgnoreCase) && capability.int8;
            return dtypeSupported
                && (!requiresInt8WeightKernel || capability.int8)
                && (capability.layouts ?? Array.Empty<string>()).Any(layout => string.Equals(layout, request.targetLayout, StringComparison.OrdinalIgnoreCase));
        }

        private static bool RequiresInt8WeightOnlyKernel(string operatorName)
        {
            return string.Equals(operatorName, "Convolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "ConvolutionDepthWise", StringComparison.Ordinal)
                || string.Equals(operatorName, "Gemm", StringComparison.Ordinal)
                || string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal);
        }

        private static bool IsInt8WeightOnlyOperator(NcnnTextureExecutionPlanRequest request, string layerName, string operatorName)
        {
            if (request == null || !request.int8WeightOnly)
                return false;
            var layerNames = request.int8WeightOnlyLayerNames;
            if (request.int8WeightOnlyLayerSelectionExplicit)
                return layerNames != null && layerNames.Any(value => string.Equals(value, layerName, StringComparison.Ordinal));
            var operators = request.int8WeightOnlyOperators;
            if (operators == null || operators.Length == 0)
                return true;
            return operators.Any(value => string.Equals(value, operatorName, StringComparison.Ordinal));
        }

        private static bool HasImmutableWeightsWithoutInt8WeightOnlyKernel(string operatorName)
        {
            return string.Equals(operatorName, "Convolution1D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Convolution3D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "DeconvolutionDepthWise", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution3D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Embed", StringComparison.Ordinal)
                || string.Equals(operatorName, "MultiHeadAttention", StringComparison.Ordinal)
                || string.Equals(operatorName, "BatchNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "GroupNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "LayerNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "Scale", StringComparison.Ordinal)
                || string.Equals(operatorName, "PReLU", StringComparison.Ordinal)
                || string.Equals(operatorName, "Normalize", StringComparison.Ordinal);
        }

        private static bool TryAcceptRuntimeVerifiedNode(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request,
            out NcnnTexturePlanTensorDescriptor[] outputs,
            out string executionPath,
            out bool usesDescriptorAlias,
            out string reason)
        {
            outputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
            executionPath = null;
            usesDescriptorAlias = false;
            reason = null;
            if (request?.nodeVerifier == null)
            {
                reason = "No loaded-runtime CommandBuffer Pack4 verifier is available.";
                return false;
            }

            NcnnTextureExecutionPlanNodeVerification verification;
            try
            {
                verification = request.nodeVerifier(layer, inputs, request);
            }
            catch (Exception exception)
            {
                reason = "The loaded-runtime verifier failed: " + exception.Message;
                return false;
            }

            if (verification == null || !verification.accepted)
            {
                reason = verification?.reason ?? "The loaded-runtime verifier rejected this node.";
                return false;
            }

            if (verification.usesDescriptorAlias)
            {
                if (!HasRuntimeAliasEvidence(verification, inputs))
                {
                    reason = "The loaded-runtime verifier did not provide descriptor alias evidence for this noop/view path.";
                    return false;
                }
            }
            else if (!IsRealCommandBufferPack4Path(verification.executionPath))
            {
                reason = "The loaded-runtime verifier did not provide a real CommandBuffer Pack4 execution path.";
                return false;
            }

            var topNames = layer?.topNames ?? Array.Empty<string>();
            if (verification.outputs == null || verification.outputs.Length != topNames.Length || verification.outputs.Length == 0)
            {
                reason = "The loaded-runtime verifier did not provide one target descriptor for every output blob.";
                return false;
            }

            for (var index = 0; index < verification.outputs.Length; index++)
            {
                var output = verification.outputs[index];
                if (output == null
                    || (!MatchesTarget(output, request) && !IsMaxPoolingIndexOutput(layer, index, output, request))
                    || !string.Equals(output.blob, topNames[index], StringComparison.Ordinal))
                {
                    reason = "The loaded-runtime verifier produced an output descriptor outside the requested target contract.";
                    return false;
                }
            }

            outputs = verification.outputs.Select(output => CloneDescriptor(output, output.blob)).ToArray();
            executionPath = verification.executionPath;
            usesDescriptorAlias = verification.usesDescriptorAlias;
            return true;
        }

        private static bool HasRuntimeAliasEvidence(
            NcnnTextureExecutionPlanNodeVerification verification,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs)
        {
            if (!string.Equals(verification.executionPath, "descriptor-alias", StringComparison.Ordinal)
                || inputs == null || inputs.Count != 1 || inputs[0] == null
                || string.IsNullOrWhiteSpace(inputs[0].aliasGroup)
                || verification.outputs == null || verification.outputs.Length == 0)
            {
                return false;
            }

            var source = inputs[0];
            return verification.outputs.All(output => output != null
                && string.Equals(output.aliasGroup, source.aliasGroup, StringComparison.Ordinal)
                && output.textureBacked == source.textureBacked
                && TryToBufferShape(output.logicalShape, out var outputShape)
                && TryToBufferShape(source.logicalShape, out var sourceShape)
                && ElementCount(outputShape) == ElementCount(sourceShape)
                && ShapesEqual(output.storageShape, source.storageShape));
        }

        private static bool TryResolveExpandDims(int[] sourceShape, NcnnParamModel.Layer layer, out int[] outputShape, out string reason)
        {
            outputShape = null;
            reason = null;
            if (!TryToBufferShape(sourceShape, out var input))
            {
                reason = "ExpandDims source logical shape is missing or invalid.";
                return false;
            }

            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, Array.Empty<int>());
            if (axes == null || axes.Length == 0)
            {
                reason = "ExpandDims requires static axes metadata.";
                return false;
            }

            try
            {
                var dims = input.dims;
                var values = new[] { input.w, input.h, input.dims == 4 ? input.d : input.c, input.dims == 4 ? input.c : 1 };
                for (var i = 0; i < axes.Length; i++)
                {
                    var outDims = dims + 1;
                    if (outDims > 4)
                        throw new InvalidOperationException("ExpandDims would exceed rank four.");
                    var ncnnAxis = axes[i] < 0 ? axes[i] + outDims : axes[i];
                    if (ncnnAxis < 0 || ncnnAxis >= outDims)
                        throw new InvalidOperationException("ExpandDims axis is out of range.");
                    var axis = NcnnRepro.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                    var next = new[] { 1, 1, 1, 1 };
                    for (var j = 0; j < outDims; j++)
                        next[j] = j == axis ? 1 : values[j < axis ? j : j - 1];
                    values = next;
                    dims = outDims;
                }

                outputShape = dims == 1
                    ? new[] { 1, values[0], 1, 1, 1 }
                    : dims == 2
                        ? new[] { 2, values[0], values[1], 1, 1 }
                        : dims == 3
                            ? new[] { 3, values[0], values[1], 1, values[2] }
                            : new[] { 4, values[0], values[1], values[2], values[3] };
                return true;
            }
            catch (Exception exception)
            {
                reason = "ExpandDims shape resolution failed: " + exception.Message;
                return false;
            }
        }

        private static bool HasIdentityTileParameters(NcnnParamModel.Layer layer)
        {
            var repeats = layer.GetInts(-23302, null) ?? layer.GetInts(2, null) ?? layer.GetInts(-23330, null) ?? layer.GetInts(30, null);
            if (repeats != null && repeats.Length > 0)
                return repeats.All(value => value == 1);
            return layer.GetInt(1, 1) == 1;
        }

        private static bool IsRealCommandBufferPack4Path(string executionPath)
        {
            if (string.IsNullOrWhiteSpace(executionPath)
                || !executionPath.StartsWith("command-buffer-pack4", StringComparison.OrdinalIgnoreCase))
                return false;

            return !RejectedComputationPaths.Any(path => executionPath.IndexOf(path, StringComparison.OrdinalIgnoreCase) >= 0);
        }

        private static bool ShapesEqual(int[] left, int[] right)
        {
            return left != null && right != null && left.SequenceEqual(right);
        }

        private static void RegisterOutputs(
            Dictionary<string, NcnnTexturePlanTensorDescriptor> descriptors,
            IEnumerable<NcnnTexturePlanTensorDescriptor> outputs)
        {
            foreach (var output in outputs ?? Array.Empty<NcnnTexturePlanTensorDescriptor>())
            {
                if (output != null && !string.IsNullOrWhiteSpace(output.blob))
                    descriptors[output.blob] = CloneDescriptor(output, output.blob);
            }
        }

        private static bool MatchesTarget(NcnnTexturePlanTensorDescriptor descriptor, NcnnTextureExecutionPlanRequest request)
        {
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                && string.Equals(descriptor.dtype, request.targetDtype, StringComparison.OrdinalIgnoreCase)
                && TryToBufferShape(descriptor.logicalShape, out _)
                && TryToBufferShape(descriptor.storageShape, out _);
        }

        private static bool IsMaxPoolingIndexInput(
            string operatorName,
            int inputIndex,
            NcnnTexturePlanTensorDescriptor descriptor,
            NcnnTextureExecutionPlanRequest request)
        {
            return string.Equals(operatorName, "MaxUnPooling", StringComparison.Ordinal)
                && inputIndex == 1
                && descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                && string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                && TryToBufferShape(descriptor.logicalShape, out _)
                && TryToBufferShape(descriptor.storageShape, out _)
                && TryToBufferShape(descriptor.sourceLogicalShape, out _);
        }

        private static bool IsMaxPoolingIndexOutput(
            NcnnParamModel.Layer layer,
            int outputIndex,
            NcnnTexturePlanTensorDescriptor descriptor,
            NcnnTextureExecutionPlanRequest request)
        {
            return string.Equals(layer?.typeName, "MaxPoolingInd", StringComparison.Ordinal)
                && outputIndex == 1
                && descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                && string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                && TryToBufferShape(descriptor.logicalShape, out _)
                && TryToBufferShape(descriptor.storageShape, out _)
                && TryToBufferShape(descriptor.sourceLogicalShape, out _);
        }

        private static NcnnTextureExecutionPlanDiagnostic CreateDiagnostic(
            NcnnTextureExecutionPlanRequest request,
            int layerIndex,
            NcnnParamModel.Layer layer,
            NcnnOperatorCapability capability,
            string operatorName,
            string code,
            string reason,
            IEnumerable<NcnnTexturePlanTensorDescriptor> inputs,
            bool blocking,
            string recommendedAction)
        {
            return new NcnnTextureExecutionPlanDiagnostic
            {
                layerIndex = layerIndex,
                layer = layer?.name ?? string.Empty,
                operatorName = operatorName ?? string.Empty,
                canonicalOperator = capability?.canonicalOperator ?? operatorName ?? string.Empty,
                capabilityStatus = capability?.status ?? NcnnOperatorCapabilityStatus.Unsupported,
                code = code,
                reason = reason,
                targetBackend = request.targetBackend,
                targetDtype = request.targetDtype,
                targetLayout = request.targetLayout,
                int8WeightOnly = request.int8WeightOnly,
                inputs = (inputs ?? Array.Empty<NcnnTexturePlanTensorDescriptor>()).Select(input => input == null ? null : CloneDescriptor(input, input.blob)).ToArray(),
                rejectedPaths = RejectedComputationPaths,
                recommendedAction = recommendedAction,
                blocking = blocking
            };
        }

        private static NcnnTexturePlanTensorDescriptor CloneDescriptor(NcnnTexturePlanTensorDescriptor descriptor, string blob)
        {
            return new NcnnTexturePlanTensorDescriptor
            {
                blob = blob ?? descriptor?.blob ?? string.Empty,
                logicalShape = CopyShape(descriptor?.logicalShape),
                storageShape = CopyShape(descriptor?.storageShape),
                sourceLogicalShape = CopyShape(descriptor?.sourceLogicalShape),
                layout = descriptor?.layout ?? string.Empty,
                dtype = descriptor?.dtype ?? string.Empty,
                aliasGroup = descriptor?.aliasGroup ?? string.Empty,
                textureBacked = descriptor != null && descriptor.textureBacked
            };
        }

        private static bool TryToBufferShape(int[] shape, out NcnnRepro.BufferShape value)
        {
            value = default;
            if (shape == null || shape.Length != 5 || shape[0] < 1 || shape[0] > 4
                || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 || shape[4] <= 0)
                return false;
            value = new NcnnRepro.BufferShape(shape[0], shape[1], shape[2], shape[3], shape[4]);
            return true;
        }

        private static int[] ToShapeArray(NcnnRepro.BufferShape shape)
        {
            return new[] { shape.dims, shape.w, shape.h, shape.d, shape.c };
        }

        private static int[] CopyShape(int[] shape)
        {
            return shape == null ? Array.Empty<int>() : shape.ToArray();
        }

        private static bool ShapesEqual(NcnnRepro.BufferShape a, NcnnRepro.BufferShape b)
        {
            return a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c;
        }

        private static int ElementCount(NcnnRepro.BufferShape shape)
        {
            return shape.w * shape.h * shape.d * shape.c;
        }
    }
}
