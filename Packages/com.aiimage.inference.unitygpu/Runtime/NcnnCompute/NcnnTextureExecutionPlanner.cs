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

            return "StrictTextureInference rejected the CommandBuffer Pack4 execution plan"
                + " | layer_index=" + first.layerIndex
                + " | layer=" + (first.layer ?? string.Empty)
                + " | operator=" + (first.operatorName ?? string.Empty)
                + " | status=" + (first.capabilityStatus ?? string.Empty)
                + " | code=" + (first.code ?? string.Empty)
                + " | target_backend=" + (first.targetBackend ?? string.Empty)
                + " | target_dtype=" + (first.targetDtype ?? string.Empty)
                + " | target_layout=" + (first.targetLayout ?? string.Empty)
                + " | rejected_paths=" + string.Join(",", first.rejectedPaths ?? Array.Empty<string>())
                + " | recommendation=" + (first.recommendedAction ?? string.Empty);
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

                    if (!string.Equals(operatorName, "Reshape", StringComparison.Ordinal))
                    {
                        diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, "missing-descriptor-alias-evidence", viewReason, inputs, true, "Use an alias-compatible view or implement a real Pack4 texture transform."));
                        nodes.Add(node);
                        continue;
                    }
                }

                var inputsMatchTarget = inputs.All(input => MatchesTarget(input, request));
                var strictCapability = NcnnOperatorCapabilities.IsStrictlySupported(
                    capability,
                    request.targetBackend,
                    request.targetDtype,
                    request.targetLayout);
                var verifiedOutputs = Array.Empty<NcnnTexturePlanTensorDescriptor>();
                var verifiedPath = string.Empty;
                var verificationReason = string.Empty;
                var verifiedUsesDescriptorAlias = false;
                var profileVerified = !strictCapability
                    && inputsMatchTarget
                    && string.Equals(capability?.status, NcnnOperatorCapabilityStatus.Partial, StringComparison.Ordinal)
                    && IsProfileTargetCompatible(capability, request)
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
                                && IsProfileTargetCompatible(capability, request)
                                ? "The loaded runtime profile cannot prove that this node reaches a real CommandBuffer Pack4 path"
                                    + (string.IsNullOrWhiteSpace(verificationReason) ? "." : ": " + verificationReason)
                            : "The capability matrix does not record a verified CommandBuffer Pack4 implementation for this dtype/layout.";
                    var code = string.Equals(capability?.status, NcnnOperatorCapabilityStatus.Partial, StringComparison.Ordinal)
                        && IsProfileTargetCompatible(capability, request)
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
                strict = request.strict,
                debugOracleRelaxed = request.debugOracleRelaxed,
                nodes = nodes.ToArray(),
                diagnostics = diagnostics.ToArray(),
                strictEligible = strictEligible,
                dispatchAllowed = dispatchAllowed,
                summary = "nodes=" + nodes.Count + " | diagnostics=" + diagnostics.Count + " | strict_eligible=" + strictEligible + " | dispatch_allowed=" + dispatchAllowed
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
            foreach (var bottomName in layer.bottomNames ?? Array.Empty<string>())
            {
                descriptors.TryGetValue(bottomName, out var descriptor);
                inputs.Add(descriptor == null ? null : CloneDescriptor(descriptor, bottomName));
            }
            return inputs;
        }

        private static bool IsViewOperator(string operatorName)
        {
            return string.Equals(operatorName, "Noop", StringComparison.Ordinal)
                || string.Equals(operatorName, "Split", StringComparison.Ordinal)
                || string.Equals(operatorName, "Reshape", StringComparison.Ordinal);
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
            if (string.Equals(layer.typeName, "Reshape", StringComparison.Ordinal) || layer.type == NcnnLayerTypes.Reshape)
            {
                if (!TryResolveReshape(source.logicalShape, layer, out outputLogicalShape, out reason))
                    return false;
            }

            if (!HasPack4AliasEvidence(source, outputLogicalShape))
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

        private static bool HasPack4AliasEvidence(NcnnTexturePlanTensorDescriptor source, int[] outputLogicalShape)
        {
            if (!source.textureBacked || !TryToBufferShape(source.logicalShape, out var input) || !TryToBufferShape(outputLogicalShape, out var output))
                return false;
            if (!TryToBufferShape(source.storageShape, out _))
                return false;
            if (input.dims > 3 || output.dims > 3)
                return ShapesEqual(input, output);
            if (input.dims != output.dims || ElementCount(input) != ElementCount(output))
                return false;
            if (ShapesEqual(input, output))
                return true;
            if ((input.dims == 3 && (input.c % 4) != 0) || (output.dims == 3 && (output.c % 4) != 0))
                return false;
            return input.w == output.w
                && input.h == output.h
                && Mathf.CeilToInt(input.c / 4f) == Mathf.CeilToInt(output.c / 4f);
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
            NcnnTextureExecutionPlanRequest request)
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
                && (capability.layouts ?? Array.Empty<string>()).Any(layout => string.Equals(layout, request.targetLayout, StringComparison.OrdinalIgnoreCase));
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
                if (output == null || !MatchesTarget(output, request) || !string.Equals(output.blob, topNames[index], StringComparison.Ordinal))
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
                && ShapesEqual(output.logicalShape, source.logicalShape)
                && ShapesEqual(output.storageShape, source.storageShape));
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
