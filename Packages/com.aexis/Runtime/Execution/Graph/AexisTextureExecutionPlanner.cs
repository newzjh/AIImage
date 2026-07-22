using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Aexis.Execution
{
    public static class AexisTexturePlanLayout
    {
        public const string Packed4 = "Packed4";
    }

    [Serializable]
    public sealed class AexisTexturePlanTensorDescriptor
    {
        public string blob;
        // Shape encoding is [dims, w, h, d, c], matching AexisGraphSession.BufferShape.
        public int[] logicalShape;
        public int[] storageShape;
        // MaxPoolingInd indices retain the source activation shape so MaxUnPooling can
        // reconstruct its exact output dimensions without a CPU-side readback.
        public int[] sourceLogicalShape;
        public string layout = AexisTexturePlanLayout.Packed4;
        // dtype describes the physical texture format. logicalDtype keeps tensor
        // semantics when Int32 values are encoded exactly in an FP32 texture.
        public string dtype = "FP16";
        public string logicalDtype = "Float16";
        public string aliasGroup;
        public bool textureBacked = true;
    }

    // Partial matrix entries need a loaded-runtime proof for a concrete node. The planner
    // accepts only a real CommandBuffer texture path returned by this callback.
    public delegate AexisTextureExecutionPlanNodeVerification AexisTextureExecutionPlanNodeVerifier(
        AexisGraphModel.Layer layer,
        IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
        AexisTextureExecutionPlanRequest request);

    public sealed class AexisTextureExecutionPlanNodeVerification
    {
        public bool accepted;
        public bool usesDescriptorAlias;
        public string executionPath;
        public string reason;
        public AexisTexturePlanTensorDescriptor[] outputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
    }

    [Serializable]
    public sealed class AexisTextureExecutionPlanRequest
    {
        public string modelName;
        public string targetBackend = AexisOperatorCapabilityBackend.CommandBuffer;
        public string targetDtype = "FP16";
        public string targetLayout = AexisTexturePlanLayout.Packed4;
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
        public bool int4WeightOnly;
        public string[] int4WeightOnlyOperators = Array.Empty<string>();
        public string[] int4WeightOnlyLayerNames = Array.Empty<string>();
        public bool int4WeightOnlyLayerSelectionExplicit;
        public AexisTexturePlanTensorDescriptor[] inputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
        [NonSerialized] public AexisTextureExecutionPlanNodeVerifier nodeVerifier;
    }

    [Serializable]
    public sealed class AexisTextureExecutionPlanNode
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
        public AexisTexturePlanTensorDescriptor[] inputs;
        public AexisTexturePlanTensorDescriptor[] outputs;
    }

    [Serializable]
    public sealed class AexisTextureExecutionPlanDiagnostic
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
        public AexisTexturePlanTensorDescriptor[] inputs;
        public string[] rejectedPaths;
        public string recommendedAction;
        public bool blocking;
    }

    [Serializable]
    public sealed class AexisTextureExecutionPlan
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
        public AexisTextureExecutionPlanNode[] nodes;
        public AexisTextureExecutionPlanDiagnostic[] diagnostics;
        public bool strictEligible;
        public bool dispatchAllowed;
        public string summary;
    }

    public sealed class StrictTextureInferencePlanException : InvalidOperationException
    {
        public StrictTextureInferencePlanException(AexisTextureExecutionPlan plan)
            : base(FormatMessage(plan))
        {
            Plan = plan;
            Diagnostics = plan?.diagnostics ?? Array.Empty<AexisTextureExecutionPlanDiagnostic>();
        }

        public AexisTextureExecutionPlan Plan { get; }
        public IReadOnlyList<AexisTextureExecutionPlanDiagnostic> Diagnostics { get; }

        private static string FormatMessage(AexisTextureExecutionPlan plan)
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
    public static class AexisTextureExecutionPlanner
    {
        public const int SchemaVersion = 1;
        public const string Contract = "aiimage.strict-texture-execution-plan/v1";

        private static readonly string[] RejectedComputationPaths =
        {
            "alias-only",
            "placeholder",
            "materialize",
            "compute-buffer",
            "buffer-fallback",
            "texture-to-buffer",
            "buffer-to-texture",
            "readback",
            "legacy-path"
        };

        public static AexisTextureExecutionPlan Compile(
            AexisGraphModel model,
            AexisTextureExecutionPlanRequest request)
        {
            var plan = Analyze(model, request);
            ThrowIfDispatchRejected(plan);
            return plan;
        }

        public static AexisTextureExecutionPlan Analyze(
            AexisGraphModel model,
            AexisTextureExecutionPlanRequest request)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));

            request ??= new AexisTextureExecutionPlanRequest();
            var descriptors = new Dictionary<string, AexisTexturePlanTensorDescriptor>(StringComparer.Ordinal);
            foreach (var input in request.inputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
            {
                if (input == null || string.IsNullOrWhiteSpace(input.blob))
                    continue;
                descriptors[input.blob] = CloneDescriptor(input, input.blob);
            }

            var nodes = new List<AexisTextureExecutionPlanNode>();
            var diagnostics = new List<AexisTextureExecutionPlanDiagnostic>();
            var layers = model.layers ?? new List<AexisGraphModel.Layer>();
            for (var index = 0; index < layers.Count; index++)
            {
                var layer = layers[index];
                if (layer == null)
                {
                    nodes.Add(new AexisTextureExecutionPlanNode { layerIndex = index, accepted = false, executionPath = "invalid" });
                    diagnostics.Add(CreateDiagnostic(request, index, null, null, null, "null-layer", "The model contains a null layer.", Array.Empty<AexisTexturePlanTensorDescriptor>(), true, "Re-export the model graph."));
                    continue;
                }

                var operatorName = string.IsNullOrWhiteSpace(layer.typeName) ? layer.type.ToString() : layer.typeName;
                AexisOperatorCapabilities.TryGet(operatorName, out var capability);
                var inputs = ResolveInputs(layer, descriptors);
                var node = new AexisTextureExecutionPlanNode
                {
                    layerIndex = index,
                    layer = layer.name ?? string.Empty,
                    operatorName = operatorName,
                    canonicalOperator = capability?.canonicalOperator ?? operatorName,
                    capabilityStatus = capability?.status ?? AexisOperatorCapabilityStatus.Unsupported,
                    inputs = inputs.Select(input => input == null ? null : CloneDescriptor(input, input.blob)).ToArray(),
                    outputs = Array.Empty<AexisTexturePlanTensorDescriptor>(),
                    executionPath = "rejected"
                };

                if (string.Equals(operatorName, "Input", StringComparison.Ordinal))
                {
                    PlanInputNode(request, layer, node, descriptors, diagnostics);
                    nodes.Add(node);
                    continue;
                }

                if (request.strict
                    && (string.Equals(operatorName, "NonZero", StringComparison.Ordinal)
                        || string.Equals(operatorName, "Compress", StringComparison.Ordinal))
                    && TryFindOutputConsumer(layers, index, layer.topNames, out var consumedBlob, out var consumerIndex, out var consumerName))
                {
                    diagnostics.Add(CreateDiagnostic(
                        request,
                        index,
                        layer,
                        capability,
                        operatorName,
                        "bounded-data-index-output-must-be-terminal",
                        "Bounded " + operatorName + " output " + consumedBlob + " is consumed by layer " + consumerIndex + " (" + consumerName + "); padded capacity values are not standard ONNX tensor semantics.",
                        inputs,
                        true,
                        "Keep both bounded value/count outputs terminal, or implement an explicit count-aware texture consumer profile."));
                    nodes.Add(node);
                    continue;
                }

                if (inputs.Any(input => input == null))
                {
                    diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, "missing-input-descriptor", "A required input has no declared texture descriptor.", inputs, true, "Declare every model input with logical/storage shape, dtype, layout, and alias group."));
                    nodes.Add(node);
                    continue;
                }

                if (request.strict && !TryValidateTextureCapacities(inputs, out var capacityReason))
                {
                    diagnostics.Add(CreateDiagnostic(
                        request,
                        index,
                        layer,
                        capability,
                        operatorName,
                        "texture-descriptor-capacity-exceeded",
                        capacityReason,
                        inputs,
                        true,
                        "Use storage extents that fit the active graphics device, or reject this model before dispatch."));
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
                var quantizesOperator = IsInt8WeightOnlyOperator(request, layer.name, operatorName)
                    || IsInt4WeightOnlyOperator(request, layer.name, operatorName);
                if (quantizesOperator && HasImmutableWeightsWithoutInt8WeightOnlyKernel(operatorName))
                {
                    diagnostics.Add(CreateDiagnostic(
                        request,
                        index,
                        layer,
                        capability,
                        operatorName,
                        "missing-quantized-weight-only-kernel",
                        "Selective quantization has no verified immutable packed-weight CommandBuffer kernel for this operator; strict quant planning refuses an FP32 parameter or Buffer fallback.",
                        inputs,
                        true,
                        "Implement and verify a packed INT8/INT4 CommandBuffer kernel before enabling this model quantization plan."));
                    nodes.Add(node);
                    continue;
                }
                var requiresInt8WeightKernel = quantizesOperator && RequiresInt8WeightOnlyKernel(operatorName);
                var strictCapability = AexisOperatorCapabilities.IsStrictlySupported(
                    capability,
                    request.targetBackend,
                    request.targetDtype,
                    request.targetLayout)
                    && (!requiresInt8WeightKernel || capability.int8);
                var isConditionalCapability = string.Equals(capability?.status, AexisOperatorCapabilityStatus.SupportedByProfile, StringComparison.Ordinal)
                    || string.Equals(capability?.status, AexisOperatorCapabilityStatus.Partial, StringComparison.Ordinal);
                var profileMatchReason = string.Empty;
                var profileTargetCompatible = isConditionalCapability
                    && (!requiresInt8WeightKernel || capability.int8)
                    && AexisOperatorCapabilities.TryMatchTextureProfile(
                        capability,
                        layer,
                        request.targetBackend,
                        request.targetDtype,
                        request.targetLayout,
                        inputs,
                        out _,
                        out profileMatchReason);
                var verifiedOutputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
                var verifiedPath = string.Empty;
                var verificationReason = string.Empty;
                var verifiedUsesDescriptorAlias = false;
                var profileVerified = !strictCapability
                    && inputsMatchTarget
                    && profileTargetCompatible
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
                else if (request.debugOracleRelaxed
                    && capability != null
                    && !string.Equals(capability.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal)
                    && inputs.Count > 0)
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
                        : string.Equals(capability.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal)
                            ? capability.limitations
                        : !inputsMatchTarget
                            ? "Input descriptor dtype/layout does not match the requested CommandBuffer Pack4 target."
                            : isConditionalCapability && profileTargetCompatible
                                ? "The loaded runtime profile cannot prove that this node reaches a real CommandBuffer Pack4 path"
                                    + (string.IsNullOrWhiteSpace(verificationReason) ? "." : ": " + verificationReason)
                            : isConditionalCapability && !string.IsNullOrWhiteSpace(profileMatchReason)
                                ? profileMatchReason
                                : "The capability matrix does not record a verified CommandBuffer Pack4 implementation for this dtype/layout.";
                    var code = capability != null && string.Equals(capability.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal)
                        ? "unsupported-operator"
                        : isConditionalCapability && profileTargetCompatible
                        ? "command-buffer-pack4-profile-rejected"
                        : "missing-command-buffer-pack4-capability";
                    diagnostics.Add(CreateDiagnostic(request, index, layer, capability, operatorName, code, reason, inputs, true, "Implement and verify a real CommandBuffer Pack4 path, then update the capability matrix."));
                }

                nodes.Add(node);
            }

            var blocking = diagnostics.Any(diagnostic => diagnostic.blocking);
            var strictEligible = !blocking && nodes.All(node => node.accepted && !node.acceptedByDebugOracle);
            var dispatchAllowed = !blocking && (!request.strict || strictEligible || request.debugOracleRelaxed);
            return new AexisTextureExecutionPlan
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

        private static bool TryFindOutputConsumer(
            IReadOnlyList<AexisGraphModel.Layer> layers,
            int producerIndex,
            string[] outputs,
            out string consumedBlob,
            out int consumerIndex,
            out string consumerName)
        {
            consumedBlob = null;
            consumerIndex = -1;
            consumerName = null;
            if (layers == null || outputs == null || outputs.Length == 0)
                return false;

            var outputNames = new HashSet<string>(outputs.Where(name => !string.IsNullOrWhiteSpace(name)), StringComparer.Ordinal);
            if (outputNames.Count == 0)
                return false;
            for (var index = producerIndex + 1; index < layers.Count; index++)
            {
                var layer = layers[index];
                if (layer?.bottomNames == null)
                    continue;
                for (var bottomIndex = 0; bottomIndex < layer.bottomNames.Length; bottomIndex++)
                {
                    var bottom = layer.bottomNames[bottomIndex];
                    if (!outputNames.Contains(bottom))
                        continue;
                    consumedBlob = bottom;
                    consumerIndex = index;
                    consumerName = layer.name ?? layer.typeName ?? string.Empty;
                    return true;
                }
            }
            return false;
        }

        public static void ThrowIfDispatchRejected(AexisTextureExecutionPlan plan)
        {
            if (plan == null)
                throw new ArgumentNullException(nameof(plan));
            if (!plan.dispatchAllowed)
                throw new StrictTextureInferencePlanException(plan);
        }

        public static string ToStableJson(AexisTextureExecutionPlan plan, bool prettyPrint = true)
        {
            if (plan == null)
                throw new ArgumentNullException(nameof(plan));
            return JsonUtility.ToJson(plan, prettyPrint) + "\n";
        }

        public static void WriteStableJson(string path, AexisTextureExecutionPlan plan)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException("An output path is required.", nameof(path));
            var directory = Path.GetDirectoryName(Path.GetFullPath(path));
            if (!string.IsNullOrEmpty(directory))
                Directory.CreateDirectory(directory);
            File.WriteAllText(path, ToStableJson(plan), new System.Text.UTF8Encoding(false));
        }

        private static void PlanInputNode(
            AexisTextureExecutionPlanRequest request,
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanNode node,
            Dictionary<string, AexisTexturePlanTensorDescriptor> descriptors,
            List<AexisTextureExecutionPlanDiagnostic> diagnostics)
        {
            var topNames = layer.topNames ?? Array.Empty<string>();
            var sourceName = topNames.FirstOrDefault(name => descriptors.ContainsKey(name)) ?? layer.name;
            if (string.IsNullOrWhiteSpace(sourceName) || !descriptors.TryGetValue(sourceName, out var source) || !MatchesTarget(source, request))
            {
                diagnostics.Add(CreateDiagnostic(request, node.layerIndex, layer, null, node.operatorName, "missing-pack4-input-descriptor", "Input requires a texture-backed descriptor matching the requested dtype/layout.", Array.Empty<AexisTexturePlanTensorDescriptor>(), true, "Supply a CommandBuffer Pack4 input descriptor for this blob."));
                return;
            }

            if (request.strict && !TryValidateTextureCapacity(source, out var capacityReason))
            {
                diagnostics.Add(CreateDiagnostic(
                    request,
                    node.layerIndex,
                    layer,
                    null,
                    node.operatorName,
                    "texture-descriptor-capacity-exceeded",
                    capacityReason,
                    new[] { source },
                    true,
                    "Use input storage extents that fit the active graphics device."));
                return;
            }

            var outputs = new List<AexisTexturePlanTensorDescriptor>();
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

        private static List<AexisTexturePlanTensorDescriptor> ResolveInputs(
            AexisGraphModel.Layer layer,
            Dictionary<string, AexisTexturePlanTensorDescriptor> descriptors)
        {
            var inputs = new List<AexisTexturePlanTensorDescriptor>();
            var bottomNames = layer?.bottomNames ?? Array.Empty<string>();
            // aten::to in the exported SD graph carries dtype/device/non-blocking metadata
            // after its data input.  Its runtime layer aliases only the first texture and
            // consumes the rest as scalar metadata, so strict descriptor planning must
            // validate precisely that same single data dependency.
            var count = layer != null && layer.type == AexisLayerTypes.AtenTo
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
            AexisTextureExecutionPlanRequest request,
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            out AexisTexturePlanTensorDescriptor[] outputs,
            out string reason)
        {
            outputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
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
                outputLogicalShape = ToShapeArray(new AexisGraphSession.BufferShape(1, ElementCount(input), 1, 1, 1));
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
                    outputLogicalShape = ToShapeArray(AexisGraphSession.ResolveSqueezeShape(input, layer));
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

            outputs = topNames.Select(topName => new AexisTexturePlanTensorDescriptor
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

        private static bool TryResolveReshape(int[] sourceShape, AexisGraphModel.Layer layer, out int[] outputShape, out string reason)
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
                var output = AexisGraphSession.ResolveReshapeShape(source, layer);
                outputShape = ToShapeArray(output);
                return true;
            }
            catch (Exception exception)
            {
                reason = "Reshape shape resolution failed: " + exception.Message;
                return false;
            }
        }

        private static bool HasPack4AliasEvidence(AexisTexturePlanTensorDescriptor source, int[] outputLogicalShape, string operatorName)
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

        private static AexisTexturePlanTensorDescriptor[] CreateComputedOutputs(
            AexisGraphModel.Layer layer,
            AexisTexturePlanTensorDescriptor source,
            AexisTextureExecutionPlanRequest request)
        {
            return (layer.topNames ?? Array.Empty<string>()).Select((topName, index) => new AexisTexturePlanTensorDescriptor
            {
                blob = topName,
                logicalShape = CopyShape(source.logicalShape),
                storageShape = CopyShape(source.storageShape),
                layout = request.targetLayout,
                dtype = request.targetDtype,
                logicalDtype = source.logicalDtype,
                aliasGroup = "computed:" + (layer.name ?? layer.typeName ?? "layer") + ":" + index,
                textureBacked = true
            }).ToArray();
        }

        private static bool RequiresInt8WeightOnlyKernel(string operatorName)
        {
            return string.Equals(operatorName, "Convolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "ConvolutionDepthWise", StringComparison.Ordinal)
                || string.Equals(operatorName, "Gemm", StringComparison.Ordinal)
                || string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal);
        }

        private static bool IsInt8WeightOnlyOperator(AexisTextureExecutionPlanRequest request, string layerName, string operatorName)
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

        private static bool IsInt4WeightOnlyOperator(AexisTextureExecutionPlanRequest request, string layerName, string operatorName)
        {
            if (request == null || !request.int4WeightOnly)
                return false;
            var layerNames = request.int4WeightOnlyLayerNames;
            if (request.int4WeightOnlyLayerSelectionExplicit)
                return layerNames != null && layerNames.Any(value => string.Equals(value, layerName, StringComparison.Ordinal));
            var operators = request.int4WeightOnlyOperators;
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
                || string.Equals(operatorName, "InstanceNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "GroupNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "LayerNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "Scale", StringComparison.Ordinal)
                || string.Equals(operatorName, "PReLU", StringComparison.Ordinal)
                || string.Equals(operatorName, "Normalize", StringComparison.Ordinal);
        }

        private static bool TryAcceptRuntimeVerifiedNode(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            out AexisTexturePlanTensorDescriptor[] outputs,
            out string executionPath,
            out bool usesDescriptorAlias,
            out string reason)
        {
            outputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
            executionPath = null;
            usesDescriptorAlias = false;
            reason = null;
            if (request?.nodeVerifier == null)
            {
                reason = "No loaded-runtime CommandBuffer Pack4 verifier is available.";
                return false;
            }

            AexisTextureExecutionPlanNodeVerification verification;
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
            else if (!IsRealCommandBufferTexturePath(verification.executionPath))
            {
                reason = "The loaded-runtime verifier did not provide a real texture-native CommandBuffer execution path.";
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
            AexisTextureExecutionPlanNodeVerification verification,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs)
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

        private static bool TryResolveExpandDims(int[] sourceShape, AexisGraphModel.Layer layer, out int[] outputShape, out string reason)
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
                    var axis = AexisGraphSession.MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
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

        private static bool HasIdentityTileParameters(AexisGraphModel.Layer layer)
        {
            var repeats = layer.GetInts(-23302, null) ?? layer.GetInts(2, null) ?? layer.GetInts(-23330, null) ?? layer.GetInts(30, null);
            if (repeats != null && repeats.Length > 0)
                return repeats.All(value => value == 1);
            return layer.GetInt(1, 1) == 1;
        }

        private static bool IsRealCommandBufferTexturePath(string executionPath)
        {
            if (string.IsNullOrWhiteSpace(executionPath))
                return false;

            var textureNative = executionPath.StartsWith("command-buffer-pack4", StringComparison.OrdinalIgnoreCase)
                || executionPath.StartsWith("command-buffer-linearmat", StringComparison.OrdinalIgnoreCase);
            if (!textureNative)
                return false;

            return !RejectedComputationPaths.Any(path => executionPath.IndexOf(path, StringComparison.OrdinalIgnoreCase) >= 0);
        }

        private static bool ShapesEqual(int[] left, int[] right)
        {
            return left != null && right != null && left.SequenceEqual(right);
        }

        private static void RegisterOutputs(
            Dictionary<string, AexisTexturePlanTensorDescriptor> descriptors,
            IEnumerable<AexisTexturePlanTensorDescriptor> outputs)
        {
            foreach (var output in outputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
            {
                if (output != null && !string.IsNullOrWhiteSpace(output.blob))
                    descriptors[output.blob] = CloneDescriptor(output, output.blob);
            }
        }

        private static bool MatchesTarget(AexisTexturePlanTensorDescriptor descriptor, AexisTextureExecutionPlanRequest request)
        {
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                && string.Equals(descriptor.dtype, request.targetDtype, StringComparison.OrdinalIgnoreCase)
                && TryToBufferShape(descriptor.logicalShape, out _)
                && TryToBufferShape(descriptor.storageShape, out _);
        }

        private static bool TryValidateTextureCapacities(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> descriptors,
            out string reason)
        {
            for (var index = 0; descriptors != null && index < descriptors.Count; index++)
            {
                if (!TryValidateTextureCapacity(descriptors[index], out reason))
                {
                    reason = "Input " + index + " (" + (descriptors[index]?.blob ?? string.Empty) + ") is invalid: " + reason;
                    return false;
                }
            }

            reason = null;
            return true;
        }

        private static bool TryValidateTextureCapacity(
            AexisTexturePlanTensorDescriptor descriptor,
            out string reason)
        {
            if (descriptor?.logicalShape == null || descriptor.storageShape == null
                || descriptor.logicalShape.Length != 5 || descriptor.storageShape.Length != 5)
            {
                reason = "The texture descriptor does not contain five-component logical and storage shapes.";
                return false;
            }

            var logical = descriptor.logicalShape;
            var storage = descriptor.storageShape;
            long logicalElements = 1;
            for (var axis = 1; axis < logical.Length; axis++)
            {
                if (logical[axis] <= 0 || logicalElements > int.MaxValue / (long)logical[axis])
                {
                    reason = "The logical tensor element count exceeds the supported 32-bit shader descriptor range.";
                    return false;
                }
                logicalElements *= logical[axis];
            }

            var maxTextureSize = GetSystemLimit(() => SystemInfo.maxTextureSize, 16384);
            if (storage[1] > maxTextureSize || storage[2] > maxTextureSize)
            {
                reason = "Texture width/height exceeds SystemInfo.maxTextureSize=" + maxTextureSize + ".";
                return false;
            }

            if (storage[0] >= 3)
            {
                var packs = Math.Max(1L, (storage[4] + 3L) / 4L);
                var slices = storage[0] == 4 ? packs * storage[3] : packs;
                var maxSlices = GetSystemLimit(() => SystemInfo.maxTextureArraySlices, 2048);
                if (slices > maxSlices)
                {
                    reason = "Texture2DArray slices=" + slices + " exceed SystemInfo.maxTextureArraySlices=" + maxSlices + ".";
                    return false;
                }
            }

            reason = null;
            return true;
        }

        private static int GetSystemLimit(Func<int> query, int fallback)
        {
            try
            {
                return Math.Max(1, query());
            }
            catch
            {
                return fallback;
            }
        }

        private static bool IsMaxPoolingIndexInput(
            string operatorName,
            int inputIndex,
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
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
            AexisGraphModel.Layer layer,
            int outputIndex,
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
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

        private static AexisTextureExecutionPlanDiagnostic CreateDiagnostic(
            AexisTextureExecutionPlanRequest request,
            int layerIndex,
            AexisGraphModel.Layer layer,
            AexisOperatorCapability capability,
            string operatorName,
            string code,
            string reason,
            IEnumerable<AexisTexturePlanTensorDescriptor> inputs,
            bool blocking,
            string recommendedAction)
        {
            return new AexisTextureExecutionPlanDiagnostic
            {
                layerIndex = layerIndex,
                layer = layer?.name ?? string.Empty,
                operatorName = operatorName ?? string.Empty,
                canonicalOperator = capability?.canonicalOperator ?? operatorName ?? string.Empty,
                capabilityStatus = capability?.status ?? AexisOperatorCapabilityStatus.Unsupported,
                code = code,
                reason = reason,
                targetBackend = request.targetBackend,
                targetDtype = request.targetDtype,
                targetLayout = request.targetLayout,
                int8WeightOnly = request.int8WeightOnly,
                inputs = (inputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>()).Select(input => input == null ? null : CloneDescriptor(input, input.blob)).ToArray(),
                rejectedPaths = RejectedComputationPaths,
                recommendedAction = recommendedAction,
                blocking = blocking
            };
        }

        private static AexisTexturePlanTensorDescriptor CloneDescriptor(AexisTexturePlanTensorDescriptor descriptor, string blob)
        {
            return new AexisTexturePlanTensorDescriptor
            {
                blob = blob ?? descriptor?.blob ?? string.Empty,
                logicalShape = CopyShape(descriptor?.logicalShape),
                storageShape = CopyShape(descriptor?.storageShape),
                sourceLogicalShape = CopyShape(descriptor?.sourceLogicalShape),
                layout = descriptor?.layout ?? string.Empty,
                dtype = descriptor?.dtype ?? string.Empty,
                logicalDtype = descriptor?.logicalDtype ?? string.Empty,
                aliasGroup = descriptor?.aliasGroup ?? string.Empty,
                textureBacked = descriptor != null && descriptor.textureBacked
            };
        }

        private static bool TryToBufferShape(int[] shape, out AexisGraphSession.BufferShape value)
        {
            value = default;
            if (shape == null || shape.Length != 5 || shape[0] < 1 || shape[0] > 4
                || shape[1] <= 0 || shape[2] <= 0 || shape[3] <= 0 || shape[4] <= 0)
                return false;
            value = new AexisGraphSession.BufferShape(shape[0], shape[1], shape[2], shape[3], shape[4]);
            return true;
        }

        private static int[] ToShapeArray(AexisGraphSession.BufferShape shape)
        {
            return new[] { shape.dims, shape.w, shape.h, shape.d, shape.c };
        }

        private static int[] CopyShape(int[] shape)
        {
            return shape == null ? Array.Empty<int>() : shape.ToArray();
        }

        private static bool ShapesEqual(AexisGraphSession.BufferShape a, AexisGraphSession.BufferShape b)
        {
            return a.dims == b.dims && a.w == b.w && a.h == b.h && a.d == b.d && a.c == b.c;
        }

        private static int ElementCount(AexisGraphSession.BufferShape shape)
        {
            return shape.w * shape.h * shape.d * shape.c;
        }
    }
}
