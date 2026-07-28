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
        // True only for a ComputeBuffer that has already been uploaded into an
        // exact RFloat LinearMat texture at the graph boundary. This is never a
        // Buffer activation; it makes the exceptional exact index storage explicit.
        public bool fixedInputUpload;
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
        // Kernel-local Pack4 RTs must be declared by the loaded-runtime verifier.
        // They participate in the same static liveness/arena proof as graph outputs.
        public AexisTexturePlanTensorDescriptor[] scratch = Array.Empty<AexisTexturePlanTensorDescriptor>();
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
        // Model-bound FP32 Pack4 activation producers inside an otherwise FP16
        // profile. This mirrors the loaded runtime's precision-sensitive and
        // explicitly configured activation decisions; it is not a fallback or
        // a second backend.
        public string[] fp32ActivationLayerNames = Array.Empty<string>();
        // Optional execution boundary for a graph prefix. Layers after the first
        // producer of this blob are not dispatched and must not block a strict plan.
        // An unknown blob deliberately preserves complete-graph validation.
        public string stopAfterTopName;
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
        public AexisTexturePlanTensorDescriptor[] scratch;
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
    public sealed class AexisTextureExecutionPlanResource
    {
        // A resource represents one physical Pack4 Texture2DArray allocation.  View
        // nodes retain the source alias group and therefore never add a second resource.
        public string aliasGroup;
        public string representativeBlob;
        public AexisTexturePlanTensorDescriptor descriptor;
        public int firstLayerIndex;
        public int lastLayerIndex;
        public bool persistent;
        public bool temporary;
        // Persistent graph outputs are allocated by the compiled arena but are
        // handed to the CommandBuffer result owner rather than released at the
        // graph boundary. External inputs are never arena-owned.
        public bool producedByGraph;
        // Caller-supplied textures can be carried through descriptor aliases, but
        // their storage must never acquire a CommandBuffer arena allocation.
        public bool externalInput;
        public bool scratch;
        // Temporary resources with identical storage descriptors may share this slot
        // when their inclusive liveness ranges do not overlap.
        public int allocationSlot;
        public long estimatedBytes;
    }

    [Serializable]
    public sealed class AexisTextureExecutionPlanMemory
    {
        public AexisTextureExecutionPlanResource[] resources;
        public long peakLiveBytes;
        public int peakLiveLayerIndex;
        public long persistentBytes;
        public long temporaryArenaBytes;
        public long totalArenaBytes;
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
        // The memory plan is metadata only. Runtime allocation remains texture-native
        // through AexisGraphSession's CommandBuffer temporary RT pool.
        public AexisTextureExecutionPlanMemory memory;
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
                + " | reason=" + (first.reason ?? string.Empty)
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
        public const int SchemaVersion = 2;
        public const string Contract = "aiimage.strict-texture-execution-plan/v2";

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
            var plannedLayerCount = ResolvePlannedLayerCount(layers, request.stopAfterTopName);
            for (var index = 0; index < plannedLayerCount; index++)
            {
                var layer = layers[index];
                if (layer == null)
                {
                    nodes.Add(new AexisTextureExecutionPlanNode { layerIndex = index, accepted = false, executionPath = "invalid" });
                    diagnostics.Add(CreateDiagnostic(request, index, null, null, null, "null-layer", "The model contains a null layer.", Array.Empty<AexisTexturePlanTensorDescriptor>(), true, "Re-export the model graph."));
                    continue;
                }

                var operatorName = ResolveCapabilityOperatorName(layer);
                AexisOperatorCapabilities.TryGet(operatorName, out var capability);
                var inputs = ResolveInputs(layer, descriptors);
                // The Qwen decoder declares all KV-cache slots as one graph Input,
                // while its first SDPA invocation deliberately has no past K/V
                // textures.  This removes only that proven absent pair so the
                // runtime verifier can admit its real GPU cache-initialization path.
                TrimOptionalInitialSdpaKvCacheInputs(layer, inputs);
                var node = new AexisTextureExecutionPlanNode
                {
                    layerIndex = index,
                    layer = layer.name ?? string.Empty,
                    operatorName = operatorName,
                    canonicalOperator = capability?.canonicalOperator ?? operatorName,
                    capabilityStatus = capability?.status ?? AexisOperatorCapabilityStatus.Unsupported,
                    inputs = inputs.Select(input => input == null ? null : CloneDescriptor(input, input.blob)).ToArray(),
                    outputs = Array.Empty<AexisTexturePlanTensorDescriptor>(),
                    scratch = Array.Empty<AexisTexturePlanTensorDescriptor>(),
                    executionPath = "rejected"
                };

                if (string.Equals(operatorName, "Input", StringComparison.Ordinal))
                {
                    PlanInputNode(request, layer, layers, node, descriptors, diagnostics);
                    nodes.Add(node);
                    continue;
                }

                if (request.strict
                    && (string.Equals(operatorName, "NonZero", StringComparison.Ordinal)
                        || string.Equals(operatorName, "Compress", StringComparison.Ordinal)
                        || string.Equals(operatorName, "Nms", StringComparison.Ordinal)
                        || string.Equals(operatorName, "NonMaxSuppression", StringComparison.Ordinal))
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
                    MatchesTarget(input, request)
                    || IsMaxPoolingIndexInput(operatorName, inputIndex, input, request)
                    || IsFixedTextureInputUpload(operatorName, inputIndex, input, request)
                    || IsQwenFp32AttentionKvCacheInput(layer, operatorName, inputIndex, input, request)
                    || IsVerifiedFp32ActivationIslandTexture(layer, input, request)
                    || IsVerifiedLinearMatTexture(input, request)).All(value => value);
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
                // Capability profiles describe the model's requested activation
                // contract. A descriptor from the explicit FP32 Pack4 set is a
                // physical-storage promotion inside that same FP16 contract. Use
                // a normalized copy only for profile selection; the real runtime
                // verifier below still receives the unmodified FP32 descriptor.
                var profileInputs = inputs.Select(input =>
                {
                    if (!IsVerifiedFp32ActivationIslandTexture(layer, input, request))
                        return input;
                    var normalized = CloneDescriptor(input, input.blob);
                    normalized.dtype = ResolvePhysicalTextureDtype(request.targetDtype);
                    return normalized;
                }).ToArray();
                var profileMatchReason = string.Empty;
                var profileTargetCompatible = isConditionalCapability
                    && (!requiresInt8WeightKernel || capability.int8)
                    && AexisOperatorCapabilities.TryMatchTextureProfile(
                        capability,
                        layer,
                        request.targetBackend,
                        request.targetDtype,
                        request.targetLayout,
                        profileInputs,
                        out _,
                        out profileMatchReason);
                var verifiedOutputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
                var verifiedScratch = Array.Empty<AexisTexturePlanTensorDescriptor>();
                var verifiedPath = string.Empty;
                var verificationReason = string.Empty;
                var verifiedUsesDescriptorAlias = false;
                var profileVerified = !strictCapability
                    && inputsMatchTarget
                    && profileTargetCompatible
                    && TryAcceptRuntimeVerifiedNode(layer, inputs, request, out verifiedOutputs, out verifiedScratch, out verifiedPath, out verifiedUsesDescriptorAlias, out verificationReason);
                if (strictCapability && inputsMatchTarget)
                {
                    var outputs = CreateComputedOutputs(layer, inputs[0], request);
                    if (!TryValidateTextureCapacities(outputs, out var outputCapacityReason))
                    {
                        diagnostics.Add(CreateDiagnostic(
                            request,
                            index,
                            layer,
                            capability,
                            operatorName,
                            "output-texture-descriptor-capacity-exceeded",
                            outputCapacityReason,
                            outputs,
                            true,
                            "Use an output shape whose Pack4 texture-array storage fits the active graphics device."));
                        nodes.Add(node);
                        continue;
                    }
                    node.accepted = true;
                    node.executionPath = "command-buffer-pack4";
                    node.outputs = outputs;
                    RegisterOutputs(descriptors, outputs);
                }
                else if (profileVerified)
                {
                    if (!TryValidateTextureCapacities(verifiedOutputs, out var outputCapacityReason))
                    {
                        diagnostics.Add(CreateDiagnostic(
                            request,
                            index,
                            layer,
                            capability,
                            operatorName,
                            "output-texture-descriptor-capacity-exceeded",
                            outputCapacityReason,
                            verifiedOutputs,
                            true,
                            "Use an output shape whose Pack4 texture-array storage fits the active graphics device."));
                        nodes.Add(node);
                        continue;
                    }
                    node.accepted = true;
                    node.usesDescriptorAlias = verifiedUsesDescriptorAlias;
                    node.executionPath = verifiedPath;
                    node.outputs = verifiedOutputs;
                    node.scratch = verifiedScratch;
                    RegisterOutputs(descriptors, verifiedOutputs);
                }
                else if (request.debugOracleRelaxed
                    && capability != null
                    && !string.Equals(capability.status, AexisOperatorCapabilityStatus.Unsupported, StringComparison.Ordinal)
                    && inputs.Count > 0)
                {
                    var outputs = CreateComputedOutputs(layer, inputs[0], request);
                    if (!TryValidateTextureCapacities(outputs, out var outputCapacityReason))
                    {
                        diagnostics.Add(CreateDiagnostic(
                            request,
                            index,
                            layer,
                            capability,
                            operatorName,
                            "output-texture-descriptor-capacity-exceeded",
                            outputCapacityReason,
                            outputs,
                            true,
                            "Use an output shape whose Pack4 texture-array storage fits the active graphics device."));
                        nodes.Add(node);
                        continue;
                    }
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
            var memory = BuildMemoryPlan(nodes);
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
                memory = memory,
                summary = "nodes=" + nodes.Count + " | diagnostics=" + diagnostics.Count + " | int8_weight_only=" + request.int8WeightOnly + " | strict_eligible=" + strictEligible + " | dispatch_allowed=" + dispatchAllowed + " | peak_rt_bytes=" + memory.peakLiveBytes + " | arena_rt_bytes=" + memory.totalArenaBytes
            };
        }

        private sealed class ResourceDraft
        {
            public AexisTextureExecutionPlanResource resource;
            public int producerLayerIndex = -1;
            // An input first observed at the graph boundary remains caller-owned
            // even if a later descriptor-alias output retains its alias group.
            public bool externalInput;
            public bool forceTemporary;
            // A terminal descriptor-alias does not allocate another texture, but
            // it does transfer ownership of its source physical texture to the
            // graph result. Keep that one alias group alive past execution.
            public bool forcePersistent;
        }

        private sealed class TemporaryArenaSlot
        {
            public int slot;
            public string signature;
            public int lastLayerIndex;
            public long bytes;
        }

        private static AexisTextureExecutionPlanMemory BuildMemoryPlan(IReadOnlyList<AexisTextureExecutionPlanNode> nodes)
        {
            var drafts = new Dictionary<string, ResourceDraft>(StringComparer.Ordinal);
            var lastInputUse = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var nodeIndex = 0; nodeIndex < (nodes?.Count ?? 0); nodeIndex++)
            {
                var node = nodes[nodeIndex];
                if (node == null || !node.accepted)
                    continue;
                foreach (var input in node.inputs ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
                {
                    if (input != null && !string.IsNullOrWhiteSpace(input.blob))
                        lastInputUse[input.blob] = nodeIndex;
                }
            }
            for (var nodeIndex = 0; nodeIndex < (nodes?.Count ?? 0); nodeIndex++)
            {
                var node = nodes[nodeIndex];
                if (node == null || !node.accepted)
                    continue;
                TouchResources(drafts, node.inputs, nodeIndex, false);
                // Descriptor aliases retain the producer's Pack4 texture and only
                // extend its liveness. Treating an alias as a new graph-produced
                // RT makes the arena demand a bind that no CommandBuffer layer can
                // perform, even though no allocation is semantically valid.
                TouchResources(
                    drafts,
                    node.outputs,
                    nodeIndex,
                    !node.usesDescriptorAlias,
                    isScratch: false,
                    isTerminalOutput: descriptor => descriptor != null
                        && (!lastInputUse.TryGetValue(descriptor.blob ?? string.Empty, out var lastUse)
                            || lastUse <= nodeIndex));
                TouchResources(drafts, node.scratch, nodeIndex, true, true);
            }

            var resources = drafts.Values
                .Select(draft => FinalizeResource(draft))
                .OrderBy(resource => resource.firstLayerIndex)
                .ThenBy(resource => resource.aliasGroup, StringComparer.Ordinal)
                .ToArray();
            AssignStaticTemporarySlots(resources);

            long peakLiveBytes = 0;
            var peakLiveLayerIndex = -1;
            var maxLayerIndex = resources.Length == 0 ? -1 : resources.Max(resource => resource.lastLayerIndex);
            for (var layerIndex = 0; layerIndex <= maxLayerIndex; layerIndex++)
            {
                long liveBytes = 0;
                foreach (var resource in resources)
                {
                    if (resource.firstLayerIndex <= layerIndex && resource.lastLayerIndex >= layerIndex)
                        liveBytes = checked(liveBytes + resource.estimatedBytes);
                }
                if (liveBytes > peakLiveBytes)
                {
                    peakLiveBytes = liveBytes;
                    peakLiveLayerIndex = layerIndex;
                }
            }

            var persistentBytes = resources
                .Where(resource => resource.persistent && !resource.externalInput)
                .Sum(resource => resource.estimatedBytes);
            var temporaryArenaBytes = resources
                .Where(resource => resource.temporary)
                .GroupBy(resource => resource.allocationSlot)
                .Sum(slot => slot.Max(resource => resource.estimatedBytes));
            return new AexisTextureExecutionPlanMemory
            {
                resources = resources,
                peakLiveBytes = peakLiveBytes,
                peakLiveLayerIndex = peakLiveLayerIndex,
                persistentBytes = persistentBytes,
                temporaryArenaBytes = temporaryArenaBytes,
                totalArenaBytes = checked(persistentBytes + temporaryArenaBytes)
            };
        }

        private static void TouchResources(
            Dictionary<string, ResourceDraft> drafts,
            IEnumerable<AexisTexturePlanTensorDescriptor> descriptors,
            int layerIndex,
            bool isOutput,
            bool isScratch = false,
            Func<AexisTexturePlanTensorDescriptor, bool> isTerminalOutput = null)
        {
            foreach (var descriptor in descriptors ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
            {
                if (descriptor == null || !descriptor.textureBacked)
                    continue;
                var aliasGroup = string.IsNullOrWhiteSpace(descriptor.aliasGroup)
                    ? "blob:" + (descriptor.blob ?? string.Empty)
                    : descriptor.aliasGroup;
                if (!drafts.TryGetValue(aliasGroup, out var draft))
                {
                    draft = new ResourceDraft
                    {
                        resource = new AexisTextureExecutionPlanResource
                        {
                            aliasGroup = aliasGroup,
                            representativeBlob = descriptor.blob ?? string.Empty,
                            descriptor = CloneDescriptor(descriptor, descriptor.blob),
                            firstLayerIndex = layerIndex,
                            lastLayerIndex = layerIndex,
                            allocationSlot = -1,
                            // Initialise this at first sight so a later alias of
                            // identical physical storage cannot replace the
                            // producer blob used by allocation diagnostics.
                            estimatedBytes = EstimateTextureBytes(descriptor)
                        }
                    };
                    drafts.Add(aliasGroup, draft);
                }
                else
                {
                    draft.resource.firstLayerIndex = Math.Min(draft.resource.firstLayerIndex, layerIndex);
                    draft.resource.lastLayerIndex = Math.Max(draft.resource.lastLayerIndex, layerIndex);
                    var descriptorBytes = EstimateTextureBytes(descriptor);
                    if (descriptorBytes > draft.resource.estimatedBytes)
                    {
                        draft.resource.descriptor = CloneDescriptor(descriptor, descriptor.blob);
                        draft.resource.representativeBlob = descriptor.blob ?? draft.resource.representativeBlob;
                    }
                }
                if (!isOutput && draft.producerLayerIndex < 0)
                    draft.externalInput = true;
                if (isOutput && draft.producerLayerIndex < 0)
                    draft.producerLayerIndex = layerIndex;
                if (isTerminalOutput?.Invoke(descriptor) == true)
                    draft.forcePersistent = true;
                if (isScratch)
                    draft.forceTemporary = true;
                if (isScratch)
                    draft.resource.scratch = true;
            }
        }

        private static AexisTextureExecutionPlanResource FinalizeResource(ResourceDraft draft)
        {
            var resource = draft.resource;
            resource.estimatedBytes = EstimateTextureBytes(resource.descriptor);
            resource.producedByGraph = draft.producerLayerIndex >= 0;
            resource.externalInput = draft.externalInput;
            // Inputs, KV cache slots, and terminal graph values stay alive beyond this
            // CommandBuffer. Every other descriptor is a temporary Pack4 RT.
            resource.persistent = !draft.forceTemporary && (draft.forcePersistent
                || draft.producerLayerIndex < 0
                || resource.aliasGroup.StartsWith("input:", StringComparison.Ordinal)
                || resource.representativeBlob.StartsWith("cache_", StringComparison.Ordinal)
                || (draft.producerLayerIndex >= 0 && resource.lastLayerIndex == draft.producerLayerIndex));
            resource.temporary = !resource.persistent;
            return resource;
        }

        private static void AssignStaticTemporarySlots(IReadOnlyList<AexisTextureExecutionPlanResource> resources)
        {
            var slots = new List<TemporaryArenaSlot>();
            var nextSlot = 0;
            foreach (var resource in resources)
            {
                if (resource.externalInput)
                {
                    resource.allocationSlot = -1;
                    continue;
                }
                if (resource.persistent)
                {
                    resource.allocationSlot = nextSlot++;
                    continue;
                }

                var signature = StorageSignature(resource.descriptor);
                var reusable = slots
                    .Where(slot => slot.lastLayerIndex < resource.firstLayerIndex
                        && string.Equals(slot.signature, signature, StringComparison.Ordinal)
                        && slot.bytes >= resource.estimatedBytes)
                    .OrderBy(slot => slot.bytes)
                    .FirstOrDefault();
                if (reusable == null)
                {
                    reusable = new TemporaryArenaSlot
                    {
                        slot = nextSlot++,
                        signature = signature,
                        bytes = resource.estimatedBytes
                    };
                    slots.Add(reusable);
                }
                reusable.lastLayerIndex = resource.lastLayerIndex;
                resource.allocationSlot = reusable.slot;
            }
        }

        private static string StorageSignature(AexisTexturePlanTensorDescriptor descriptor)
        {
            var storage = descriptor?.storageShape ?? Array.Empty<int>();
            return (descriptor?.dtype ?? string.Empty) + "|" + (descriptor?.layout ?? string.Empty) + "|" + string.Join(",", storage);
        }

        private static long EstimateTextureBytes(AexisTexturePlanTensorDescriptor descriptor)
        {
            if (!TryToBufferShape(descriptor?.storageShape, out var storage))
                return 0;
            var packs = Math.Max(1, (storage.c + 3) / 4);
            var bytesPerTexel = string.Equals(descriptor.dtype, "FP16", StringComparison.OrdinalIgnoreCase) ? 8L : 16L;
            try
            {
                return checked((long)storage.w * storage.h * storage.d * packs * bytesPerTexel);
            }
            catch (OverflowException)
            {
                return long.MaxValue;
            }
        }

        private static int ResolvePlannedLayerCount(
            IReadOnlyList<AexisGraphModel.Layer> layers,
            string stopAfterTopName)
        {
            if (layers == null || layers.Count == 0 || string.IsNullOrWhiteSpace(stopAfterTopName))
                return layers?.Count ?? 0;

            for (var index = 0; index < layers.Count; index++)
            {
                var topNames = layers[index]?.topNames;
                if (topNames != null && topNames.Contains(stopAfterTopName))
                    return index + 1;
            }

            // Preserve whole-model validation when the requested boundary is invalid.
            return layers.Count;
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
            IReadOnlyList<AexisGraphModel.Layer> layers,
            AexisTextureExecutionPlanNode node,
            Dictionary<string, AexisTexturePlanTensorDescriptor> descriptors,
            List<AexisTextureExecutionPlanDiagnostic> diagnostics)
        {
            var topNames = layer.topNames ?? Array.Empty<string>();
            if (IsQwenKvCacheInputGroup(layer))
            {
                // Do not synthesize zero textures or CPU-side cache state.  SDPA
                // receives no past K/V on its initial step and creates both cache
                // textures with its native Pack4 GPU dispatch.
                node.accepted = true;
                node.executionPath = "gpu-kv-cache-initial-state";
                node.outputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
                return;
            }

            var sourceName = topNames.FirstOrDefault(name => descriptors.ContainsKey(name)) ?? layer.name;
            // Exact integer indices may originate either from the narrowly allowed
            // fixed-buffer upload or from a caller-provided RFloat texture boundary.
            // In both cases Embed sees texture-native FP32 LinearMat storage; do not
            // force a native texture input back through a ComputeBuffer just to admit
            // an FP16 graph plan.
            var isExactTextureIndex = descriptors.TryGetValue(sourceName ?? string.Empty, out var source)
                && string.Equals(source.logicalDtype, "Int32", StringComparison.Ordinal)
                && source.aliasGroup != null
                && source.aliasGroup.StartsWith("input:", StringComparison.Ordinal)
                && IsVerifiedLinearMatTexture(source, request);
            // Qwen's vision patch embed has one verified FP32 LinearMat image
            // boundary before its Convolution3D Pack4 path.  Do not turn that
            // narrowly-scoped native texture intake into a general FP32
            // activation exception for an FP16 graph: token/index inputs must
            // remain Int32 and use the Embed-specific proof above.
            var isVerifiedLinearMatInput = IsVerifiedNativeFp32ImageInput(
                source,
                layer,
                layers,
                request);
            if (string.IsNullOrWhiteSpace(sourceName)
                || source == null
                || (!MatchesTarget(source, request) && !isExactTextureIndex && !isVerifiedLinearMatInput))
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
            // Input publishes the caller-owned Pack4 texture under the graph
            // blob name. It is a descriptor alias, never a graph RT allocation.
            node.usesDescriptorAlias = true;
            node.executionPath = "external-pack4-input";
            node.outputs = outputs.ToArray();
            RegisterOutputs(descriptors, node.outputs);
        }

        private static bool IsVerifiedNativeFp32ImageInput(
            AexisTexturePlanTensorDescriptor descriptor,
            AexisGraphModel.Layer inputLayer,
            IReadOnlyList<AexisGraphModel.Layer> layers,
            AexisTextureExecutionPlanRequest request)
        {
            if (!IsVerifiedLinearMatTexture(descriptor, request)
                || !string.Equals(descriptor.logicalDtype, "Float32", StringComparison.Ordinal)
                || inputLayer?.topNames == null
                || layers == null)
            {
                return false;
            }

            // The native FP32 image boundary is intentionally limited to the
            // Convolution3D patch embedding profile.  Every other FP32 input
            // must match the requested activation dtype, while Embed token ids
            // are admitted only by the exact Int32 index contract above.
            foreach (var topName in inputLayer.topNames)
            {
                if (string.IsNullOrWhiteSpace(topName))
                    continue;
                foreach (var candidate in layers)
                {
                    if (candidate?.bottomNames == null || !candidate.bottomNames.Contains(topName))
                        continue;
                    var operatorName = string.IsNullOrWhiteSpace(candidate.typeName)
                        ? candidate.type.ToString()
                        : candidate.typeName;
                    if (string.Equals(operatorName, "Convolution3D", StringComparison.Ordinal))
                        return true;
                }
            }

            return false;
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

        private static void TrimOptionalInitialSdpaKvCacheInputs(
            AexisGraphModel.Layer layer,
            List<AexisTexturePlanTensorDescriptor> inputs)
        {
            if (!IsKvCacheSdpaLayer(layer)
                || inputs == null
                || inputs.Count != 6
                || inputs[4] != null
                || inputs[5] != null)
                return;

            // Qwen's exported KV-cache SDPA always carries an attention mask, so
            // Q/K/V/mask are the four mandatory descriptors.  Never trim a missing
            // activation or a partial cache pair.
            for (var index = 0; index < 4; index++)
            {
                if (inputs[index] == null)
                    return;
            }

            inputs.RemoveRange(4, 2);
        }

        private static bool IsQwenKvCacheInputGroup(AexisGraphModel.Layer inputLayer)
        {
            // pnnx emits this logical group, but Qwen's decoder session binds the
            // individual cache_kN/cache_vN textures directly. The group name itself
            // never has storage, on either the initial step or later decode steps.
            // Individual SDPA cache pair validation remains strict below.
            return inputLayer?.name?.IndexOf("kv_cache", StringComparison.Ordinal) >= 0;
        }

        private static bool IsKvCacheSdpaLayer(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (!string.Equals(operatorName, "SDPA", StringComparison.Ordinal)
                || layer.bottomNames == null
                || layer.bottomNames.Length != 6)
            {
                return false;
            }

            // The Qwen export represents past attention state with six independent
            // cache pairs. This is deliberately keyed to the two actual SDPA input
            // names rather than exporter-specific parameter ids or output aliases.
            // Every other six-input SDPA node still requires descriptors for each
            // input.
            var key = layer.bottomNames[4];
            var value = layer.bottomNames[5];
            return !string.IsNullOrWhiteSpace(key)
                && key.StartsWith("cache_k", StringComparison.Ordinal)
                && !string.IsNullOrWhiteSpace(value)
                && value.StartsWith("cache_v", StringComparison.Ordinal)
                && string.Equals(key.Substring("cache_k".Length), value.Substring("cache_v".Length), StringComparison.Ordinal);
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
            // LinearMat tensors use RFloat texture storage even in FP16 graphs.
            // A view only carries the same native texture descriptor forward, so
            // accept that verified storage contract without materializing it.
            if ((!MatchesTarget(source, request)
                    && !IsVerifiedFp32ActivationIslandTexture(layer, source, request)
                    && !IsVerifiedLinearMatTexture(source, request))
                || string.IsNullOrWhiteSpace(source.aliasGroup))
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
                dtype = ResolvePhysicalTextureDtype(request.targetDtype),
                logicalDtype = string.Equals(request.targetDtype, "BF16", StringComparison.OrdinalIgnoreCase) ? "BFloat16" : source.logicalDtype,
                // Blob names are graph-unique; layer names are not guaranteed to be.
                // Keying allocations by the latter would incorrectly alias adjacent
                // anonymous/repeated layers and under-report the Pack4 RT arena.
                aliasGroup = "computed:" + (topName ?? (layer.name ?? layer.typeName ?? "layer") + ":" + index),
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
            out AexisTexturePlanTensorDescriptor[] scratch,
            out string executionPath,
            out bool usesDescriptorAlias,
            out string reason)
        {
            outputs = Array.Empty<AexisTexturePlanTensorDescriptor>();
            scratch = Array.Empty<AexisTexturePlanTensorDescriptor>();
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
                    || (!MatchesTarget(output, request)
                        && !IsMaxPoolingIndexOutput(layer, index, output, request)
                        && !IsVerifiedFp32ActivationIslandTexture(layer, output, request)
                        && !IsVerifiedLinearMatTexture(output, request))
                    || !string.Equals(output.blob, topNames[index], StringComparison.Ordinal))
                {
                    reason = "The loaded-runtime verifier produced an output descriptor outside the requested target contract.";
                    return false;
                }
            }

            foreach (var descriptor in verification.scratch ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
            {
                if (!IsValidDeclaredScratch(descriptor, request))
                {
                    reason = "The loaded-runtime verifier declared scratch outside the Pack4 texture-only arena contract.";
                    return false;
                }
            }

            outputs = verification.outputs.Select(output => CloneDescriptor(output, output.blob)).ToArray();
            scratch = (verification.scratch ?? Array.Empty<AexisTexturePlanTensorDescriptor>())
                .Select(descriptor => CloneDescriptor(descriptor, descriptor.blob))
                .ToArray();
            executionPath = verification.executionPath;
            usesDescriptorAlias = verification.usesDescriptorAlias;
            return true;
        }

        private static bool IsValidDeclaredScratch(
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
        {
            if (descriptor == null || !descriptor.textureBacked
                || string.IsNullOrWhiteSpace(descriptor.blob)
                || string.IsNullOrWhiteSpace(descriptor.aliasGroup)
                || !descriptor.aliasGroup.StartsWith("scratch:", StringComparison.Ordinal)
                || !TryToBufferShape(descriptor.logicalShape, out var logical)
                || !TryToBufferShape(descriptor.storageShape, out var storage)
                || logical.dims < 1 || logical.dims > 4
                || storage.dims < 1 || storage.dims > 4
                || !string.Equals(descriptor.layout, request?.targetLayout, StringComparison.OrdinalIgnoreCase)
                || (!string.Equals(descriptor.dtype, "FP16", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)))
            {
                return false;
            }
            return TryValidateTextureCapacities(new[] { descriptor }, out _);
        }

        private static bool HasRuntimeAliasEvidence(
            AexisTextureExecutionPlanNodeVerification verification,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs)
        {
            if (!string.Equals(verification.executionPath, "descriptor-alias", StringComparison.Ordinal)
                // Some texture-native noops (notably Interp size_expr=1w,1h)
                // consume optional descriptor-only shape references. The first
                // input remains the sole aliased activation; the extra inputs
                // are never activation data sources or CPU readback targets.
                || inputs == null || inputs.Count < 1 || inputs[0] == null
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
                && string.Equals(descriptor.dtype, ResolvePhysicalTextureDtype(request.targetDtype), StringComparison.OrdinalIgnoreCase)
                && TryToBufferShape(descriptor.logicalShape, out _)
                && TryToBufferShape(descriptor.storageShape, out _);
        }

        private static bool IsVerifiedFp32ActivationIslandTexture(
            AexisGraphModel.Layer consumer,
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
        {
            if (descriptor == null
                || !descriptor.textureBacked
                || !string.Equals(request?.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(descriptor.layout, request?.targetLayout, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !TryToBufferShape(descriptor.logicalShape, out _)
                || !TryToBufferShape(descriptor.storageShape, out var storage)
                || storage.dims < 3
                || string.IsNullOrWhiteSpace(descriptor.aliasGroup))
            {
                return false;
            }

            // The FP32 activation must be produced by a named member of this
            // exact model island. External FP32 textures remain rejected.
            foreach (var producer in request.fp32ActivationLayerNames ?? Array.Empty<string>())
            {
                if (!string.IsNullOrWhiteSpace(producer)
                    && descriptor.aliasGroup.StartsWith("computed:" + producer + ":", StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        private static bool IsFixedTextureInputUpload(
            string operatorName,
            int inputIndex,
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
        {
            // An index buffer is uploaded before model execution and Embed samples
            // the resulting texture. Do not generalize this exception to FP32
            // activations or to arbitrary operators in an FP16 graph.
            return descriptor != null
                && descriptor.fixedInputUpload
                && string.Equals(operatorName, "Embed", StringComparison.Ordinal)
                && inputIndex == 0
                && string.Equals(descriptor.logicalDtype, "Int32", StringComparison.Ordinal)
                && descriptor.aliasGroup != null
                && descriptor.aliasGroup.StartsWith("input:", StringComparison.Ordinal)
                && IsVerifiedLinearMatTexture(descriptor, request);
        }

        private static bool IsVerifiedLinearMatTexture(
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
        {
            if (descriptor == null
                || !descriptor.textureBacked
                || !string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !TryToBufferShape(descriptor.logicalShape, out var logical)
                || !TryToBufferShape(descriptor.storageShape, out var storage))
            {
                return false;
            }

            var expected = AexisGraphSession.ResolveLinearMatStorageShape(logical);
            return storage.dims == expected.dims
                && storage.w == expected.w
                && storage.h == expected.h
                && storage.d == expected.d
                && storage.c == expected.c;
        }

        private static bool IsQwenFp32AttentionKvCacheInput(
            AexisGraphModel.Layer layer,
            string operatorName,
            int inputIndex,
            AexisTexturePlanTensorDescriptor descriptor,
            AexisTextureExecutionPlanRequest request)
        {
            // Qwen retains the output of its cache-producing SDPA nodes as FP32 for
            // numerical stability, while Q4's ordinary activations remain FP16.
            // This is a persistent Texture2DArray cache consumed by the same SDPA
            // texture kernel, not a promoted activation or a buffer fallback.
            if (!string.Equals(operatorName, "SDPA", StringComparison.Ordinal)
                || !string.Equals(request?.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                || inputIndex < 4 || inputIndex > 5
                || layer?.bottomNames == null || layer.bottomNames.Length <= inputIndex
                || descriptor == null || !descriptor.textureBacked
                || !string.Equals(descriptor.dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(descriptor.logicalDtype, "Float32", StringComparison.Ordinal)
                || !string.Equals(descriptor.layout, request.targetLayout, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(descriptor.blob, layer.bottomNames[inputIndex], StringComparison.Ordinal)
                || !TryToBufferShape(descriptor.logicalShape, out var logical)
                || !TryToBufferShape(descriptor.storageShape, out var storage))
            {
                return false;
            }

            var expectedPrefix = inputIndex == 4 ? "cache_k" : "cache_v";
            return descriptor.blob.StartsWith(expectedPrefix, StringComparison.Ordinal)
                && logical.dims == 3
                && logical.w > 0 && logical.h > 0 && logical.d == 1 && logical.c > 0
                && storage.dims == 3
                && storage.w == logical.w
                && storage.h >= logical.h
                && storage.d == 1
                && storage.c == logical.c;
        }

        private static string ResolvePhysicalTextureDtype(string targetDtype)
        {
            // Unity has no portable BF16 RenderTexture. The runtime uses FP32
            // texture storage and records BFloat16 as the logical dtype.
            return string.Equals(targetDtype, "BF16", StringComparison.OrdinalIgnoreCase) ? "FP32" : targetDtype;
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

        private static string ResolveCapabilityOperatorName(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (!string.Equals(operatorName, "RandomLike", StringComparison.Ordinal)
                || layer?.stringParams == null
                || !layer.stringParams.TryGetValue("aexis.random.operator", out var randomOperator)
                || string.IsNullOrWhiteSpace(randomOperator))
                return operatorName;

            switch (randomOperator)
            {
                case "RandomUniform":
                case "RandomNormal":
                case "RandomUniformLike":
                case "RandomNormalLike":
                case "Bernoulli":
                    return randomOperator;
                default:
                    return operatorName;
            }
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
                fixedInputUpload = descriptor != null && descriptor.fixedInputUpload,
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
