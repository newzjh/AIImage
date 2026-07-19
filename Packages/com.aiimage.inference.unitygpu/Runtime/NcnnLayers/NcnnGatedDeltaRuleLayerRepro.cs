using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    /// Texture-only FP32 recurrent Gated Delta Rule.  Prefill dispatches one
    /// token at a time so the state transition is deterministic and auditable.
    public sealed class NcnnGatedDeltaRuleLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnGatedDeltaRuleLayerRepro()
            : base(NcnnLayerTypes.GatedDeltaRule, supportsBufferPath: false, supportsCommandBufferPath: true)
        {
        }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            owner._extraPacks[layer.name] = new NcnnRepro.GatedDeltaRulePack
            {
                epsilon = 1e-6f
            };
            return default;
        }

        public override void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            // InferWithMultiInputs uses this dispatch entry for both storage modes.
            // Route it to the strict texture implementation; there is no buffer kernel.
            ExecuteRenderTexturePath(owner, layer, context);
        }

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length < 8 || layer.topNames == null || layer.topNames.Length < 2)
                throw new InvalidOperationException("GatedDeltaRule requires eight inputs and two outputs: " + layer.name);
            var alog = RequireScalar(owner, context, layer.bottomNames[0], layer.name);
            var dt = RequireScalar(owner, context, layer.bottomNames[1], layer.name);
            var b = RequireScalar(owner, context, layer.bottomNames[2], layer.name);
            var a = RequireScalar(owner, context, layer.bottomNames[3], layer.name);
            var q = RequireArray(context, layer.bottomNames[4], layer.name);
            var k = RequireArray(context, layer.bottomNames[5], layer.name);
            var v = RequireArray(context, layer.bottomNames[6], layer.name);
            var state = RequireArray(context, layer.bottomNames[7], layer.name);
            var qShape = NcnnRepro.GetTextureShape(context.textureShapes, q, layer.bottomNames[4]);
            var vShape = NcnnRepro.GetTextureShape(context.textureShapes, v, layer.bottomNames[6]);
            var stateShape = NcnnRepro.GetTextureShape(context.textureShapes, state, layer.bottomNames[7]);
            // ncnn d3 tensors are [c=sequence, h=heads, w=dimension]. Pack4
            // storage therefore packs sequence positions into array slices.
            var heads = Mathf.Max(1, qShape.h);
            var keyDim = qShape.w;
            var valueDim = vShape.w;
            var sequence = Mathf.Max(1, qShape.c);
            owner.DebugLog?.Invoke(
                "[Ncnn][GatedDeltaRule] layer=" + layer.name
                + " q=d" + qShape.dims + ":" + qShape.w + "x" + qShape.h + "x" + qShape.d + "x" + qShape.c
                + " qTexture=" + q.texture.width + "x" + q.texture.height + "x" + q.texture.volumeDepth
                + " v=d" + vShape.dims + ":" + vShape.w + "x" + vShape.h + "x" + vShape.d + "x" + vShape.c
                + " vTexture=" + v.texture.width + "x" + v.texture.height + "x" + v.texture.volumeDepth
                + " state=d" + stateShape.dims + ":" + stateShape.w + "x" + stateShape.h + "x" + stateShape.d + "x" + stateShape.c
                + " stateTexture=" + state.texture.width + "x" + state.texture.height + "x" + state.texture.volumeDepth
                + " alogTexture=" + alog.texture.width + "x" + alog.texture.height + " " + alog.texture.dimension + " " + alog.texture.format
                + " dtTexture=" + dt.texture.width + "x" + dt.texture.height + " " + dt.texture.dimension + " " + dt.texture.format
                + " bTexture=" + b.texture.width + "x" + b.texture.height + " " + b.texture.dimension + " " + b.texture.format
                + " aTexture=" + a.texture.width + "x" + a.texture.height + " " + a.texture.dimension + " " + a.texture.format
                + " heads=" + heads + " sequence=" + sequence + " keyDim=" + keyDim + " valueDim=" + valueDim);
            var output = owner.RentTempArray(v.texture.width, v.texture.height, Mathf.Max(1, v.texture.volumeDepth), owner.TensorTextureFormat);
            var stateA = owner.RentTempArray(state.texture.width, state.texture.height, Mathf.Max(1, state.texture.volumeDepth), owner.TensorTextureFormat);
            var stateB = owner.RentTempArray(state.texture.width, state.texture.height, Mathf.Max(1, state.texture.volumeDepth), owner.TensorTextureFormat);
            RenderTexture finalState = state.texture;
            try
            {
                for (var token = 0; token < sequence; token++)
                {
                    var source = token == 0 ? state.texture : ((token & 1) == 1 ? stateA : stateB);
                    var target = (token & 1) == 0 ? stateA : stateB;
                    owner.Ops.GatedDeltaRulePack4(alog.texture, dt.texture, b.texture, a.texture, q.texture, k.texture, v.texture, source, output, target, heads, keyDim, valueDim, sequence, token, 1e-6f);
                    finalState = target;
                }
                // The final state is in current.  Copying is intentionally not
                // done through a buffer; publish the final texture directly.
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, vShape, NcnnRepro.GetTextureStorageShape(v, vShape));
                NcnnRepro.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[1], finalState, stateShape, NcnnRepro.GetTextureStorageShape(state, stateShape));
                output = null;
                if (finalState == stateA) stateA = null;
                if (finalState == stateB) stateB = null;
            }
            finally
            {
                if (output != null) owner.ReturnTempArray(output);
                if (stateA != null) owner.ReturnTempArray(stateA);
                if (stateB != null) owner.ReturnTempArray(stateB);
                if (alog.temporary) owner.ReturnTempArray(alog.texture);
                if (dt.temporary) owner.ReturnTempArray(dt.texture);
                if (b.temporary) owner.ReturnTempArray(b.texture);
                if (a.temporary) owner.ReturnTempArray(a.texture);
            }
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (layer.bottomNames == null || layer.bottomNames.Length < 8 || layer.topNames == null || layer.topNames.Length < 2)
                throw new InvalidOperationException("GatedDeltaRule command contract is invalid: " + layer.name);
            var alog = RequireCmd2D(context, layer.bottomNames[0], layer.name);
            var dt = RequireCmd2D(context, layer.bottomNames[1], layer.name);
            var b = RequireCmd2D(context, layer.bottomNames[2], layer.name);
            var a = RequireCmd2D(context, layer.bottomNames[3], layer.name);
            var q = RequireCmdArray(context, layer.bottomNames[4], layer.name);
            var k = RequireCmdArray(context, layer.bottomNames[5], layer.name);
            var v = RequireCmdArray(context, layer.bottomNames[6], layer.name);
            var state = RequireCmdArray(context, layer.bottomNames[7], layer.name);
            var qShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[4]);
            var vShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[6]);
            var stateShape = NcnnRepro.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[7]);
            var heads = Mathf.Max(1, qShape.h);
            var sequence = Mathf.Max(1, qShape.c);
            var output = owner.RentTempArray(context.commandBuffer, v.texture.width, v.texture.height, v.texture.depth, owner.TensorTextureFormat);
            var stateA = owner.RentTempArray(context.commandBuffer, state.texture.width, state.texture.height, state.texture.depth, owner.TensorTextureFormat);
            var stateB = owner.RentTempArray(context.commandBuffer, state.texture.width, state.texture.height, state.texture.depth, owner.TensorTextureFormat);
            ComputeTexture finalState = state.texture;
            for (var token = 0; token < sequence; token++)
            {
                var source = token == 0 ? state.texture : ((token & 1) == 1 ? stateA : stateB);
                var target = (token & 1) == 0 ? stateA : stateB;
                owner.Ops.GatedDeltaRulePack4(context.commandBuffer, alog.texture, dt.texture, b.texture, a.texture, q.texture, k.texture, v.texture, source, output, target, heads, qShape.w, vShape.w, sequence, token, 1e-6f);
                finalState = target;
            }
            context.blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(output, vShape, new NcnnRepro.BufferShape(vShape.dims, output.width, output.height, vShape.d, output.depth * 4), owned: true);
            context.blobs[layer.topNames[1]] = NcnnRepro.CreateCmdTensorRef(finalState, stateShape, new NcnnRepro.BufferShape(stateShape.dims, finalState.width, finalState.height, stateShape.d, finalState.depth * 4), owned: true);
            if (context.shapes != null) { context.shapes[layer.topNames[0]] = vShape; context.shapes[layer.topNames[1]] = stateShape; }
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }

        private struct ScalarView
        {
            public RenderTexture texture;
            public bool temporary;
        }

        private static ScalarView RequireScalar(NcnnRepro owner, NcnnLayerBufferContext context, string name, string layer)
        {
            if (context.textureBlobs == null || !context.textureBlobs.TryGetValue(name, out var tensor) || tensor == null || tensor.texture == null)
                throw new InvalidOperationException("GatedDeltaRule requires texture scalar input (buffer fallback prohibited): layer=" + layer + " blob=" + name);
            if (tensor.texture.dimension == TextureDimension.Tex2D)
                return new ScalarView { texture = tensor.texture, temporary = false };

            var shape = NcnnRepro.GetTextureShape(context.textureShapes, tensor, name);
            var elements = Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            var linear = owner.RentTempMat(elements, 1, NcnnRepro.ResolveLinearMatTextureFormat());
            owner.Ops.ReshapePack4ToLinearMat(
                tensor.texture,
                shape.w,
                Mathf.Max(1, shape.h),
                Mathf.Max(1, shape.d),
                Mathf.Max(1, shape.c),
                shape.dims,
                linear,
                inputPack4Linear: NcnnRepro.IsPack4LinearMatTexture(tensor, shape));
            return new ScalarView { texture = linear, temporary = true };
        }

        private static NcnnRepro.CmdTensorRef RequireCmd2D(NcnnLayerCommandBufferContext context, string name, string layer)
        {
            var tensor = NcnnRepro.GetCmdTensor(context.blobs, name);
            if (tensor == null || tensor.texture == null || tensor.texture.dimension != TextureDimension.Tex2D)
                throw new InvalidOperationException("GatedDeltaRule command path requires Texture2D scalar input: layer=" + layer + " blob=" + name);
            return tensor;
        }

        private static NcnnRepro.CmdTensorRef RequireCmdArray(NcnnLayerCommandBufferContext context, string name, string layer)
        {
            var tensor = NcnnRepro.GetCmdTensor(context.blobs, name);
            if (tensor == null || tensor.texture == null || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("GatedDeltaRule command path requires Texture2DArray input: layer=" + layer + " blob=" + name);
            return tensor;
        }

        private static NcnnRepro.TensorRef RequireArray(NcnnLayerBufferContext context, string name, string layer)
        {
            if (context.textureBlobs == null || !context.textureBlobs.TryGetValue(name, out var tensor) || tensor == null || tensor.texture == null || tensor.texture.dimension != TextureDimension.Tex2DArray)
                throw new InvalidOperationException("GatedDeltaRule requires Texture2DArray input (buffer fallback prohibited): layer=" + layer + " blob=" + name);
            return tensor;
        }
    }
}
