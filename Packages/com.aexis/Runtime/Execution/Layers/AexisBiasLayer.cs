using System;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // ncnn Bias is a channel-wise add. This implementation intentionally exposes
    // only Pack4 texture execution; ordinary inference buffers are never used.
    public sealed class AexisBiasLayer : AexisBaseLayer
    {
        public AexisBiasLayer() : base(AexisLayerTypes.Bias, supportsBufferPath: false, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(
            AexisGraphSession owner,
            AexisGraphModel.Layer layer,
            AexisWeightReader br)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            if (layer == null) throw new ArgumentNullException(nameof(layer));
            if (br == null) throw new ArgumentNullException(nameof(br));

            var channels = layer.GetInt(0, 0);
            if (channels <= 0)
                throw new InvalidOperationException("Bias requires a positive bias_data_size: " + layer.name);

            var bytesStart = br.Position;
            var timer = Stopwatch.StartNew();
            var values = br.ReadTensorAsFloat32(channels, 0, 0, 0, 1);
            timer.Stop();
            var readMs = timer.ElapsedMilliseconds;

            timer.Restart();
            var packed = AexisGraphSession.PackBiasToO4(values, channels, (channels + 3) / 4);
            var pack = new AexisGraphSession.BiasPack
            {
                channels = channels,
                bias4 = new ComputeBuffer(packed.Length, sizeof(float) * 4, ComputeBufferType.Structured)
            };
            AexisGpuResourceTracker.RegisterBuffer(
                pack.bias4,
                packed.Length,
                sizeof(float) * 4,
                "AexisGraphSession.Bias4:" + layer.name);
            try
            {
                pack.bias4.SetData(packed);
                owner._bias[layer.name] = pack;
            }
            catch
            {
                pack.Dispose();
                throw;
            }
            timer.Stop();
            return new AexisGraphSession.LayerLoadMetrics(
                Math.Max(0, br.Position - bytesStart),
                readMs,
                timer.ElapsedMilliseconds,
                timer.ElapsedMilliseconds);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._bias.TryGetValue(layer.name, out var bias) || bias?.bias4 == null)
                throw new InvalidOperationException("Immutable Bias constants are not loaded: " + layer.name);
            if (!owner.TryGetPack4Texture(
                    layer.bottomNames[0],
                    context.textureBlobs,
                    context.textureShapes,
                    context.bufferBlobs,
                    context.bufferViews,
                    out var src,
                    out var srcShape))
            {
                throw new InvalidOperationException("Bias render-texture path requires a Pack4 texture input: " + layer.name);
            }
            if ((srcShape.dims != 3 && srcShape.dims != 4) || srcShape.c != bias.channels
                || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
                throw new InvalidOperationException("Bias requires a descriptor-valid rank-3/rank-4 Pack4 activation matching its immutable channel constants: " + layer.name);

            var storageShape = AexisGraphSession.GetTextureStorageShape(src, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
            var output = owner.RentTempArray(src.width, src.height, outputDepth, src.texture.format);
            owner.Ops.AddBiasPack4(src.texture, bias.bias4, src.packs, output);
            AexisGraphSession.SetTextureBlob(context.textureBlobs, context.textureShapes, layer.topNames[0], output, srcShape, storageShape);
            owner.Consume(context.textureBlobs, context.bufferBlobs, context.bufferRefs, context.bufferViews, context.remaining, layer.bottomNames, context.pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var src = AexisGraphSession.GetCmdTensor(context.blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(context.shapes, context.blobs, layer.bottomNames[0]);
            if (!owner._bias.TryGetValue(layer.name, out var bias) || bias?.bias4 == null)
                throw new InvalidOperationException("Immutable Bias constants are not loaded: " + layer.name);
            if ((srcShape.dims != 3 && srcShape.dims != 4) || srcShape.c != bias.channels
                || !AexisGraphSession.MatchesPack4TextureStorage(src, srcShape))
                throw new InvalidOperationException("Bias requires a descriptor-valid rank-3/rank-4 Pack4 activation matching its immutable channel constants: " + layer.name);

            var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
            var outputDepth = srcShape.dims == 4 ? srcShape.d * src.packs : src.packs;
            var output = owner.RentTempArray(context.commandBuffer, src.width, src.height, outputDepth, src.texture.format);
            owner.Ops.AddBiasPack4(context.commandBuffer, src.texture, bias.bias4, src.packs, output);
            context.blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorRef(
                output,
                srcShape,
                storageShape,
                owned: true,
                blobName: layer.topNames[0]);
            if (context.shapes != null)
                context.shapes[layer.topNames[0]] = srcShape;
            owner.ConsumeCmd(context.commandBuffer, context.blobs, context.remaining, layer.bottomNames, context.pinnedNames, context.shapes);
        }
    }
}
