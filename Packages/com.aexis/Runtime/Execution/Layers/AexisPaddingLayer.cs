using System;
using System.Collections.Generic;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class AexisPaddingLayer : AexisBaseLayer
    {
        private sealed class PaddingPack : IDisposable
        {
            public int top;
            public int bottom;
            public int left;
            public int right;
            public int type;
            public float value;
            public int perChannelPadDataSize;
            public int front;
            public int behind;
            public float[] perChannelPadDataCpu;
            public ComputeBuffer perChannelPadData;

            public void Dispose()
            {
                try { perChannelPadData?.Dispose(); } catch { }
            }
        }

        public AexisPaddingLayer() : base(AexisLayerTypes.Padding, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override AexisGraphSession.LayerLoadMetrics LoadLayer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            var pp = new PaddingPack
            {
                top = layer.GetInt(0, 0),
                bottom = layer.GetInt(1, 0),
                left = layer.GetInt(2, 0),
                right = layer.GetInt(3, 0),
                type = layer.GetInt(4, 0),
                value = layer.GetFloat(5, 0f),
                perChannelPadDataSize = layer.GetInt(6, 0),
                front = layer.GetInt(7, 0),
                behind = layer.GetInt(8, 0)
            };

            if (pp.perChannelPadDataSize > 0)
            {
                phaseSw.Restart();
                pp.perChannelPadDataCpu = br.ReadTensorAsFloat32(pp.perChannelPadDataSize, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                pp.perChannelPadData = AexisGraphSession.NewBuffer(pp.perChannelPadDataCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
            }

            owner._extraPacks[layer.name] = pp;
            return new AexisGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        public override void ExecuteBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PaddingPack pp)
            {
                pp = new PaddingPack
                {
                    top = layer.GetInt(0, 0),
                    bottom = layer.GetInt(1, 0),
                    left = layer.GetInt(2, 0),
                    right = layer.GetInt(3, 0),
                    type = layer.GetInt(4, 0),
                    value = layer.GetFloat(5, 0f),
                    perChannelPadDataSize = layer.GetInt(6, 0),
                    front = layer.GetInt(7, 0),
                    behind = layer.GetInt(8, 0)
                };
                owner._extraPacks[layer.name] = pp;
            }

            if (pp.top == 0 && pp.bottom == 0 && pp.left == 0 && pp.right == 0 && pp.front == 0 && pp.behind == 0)
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            if (CanUseSimplePack4TexturePath(owner, layer.bottomNames[0], pp, context.textureBlobs, context.textureShapes, context.bufferBlobs, context.bufferViews, out _, out _))
            {
                ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            #pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
            #pragma warning restore CS0618
        }

        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;
            var tempOwned = context.tempOwned;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PaddingPack pp)
                throw new InvalidOperationException("Padding pack not found: " + layer.name);

            if (pp.top == 0 && pp.bottom == 0 && pp.left == 0 && pp.right == 0 && pp.front == 0 && pp.behind == 0)
            {
#pragma warning disable CS0618
                new AexisNoopLayer().ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
                return;
            }

            var srcBuf = owner.GetOrConvertToBuffer(layer.bottomNames[0], textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            var srcView = AexisGraphSession.TryGetBufferView(layer.bottomNames[0], bufferBlobs, bufferViews);
            if (srcBuf == null || srcView == null)
                throw new InvalidOperationException("Padding source not found: " + layer.name);

            var outDims = srcView.dims;
            var outW = srcView.w + pp.left + pp.right;
            var outH = srcView.dims >= 2 ? srcView.h + pp.top + pp.bottom : 1;
            var outD = srcView.dims == 4 ? srcView.d + pp.front + pp.behind : 1;
            var outC = srcView.dims == 3 ? srcView.c + pp.front + pp.behind : srcView.c;
            if (outW <= 0 || outH <= 0 || outD <= 0 || outC <= 0)
                throw new InvalidOperationException("Padding invalid output shape: " + layer.name);

            var srcData = AexisGraphSession.ReadFloatBuffer(srcBuf);
            var outTensor = owner.RentTempTensorBuffer(outDims, outW, outH, outD, outC);
            var outData = new float[outTensor.elementCount];

            if (srcView.dims == 1)
            {
                for (var ox = 0; ox < outW; ox++)
                {
                    if (TryMapSpatialIndex(ox - pp.left, srcView.w, pp.type, out var sx))
                        outData[ox] = srcData[sx];
                    else
                        outData[ox] = pp.value;
                }
            }
            else if (srcView.dims == 2)
            {
                for (var oy = 0; oy < outH; oy++)
                {
                    for (var ox = 0; ox < outW; ox++)
                    {
                        var outIndex = oy * outW + ox;
                        if (TryMapSpatialIndex(ox - pp.left, srcView.w, pp.type, out var sx)
                            && TryMapSpatialIndex(oy - pp.top, srcView.h, pp.type, out var sy))
                        {
                            outData[outIndex] = srcData[sy * srcView.w + sx];
                        }
                        else
                        {
                            outData[outIndex] = pp.value;
                        }
                    }
                }
            }
            else if (srcView.dims == 3)
            {
                var srcPlane = srcView.w * srcView.h;
                var outPlane = outW * outH;
                for (var oc = 0; oc < outC; oc++)
                {
                    var padValue = ResolvePadValue(pp, oc);
                    if (!TryMapChannelIndex(oc - pp.front, srcView.c, pp.type, out var sc))
                    {
                        FillChannel(outData, oc * outPlane, outPlane, padValue);
                        continue;
                    }

                    var srcBase = sc * srcPlane;
                    var outBase = oc * outPlane;
                    for (var oy = 0; oy < outH; oy++)
                    {
                        for (var ox = 0; ox < outW; ox++)
                        {
                            var outIndex = outBase + oy * outW + ox;
                            if (TryMapSpatialIndex(ox - pp.left, srcView.w, pp.type, out var sx)
                                && TryMapSpatialIndex(oy - pp.top, srcView.h, pp.type, out var sy))
                            {
                                outData[outIndex] = srcData[srcBase + sy * srcView.w + sx];
                            }
                            else
                            {
                                outData[outIndex] = padValue;
                            }
                        }
                    }
                }
            }
            else
            {
                var srcPlane = srcView.w * srcView.h;
                var srcDepthStride = srcPlane;
                var srcChannelStride = srcView.d * srcPlane;
                var outPlane = outW * outH;
                var outDepthStride = outPlane;
                var outChannelStride = outD * outPlane;

                for (var oc = 0; oc < outC; oc++)
                {
                    var padValue = ResolvePadValue(pp, oc);
                    var srcChannelBase = oc * srcChannelStride;
                    var outChannelBase = oc * outChannelStride;
                    for (var oz = 0; oz < outD; oz++)
                    {
                        if (!TryMapChannelIndex(oz - pp.front, srcView.d, pp.type, out var sz))
                        {
                            FillChannel(outData, outChannelBase + oz * outDepthStride, outPlane, padValue);
                            continue;
                        }

                        var srcDepthBase = srcChannelBase + sz * srcDepthStride;
                        var outDepthBase = outChannelBase + oz * outDepthStride;
                        for (var oy = 0; oy < outH; oy++)
                        {
                            for (var ox = 0; ox < outW; ox++)
                            {
                                var outIndex = outDepthBase + oy * outW + ox;
                                if (TryMapSpatialIndex(ox - pp.left, srcView.w, pp.type, out var sx)
                                    && TryMapSpatialIndex(oy - pp.top, srcView.h, pp.type, out var sy))
                                {
                                    outData[outIndex] = srcData[srcDepthBase + sy * srcView.w + sx];
                                }
                                else
                                {
                                    outData[outIndex] = padValue;
                                }
                            }
                        }
                    }
                }
            }

            outTensor.buffer.SetData(outData);
            owner.PublishTensorBufferOutput(
                layer.topNames[0],
                outTensor,
                preferTexture: outTensor.dims <= 3,
                textureBlobs,
                textureShapes,
                bufferBlobs,
                bufferRefs,
                bufferViews,
                tempOwned);
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteRenderTexturePath(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerBufferContext context)
        {
            var textureBlobs = context.textureBlobs;
            var textureShapes = context.textureShapes;
            var bufferBlobs = context.bufferBlobs;
            var bufferRefs = context.bufferRefs;
            var bufferViews = context.bufferViews;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PaddingPack pp)
                throw new InvalidOperationException("Padding pack not found: " + layer.name);

            if (pp.top == 0 && pp.bottom == 0 && pp.left == 0 && pp.right == 0 && pp.front == 0 && pp.behind == 0)
            {
                new AexisNoopLayer().ExecuteRenderTexturePath(owner, layer, context);
                return;
            }

            if (!CanUseSimplePack4TexturePath(owner, layer.bottomNames[0], pp, textureBlobs, textureShapes, bufferBlobs, bufferViews, out var srcTex, out var srcShape))
                throw new InvalidOperationException("Padding render-texture path requires simple pack4 input: " + layer.name);

            var outRt = owner.RentTempArray(srcTex.width + pp.left + pp.right, srcTex.height + pp.top + pp.bottom, srcTex.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.PaddingPack4(srcTex.texture, srcTex.packs, pp.left, pp.right, pp.top, pp.bottom, pp.type, new Vector4(pp.value, pp.value, pp.value, pp.value), outRt);
            AexisGraphSession.SetTextureBlob(
                textureBlobs,
                textureShapes,
                layer.topNames[0],
                outRt,
                new AexisGraphSession.BufferShape(3, srcShape.w + pp.left + pp.right, srcShape.h + pp.top + pp.bottom, 1, srcShape.c));
            owner.Consume(textureBlobs, bufferBlobs, bufferRefs, bufferViews, remaining, layer.bottomNames, pinnedNames);
        }

        public override void ExecuteCommandBuffer(AexisGraphSession owner, AexisGraphModel.Layer layer, AexisLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;
            var remaining = context.remaining;
            var pinnedNames = context.pinnedNames;

            if (!owner._extraPacks.TryGetValue(layer.name, out var packObj) || packObj is not PaddingPack pp)
            {
                pp = new PaddingPack
                {
                    top = layer.GetInt(0, 0),
                    bottom = layer.GetInt(1, 0),
                    left = layer.GetInt(2, 0),
                    right = layer.GetInt(3, 0),
                    type = layer.GetInt(4, 0),
                    value = layer.GetFloat(5, 0f),
                    perChannelPadDataSize = layer.GetInt(6, 0),
                    front = layer.GetInt(7, 0),
                    behind = layer.GetInt(8, 0)
                };
                owner._extraPacks[layer.name] = pp;
            }

            var src = AexisGraphSession.GetCmdTensor(blobs, layer.bottomNames[0]);
            var srcShape = AexisGraphSession.GetCmdShape(shapes, blobs, layer.bottomNames[0]);
            if (pp.top == 0 && pp.bottom == 0 && pp.left == 0 && pp.right == 0 && pp.front == 0 && pp.behind == 0)
            {
                var storageShape = AexisGraphSession.GetCmdStorageShape(src, srcShape);
                blobs[layer.topNames[0]] = AexisGraphSession.CreateCmdTensorAlias(src, srcShape, storageShape);
                if (shapes != null)
                    shapes[layer.topNames[0]] = srcShape;
                owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
                return;
            }

            var outShape = ResolveCmdOutputShape(srcShape, pp);
            if (!CanUseSimplePack4CmdPath(src, srcShape, pp))
            {
                throw new InvalidOperationException(
                    "Padding command-buffer Pack4 profile rejected the input descriptor"
                    + " | layer=" + layer.name
                    + " | logical=d" + srcShape.dims + ":" + srcShape.w + "x" + srcShape.h + "x" + srcShape.d + "x" + srcShape.c
                    + " | texture=" + src.width + "x" + src.height + "x" + src.packs
                    + " | front=" + pp.front + " | behind=" + pp.behind + " | per_channel=" + pp.perChannelPadDataSize
                    + " | rejectedFallback=placeholder");
            }

            var outW = srcShape.w + pp.left + pp.right;
            var outH = srcShape.h + pp.top + pp.bottom;
            if (outW <= 0 || outH <= 0)
                throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

            var outArr = owner.RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.PaddingPack4(cmd, src.texture, src.packs, pp.left, pp.right, pp.top, pp.bottom, pp.type, new Vector4(pp.value, pp.value, pp.value, pp.value), outArr);
            blobs[layer.topNames[0]] = new AexisGraphSession.CmdTensorRef
            {
                texture = outArr,
                width = outW,
                height = outH,
                packs = src.packs,
                refs = 1,
                owned = true
            };
            if (shapes != null)
                shapes[layer.topNames[0]] = outShape;
            owner.ConsumeCmd(cmd, blobs, remaining, layer.bottomNames, pinnedNames, shapes);
        }

        private static bool CanUseSimplePack4TexturePath(
            AexisGraphSession owner,
            string bottomName,
            PaddingPack pp,
            Dictionary<string, AexisGraphSession.TensorRef> textureBlobs,
            Dictionary<string, AexisGraphSession.BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
            out AexisGraphSession.TensorRef srcTex,
            out AexisGraphSession.BufferShape srcShape)
        {
            srcTex = null;
            srcShape = default;
            if (pp.front != 0 || pp.behind != 0 || pp.perChannelPadDataSize != 0)
                return false;
            if (!owner.TryGetPack4Texture(bottomName, textureBlobs, textureShapes, bufferBlobs, bufferViews, out srcTex, out srcShape))
                return false;
            return srcShape.dims == 3;
        }

        private static bool CanUseSimplePack4CmdPath(AexisGraphSession.CmdTensorRef src, AexisGraphSession.BufferShape srcShape, PaddingPack pp)
        {
            return src != null
                && src.texture != null
                && pp.front == 0
                && pp.behind == 0
                && pp.perChannelPadDataSize == 0
                && srcShape.dims == 3
                && srcShape.w == src.width
                && srcShape.h == src.height
                && srcShape.d == 1
                && srcShape.c > 0
                && srcShape.c <= src.packs * 4;
        }

        private static AexisGraphSession.BufferShape ResolveCmdOutputShape(AexisGraphSession.BufferShape srcShape, PaddingPack pp)
        {
            var outW = Mathf.Max(1, srcShape.w + pp.left + pp.right);
            if (srcShape.dims == 1)
                return new AexisGraphSession.BufferShape(1, outW, 1, 1, 1);

            var outH = Mathf.Max(1, srcShape.h + pp.top + pp.bottom);
            if (srcShape.dims == 2)
                return new AexisGraphSession.BufferShape(2, outW, outH, 1, 1);

            if (srcShape.dims == 3)
                return new AexisGraphSession.BufferShape(3, outW, outH, 1, Mathf.Max(1, srcShape.c + pp.front + pp.behind));

            return new AexisGraphSession.BufferShape(4, outW, outH, Mathf.Max(1, srcShape.d + pp.front + pp.behind), srcShape.c);
        }

        private static float ResolvePadValue(PaddingPack pack, int channel)
        {
            if (pack.perChannelPadDataCpu != null && channel >= 0 && channel < pack.perChannelPadDataCpu.Length)
                return pack.perChannelPadDataCpu[channel];
            return pack.value;
        }

        private static void FillChannel(float[] data, int offset, int count, float value)
        {
            for (var i = 0; i < count; i++)
                data[offset + i] = value;
        }

        private static bool TryMapSpatialIndex(int coord, int length, int type, out int mapped)
        {
            mapped = 0;
            if (length <= 0)
                return false;

            if (coord >= 0 && coord < length)
            {
                mapped = coord;
                return true;
            }

            if (type == 0)
                return false;

            if (type == 1)
            {
                mapped = Mathf.Clamp(coord, 0, length - 1);
                return true;
            }

            mapped = Reflect101Index(coord, length);
            return true;
        }

        private static bool TryMapChannelIndex(int coord, int length, int type, out int mapped)
        {
            mapped = 0;
            if (length <= 0)
                return false;

            if (coord >= 0 && coord < length)
            {
                mapped = coord;
                return true;
            }

            if (type == 0)
                return false;

            if (type == 1)
            {
                mapped = Mathf.Clamp(coord, 0, length - 1);
                return true;
            }

            mapped = Reflect101Index(coord, length);
            return true;
        }

        private static int Reflect101Index(int x, int len)
        {
            if (len <= 1)
                return 0;
            var period = len * 2 - 2;
            var y = Mathf.Abs(x);
            var m = y % period;
            if (m >= len)
                m = period - m;
            return m;
        }
    }
}
