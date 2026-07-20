using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMemoryDataLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMemoryDataLayerRepro() : base(NcnnLayerTypes.MemoryData, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnGraphSession.LayerLoadMetrics LoadLayer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
                        var bytesStart = br.Position;
                        long readMs = 0;
                        long uploadMs = 0;
                        long packMs = 0;
                        var phaseSw = new Stopwatch();

                                        var w = layer.GetInt(0, 0);
                                        var h = layer.GetInt(1, 0);
                                        var d = layer.GetInt(11, 0);
                                        var c = layer.GetInt(2, 0);
                                        var loadType = layer.GetInt(21, 1);

                                        phaseSw.Restart();
                                        var a = NcnnGraphSession.ReadClipMatAsFloat32(br, w, h, d, c, loadType);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        var dims = 1;
                                        if (h > 0) dims = 2;
                                        if (c > 0) dims = d > 0 ? 4 : 3;
                                        var memoryPack = new NcnnGraphSession.MemoryDataPack
                                        {
                                            dims = dims,
                                            w = Mathf.Max(1, w),
                                            h = Mathf.Max(1, h),
                                            d = Mathf.Max(1, d),
                                            c = Mathf.Max(1, c),
                                            cpuData = a
                                        };
                                        if (TryCreateChannelVectorTexture(memoryPack))
                                        {
                                            owner._memoryData[layer.name] = memoryPack;
                                            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
                                        }
                                        if (TryCreateLinearMatTexture(memoryPack))
                                        {
                                            owner._memoryData[layer.name] = memoryPack;
                                            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
                                        }
                                        phaseSw.Restart();
                                        var createdPack4Texture = TryCreatePack4Texture(memoryPack);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;
                                        if (createdPack4Texture)
                                        {
                                            owner._memoryData[layer.name] = memoryPack;
                                            return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
                                        }
                                        if (ShouldUseVistaPromptPack4RtOnly(owner, layer, memoryPack))
                                        {
                                            phaseSw.Restart();
                                            var createdVistaPromptRt = TryGetOrCreateVistaPromptPack4Rt(memoryPack, memoryPack.w, out _);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                            if (createdVistaPromptRt)
                                            {
                                                owner._memoryData[layer.name] = memoryPack;
                                                return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
                                            }
                                        }

                                        phaseSw.Restart();
                                        var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                                        buf.SetData(a);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        memoryPack.data = buf;
                                        owner._memoryData[layer.name] = memoryPack;
                                        return new NcnnGraphSession.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
                        var textureBlobs = context.textureBlobs;
                        var textureShapes = context.textureShapes;
                        var bufferBlobs = context.bufferBlobs;
                        var bufferRefs = context.bufferRefs;
                        var bufferViews = context.bufferViews;
                        var indexBlobs = context.indexBlobs;
                        var remaining = context.remaining;
                        var pinnedNames = context.pinnedNames;
                        var tempOwned = context.tempOwned;

                        do
                        {
                                                if (!owner._memoryData.TryGetValue(layer.name, out var mp))
                                                    throw new InvalidOperationException("MemoryData not found: " + layer.name);

                                                if (TryPublishVistaPromptPack4RtBlob(owner, layer, mp, textureBlobs, textureShapes))
                                                    continue;
                                                if (TryPublishPack4RtBlob(layer, mp, textureBlobs, textureShapes))
                                                    continue;

                                                if (mp.data == null && !TryCreateMemoryDataBuffer(mp))
                                                    throw new InvalidOperationException("MemoryData buffer not found: " + layer.name);

                                                if (owner.ShouldBlockPack4BufferFallback() && mp.dims <= 4)
                                                {
                                                    var logicalShape = new NcnnGraphSession.BufferShape(mp.dims, mp.w, mp.h, mp.d, mp.c);
                                                    owner.PublishScratchTextureOutput(
                                                        layer.topNames[0],
                                                        mp.data,
                                                        logicalShape,
                                                        textureBlobs,
                                                        textureShapes);
                                                    continue;
                                                }

                                                var tensor = new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false);
                                                owner.PublishTensorBufferOutput(
                                                    layer.topNames[0],
                                                    tensor,
                                                    preferTexture: mp.dims <= 4,
                                                    textureBlobs,
                                                    textureShapes,
                                                    bufferBlobs,
                                                    bufferRefs,
                                                    bufferViews,
                                                    tempOwned);
                                                continue;
                        } while (false);
        }

        public override void ExecuteRenderTexturePath(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;

            if (!owner._memoryData.TryGetValue(layer.name, out var mp))
                throw new InvalidOperationException("MemoryData not found: " + layer.name);
            if (TryPublishChannelVectorCmdBlob(owner, cmd, layer, mp, blobs, shapes))
                return;
            if (TryPublishLinearMatCmdBlob(owner, cmd, layer, mp, blobs, shapes))
                return;
            if (TryPublishVistaPromptPack4CmdBlob(owner, cmd, layer, mp, blobs, shapes))
                return;
            if (TryPublishPack4CmdBlob(owner, cmd, layer, mp, blobs, shapes))
                return;
            if (mp.data == null && !TryCreateMemoryDataBuffer(mp))
                throw new InvalidOperationException("MemoryData not found: " + layer.name);

            owner.PublishCmdTensorBufferOutput(
                cmd,
                layer.topNames[0],
                new NcnnTensorBuffer(mp.data, mp.dims, mp.w, mp.h, mp.d, mp.c, false),
                preferTexture: true,
                blobs,
                shapes);
        }

        private static bool TryCreateLinearMatTexture(NcnnGraphSession.MemoryDataPack memoryPack)
        {
            if (memoryPack == null
                || (memoryPack.dims != 1 && memoryPack.dims != 2)
                || memoryPack.w <= 0
                || memoryPack.h <= 0
                || memoryPack.cpuData == null
                || memoryPack.cpuData.Length < memoryPack.w * memoryPack.h)
            {
                return false;
            }

            Texture2D upload = null;
            RenderTexture texture = null;
            try
            {
                var descriptor = new RenderTextureDescriptor(memoryPack.w, memoryPack.h, RenderTextureFormat.RFloat, 0)
                {
                    dimension = TextureDimension.Tex2D,
                    enableRandomWrite = false,
                    msaaSamples = 1
                };
                texture = new RenderTexture(descriptor)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "NcnnMemoryDataLinearMat"
                };
                texture.Create();
                NcnnGpuResourceTracker.RegisterTexture(texture, "NcnnMemoryDataLinearMat");

                upload = new Texture2D(memoryPack.w, memoryPack.h, TextureFormat.RFloat, false, true)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0,
                    name = "NcnnMemoryDataLinearMatUpload"
                };
                upload.SetPixelData(memoryPack.cpuData, 0);
                upload.Apply(false, true);
                Graphics.CopyTexture(upload, 0, 0, texture, 0, 0);

                memoryPack.linearMatRt = texture;
                texture = null;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                if (upload != null)
                    UnityEngine.Object.DestroyImmediate(upload);
                if (texture != null)
                {
                    NcnnGpuResourceTracker.ReleaseTexture(texture, "NcnnMemoryDataLinearMat.create-failed");
                    texture.Release();
                    UnityEngine.Object.DestroyImmediate(texture);
                }
            }
        }

        private static bool TryCreatePack4Texture(NcnnGraphSession.MemoryDataPack memoryPack)
        {
            if (memoryPack == null
                || (memoryPack.dims != 3 && memoryPack.dims != 4)
                || memoryPack.w <= 0
                || memoryPack.h <= 0
                || memoryPack.d <= 0
                || memoryPack.c <= 0
                || memoryPack.cpuData == null
                || memoryPack.cpuData.Length < memoryPack.w * memoryPack.h * memoryPack.d * memoryPack.c)
            {
                return false;
            }

            var packs = Mathf.Max(1, Mathf.CeilToInt(memoryPack.c / 4f));
            var slices = Mathf.Max(1, memoryPack.d) * packs;
            RenderTexture texture = null;
            Texture2DArray upload = null;
            try
            {
                var descriptor = new RenderTextureDescriptor(memoryPack.w, memoryPack.h, RenderTextureFormat.ARGBFloat, 0)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = slices,
                    enableRandomWrite = false,
                    msaaSamples = 1
                };
                texture = new RenderTexture(descriptor)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "NcnnMemoryDataPack4"
                };
                texture.Create();
                NcnnGpuResourceTracker.RegisterTexture(texture, "NcnnMemoryDataPack4");

                upload = new Texture2DArray(memoryPack.w, memoryPack.h, slices, TextureFormat.RGBAFloat, false, true)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0,
                    name = "NcnnMemoryDataPack4Upload"
                };
                var spatialCount = memoryPack.w * memoryPack.h;
                for (var z = 0; z < memoryPack.d; z++)
                {
                    for (var pack = 0; pack < packs; pack++)
                    {
                        var pixels = new Color[spatialCount];
                        for (var y = 0; y < memoryPack.h; y++)
                        {
                            for (var x = 0; x < memoryPack.w; x++)
                            {
                                var spatial = y * memoryPack.w + x;
                                var channel = pack * 4;
                                var r = channel < memoryPack.c ? memoryPack.cpuData[((channel * memoryPack.d + z) * memoryPack.h + y) * memoryPack.w + x] : 0f;
                                var g = channel + 1 < memoryPack.c ? memoryPack.cpuData[(((channel + 1) * memoryPack.d + z) * memoryPack.h + y) * memoryPack.w + x] : 0f;
                                var b = channel + 2 < memoryPack.c ? memoryPack.cpuData[(((channel + 2) * memoryPack.d + z) * memoryPack.h + y) * memoryPack.w + x] : 0f;
                                var a = channel + 3 < memoryPack.c ? memoryPack.cpuData[(((channel + 3) * memoryPack.d + z) * memoryPack.h + y) * memoryPack.w + x] : 0f;
                                pixels[spatial] = new Color(r, g, b, a);
                            }
                        }
                        upload.SetPixels(pixels, z * packs + pack, 0);
                    }
                }
                upload.Apply(false, true);
                for (var slice = 0; slice < slices; slice++)
                    Graphics.CopyTexture(upload, slice, 0, texture, slice, 0);

                memoryPack.pack4Rt = texture;
                memoryPack.pack4RtChannels = memoryPack.c;
                memoryPack.pack4RtDepth = slices;
                texture = null;
                return true;
            }
            catch
            {
                return false;
            }
            finally
            {
                if (upload != null) UnityEngine.Object.DestroyImmediate(upload);
                if (texture != null)
                {
                    NcnnGpuResourceTracker.ReleaseTexture(texture, "NcnnMemoryDataPack4.create-failed");
                    texture.Release();
                    UnityEngine.Object.DestroyImmediate(texture);
                }
            }
        }

        private static bool TryPublishPack4RtBlob(
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            if (memoryPack?.pack4Rt == null
                || !memoryPack.pack4Rt.IsCreated()
                || (memoryPack.dims != 3 && memoryPack.dims != 4)
                || layer?.topNames == null
                || layer.topNames.Length == 0)
            {
                return false;
            }

            var shape = new NcnnGraphSession.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            textureBlobs[layer.topNames[0]] = NcnnGraphSession.CreateTextureRef(memoryPack.pack4Rt, shape, shape, owned: false, blobName: layer.topNames[0]);
            textureShapes[layer.topNames[0]] = shape;
            return true;
        }

        private static bool TryPublishPack4CmdBlob(
            NcnnGraphSession owner,
            CommandBuffer commandBuffer,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            if (memoryPack?.pack4Rt == null
                || !memoryPack.pack4Rt.IsCreated()
                || (memoryPack.dims != 3 && memoryPack.dims != 4)
                || layer?.topNames == null
                || layer.topNames.Length == 0)
            {
                return false;
            }

            var texture = owner.RentTempArray(
                commandBuffer,
                memoryPack.w,
                memoryPack.h,
                memoryPack.pack4RtDepth,
                memoryPack.pack4Rt.format);
            for (var slice = 0; slice < memoryPack.pack4RtDepth; slice++)
                commandBuffer.CopyTexture(memoryPack.pack4Rt, slice, 0, texture.nameID, slice, 0);
            var shape = new NcnnGraphSession.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(texture, shape, shape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null) shapes[layer.topNames[0]] = shape;
            return true;
        }

        private static bool TryPublishLinearMatCmdBlob(
            NcnnGraphSession owner,
            CommandBuffer cmd,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            if (memoryPack?.linearMatRt == null
                || !memoryPack.linearMatRt.IsCreated()
                || (memoryPack.dims != 1 && memoryPack.dims != 2)
                || memoryPack.w <= 0
                || memoryPack.h <= 0
                || layer?.topNames == null
                || layer.topNames.Length == 0
                || string.IsNullOrWhiteSpace(layer.topNames[0]))
            {
                return false;
            }

            var output = owner.RentTempMat(
                cmd,
                memoryPack.w,
                memoryPack.h,
                NcnnGraphSession.ResolveLinearMatTextureFormat());
            cmd.CopyTexture(memoryPack.linearMatRt, 0, 0, output.nameID, 0, 0);
            var logicalShape = new NcnnGraphSession.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, 1, 1);
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(output, logicalShape, logicalShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.DebugLog?.Invoke(
                "[MemoryDataLinearMat][cmd] texture copy"
                + " | layer=" + layer.name
                + " | top=" + layer.topNames[0]
                + " | shape=" + memoryPack.w.ToString(CultureInfo.InvariantCulture) + "x" + memoryPack.h.ToString(CultureInfo.InvariantCulture));
            return true;
        }

        private static bool TryCreateChannelVectorTexture(NcnnGraphSession.MemoryDataPack memoryPack)
        {
            if (memoryPack == null
                || memoryPack.dims != 3
                || memoryPack.w != 1
                || memoryPack.h != 1
                || memoryPack.d != 1
                || memoryPack.c <= 0
                || memoryPack.cpuData == null
                || memoryPack.cpuData.Length < memoryPack.c)
            {
                return false;
            }

            try
            {
                var pixels = new Color[memoryPack.c];
                for (var index = 0; index < pixels.Length; index++)
                    pixels[index] = new Color(memoryPack.cpuData[index], 0f, 0f, 0f);

                var texture = new Texture2D(memoryPack.c, 1, TextureFormat.RGBAHalf, false, true)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0,
                    name = "NcnnMemoryDataChannelVector"
                };
                texture.SetPixels(pixels);
                texture.Apply(false, true);
                memoryPack.channelVectorTexture = texture;
                return true;
            }
            catch
            {
                return false;
            }
        }

        private static bool TryPublishChannelVectorCmdBlob(
            NcnnGraphSession owner,
            CommandBuffer cmd,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            if (memoryPack?.channelVectorTexture == null
                || memoryPack.dims != 3
                || memoryPack.w != 1
                || memoryPack.h != 1
                || memoryPack.d != 1
                || memoryPack.c <= 0
                || layer?.topNames == null
                || layer.topNames.Length == 0
                || string.IsNullOrWhiteSpace(layer.topNames[0]))
            {
                return false;
            }

            var packs = Mathf.Max(1, Mathf.CeilToInt(memoryPack.c / 4f));
            if (!TryGetOrCreateChannelVectorPack4Rt(memoryPack, out var packedVector) || packedVector == null)
                return false;

            // Match the Pack4 RT representation exactly: a channel vector occupies
            // one texel across ceil(C/4) array slices, rather than C scalar texels.
            var vector = owner.RentTempArray(cmd, 1, 1, packs, RenderTextureFormat.ARGBHalf);
            owner.Ops.CopyPack4(cmd, packedVector, 0, vector, 0, packs);
            var logicalShape = new NcnnGraphSession.BufferShape(3, 1, 1, 1, memoryPack.c);
            var storageShape = logicalShape;
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(vector, logicalShape, storageShape, owned: true, blobName: layer.topNames[0]);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.DebugLog?.Invoke(
                "[MemoryDataChannelVector][cmd] pack4 texture copy"
                + " | layer=" + layer.name
                + " | top=" + layer.topNames[0]
                + " | channels=" + memoryPack.c.ToString(CultureInfo.InvariantCulture)
                + " | packs=" + packs.ToString(CultureInfo.InvariantCulture));
            return true;
        }

        private static bool TryGetOrCreateChannelVectorPack4Rt(NcnnGraphSession.MemoryDataPack memoryPack, out RenderTexture packedVector)
        {
            packedVector = null;
            if (memoryPack == null
                || memoryPack.dims != 3
                || memoryPack.w != 1
                || memoryPack.h != 1
                || memoryPack.d != 1
                || memoryPack.c <= 0
                || memoryPack.cpuData == null
                || memoryPack.cpuData.Length < memoryPack.c)
            {
                return false;
            }

            var channels = memoryPack.c;
            var packs = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            if (memoryPack.pack4Rt != null
                && memoryPack.pack4RtChannels == channels
                && memoryPack.pack4RtDepth == packs
                && memoryPack.pack4Rt.IsCreated())
            {
                packedVector = memoryPack.pack4Rt;
                return true;
            }

            try
            {
                if (memoryPack.pack4Rt != null)
                {
                    NcnnGpuResourceTracker.ReleaseTexture(memoryPack.pack4Rt, "NcnnMemoryDataChannelVector.recreate");
                    memoryPack.pack4Rt.Release();
                    UnityEngine.Object.DestroyImmediate(memoryPack.pack4Rt);
                    memoryPack.pack4Rt = null;
                }

                var descriptor = new RenderTextureDescriptor(1, 1, RenderTextureFormat.ARGBFloat, 0)
                {
                    dimension = TextureDimension.Tex2DArray,
                    volumeDepth = packs,
                    enableRandomWrite = false,
                    msaaSamples = 1
                };
                var texture = new RenderTexture(descriptor)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    name = "NcnnMemoryDataChannelVectorPack4"
                };
                texture.Create();
                NcnnGpuResourceTracker.RegisterTexture(texture, "NcnnMemoryDataChannelVectorPack4");

                var upload = new Texture2DArray(1, 1, packs, TextureFormat.RGBAFloat, false, true)
                {
                    filterMode = FilterMode.Point,
                    wrapMode = TextureWrapMode.Clamp,
                    anisoLevel = 0,
                    name = "NcnnMemoryDataChannelVectorPack4Upload"
                };
                try
                {
                    for (var pack = 0; pack < packs; pack++)
                    {
                        var baseIndex = pack * 4;
                        var x = baseIndex < channels ? memoryPack.cpuData[baseIndex] : 0f;
                        var y = baseIndex + 1 < channels ? memoryPack.cpuData[baseIndex + 1] : 0f;
                        var z = baseIndex + 2 < channels ? memoryPack.cpuData[baseIndex + 2] : 0f;
                        var w = baseIndex + 3 < channels ? memoryPack.cpuData[baseIndex + 3] : 0f;
                        upload.SetPixels(new[] { new Color(x, y, z, w) }, pack, 0);
                    }
                    upload.Apply(false, true);
                    for (var pack = 0; pack < packs; pack++)
                        Graphics.CopyTexture(upload, pack, 0, texture, pack, 0);
                }
                finally
                {
                    UnityEngine.Object.DestroyImmediate(upload);
                }

                memoryPack.pack4Rt = texture;
                memoryPack.pack4RtChannels = channels;
                memoryPack.pack4RtDepth = packs;
                packedVector = texture;
                return true;
            }
            catch
            {
                if (memoryPack.pack4Rt != null)
                {
                    try { memoryPack.pack4Rt.Release(); } catch { }
                    UnityEngine.Object.DestroyImmediate(memoryPack.pack4Rt);
                    memoryPack.pack4Rt = null;
                }
                return false;
            }
        }

        private static bool ShouldUseVistaPromptPack4RtOnly(NcnnGraphSession owner, NcnnParamModel.Layer layer, NcnnGraphSession.MemoryDataPack memoryPack)
        {
            return owner != null
                && owner.EnableVistaTailPack4Specializations
                && layer != null
                && string.Equals(layer.name, "squeeze", StringComparison.Ordinal)
                && memoryPack != null
                && memoryPack.dims == 1
                && memoryPack.w > 0
                && memoryPack.cpuData != null
                && memoryPack.cpuData.Length >= memoryPack.w;
        }

        private static bool TryPublishVistaPromptPack4RtBlob(
            NcnnGraphSession owner,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.TensorRef> textureBlobs,
            Dictionary<string, NcnnGraphSession.BufferShape> textureShapes)
        {
            if (!ShouldUseVistaPromptPack4RtOnly(owner, layer, memoryPack))
                return false;
            if (!TryGetOrCreateVistaPromptPack4Rt(memoryPack, memoryPack.w, out var promptRt) || promptRt == null)
                return false;
            if (layer.topNames == null || layer.topNames.Length == 0 || string.IsNullOrWhiteSpace(layer.topNames[0]))
                return false;

            var logicalShape = new NcnnGraphSession.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            textureBlobs[layer.topNames[0]] = new NcnnGraphSession.TensorRef
            {
                texture = promptRt,
                width = promptRt.width,
                height = promptRt.height,
                packs = memoryPack.pack4RtDepth,
                refs = 1,
                owned = false,
                hasLogicalShape = true,
                logicalShape = logicalShape,
                hasStorageShape = true,
                storageShape = logicalShape
            };
            textureShapes[layer.topNames[0]] = logicalShape;
            owner.DebugLog?.Invoke(
                "[MemoryDataPack4Rt] direct publish"
                + " | layer=" + layer.name
                + " | top=" + layer.topNames[0]
                + " | packs=" + memoryPack.pack4RtDepth.ToString(CultureInfo.InvariantCulture));
            return true;
        }

        private static bool TryPublishVistaPromptPack4CmdBlob(
            NcnnGraphSession owner,
            CommandBuffer cmd,
            NcnnParamModel.Layer layer,
            NcnnGraphSession.MemoryDataPack memoryPack,
            Dictionary<string, NcnnGraphSession.CmdTensorRef> blobs,
            Dictionary<string, NcnnGraphSession.BufferShape> shapes)
        {
            if (!ShouldUseVistaPromptPack4RtOnly(owner, layer, memoryPack))
                return false;
            if (!TryGetOrCreateVistaPromptPack4Rt(memoryPack, memoryPack.w, out var promptRt) || promptRt == null)
                return false;
            if (layer.topNames == null || layer.topNames.Length == 0 || string.IsNullOrWhiteSpace(layer.topNames[0]))
                return false;

            var packs = Mathf.Max(1, memoryPack.pack4RtDepth);
            var outArr = owner.RentTempArray(cmd, promptRt.width, promptRt.height, packs, promptRt.format);
            for (var pack = 0; pack < packs; pack++)
                cmd.CopyTexture(promptRt, pack, 0, outArr.nameID, pack, 0);

            var logicalShape = new NcnnGraphSession.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            blobs[layer.topNames[0]] = NcnnGraphSession.CreateCmdTensorRef(outArr, logicalShape, logicalShape, owned: true);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.DebugLog?.Invoke(
                "[MemoryDataPack4Rt][cmd] direct publish"
                + " | layer=" + layer.name
                + " | top=" + layer.topNames[0]
                + " | packs=" + packs.ToString(CultureInfo.InvariantCulture));
            return true;
        }

        private static bool TryCreateMemoryDataBuffer(NcnnGraphSession.MemoryDataPack memoryPack)
        {
            if (memoryPack == null || memoryPack.data != null || memoryPack.cpuData == null || memoryPack.cpuData.Length == 0)
                return memoryPack?.data != null;

            try
            {
                var buf = new ComputeBuffer(memoryPack.cpuData.Length, sizeof(float), ComputeBufferType.Structured);
                buf.SetData(memoryPack.cpuData);
                memoryPack.data = buf;
                return true;
            }
            catch
            {
                return false;
            }
        }

        internal static bool TryGetOrCreateVistaPromptPack4Rt(NcnnGraphSession.MemoryDataPack promptMemoryPack, int featureChannels, out RenderTexture promptRt)
        {
            promptRt = null;
            if (promptMemoryPack == null || promptMemoryPack.cpuData == null)
                return false;
            if (promptMemoryPack.dims != 1)
                return false;

            var channels = Mathf.Max(1, featureChannels);
            if (promptMemoryPack.w < channels || promptMemoryPack.cpuData.Length < channels)
                return false;

            var packs = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            if (promptMemoryPack.pack4Rt != null
                && promptMemoryPack.pack4RtChannels == channels
                && promptMemoryPack.pack4RtDepth == packs
                && promptMemoryPack.pack4Rt.IsCreated())
            {
                promptRt = promptMemoryPack.pack4Rt;
                return true;
            }

            try
            {
                if (promptMemoryPack.pack4Rt != null)
                {
                    NcnnGpuResourceTracker.ReleaseTexture(promptMemoryPack.pack4Rt, "VistaPromptPack4Rt.recreate");
                    promptMemoryPack.pack4Rt.Release();
                    UnityEngine.Object.DestroyImmediate(promptMemoryPack.pack4Rt);
                }
            }
            catch { }

            var desc = new RenderTextureDescriptor(1, 1, RenderTextureFormat.ARGBFloat, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = packs,
                enableRandomWrite = false,
                msaaSamples = 1,
            };
            var rt = new RenderTexture(desc)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                name = "VistaPromptPack4Rt"
            };
            rt.Create();
            NcnnGpuResourceTracker.RegisterTexture(rt, "VistaPromptPack4Rt");

            var upload = new Texture2DArray(1, 1, packs, TextureFormat.RGBAFloat, false, true)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                name = "VistaPromptPack4Upload"
            };
            try
            {
                for (var pack = 0; pack < packs; pack++)
                {
                    var baseIndex = pack * 4;
                    var x = baseIndex + 0 < channels ? promptMemoryPack.cpuData[baseIndex + 0] : 0f;
                    var y = baseIndex + 1 < channels ? promptMemoryPack.cpuData[baseIndex + 1] : 0f;
                    var z = baseIndex + 2 < channels ? promptMemoryPack.cpuData[baseIndex + 2] : 0f;
                    var w = baseIndex + 3 < channels ? promptMemoryPack.cpuData[baseIndex + 3] : 0f;
                    upload.SetPixels(new[] { new Color(x, y, z, w) }, pack, 0);
                }
                upload.Apply(false, true);

                for (var pack = 0; pack < packs; pack++)
                    Graphics.CopyTexture(upload, pack, 0, rt, pack, 0);
            }
            finally
            {
                UnityEngine.Object.DestroyImmediate(upload);
            }

            promptMemoryPack.pack4Rt = rt;
            promptMemoryPack.pack4RtChannels = channels;
            promptMemoryPack.pack4RtDepth = packs;
            promptRt = rt;
            return true;
        }
    }
}
