using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    // Migration note: avoid expanding the legacy compute-buffer path; prefer pack4 RT execution, and plan for ComputeTexture command-buffer pack4 RT for async compute and temporary RT allocation support.
    public sealed class NcnnMemoryDataLayerRepro : NcnnBaseLayerRepro
    {
        public NcnnMemoryDataLayerRepro() : base(NcnnLayerTypes.MemoryData, supportsBufferPath: true, supportsCommandBufferPath: true) { }

        public override NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
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
                                        var a = NcnnRepro.ReadClipMatAsFloat32(br, w, h, d, c, loadType);
                                        phaseSw.Stop();
                                        readMs += phaseSw.ElapsedMilliseconds;

                                        var dims = 1;
                                        if (h > 0) dims = 2;
                                        if (c > 0) dims = d > 0 ? 4 : 3;
                                        var memoryPack = new NcnnRepro.MemoryDataPack
                                        {
                                            dims = dims,
                                            w = Mathf.Max(1, w),
                                            h = Mathf.Max(1, h),
                                            d = Mathf.Max(1, d),
                                            c = Mathf.Max(1, c),
                                            cpuData = a
                                        };
                                        if (ShouldUseVistaPromptPack4RtOnly(owner, layer, memoryPack))
                                        {
                                            phaseSw.Restart();
                                            var createdVistaPromptRt = TryGetOrCreateVistaPromptPack4Rt(memoryPack, memoryPack.w, out _);
                                            phaseSw.Stop();
                                            uploadMs += phaseSw.ElapsedMilliseconds;
                                            if (createdVistaPromptRt)
                                            {
                                                owner._memoryData[layer.name] = memoryPack;
                                                return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
                                            }
                                        }

                                        phaseSw.Restart();
                                        var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                                        buf.SetData(a);
                                        phaseSw.Stop();
                                        uploadMs += phaseSw.ElapsedMilliseconds;

                                        memoryPack.data = buf;
                                        owner._memoryData[layer.name] = memoryPack;
                                        return new NcnnRepro.LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }
        [Obsolete(ComputeBufferPathObsoleteMessage)]
        public override void ExecuteComputeBufferPath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
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

                                                if (mp.data == null && !TryCreateMemoryDataBuffer(mp))
                                                    throw new InvalidOperationException("MemoryData buffer not found: " + layer.name);

                                                if (owner.ShouldBlockPack4BufferFallback() && mp.dims <= 4)
                                                {
                                                    var logicalShape = new NcnnRepro.BufferShape(mp.dims, mp.w, mp.h, mp.d, mp.c);
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

        public override void ExecuteRenderTexturePath(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
#pragma warning disable CS0618
            ExecuteComputeBufferPath(owner, layer, context);
#pragma warning restore CS0618
        }

        public override void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            var cmd = context.commandBuffer;
            var blobs = context.blobs;
            var shapes = context.shapes;

            if (!owner._memoryData.TryGetValue(layer.name, out var mp))
                throw new InvalidOperationException("MemoryData not found: " + layer.name);
            if (TryPublishVistaPromptPack4CmdBlob(owner, cmd, layer, mp, blobs, shapes))
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

        private static bool ShouldUseVistaPromptPack4RtOnly(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnRepro.MemoryDataPack memoryPack)
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
            NcnnRepro owner,
            NcnnParamModel.Layer layer,
            NcnnRepro.MemoryDataPack memoryPack,
            Dictionary<string, NcnnRepro.TensorRef> textureBlobs,
            Dictionary<string, NcnnRepro.BufferShape> textureShapes)
        {
            if (!ShouldUseVistaPromptPack4RtOnly(owner, layer, memoryPack))
                return false;
            if (!TryGetOrCreateVistaPromptPack4Rt(memoryPack, memoryPack.w, out var promptRt) || promptRt == null)
                return false;
            if (layer.topNames == null || layer.topNames.Length == 0 || string.IsNullOrWhiteSpace(layer.topNames[0]))
                return false;

            var logicalShape = new NcnnRepro.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            textureBlobs[layer.topNames[0]] = new NcnnRepro.TensorRef
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
            NcnnRepro owner,
            CommandBuffer cmd,
            NcnnParamModel.Layer layer,
            NcnnRepro.MemoryDataPack memoryPack,
            Dictionary<string, NcnnRepro.CmdTensorRef> blobs,
            Dictionary<string, NcnnRepro.BufferShape> shapes)
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

            var logicalShape = new NcnnRepro.BufferShape(memoryPack.dims, memoryPack.w, memoryPack.h, memoryPack.d, memoryPack.c);
            blobs[layer.topNames[0]] = NcnnRepro.CreateCmdTensorRef(outArr, logicalShape, logicalShape, owned: true);
            if (shapes != null)
                shapes[layer.topNames[0]] = logicalShape;
            owner.DebugLog?.Invoke(
                "[MemoryDataPack4Rt][cmd] direct publish"
                + " | layer=" + layer.name
                + " | top=" + layer.topNames[0]
                + " | packs=" + packs.ToString(CultureInfo.InvariantCulture));
            return true;
        }

        private static bool TryCreateMemoryDataBuffer(NcnnRepro.MemoryDataPack memoryPack)
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

        internal static bool TryGetOrCreateVistaPromptPack4Rt(NcnnRepro.MemoryDataPack promptMemoryPack, int featureChannels, out RenderTexture promptRt)
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
