using System;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public enum NcnnLayerPathPreference
    {
        Auto,
        Pack4Rt,
        Buffer
    }

    public sealed class NcnnLayerBufferContext
    {
        public Dictionary<string, NcnnRepro.TensorRef> textureBlobs;
        public Dictionary<string, NcnnRepro.BufferShape> textureShapes;
        public Dictionary<string, ComputeBuffer> bufferBlobs;
        public Dictionary<string, NcnnRepro.BufferRef> bufferRefs;
        public Dictionary<string, NcnnTensorBuffer> bufferViews;
        public Dictionary<string, NcnnRepro.IndexRef> indexBlobs;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
        public List<IDisposable> tempOwned;
    }

    public sealed class NcnnLayerCommandBufferContext
    {
        public CommandBuffer commandBuffer;
        public Dictionary<string, NcnnRepro.CmdTensorRef> blobs;
        public Dictionary<string, NcnnRepro.BufferShape> shapes;
        public Dictionary<string, int> remaining;
        public ICollection<string> pinnedNames;
    }

    // Migration guidance for all LayerRepro implementations:
    // 1. Keep the compute-buffer path only as a compatibility / truth-path fallback and avoid expanding new buffer-only branches.
    // 2. Prefer migrating execution to the pack4 RenderTexture path first, because it is the near-term primary runtime path.
    // 3. Long term, migrate toward the ComputeTexture-based ExecuteCommandBuffer pack4 RT path so async compute and command-buffer temporary RT allocation are both supported.
    public abstract class NcnnBaseLayerRepro
    {
        protected NcnnBaseLayerRepro(
            NcnnLayerTypeKey typeKey,
            bool supportsBufferPath,
            bool supportsCommandBufferPath)
        {
            TypeKey = typeKey;
            SupportsBufferPath = supportsBufferPath;
            SupportsCommandBufferPath = supportsCommandBufferPath;
        }

        public NcnnLayerTypeKey TypeKey { get; }
        public bool SupportsBufferPath { get; }
        public bool SupportsCommandBufferPath { get; }
        public NcnnLayerPathPreference PreferredPath { get; set; } = NcnnLayerPathPreference.Auto;

        public virtual NcnnRepro.LayerLoadMetrics LoadLayer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            return default;
        }

        public virtual void ExecuteBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerBufferContext context)
        {
            throw new NotSupportedException("Buffer path is not implemented for layer type: " + TypeKey);
        }

        public virtual void ExecuteCommandBuffer(NcnnRepro owner, NcnnParamModel.Layer layer, NcnnLayerCommandBufferContext context)
        {
            if (owner == null)
                throw new ArgumentNullException(nameof(owner));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (context == null)
                throw new ArgumentNullException(nameof(context));

            if (layer.bottomNames != null && layer.bottomNames.Length > 0)
            {
                new NcnnNoopLayerRepro().ExecuteCommandBuffer(owner, layer, context);
                return;
            }

            if (layer.topNames != null && layer.topNames.Length > 0)
                owner.PublishCmdTensorLikeInput(context.commandBuffer, layer.topNames[0], 1, 1, 1, context.blobs, context.shapes);
        }
    }
}
