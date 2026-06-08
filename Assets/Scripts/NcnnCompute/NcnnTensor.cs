using System;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnTensor : IDisposable
    {
        public int width { get; }
        public int height { get; }
        public RenderTexture rt { get; }

        public NcnnTensor(int width, int height)
        {
            if (width <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (height <= 0) throw new ArgumentOutOfRangeException(nameof(height));
            this.width = width;
            this.height = height;

            rt = new RenderTexture(width, height, 0, RenderTextureFormat.ARGBFloat, RenderTextureReadWrite.Linear);
            rt.enableRandomWrite = true;
            rt.wrapMode = TextureWrapMode.Clamp;
            rt.filterMode = FilterMode.Point;
            rt.Create();
            NcnnGpuResourceTracker.RegisterTexture(rt, "NcnnTensor");
        }

        public void Dispose()
        {
            try
            {
                if (rt != null)
                {
                    NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnTensor");
                    rt.Release();
                    UnityEngine.Object.Destroy(rt);
                }
            }
            catch
            {
            }
        }
    }
}
