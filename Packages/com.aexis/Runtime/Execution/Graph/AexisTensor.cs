using System;
using UnityEngine;

namespace Aexis.Execution
{
    public sealed class AexisTensor : IDisposable
    {
        public int width { get; }
        public int height { get; }
        public RenderTexture rt { get; }
        public AexisTextureMat repoVkMat { get; }

        public AexisTensor(int width, int height)
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
            AexisGpuResourceTracker.RegisterTexture(rt, "AexisTensor");
            repoVkMat = new AexisTextureMat(
                rt,
                new AexisGraphSession.BufferShape(2, width, height, 1, 1),
                new AexisGraphSession.BufferShape(2, width, height, 1, 1));
        }

        public void Dispose()
        {
            try
            {
                if (rt != null)
                {
                    AexisGpuResourceTracker.ReleaseTexture(rt, "AexisTensor");
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
