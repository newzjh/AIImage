using System;
using UnityEngine;

namespace Aexis.Execution
{
    public sealed class AexisComputeBackend
    {
        private readonly ComputeShader _cs;
        private readonly int _kPassthrough;

        public AexisComputeBackend()
        {
            _cs = AexisComputeShaderLoader.LoadOrThrow();
            _kPassthrough = _cs.FindKernel("AexisPassthrough");
        }

        public AexisTensor Passthrough(Texture src)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            var outT = new AexisTensor(src.width, src.height);
            _cs.SetTexture(_kPassthrough, "_AexisIn", src);
            _cs.SetTexture(_kPassthrough, "_AexisOut", outT.rt);
            Dispatch2D(_cs, _kPassthrough, src.width, src.height, 8, 8);
            return outT;
        }

        private static void Dispatch2D(ComputeShader cs, int kernel, int w, int h, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cs.Dispatch(kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);
        }
    }
}
