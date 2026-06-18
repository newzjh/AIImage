using System;
using UnityEngine;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace NcnnCompute
{
    internal static class NcnnComputeShaderLoader
    {
        private const string ResourceName = "NcnnCompute";
        private const string EditorAssetPath = "Assets/Resources/NcnnCompute.compute";

        public static ComputeShader LoadOrThrow()
        {
            var shader = Resources.Load<ComputeShader>(ResourceName);
#if UNITY_EDITOR
            if (shader == null)
                shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(EditorAssetPath);
#endif
            if (shader == null)
                throw new InvalidOperationException("ComputeShader not found: Assets/Resources/NcnnCompute.compute");
            return shader;
        }
    }
}
