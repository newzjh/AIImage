using System;
using UnityEngine;
using UnityEngine.Rendering;
using UnityEngine.Experimental.Rendering;
#if UNITY_EDITOR
using UnityEditor;
#endif

namespace AIImage.Inference.UnityGpu
{
    // Texture-native D3 lowering. Dynamic logical lengths are written to RInt shape
    // textures and remain GPU-resident for downstream nodes.
    public sealed class SentisTextureBackend
    {
        private const string ResourceName = "SentisOps";
        private const string EditorAssetPath = "Packages/com.aiimage.inference.kernels/Runtime/Resources/SentisOps.compute";
        private readonly ComputeShader shader;

        public SentisTextureBackend()
        {
            shader = Resources.Load<ComputeShader>(ResourceName);
#if UNITY_EDITOR
            if (shader == null) shader = AssetDatabase.LoadAssetAtPath<ComputeShader>(EditorAssetPath);
#endif
            if (shader == null) throw new InvalidOperationException("SentisOps.compute is required for texture-native ONNX/Sentis D3 execution.");
        }

        public static RenderTexture CreateShapeTexture(string name)
        {
            var texture = new RenderTexture(new RenderTextureDescriptor(2, 1, GraphicsFormat.R32_SInt, 0) { enableRandomWrite = true }) { name = name ?? "SentisShape" };
            texture.Create();
            return texture;
        }

        public void NonZero(CommandBuffer cmd, RenderTexture values, int inputLength, int capacity, RenderTexture outputIndices, RenderTexture shape)
        {
            var kernel = shader.FindKernel("SentisNonZeroLinear");
            Bind(cmd, kernel, values, null, null, null, inputLength, capacity, 0, null, outputIndices, shape);
            cmd.DispatchCompute(shader, kernel, 1, 1, 1);
        }
        public void Compress(CommandBuffer cmd, RenderTexture values, RenderTexture condition, int inputLength, int capacity, RenderTexture output, RenderTexture shape)
            => DispatchSerial(cmd, "SentisCompressLinear", values, condition, null, inputLength, capacity, 0, output, null, shape);
        public void GatherND(CommandBuffer cmd, RenderTexture values, RenderTexture indices, int inputLength, int outputLength, RenderTexture output)
            => DispatchLinear(cmd, "SentisGatherNDLinear", values, indices, null, inputLength, outputLength, 0, output, null, null, 64);
        public void ScatterUnique(CommandBuffer cmd, RenderTexture data, RenderTexture indices, RenderTexture updates, int inputLength, int updateLength, RenderTexture output)
            => DispatchSerial(cmd, "SentisScatterLinear", data, indices, updates, inputLength, updateLength, 0, output, null, null);
        public void TopK(CommandBuffer cmd, RenderTexture values, int inputLength, RenderTexture gpuK, int capacity, bool largest, RenderTexture outputValues, RenderTexture outputIndices, RenderTexture shape)
        {
            var kernel = shader.FindKernel("SentisTopKLinear");
            Bind(cmd, kernel, values, null, gpuK, null, inputLength, capacity, largest ? 1 : 0, outputValues, outputIndices, shape);
            cmd.DispatchCompute(shader, kernel, 1, 1, 1);
        }
        public void OneHot(CommandBuffer cmd, RenderTexture indices, int inputLength, RenderTexture gpuDepth, int capacity, float offValue, float onValue, RenderTexture output, RenderTexture shape)
        {
            var kernel = shader.FindKernel("SentisOneHotLinear"); Bind(cmd, kernel, null, indices, gpuDepth, null, inputLength, capacity, 0, output, null, shape); cmd.SetComputeFloatParam(shader, "_OffValue", offValue); cmd.SetComputeFloatParam(shader, "_OnValue", onValue); cmd.DispatchCompute(shader, kernel, Mathf.Max(1, (inputLength * capacity + 63) / 64), 1, 1);
        }

        private void DispatchSerial(CommandBuffer cmd, string name, RenderTexture values, RenderTexture indices, RenderTexture updates, int inputLength, int capacity, int mode, RenderTexture output, RenderTexture indexOutput, RenderTexture shape)
        { var kernel = shader.FindKernel(name); Bind(cmd, kernel, values, indices, null, updates, inputLength, capacity, mode, output, indexOutput, shape); cmd.DispatchCompute(shader, kernel, 1, 1, 1); }
        private void DispatchLinear(CommandBuffer cmd, string name, RenderTexture values, RenderTexture indices, RenderTexture updates, int inputLength, int capacity, int mode, RenderTexture output, RenderTexture indexOutput, RenderTexture shape, int threads)
        { var kernel = shader.FindKernel(name); Bind(cmd, kernel, values, indices, null, updates, inputLength, capacity, mode, output, indexOutput, shape); cmd.DispatchCompute(shader, kernel, Mathf.Max(1, (capacity + threads - 1) / threads), 1, 1); }
        private void Bind(CommandBuffer cmd, int kernel, RenderTexture values, RenderTexture indices, RenderTexture parameter, RenderTexture updates, int inputLength, int capacity, int mode, RenderTexture output, RenderTexture indexOutput, RenderTexture shape)
        {
            if (cmd == null || (output == null && indexOutput == null)) throw new ArgumentNullException(cmd == null ? nameof(cmd) : nameof(output));
            cmd.SetComputeIntParam(shader, "_InputLength", inputLength); cmd.SetComputeIntParam(shader, "_Capacity", capacity); cmd.SetComputeIntParam(shader, "_Largest", mode);
            if (values != null) cmd.SetComputeTextureParam(shader, kernel, "_Values", values);
            if (indices != null) cmd.SetComputeTextureParam(shader, kernel, "_Indices", indices);
            if (parameter != null) cmd.SetComputeTextureParam(shader, kernel, "_Parameter", parameter);
            if (updates != null) cmd.SetComputeTextureParam(shader, kernel, "_Updates", updates);
            if (output != null) cmd.SetComputeTextureParam(shader, kernel, "_Output", output);
            if (indexOutput != null) cmd.SetComputeTextureParam(shader, kernel, "_IndexOutput", indexOutput);
            if (shape != null) cmd.SetComputeTextureParam(shader, kernel, "_ShapeOut", shape);
        }
    }
}
