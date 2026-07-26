using System;
using System.Collections.Generic;
using Aexis.Samples.Json.Linq;
using UnityEngine;
using UnityEngine.Rendering;

namespace AIImage.Qwen35
{
    public readonly struct Qwen35DeviceCapabilities
    {
        public Qwen35DeviceCapabilities(
            RuntimePlatform platform,
            GraphicsDeviceType graphicsApi,
            int systemMemoryMb,
            int graphicsMemoryMb,
            int maxTextureSize,
            int maxTextureArraySlices,
            long maxGraphicsBufferBytes,
            int graphicsShaderLevel,
            bool supportsComputeShaders,
            bool supportsRgbaFloat)
        {
            Platform = platform;
            GraphicsApi = graphicsApi;
            SystemMemoryMb = systemMemoryMb;
            GraphicsMemoryMb = graphicsMemoryMb;
            MaxTextureSize = maxTextureSize;
            MaxTextureArraySlices = maxTextureArraySlices;
            MaxGraphicsBufferBytes = maxGraphicsBufferBytes;
            GraphicsShaderLevel = graphicsShaderLevel;
            SupportsComputeShaders = supportsComputeShaders;
            SupportsRgbaFloat = supportsRgbaFloat;
        }

        public RuntimePlatform Platform { get; }
        public GraphicsDeviceType GraphicsApi { get; }
        public int SystemMemoryMb { get; }
        public int GraphicsMemoryMb { get; }
        public int MaxTextureSize { get; }
        public int MaxTextureArraySlices { get; }
        public long MaxGraphicsBufferBytes { get; }
        public int GraphicsShaderLevel { get; }
        public bool SupportsComputeShaders { get; }
        public bool SupportsRgbaFloat { get; }

        public static Qwen35DeviceCapabilities CaptureCurrent()
        {
            return new Qwen35DeviceCapabilities(
                Application.platform,
                SystemInfo.graphicsDeviceType,
                SystemInfo.systemMemorySize,
                SystemInfo.graphicsMemorySize,
                SystemInfo.maxTextureSize,
                SystemInfo.maxTextureArraySlices,
                SystemInfo.maxGraphicsBufferSize,
                SystemInfo.graphicsShaderLevel,
                SystemInfo.supportsComputeShaders,
                SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.ARGBFloat));
        }
    }

    public sealed class Qwen35DeviceCompatibility
    {
        // Desktop Q8 runs peak near 4.7 GiB private memory plus substantial
        // texture/driver residency. Do not start the mobile pipeline on 6-8 GiB
        // devices until a lower on-device tier is measured and qualified.
        public const int MinimumSystemMemoryMb = 12288;
        public const int MinimumTextureSize = 4096;
        public const int MinimumTextureArraySlices = 256;
        public const long MinimumGraphicsBufferBytes = 254279680L;

        public readonly List<string> UnsupportedReasons = new List<string>();
        public RuntimePlatform Platform { get; private set; }
        public GraphicsDeviceType GraphicsApi { get; private set; }
        public int SystemMemoryMb { get; private set; }
        public int GraphicsMemoryMb { get; private set; }
        public int MaxTextureSize { get; private set; }
        public int MaxTextureArraySlices { get; private set; }
        public long MaxGraphicsBufferBytes { get; private set; }
        public int GraphicsShaderLevel { get; private set; }
        public bool SupportsComputeShaders { get; private set; }
        public bool SupportsRgbaFloat { get; private set; }
        public bool MobileAssetSet { get; private set; }
        public bool Supported => UnsupportedReasons.Count == 0;

        public static Qwen35DeviceCompatibility Evaluate(Qwen35ModelContract contract)
        {
            if (contract == null) throw new ArgumentNullException(nameof(contract));
            return EvaluateCapabilities(
                contract.MobileAssets != null && contract.MobileAssets.WeightOnly,
                Qwen35DeviceCapabilities.CaptureCurrent());
        }

        // This is intentionally internal and reserved for the Android smoke harness.
        // It permits an on-device qualification run below the shipping memory floor
        // without changing the normal player compatibility decision.
        internal static Qwen35DeviceCompatibility EvaluateForDeviceQualification(Qwen35ModelContract contract)
        {
            if (contract == null) throw new ArgumentNullException(nameof(contract));
            return EvaluateCapabilitiesForDeviceQualification(
                contract.MobileAssets != null && contract.MobileAssets.WeightOnly,
                Qwen35DeviceCapabilities.CaptureCurrent());
        }

        internal static Qwen35DeviceCompatibility EvaluateCapabilitiesForDeviceQualification(
            bool hasValidatedMobileAssets,
            Qwen35DeviceCapabilities capabilities)
        {
            return EvaluateCapabilities(hasValidatedMobileAssets, capabilities, ignoreSystemMemoryFloor: true);
        }

        public static Qwen35DeviceCompatibility EvaluateCapabilities(
            bool hasValidatedMobileAssets,
            Qwen35DeviceCapabilities capabilities)
        {
            return EvaluateCapabilities(hasValidatedMobileAssets, capabilities, ignoreSystemMemoryFloor: false);
        }

        private static Qwen35DeviceCompatibility EvaluateCapabilities(
            bool hasValidatedMobileAssets,
            Qwen35DeviceCapabilities capabilities,
            bool ignoreSystemMemoryFloor)
        {
            var result = new Qwen35DeviceCompatibility
            {
                Platform = capabilities.Platform,
                GraphicsApi = capabilities.GraphicsApi,
                SystemMemoryMb = capabilities.SystemMemoryMb,
                GraphicsMemoryMb = capabilities.GraphicsMemoryMb,
                MaxTextureSize = capabilities.MaxTextureSize,
                MaxTextureArraySlices = capabilities.MaxTextureArraySlices,
                MaxGraphicsBufferBytes = capabilities.MaxGraphicsBufferBytes,
                GraphicsShaderLevel = capabilities.GraphicsShaderLevel,
                SupportsComputeShaders = capabilities.SupportsComputeShaders,
                SupportsRgbaFloat = capabilities.SupportsRgbaFloat,
                MobileAssetSet = hasValidatedMobileAssets
            };

            var mobilePlayer = result.Platform == RuntimePlatform.Android || result.Platform == RuntimePlatform.IPhonePlayer;
            if (mobilePlayer && !result.MobileAssetSet)
                result.UnsupportedReasons.Add("Qwen3.5 mobile player requires the validated q8 sharded asset set.");
            if (!ignoreSystemMemoryFloor
                && result.SystemMemoryMb > 0
                && result.SystemMemoryMb < MinimumSystemMemoryMb)
                result.UnsupportedReasons.Add("system memory is below " + MinimumSystemMemoryMb + " MiB: " + result.SystemMemoryMb + " MiB");
            if (!result.SupportsComputeShaders)
                result.UnsupportedReasons.Add("ComputeShader support is unavailable.");
            if (!result.SupportsRgbaFloat)
                result.UnsupportedReasons.Add("RGBAFloat RenderTexture support is unavailable.");
            if (result.GraphicsShaderLevel < 45)
                result.UnsupportedReasons.Add("graphics shader level is below 4.5: " + result.GraphicsShaderLevel);
            if (result.MaxTextureSize < MinimumTextureSize)
                result.UnsupportedReasons.Add("maxTextureSize is below " + MinimumTextureSize + ": " + result.MaxTextureSize);
            if (result.MaxTextureArraySlices < MinimumTextureArraySlices)
                result.UnsupportedReasons.Add("maxTextureArraySlices is below " + MinimumTextureArraySlices + ": " + result.MaxTextureArraySlices);
            if (result.MaxGraphicsBufferBytes <= 0)
                result.UnsupportedReasons.Add("maxGraphicsBufferSize is unavailable; shared q8 token matrix capacity cannot be verified.");
            else if (result.MaxGraphicsBufferBytes < MinimumGraphicsBufferBytes)
                result.UnsupportedReasons.Add("maxGraphicsBufferSize cannot hold the shared q8 token matrix: " + result.MaxGraphicsBufferBytes);

            if (result.Platform == RuntimePlatform.Android && result.GraphicsApi != GraphicsDeviceType.Vulkan)
                result.UnsupportedReasons.Add("Android Qwen3.5 inference requires Vulkan; active API is " + result.GraphicsApi + ".");
            else if (result.Platform == RuntimePlatform.IPhonePlayer && result.GraphicsApi != GraphicsDeviceType.Metal)
                result.UnsupportedReasons.Add("iOS Qwen3.5 inference requires Metal; active API is " + result.GraphicsApi + ".");
            else if (result.Platform == RuntimePlatform.WindowsEditor
                || result.Platform == RuntimePlatform.OSXEditor
                || result.Platform == RuntimePlatform.LinuxEditor)
            {
                if (result.GraphicsApi != GraphicsDeviceType.Direct3D11
                    && result.GraphicsApi != GraphicsDeviceType.Direct3D12
                    && result.GraphicsApi != GraphicsDeviceType.Vulkan
                    && result.GraphicsApi != GraphicsDeviceType.Metal)
                    result.UnsupportedReasons.Add("Editor graphics API is outside the validated compute set: " + result.GraphicsApi + ".");
            }
            return result;
        }

        public void ThrowIfUnsupported()
        {
            if (!Supported)
                throw new NotSupportedException("Qwen3.5 device is unsupported:\n" + string.Join("\n", UnsupportedReasons));
        }

        public JObject ToJson()
        {
            return new JObject
            {
                ["supported"] = Supported,
                ["platform"] = Platform.ToString(),
                ["graphics_api"] = GraphicsApi.ToString(),
                ["system_memory_mb"] = SystemMemoryMb,
                ["graphics_memory_mb"] = GraphicsMemoryMb,
                ["max_texture_size"] = MaxTextureSize,
                ["max_texture_array_slices"] = MaxTextureArraySlices,
                ["max_graphics_buffer_bytes"] = MaxGraphicsBufferBytes,
                ["graphics_shader_level"] = GraphicsShaderLevel,
                ["supports_compute_shaders"] = SupportsComputeShaders,
                ["supports_rgba_float"] = SupportsRgbaFloat,
                ["mobile_q8_assets"] = MobileAssetSet,
                ["minimum_system_memory_mb"] = MinimumSystemMemoryMb,
                ["minimum_texture_size"] = MinimumTextureSize,
                ["minimum_texture_array_slices"] = MinimumTextureArraySlices,
                ["minimum_graphics_buffer_bytes"] = MinimumGraphicsBufferBytes,
                ["unsupported_reasons"] = new JArray(UnsupportedReasons)
            };
        }
    }
}
