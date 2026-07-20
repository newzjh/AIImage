using Aexis.Ncnn;
using NUnit.Framework;
using AIImage.Qwen35;
using UnityEngine;
using UnityEngine.Rendering;

public sealed class Qwen35DeviceCompatibilityTests
{
    [Test]
    public void AndroidVulkanWithValidatedQ8AssetsIsSupported()
    {
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, SupportedAndroid());
        Assert.That(result.Supported, Is.True, string.Join("\n", result.UnsupportedReasons));
    }

    [Test]
    public void MobilePlayerRejectsUnquantizedAssets()
    {
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(false, SupportedAndroid());
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("q8 sharded asset set"));
    }

    [Test]
    public void AndroidWithEightGigabytesIsRejectedUntilLowerTierIsQualified()
    {
        var probe = new Qwen35DeviceCapabilities(
            RuntimePlatform.Android,
            GraphicsDeviceType.Vulkan,
            8192,
            2048,
            8192,
            2048,
            268435456,
            50,
            true,
            true);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("system memory"));
    }

    [Test]
    public void AndroidRejectsNonVulkanGraphicsApi()
    {
        var probe = SupportedAndroid(GraphicsDeviceType.OpenGLES3);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("requires Vulkan"));
    }

    [Test]
    public void IosRejectsNonMetalGraphicsApi()
    {
        var probe = SupportedMobile(RuntimePlatform.IPhonePlayer, GraphicsDeviceType.Vulkan);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("requires Metal"));
    }

    [Test]
    public void SharedTokenBufferCapacityIsRequired()
    {
        var probe = new Qwen35DeviceCapabilities(
            RuntimePlatform.Android,
            GraphicsDeviceType.Vulkan,
            Qwen35DeviceCompatibility.MinimumSystemMemoryMb,
            2048,
            8192,
            2048,
            Qwen35DeviceCompatibility.MinimumGraphicsBufferBytes - 1,
            50,
            true,
            true);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("shared q8 token matrix"));
    }

    [Test]
    public void UnknownSharedTokenBufferCapacityIsRejected()
    {
        var probe = new Qwen35DeviceCapabilities(
            RuntimePlatform.Android,
            GraphicsDeviceType.Vulkan,
            8192,
            2048,
            8192,
            2048,
            0,
            50,
            true,
            true);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("cannot be verified"));
    }

    [Test]
    public void MinimumMemoryAndTextureLimitsAreEnforced()
    {
        var probe = new Qwen35DeviceCapabilities(
            RuntimePlatform.Android,
            GraphicsDeviceType.Vulkan,
            Qwen35DeviceCompatibility.MinimumSystemMemoryMb - 1,
            2048,
            Qwen35DeviceCompatibility.MinimumTextureSize - 1,
            Qwen35DeviceCompatibility.MinimumTextureArraySlices - 1,
            Qwen35DeviceCompatibility.MinimumGraphicsBufferBytes,
            50,
            true,
            true);
        var result = Qwen35DeviceCompatibility.EvaluateCapabilities(true, probe);
        Assert.That(result.Supported, Is.False);
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("system memory"));
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("maxTextureSize"));
        Assert.That(result.UnsupportedReasons, Has.Some.Contains("maxTextureArraySlices"));
    }

    private static Qwen35DeviceCapabilities SupportedAndroid(GraphicsDeviceType api = GraphicsDeviceType.Vulkan)
    {
        return SupportedMobile(RuntimePlatform.Android, api);
    }

    private static Qwen35DeviceCapabilities SupportedMobile(RuntimePlatform platform, GraphicsDeviceType api)
    {
        return new Qwen35DeviceCapabilities(
            platform,
            api,
            Qwen35DeviceCompatibility.MinimumSystemMemoryMb,
            2048,
            8192,
            2048,
            268435456,
            50,
            true,
            true);
    }
}
