#if UNITY_EDITOR && AEXIS_INCLUDE_EDITOR_TESTS
using System;
using System.IO;
using System.Security.Cryptography;
using Aexis.Ncnn;
using Aexis.Samples.Json.Linq;
using NUnit.Framework;
using AIImage.Qwen35;
using UnityEngine;
using Aexis.Execution;

public sealed class Qwen35Q8ArchiveTests
{
    [Test]
    public void TextureArgMaxReducesVocabularyAxis()
    {
        const int vocabularySize = 17;
        const int expectedToken = 13;
        var source = new Texture2D(vocabularySize, 1, TextureFormat.RFloat, false, true);
        var logits = new RenderTexture(vocabularySize, 1, 0, RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true
        };
        var output = new RenderTexture(1, 1, 0, RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true
        };
        try
        {
            var pixels = new Color[vocabularySize];
            for (var i = 0; i < pixels.Length; i++) pixels[i].r = -i;
            pixels[expectedToken].r = 100f;
            source.SetPixels(pixels);
            source.Apply(false, false);
            Assert.That(logits.Create(), Is.True);
            Assert.That(output.Create(), Is.True);
            Graphics.CopyTexture(source, logits);

            using var ops = new AexisOps();
            var inputShape = new AexisGraphSession.BufferShape(2, vocabularySize, 1, 1, 1);
            var outputShape = new AexisGraphSession.BufferShape(2, 1, 1, 1, 1);
            ops.AexisArgReduceLinearMat(
                logits,
                inputShape,
                inputShape,
                inputShape.dims - 1,
                true,
                false,
                true,
                outputShape,
                outputShape,
                output);

            Assert.That(Mathf.RoundToInt(AexisGraphSession.ReadScalarTexture(output)), Is.EqualTo(expectedToken));
        }
        finally
        {
            logits.Release();
            output.Release();
            UnityEngine.Object.DestroyImmediate(logits);
            UnityEngine.Object.DestroyImmediate(output);
            UnityEngine.Object.DestroyImmediate(source);
        }
    }

    [Test]
    public void TextureArgMaxReducesTiledPack4Vocabulary()
    {
        const int vocabularySize = 37;
        const int expectedToken = 34;
        const int tileWidth = 4;
        var packCount = (vocabularySize + 3) / 4;
        var tileRows = Mathf.CeilToInt(packCount / (float)tileWidth);
        var source = new Texture2DArray(tileWidth, tileRows, 1, TextureFormat.RGBAFloat, false, true);
        var descriptor = new RenderTextureDescriptor(tileWidth, tileRows, RenderTextureFormat.ARGBFloat, 0)
        {
            dimension = UnityEngine.Rendering.TextureDimension.Tex2DArray,
            volumeDepth = 1,
            enableRandomWrite = true,
            msaaSamples = 1
        };
        var logits = new RenderTexture(descriptor);
        var output = new RenderTexture(1, 1, 0, RenderTextureFormat.RFloat, RenderTextureReadWrite.Linear)
        {
            enableRandomWrite = true
        };
        try
        {
            var pixels = new Color[tileWidth * tileRows];
            for (var pack = 0; pack < packCount; pack++)
            {
                var values = new Color(
                    -(pack * 4 + 1),
                    -(pack * 4 + 2),
                    -(pack * 4 + 3),
                    -(pack * 4 + 4));
                pixels[pack] = values;
            }
            var expectedPack = expectedToken / 4;
            var expectedLane = expectedToken % 4;
            var expectedValues = pixels[expectedPack];
            expectedValues[expectedLane] = 100f;
            pixels[expectedPack] = expectedValues;
            source.SetPixels(pixels, 0);
            source.Apply(false, false);
            Assert.That(logits.Create(), Is.True);
            Assert.That(output.Create(), Is.True);
            Graphics.CopyTexture(source, logits);

            using var ops = new AexisOps();
            ops.ArgMaxPack4LinearMat(logits, vocabularySize, 1, tileRows, output);

            Assert.That(Mathf.RoundToInt(AexisGraphSession.ReadScalarTexture(output)), Is.EqualTo(expectedToken));
        }
        finally
        {
            logits.Release();
            output.Release();
            UnityEngine.Object.DestroyImmediate(logits);
            UnityEngine.Object.DestroyImmediate(output);
            UnityEngine.Object.DestroyImmediate(source);
        }
    }

    [Test]
    public void CaptureRoundTripPreservesRecordBoundariesAndTolerance()
    {
        var sourceValues = new float[8193];
        for (var i = 0; i < sourceValues.Length; i++) sourceValues[i] = (float)Math.Sin(i * 0.017) * (1f + i % 13);
        using var source = BuildRawNcnnMat(sourceValues);
        using var archive = new MemoryStream();
        using (var writer = new NcnnQ8ArchiveWriter(archive, source.Length, defaultBlockSize: 256, fp32Threshold: 16))
        using (var reader = new NcnnBinReader(source, writer))
        {
            var captured = reader.ReadNcnnMatAsFloat32(sourceValues.Length, 0, 0, 0, 0);
            Assert.That(captured, Is.EqualTo(sourceValues));
            Assert.That(source.Position, Is.EqualTo(source.Length));
        }

        archive.Position = 0;
        using var q8Reader = new NcnnBinReader(archive);
        Assert.That(q8Reader.IsQ8Archive, Is.True);
        var restored = q8Reader.ReadNcnnMatAsFloat32(sourceValues.Length, 0, 0, 0, 0);
        var maxError = 0f;
        for (var i = 0; i < restored.Length; i++) maxError = Math.Max(maxError, Math.Abs(restored[i] - sourceValues[i]));
        Assert.That(maxError, Is.LessThanOrEqualTo(13f / 127f + 1e-6f));
    }

    [Test]
    public void SmallFp32RecordIsBitExact()
    {
        var expected = new[] { -0f, 1.25f, -9.5f, float.Epsilon };
        using var archive = new MemoryStream();
        using (var writer = new NcnnQ8ArchiveWriter(archive, 20, defaultBlockSize: 256, fp32Threshold: 4096))
            writer.WriteNcnnArray(expected);
        archive.Position = 0;
        using var reader = new NcnnBinReader(archive);
        var actual = reader.ReadNcnnMatAsFloat32(expected.Length, 0, 0, 0, 0);
        CollectionAssert.AreEqual(expected, actual);
    }

    [Test]
    public void RawLoadTypeConstantRemainsFp32AboveQuantizationThreshold()
    {
        var expected = new float[5000];
        for (var i = 0; i < expected.Length; i++) expected[i] = BitConverter.Int32BitsToSingle(0x3f000000 + i);
        using var source = new MemoryStream();
        using (var sourceWriter = new BinaryWriter(source, System.Text.Encoding.UTF8, true))
            foreach (var value in expected) sourceWriter.Write(value);
        source.Position = 0;
        using var archive = new MemoryStream();
        using (var writer = new NcnnQ8ArchiveWriter(archive, source.Length, defaultBlockSize: 256, fp32Threshold: 16))
        using (var capture = new NcnnBinReader(source, writer))
            CollectionAssert.AreEqual(expected, capture.ReadNcnnMatAsFloat32(expected.Length, 0, 0, 0, 1));
        archive.Position = 0;
        using var reader = new NcnnBinReader(archive);
        CollectionAssert.AreEqual(expected, reader.ReadNcnnMatAsFloat32(expected.Length, 0, 0, 0, 1));
    }

    [Test]
    public void PackedRowWiseReadUsesOneScalePerRow()
    {
        const int rows = 3;
        const int columns = 8;
        var values = new float[rows * columns];
        for (var row = 0; row < rows; row++)
            for (var column = 0; column < columns; column++)
                values[row * columns + column] = (row + 1) * (column - 3.5f);
        using var archive = new MemoryStream();
        using (var writer = new NcnnQ8ArchiveWriter(archive, values.Length * sizeof(float), columns, 0))
            writer.WriteQ8(values, columns);
        archive.Position = 0;
        using var reader = new NcnnBinReader(archive);
        var packed = reader.ReadQ8NcnnMatPacked(values.Length, columns);
        Assert.That(packed.Scales.Length, Is.EqualTo(rows));
        Assert.That(packed.PackedValues.Length, Is.EqualTo((values.Length + 3) / 4));
    }

    [Test]
    public void ManifestBackedShardStreamSeeksAcrossPartBoundary()
    {
        var root = Path.Combine(Path.GetTempPath(), "aiimage-qwen35-shards-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(Path.Combine(root, "weights"));
        try
        {
            var part0 = Path.Combine(root, "weights", "p0");
            var part1 = Path.Combine(root, "weights", "p1");
            File.WriteAllBytes(part0, new byte[] { 1, 2, 3 });
            File.WriteAllBytes(part1, new byte[] { 4, 5, 6, 7 });
            var manifest = new JObject
            {
                ["schema"] = Qwen35MobileAssetSet.ManifestSchema,
                ["weight_only"] = true,
                ["logical_files"] = new JObject
                {
                    ["test.bin"] = new JObject
                    {
                        ["stored_bytes"] = 7,
                        ["parts"] = new JArray(
                            Part("weights/p0", part0),
                            Part("weights/p1", part1))
                    }
                }
            };
            File.WriteAllText(Path.Combine(root, Qwen35MobileAssetSet.ManifestFileName), manifest.ToString());
            var assets = Qwen35MobileAssetSet.TryLoad(root, verifyHashes: true);
            using var stream = assets.OpenRead("test.bin");
            stream.Seek(2, SeekOrigin.Begin);
            var bytes = new byte[4];
            Assert.That(stream.Read(bytes, 0, bytes.Length), Is.EqualTo(bytes.Length));
            CollectionAssert.AreEqual(new byte[] { 3, 4, 5, 6 }, bytes);
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    [Test]
    public void ModelDirectoryResolverSelectsRequestedModelFromCollectionRoot()
    {
        var root = Path.Combine(Path.GetTempPath(), "aiimage-qwen35-models-" + Guid.NewGuid().ToString("N"));
        var modelName = "qwen3.5_0.8b_mobile_q8";
        var modelDirectory = Path.Combine(root, modelName);
        Directory.CreateDirectory(modelDirectory);
        try
        {
            File.WriteAllText(Path.Combine(modelDirectory, "model.json"), "{}");
            Assert.That(Qwen35ModelDirectoryResolver.Resolve(root, modelName), Is.EqualTo(modelDirectory));
            Assert.That(Qwen35ModelDirectoryResolver.Resolve(modelDirectory, "other"), Is.EqualTo(modelDirectory));
        }
        finally
        {
            if (Directory.Exists(root)) Directory.Delete(root, true);
        }
    }

    [Test]
    public void VisionTargetFitsPlayerTextureLimitForLargeImage()
    {
        const int maxTextureSize = 16384;
        const int maxTextureArraySlices = 2048;
        const int maxPatches = maxTextureArraySlices * 4 / 3;
        var target = Qwen35VisionPreprocessor.TargetImageSize(
            3648,
            2752,
            Qwen35VisionEncoderSession.PatchSize,
            maxPatches,
            maxTextureSize);

        Assert.That(target.x, Is.LessThanOrEqualTo(maxTextureSize));
        Assert.That(target.y, Is.LessThanOrEqualTo(maxTextureSize));
        Assert.That(target.x % (Qwen35VisionEncoderSession.PatchSize * Qwen35VisionEncoderSession.SpatialMergeSize), Is.EqualTo(0));
        Assert.That(target.y % (Qwen35VisionEncoderSession.PatchSize * Qwen35VisionEncoderSession.SpatialMergeSize), Is.EqualTo(0));
        Assert.That(
            (long)(target.x / Qwen35VisionEncoderSession.PatchSize)
            * (target.y / Qwen35VisionEncoderSession.PatchSize),
            Is.LessThanOrEqualTo(maxPatches));
    }

    [Test]
    public void VisionTargetFitsPlayerTextureArraySliceLimitForP1010085()
    {
        const int maxTextureSize = 16384;
        const int maxTextureArraySlices = 2048;
        const int maxPatches = maxTextureArraySlices * 4 / 3;
        var target = Qwen35VisionPreprocessor.TargetImageSize(
            1773,
            2364,
            Qwen35VisionEncoderSession.PatchSize,
            maxPatches,
            maxTextureSize);
        var patchCount = (long)(target.x / Qwen35VisionEncoderSession.PatchSize)
            * (target.y / Qwen35VisionEncoderSession.PatchSize);

        Assert.That(patchCount, Is.LessThanOrEqualTo(maxPatches));
        Assert.That((patchCount * 3 + 3) / 4, Is.LessThanOrEqualTo(maxTextureArraySlices));
    }

    private static MemoryStream BuildRawNcnnMat(float[] values)
    {
        var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, System.Text.Encoding.UTF8, true))
        {
            writer.Write(0u);
            foreach (var value in values) writer.Write(value);
        }
        stream.Position = 0;
        return stream;
    }

    private static JObject Part(string relative, string path)
    {
        using var sha = SHA256.Create();
        using var stream = File.OpenRead(path);
        var hash = BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", string.Empty).ToLowerInvariant();
        return new JObject { ["file"] = relative, ["bytes"] = new FileInfo(path).Length, ["sha256"] = hash };
    }
}
#endif
