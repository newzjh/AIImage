using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using Aexis.Ncnn;
using UnityEngine;
using UnityEngine.Rendering;
using Aexis.Execution;

namespace AIImage.Qwen35
{
    public sealed class Qwen35VisionEncoding : IDisposable
    {
        public Qwen35OwnedTexture Embeddings { get; private set; }
        public int SourceWidth { get; }
        public int SourceHeight { get; }
        public int TargetWidth { get; }
        public int TargetHeight { get; }
        public int GridWidth { get; }
        public int GridHeight { get; }
        public int EmbeddingCount => (GridWidth / 2) * (GridHeight / 2);

        internal Qwen35VisionEncoding(
            Qwen35OwnedTexture embeddings,
            int sourceWidth,
            int sourceHeight,
            int targetWidth,
            int targetHeight,
            int gridWidth,
            int gridHeight)
        {
            Embeddings = embeddings ?? throw new ArgumentNullException(nameof(embeddings));
            SourceWidth = sourceWidth;
            SourceHeight = sourceHeight;
            TargetWidth = targetWidth;
            TargetHeight = targetHeight;
            GridWidth = gridWidth;
            GridHeight = gridHeight;
        }

        public void Dispose()
        {
            Embeddings?.Dispose();
            Embeddings = null;
        }

        public Qwen35VisionEncoding CloneStandalone()
        {
            if (Embeddings == null || Embeddings.Texture == null)
                throw new ObjectDisposedException(nameof(Qwen35VisionEncoding));

            var source = Embeddings.Texture;
            var descriptor = source.descriptor;
            descriptor.enableRandomWrite = true;
            descriptor.msaaSamples = 1;
            var copy = new RenderTexture(descriptor)
            {
                name = "Qwen35VisionEmbeddingStandalone"
            };
            try
            {
                if (!copy.Create())
                    throw new InvalidOperationException("Failed to allocate standalone Qwen3.5 vision embedding texture.");
                AexisGpuResourceTracker.RegisterTexture(copy, copy.name);
                Graphics.CopyTexture(source, copy);
                return new Qwen35VisionEncoding(
                    new Qwen35OwnedTexture(copy, Embeddings.LogicalShape, ReleaseStandaloneTexture),
                    SourceWidth,
                    SourceHeight,
                    TargetWidth,
                    TargetHeight,
                    GridWidth,
                    GridHeight);
            }
            catch
            {
                ReleaseStandaloneTexture(copy);
                throw;
            }
        }

        private static void ReleaseStandaloneTexture(RenderTexture texture)
        {
            if (texture == null) return;
            AexisGpuResourceTracker.ReleaseTexture(texture, "Qwen35VisionEmbeddingStandalone.Dispose");
            try { texture.Release(); } catch { }
            if (Application.isPlaying) UnityEngine.Object.Destroy(texture);
            else UnityEngine.Object.DestroyImmediate(texture);
        }

    }

    public sealed class Qwen35VisionEncoderSession : IDisposable
    {
        public const int PatchSize = 16;
        public const int PatchDimension = 768;
        public const int SpatialMergeSize = 2;
        private const int ModelMaximumPatchCount = 49152;
        // reshape_100 expands the patch sequence to three scalar channels stored in Pack4 arrays.
        private const int VisionEncoderArraySliceChannelsPerPatch = 3;
        private const int PackedTextureChannels = 4;

        private readonly AexisOps _ops;
        private readonly AexisGraphSession _patch;
        private readonly AexisGraphSession _position;
        private readonly AexisGraphSession _encoder;
        private readonly int _maximumPatchCount;
        private bool _disposed;

        public AexisGraphSession.ModelLoadProfile PatchEmbeddingLoadProfile => _patch.LastLoadProfile;
        public AexisGraphSession.ModelLoadProfile PositionEmbeddingLoadProfile => _position.LastLoadProfile;
        public AexisGraphSession.ModelLoadProfile EncoderLoadProfile => _encoder.LastLoadProfile;

        public Qwen35VisionEncoderSession(string modelDirectory)
            : this(modelDirectory, true)
        {
        }

        private Qwen35VisionEncoderSession(string modelDirectory, bool loadSynchronously)
        {
            if (string.IsNullOrWhiteSpace(modelDirectory))
                throw new ArgumentException("Model directory is empty.", nameof(modelDirectory));

            _ops = new AexisOps();
            try
            {
                _maximumPatchCount = Qwen35RuntimeTuning.ResolveMaximumVisionPatchCount(modelDirectory);
                _patch = CreateRepro(modelDirectory);
                _position = CreateRepro(modelDirectory);
                _encoder = CreateRepro(modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_patch, modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_position, modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_encoder, modelDirectory);
                if (loadSynchronously)
                {
                    Load(_patch, modelDirectory, "qwen3.5_vision_embed_patch.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.bin");
                    Load(_position, modelDirectory, "qwen3.5_vision_embed_pos.ncnn.param", "qwen3.5_vision_embed_pos.ncnn.bin");
                    Load(_encoder, modelDirectory, "qwen3.5_vision_encoder.ncnn.param", "qwen3.5_vision_encoder.ncnn.bin");
                }
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public static async UniTask<Qwen35VisionEncoderSession> CreateAsync(
            string modelDirectory,
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            var session = new Qwen35VisionEncoderSession(modelDirectory, false);
            try
            {
                var networks = new[]
                {
                    new NetworkLoad("vision_embed_patch", "qwen3.5_vision_embed_patch.ncnn.param", "qwen3.5_vision_embed_patch.ncnn.bin", session._patch),
                    new NetworkLoad("vision_embed_pos", "qwen3.5_vision_embed_pos.ncnn.param", "qwen3.5_vision_embed_pos.ncnn.bin", session._position),
                    new NetworkLoad("vision_encoder", "qwen3.5_vision_encoder.ncnn.param", "qwen3.5_vision_encoder.ncnn.bin", session._encoder)
                };
                long totalBytes = 0;
                for (var i = 0; i < networks.Length; i++)
                {
                    networks[i].StoredBytes = Math.Max(1, Qwen35ModelAssetResolver.GetStoredBytes(modelDirectory, networks[i].BinName));
                    totalBytes = checked(totalBytes + networks[i].StoredBytes);
                }

                long completedBytes = 0;
                for (var i = 0; i < networks.Length; i++)
                {
                    var network = networks[i];
                    var start = totalBytes > 0 ? (float)((double)completedBytes / totalBytes) : 0f;
                    var end = totalBytes > 0 ? (float)((double)(completedBytes + network.StoredBytes) / totalBytes) : 1f;
                    await LoadAsync(
                        network.Repro,
                        modelDirectory,
                        network.ParamName,
                        network.BinName,
                        progress => onProgress?.Invoke(new Qwen35Progress(
                            "loading_vision",
                            network.Name + " " + (progress.layerName ?? progress.stage),
                            Mathf.Lerp(start, end, progress.progress01),
                            progress.layerIndex,
                            progress.layerCount)),
                        cancellationToken);
                    completedBytes += network.StoredBytes;
                }
                onProgress?.Invoke(new Qwen35Progress("loading_vision", "Vision networks ready", 1f));
                return session;
            }
            catch
            {
                session.Dispose();
                throw;
            }
        }

        public Action<string> DebugLog
        {
            set
            {
                _patch.DebugLog = value;
                _position.DebugLog = value;
                _encoder.DebugLog = value;
            }
        }

        // Explicit audit-only hooks for the patch, position and encoder networks.
        public void ConfigureDebugLayerReadback(
            ISet<string> patchBlobNames,
            ISet<string> positionBlobNames,
            ISet<string> encoderBlobNames,
            Action<string, string, string, float[]> callback)
        {
            _patch.DebugLayerReadbackBlobs = patchBlobNames;
            _patch.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback("vision_embed_patch", layerName, blobName, values);
            _position.DebugLayerReadbackBlobs = positionBlobNames;
            _position.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback("vision_embed_pos", layerName, blobName, values);
            _encoder.DebugLayerReadbackBlobs = encoderBlobNames;
            _encoder.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback("vision_encoder", layerName, blobName, values);
        }

        public Qwen35VisionEncoding EncodeFile(string imagePath)
        {
            ThrowIfDisposed();
            if (string.IsNullOrWhiteSpace(imagePath) || !File.Exists(imagePath))
                throw new FileNotFoundException("Qwen3.5 image input is missing.", imagePath);
            var image = new Texture2D(2, 2, TextureFormat.RGBA32, false, true);
            try
            {
                if (!ImageConversion.LoadImage(image, File.ReadAllBytes(imagePath), false))
                    throw new InvalidDataException("Unity failed to decode Qwen3.5 image: " + imagePath);
                return Encode(image);
            }
            finally
            {
                DestroyUnityObject(image);
            }
        }

        public Qwen35VisionEncoding Encode(Texture2D image)
        {
            ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));

            var maxTextureSize = Mathf.Max(1, SystemInfo.maxTextureSize);
            var maxTextureArraySlices = Mathf.Max(1, SystemInfo.maxTextureArraySlices);
            var maxPatchCount = ResolveMaximumPatchCount(maxTextureSize, maxTextureArraySlices);
            var target = Qwen35VisionPreprocessor.TargetImageSize(
                image.height,
                image.width,
                PatchSize,
                maxPatchCount,
                maxTextureSize);
            var gridWidth = target.x / PatchSize;
            var gridHeight = target.y / PatchSize;
            var patchCount = checked(gridWidth * gridHeight);
            if (patchCount > maxPatchCount)
                throw new InvalidOperationException(
                    "Qwen3.5 vision patch count exceeds the active texture budget after preprocessing: patches="
                    + patchCount + " maxTextureSize=" + maxTextureSize
                    + " maxTextureArraySlices=" + maxTextureArraySlices + ".");
            var embeddingCount = checked((gridWidth / SpatialMergeSize) * (gridHeight / SpatialMergeSize));
            var normalized = Qwen35VisionPreprocessor.ResizeNormalize(image, target.x, target.y);
            var patches = Qwen35VisionPreprocessor.BuildDuplicatedPatches(normalized, target.x, target.y);
            Qwen35VisionPreprocessor.BuildVisionRope2D(gridHeight, gridWidth, out var ropeCosine, out var ropeSine);

            RenderTexture atlas = null;
            RenderTexture patchSpatial = null;
            RenderTexture patchLinear = null;
            RenderTexture positionGrid = null;
            RenderTexture positionRaw = null;
            RenderTexture positionMerged = null;
            RenderTexture cosine = null;
            RenderTexture sine = null;
            Qwen35OwnedTexture output = null;
            try
            {
                atlas = CreatePatchAtlasTexture(patches, target.x, target.y);
                using (var result = _patch.InferWithMultiInputs(
                    new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = atlas },
                    null,
                    null,
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(4, target.x, target.y, 2, 3)
                    }))
                {
                    patchSpatial = result.ExtractTexture("out0");
                }
                patchLinear = _patch.RentTempArray(PatchDimension / 4, patchCount, 1, RenderTextureFormat.ARGBFloat);
                _ops.Pack4SpatialToPack4Linear(patchSpatial, patchLinear);

                positionGrid = _position.RentTempMat(gridWidth, gridHeight, AexisGraphSession.ResolveLinearMatTextureFormat());
                _ops.FillScalarTexture(new[] { 0f }, positionGrid);
                using (var result = _position.InferWithMultiInputs(
                    new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = positionGrid },
                    null,
                    null,
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(2, gridWidth, gridHeight, 1, 1)
                    }))
                {
                    positionRaw = result.ExtractTexture("out0");
                }
                if (positionRaw.dimension != TextureDimension.Tex2D
                    || positionRaw.width != PatchDimension
                    || positionRaw.height != patchCount)
                    throw new InvalidOperationException(
                        "Qwen3.5 position embedding has unexpected storage: "
                        + positionRaw.dimension + " " + positionRaw.width + "x" + positionRaw.height + "x" + positionRaw.volumeDepth);
                positionMerged = _position.RentTempMat(PatchDimension, patchCount, AexisGraphSession.ResolveLinearMatTextureFormat());
                _ops.LinearMatReorderMergeRows(positionRaw, positionMerged, gridWidth, gridHeight, SpatialMergeSize);

                cosine = CreateScalarRowsTexture(ropeCosine, 64, 32, patchCount, "Qwen35VisionCosUpload");
                sine = CreateScalarRowsTexture(ropeSine, 64, 32, patchCount, "Qwen35VisionSinUpload");
                var inputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
                {
                    ["in0"] = patchLinear,
                    ["in1"] = positionMerged,
                    ["in2"] = cosine,
                    ["in3"] = sine
                };
                var shapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                {
                    ["in0"] = new AexisGraphSession.BufferShape(2, PatchDimension, patchCount, 1, 1),
                    ["in1"] = new AexisGraphSession.BufferShape(2, PatchDimension, patchCount, 1, 1),
                    ["in2"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1),
                    ["in3"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1)
                };
                using (var result = _encoder.InferWithMultiInputs(inputs, null, null, shapes))
                {
                    if (!result.TryGetLogicalShape("out0", out var dims, out var w, out var h, out var d, out var c))
                        throw new InvalidOperationException("Qwen3.5 vision encoder output shape is unavailable.");
                    if (dims != 2 || w != Qwen35DecoderSession.HiddenSize || h != embeddingCount || d != 1 || c != 1)
                        throw new InvalidOperationException(
                            "Qwen3.5 vision encoder output shape mismatch: dims=" + dims + " w=" + w + " h=" + h + " d=" + d + " c=" + c);
                    output = new Qwen35OwnedTexture(
                        _encoder,
                        result.ExtractTexture("out0"),
                        new AexisGraphSession.BufferShape(dims, w, h, d, c));
                }

                var encoding = new Qwen35VisionEncoding(
                    output,
                    image.width,
                    image.height,
                    target.x,
                    target.y,
                    gridWidth,
                    gridHeight);
                output = null;
                return encoding;
            }
            finally
            {
                output?.Dispose();
                if (atlas != null) _patch.ReturnTempArray(atlas);
                if (patchSpatial != null) _patch.ReturnTempArray(patchSpatial);
                if (patchLinear != null) _patch.ReturnTempArray(patchLinear);
                if (positionGrid != null) _position.ReturnTempArray(positionGrid);
                if (positionRaw != null) _position.ReturnTempArray(positionRaw);
                if (positionMerged != null) _position.ReturnTempArray(positionMerged);
                if (cosine != null) _encoder.ReturnTempArray(cosine);
                if (sine != null) _encoder.ReturnTempArray(sine);
            }
        }

        /// <summary>
        /// Texture-only vision encoding that returns to the Player loop between the
        /// CPU upload stages and during graph execution. Unity textures stay on the
        /// main thread; this is cooperative scheduling, not a worker-thread path.
        /// </summary>
        public async UniTask<Qwen35VisionEncoding> EncodeAsync(
            Texture2D image,
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            ThrowIfDisposed();
            if (image == null) throw new ArgumentNullException(nameof(image));
            cancellationToken.ThrowIfCancellationRequested();

            var maxTextureSize = Mathf.Max(1, SystemInfo.maxTextureSize);
            var maxTextureArraySlices = Mathf.Max(1, SystemInfo.maxTextureArraySlices);
            var maxPatchCount = ResolveMaximumPatchCount(maxTextureSize, maxTextureArraySlices);
            var target = Qwen35VisionPreprocessor.TargetImageSize(
                image.height,
                image.width,
                PatchSize,
                maxPatchCount,
                maxTextureSize);
            var gridWidth = target.x / PatchSize;
            var gridHeight = target.y / PatchSize;
            var patchCount = checked(gridWidth * gridHeight);
            if (patchCount > maxPatchCount)
                throw new InvalidOperationException(
                    "Qwen3.5 vision patch count exceeds the active texture budget after preprocessing: patches="
                    + patchCount + " maxTextureSize=" + maxTextureSize
                    + " maxTextureArraySlices=" + maxTextureArraySlices + ".");
            var embeddingCount = checked((gridWidth / SpatialMergeSize) * (gridHeight / SpatialMergeSize));

            onProgress?.Invoke(new Qwen35Progress("encoding_image", "Resizing and normalizing image", 0.04f));
            await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
            var normalized = Qwen35VisionPreprocessor.ResizeNormalize(image, target.x, target.y);
            cancellationToken.ThrowIfCancellationRequested();

            onProgress?.Invoke(new Qwen35Progress("encoding_image", "Building vision patches", 0.13f));
            await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
            var patches = Qwen35VisionPreprocessor.BuildDuplicatedPatches(normalized, target.x, target.y);
            Qwen35VisionPreprocessor.BuildVisionRope2D(gridHeight, gridWidth, out var ropeCosine, out var ropeSine);
            cancellationToken.ThrowIfCancellationRequested();

            RenderTexture atlas = null;
            RenderTexture patchSpatial = null;
            RenderTexture patchLinear = null;
            RenderTexture positionGrid = null;
            RenderTexture positionRaw = null;
            RenderTexture positionMerged = null;
            RenderTexture cosine = null;
            RenderTexture sine = null;
            Qwen35OwnedTexture output = null;
            try
            {
                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Uploading vision patches", 0.2f));
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                atlas = CreatePatchAtlasTexture(patches, target.x, target.y);
                cancellationToken.ThrowIfCancellationRequested();

                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Running patch embedding", 0.3f));
                using (var result = await _patch.InferWithMultiInputsAsync(
                    new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = atlas },
                    null,
                    null,
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(4, target.x, target.y, 2, 3)
                    },
                    cancellationToken: cancellationToken))
                {
                    patchSpatial = result.ExtractTexture("out0");
                }
                cancellationToken.ThrowIfCancellationRequested();

                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Reordering vision patches", 0.43f));
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                patchLinear = _patch.RentTempArray(PatchDimension / 4, patchCount, 1, RenderTextureFormat.ARGBFloat);
                _ops.Pack4SpatialToPack4Linear(patchSpatial, patchLinear);
                cancellationToken.ThrowIfCancellationRequested();

                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Running position embedding", 0.5f));
                positionGrid = _position.RentTempMat(gridWidth, gridHeight, AexisGraphSession.ResolveLinearMatTextureFormat());
                _ops.FillScalarTexture(new[] { 0f }, positionGrid);
                using (var result = await _position.InferWithMultiInputsAsync(
                    new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = positionGrid },
                    null,
                    null,
                    new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                    {
                        ["in0"] = new AexisGraphSession.BufferShape(2, gridWidth, gridHeight, 1, 1)
                    },
                    cancellationToken: cancellationToken))
                {
                    positionRaw = result.ExtractTexture("out0");
                }
                cancellationToken.ThrowIfCancellationRequested();
                if (positionRaw.dimension != TextureDimension.Tex2D
                    || positionRaw.width != PatchDimension
                    || positionRaw.height != patchCount)
                    throw new InvalidOperationException(
                        "Qwen3.5 position embedding has unexpected storage: "
                        + positionRaw.dimension + " " + positionRaw.width + "x" + positionRaw.height + "x" + positionRaw.volumeDepth);

                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Preparing vision position data", 0.62f));
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                positionMerged = _position.RentTempMat(PatchDimension, patchCount, AexisGraphSession.ResolveLinearMatTextureFormat());
                _ops.LinearMatReorderMergeRows(positionRaw, positionMerged, gridWidth, gridHeight, SpatialMergeSize);
                cosine = CreateScalarRowsTexture(ropeCosine, 64, 32, patchCount, "Qwen35VisionCosUpload");
                sine = CreateScalarRowsTexture(ropeSine, 64, 32, patchCount, "Qwen35VisionSinUpload");
                cancellationToken.ThrowIfCancellationRequested();

                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Running vision encoder", 0.72f));
                var inputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
                {
                    ["in0"] = patchLinear,
                    ["in1"] = positionMerged,
                    ["in2"] = cosine,
                    ["in3"] = sine
                };
                var shapes = new Dictionary<string, AexisGraphSession.BufferShape>(StringComparer.Ordinal)
                {
                    ["in0"] = new AexisGraphSession.BufferShape(2, PatchDimension, patchCount, 1, 1),
                    ["in1"] = new AexisGraphSession.BufferShape(2, PatchDimension, patchCount, 1, 1),
                    ["in2"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1),
                    ["in3"] = new AexisGraphSession.BufferShape(3, 32, patchCount, 1, 1)
                };
                using (var result = await _encoder.InferWithMultiInputsAsync(
                    inputs,
                    null,
                    null,
                    shapes,
                    cancellationToken: cancellationToken))
                {
                    if (!result.TryGetLogicalShape("out0", out var dims, out var w, out var h, out var d, out var c))
                        throw new InvalidOperationException("Qwen3.5 vision encoder output shape is unavailable.");
                    if (dims != 2 || w != Qwen35DecoderSession.HiddenSize || h != embeddingCount || d != 1 || c != 1)
                        throw new InvalidOperationException(
                            "Qwen3.5 vision encoder output shape mismatch: dims=" + dims + " w=" + w + " h=" + h + " d=" + d + " c=" + c);
                    output = new Qwen35OwnedTexture(
                        _encoder,
                        result.ExtractTexture("out0"),
                        new AexisGraphSession.BufferShape(dims, w, h, d, c));
                }
                cancellationToken.ThrowIfCancellationRequested();

                var encoding = new Qwen35VisionEncoding(
                    output,
                    image.width,
                    image.height,
                    target.x,
                    target.y,
                    gridWidth,
                    gridHeight);
                output = null;
                onProgress?.Invoke(new Qwen35Progress("encoding_image", "Vision encoding ready", 1f));
                return encoding;
            }
            finally
            {
                output?.Dispose();
                if (atlas != null) _patch.ReturnTempArray(atlas);
                if (patchSpatial != null) _patch.ReturnTempArray(patchSpatial);
                if (patchLinear != null) _patch.ReturnTempArray(patchLinear);
                if (positionGrid != null) _position.ReturnTempArray(positionGrid);
                if (positionRaw != null) _position.ReturnTempArray(positionRaw);
                if (positionMerged != null) _position.ReturnTempArray(positionMerged);
                if (cosine != null) _encoder.ReturnTempArray(cosine);
                if (sine != null) _encoder.ReturnTempArray(sine);
            }
        }

        private RenderTexture CreatePatchAtlasTexture(float[] patches, int atlasWidth, int atlasHeight)
        {
            var gridWidth = atlasWidth / PatchSize;
            var gridHeight = atlasHeight / PatchSize;
            var patchCount = gridWidth * gridHeight;
            var elementsPerPatch = PatchSize * PatchSize * 2 * 3;
            if (atlasWidth % PatchSize != 0 || atlasHeight % PatchSize != 0)
                throw new ArgumentException("Qwen3.5 patch atlas must be patch aligned.");
            if (patches == null || patches.Length != patchCount * elementsPerPatch)
                throw new ArgumentException("Qwen3.5 patch atlas value count mismatch.", nameof(patches));

            var texture = _patch.RentTempArray(atlasWidth, atlasHeight, 2, RenderTextureFormat.ARGBFloat);
            var uploadFormat = texture.format == RenderTextureFormat.ARGBHalf
                ? TextureFormat.RGBAHalf
                : TextureFormat.RGBAFloat;
            var upload = new Texture2DArray(atlasWidth, atlasHeight, 2, uploadFormat, false, true)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                name = "Qwen35PatchAtlasUpload"
            };
            try
            {
                for (var temporal = 0; temporal < 2; temporal++)
                {
                    var pixels = new Color[atlasWidth * atlasHeight];
                    for (var y = 0; y < atlasHeight; y++)
                    {
                        var patchY = y / PatchSize;
                        var localY = y - patchY * PatchSize;
                        for (var x = 0; x < atlasWidth; x++)
                        {
                            var patchX = x / PatchSize;
                            var localX = x - patchX * PatchSize;
                            var patchIndex = patchY * gridWidth + patchX;
                            var patchBase = patchIndex * elementsPerPatch;
                            float Read(int channel)
                            {
                                if (channel >= 3) return 0f;
                                return patches[patchBase + (((channel * 2 + temporal) * PatchSize + localY) * PatchSize + localX)];
                            }
                            pixels[y * atlasWidth + x] = new Color(Read(0), Read(1), Read(2), 0f);
                        }
                    }
                    upload.SetPixels(pixels, temporal, 0);
                }
                upload.Apply(false, true);
                Graphics.CopyTexture(upload, 0, 0, texture, 0, 0);
                Graphics.CopyTexture(upload, 1, 0, texture, 1, 0);
                return texture;
            }
            catch
            {
                _patch.ReturnTempArray(texture);
                throw;
            }
            finally
            {
                DestroyUnityObject(upload);
            }
        }

        private RenderTexture CreateScalarRowsTexture(
            float[] values,
            int sourceRowWidth,
            int outputWidth,
            int rows,
            string uploadName)
        {
            if (values == null || values.Length != sourceRowWidth * rows)
                throw new ArgumentException("Qwen3.5 scalar-row upload value count mismatch.", nameof(values));
            var texture = _encoder.RentTempArray(outputWidth, rows, 1, RenderTextureFormat.ARGBFloat);
            var uploadFormat = texture.format == RenderTextureFormat.ARGBHalf
                ? TextureFormat.RGBAHalf
                : TextureFormat.RGBAFloat;
            var upload = new Texture2DArray(outputWidth, rows, 1, uploadFormat, false, true)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                name = uploadName
            };
            try
            {
                var pixels = new Color[outputWidth * rows];
                for (var row = 0; row < rows; row++)
                for (var x = 0; x < outputWidth; x++)
                    pixels[row * outputWidth + x] = new Color(values[row * sourceRowWidth + x], 0f, 0f, 0f);
                upload.SetPixels(pixels, 0, 0);
                upload.Apply(false, true);
                Graphics.CopyTexture(upload, 0, 0, texture, 0, 0);
                return texture;
            }
            catch
            {
                _encoder.ReturnTempArray(texture);
                throw;
            }
            finally
            {
                DestroyUnityObject(upload);
            }
        }

        private AexisGraphSession CreateRepro(string modelDirectory)
        {
            return new AexisGraphSession(_ops)
            {
                ExecutionMode = AexisInferenceExecutionMode.ProductionTextureOnly,
                DisallowInferenceTempComputeBuffers = true,
                DisallowBufferToTextureMaterialization = true,
                DisallowBufferOutputs = true,
                EnableAttentionMatMulPack4Specializations = true,
                EnableConv1x1TextureConvolution = true,
                EnableDepthWiseTextureConvolution = true,
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                StrictTextureTargetDtype = "FP32",
                ManagedLoadGarbageCollectionIntervalBytes = Qwen35RuntimeTuning.ResolveManagedLoadGarbageCollectionIntervalBytes(modelDirectory)
            };
        }

        private static void Load(AexisGraphSession repro, string directory, string paramName, string binName)
        {
            using (var stream = Qwen35ModelAssetResolver.OpenBin(directory, binName))
            using (var reader = new NcnnBinReader(stream))
                repro.LoadModel(File.ReadAllText(Path.Combine(directory, paramName)), reader);
        }

        private static async UniTask LoadAsync(
            AexisGraphSession repro,
            string directory,
            string paramName,
            string binName,
            Action<AexisGraphSession.LoadProgress> onProgress,
            CancellationToken cancellationToken)
        {
            using (var stream = Qwen35ModelAssetResolver.OpenBin(directory, binName))
            using (var reader = new NcnnBinReader(stream))
                await repro.LoadModelAsync(
                    File.ReadAllText(Path.Combine(directory, paramName)),
                    reader,
                    onProgress,
                    cancellationToken);
        }

        private sealed class NetworkLoad
        {
            public readonly string Name;
            public readonly string ParamName;
            public readonly string BinName;
            public readonly AexisGraphSession Repro;
            public long StoredBytes;

            public NetworkLoad(string name, string paramName, string binName, AexisGraphSession repro)
            {
                Name = name;
                ParamName = paramName;
                BinName = binName;
                Repro = repro;
            }
        }

        private static void DestroyUnityObject(UnityEngine.Object value)
        {
            if (value == null) return;
            if (Application.isPlaying) UnityEngine.Object.Destroy(value);
            else UnityEngine.Object.DestroyImmediate(value);
        }

        private int ResolveMaximumPatchCount(int maxTextureSize, int maxTextureArraySlices)
        {
            // The first vision-encoder reshape expands a patch sequence into three scalar
            // channels. Pack4 texture arrays therefore consume ceil(3 * patches / 4) slices.
            var byArraySlices = (int)Math.Min(
                int.MaxValue,
                (long)maxTextureArraySlices * PackedTextureChannels / VisionEncoderArraySliceChannelsPerPatch);
            return Mathf.Min(ModelMaximumPatchCount, _maximumPatchCount, maxTextureSize, byArraySlices);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Qwen35VisionEncoderSession));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _encoder?.Dispose(); } catch { }
            try { _position?.Dispose(); } catch { }
            try { _patch?.Dispose(); } catch { }
            try { _ops?.Dispose(); } catch { }
        }
    }
}
