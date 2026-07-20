using System;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using Aexis.Ncnn;
using UnityEngine;

namespace AIImage.Qwen35
{
    public sealed class Qwen35OwnedTexture : IDisposable
    {
        private Action<RenderTexture> _release;

        public RenderTexture Texture { get; private set; }
        public NcnnGraphSession.BufferShape LogicalShape { get; }

        internal Qwen35OwnedTexture(NcnnGraphSession owner, RenderTexture texture, NcnnGraphSession.BufferShape logicalShape)
        {
            if (owner == null) throw new ArgumentNullException(nameof(owner));
            Texture = texture ?? throw new ArgumentNullException(nameof(texture));
            LogicalShape = logicalShape;
            _release = owner.ReturnTempArray;
        }

        internal Qwen35OwnedTexture(
            RenderTexture texture,
            NcnnGraphSession.BufferShape logicalShape,
            Action<RenderTexture> release)
        {
            Texture = texture ?? throw new ArgumentNullException(nameof(texture));
            LogicalShape = logicalShape;
            _release = release ?? throw new ArgumentNullException(nameof(release));
        }

        public void Dispose()
        {
            var texture = Texture;
            var release = _release;
            Texture = null;
            _release = null;
            if (texture != null)
                release?.Invoke(texture);
        }
    }

    public sealed class Qwen35DecoderState : IDisposable
    {
        private readonly NcnnGraphSession _owner;
        private readonly Dictionary<string, RenderTexture> _textures =
            new Dictionary<string, RenderTexture>(StringComparer.Ordinal);
        private readonly Dictionary<string, NcnnGraphSession.BufferShape> _shapes =
            new Dictionary<string, NcnnGraphSession.BufferShape>(StringComparer.Ordinal);

        internal Qwen35DecoderState(NcnnGraphSession owner, int sequenceLength, int positionId)
        {
            _owner = owner ?? throw new ArgumentNullException(nameof(owner));
            SequenceLength = sequenceLength;
            PositionId = positionId;
        }

        public int SequenceLength { get; }
        public int PositionId { get; }
        public int TextureCount => _textures.Count;
        internal Dictionary<string, RenderTexture> Textures => _textures;
        internal Dictionary<string, NcnnGraphSession.BufferShape> Shapes => _shapes;

        internal void Add(string inputName, RenderTexture texture, NcnnGraphSession.BufferShape shape)
        {
            if (string.IsNullOrWhiteSpace(inputName))
                throw new ArgumentException("Cache input name is empty.", nameof(inputName));
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));
            _textures.Add(inputName, texture);
            _shapes.Add(inputName, shape);
        }

        internal void TransferTo(Qwen35DecoderState destination, string inputName, NcnnGraphSession.BufferShape nextShape)
        {
            if (destination == null) throw new ArgumentNullException(nameof(destination));
            if (!_textures.TryGetValue(inputName, out var texture) || !_shapes.TryGetValue(inputName, out var shape))
                throw new InvalidOperationException("Qwen3.5 decoder cache is missing: " + inputName);
            _textures.Remove(inputName);
            _shapes.Remove(inputName);
            destination.Add(inputName, texture, nextShape);
        }

        public void Dispose()
        {
            var released = new HashSet<RenderTexture>();
            foreach (var texture in _textures.Values)
            {
                if (texture != null && released.Add(texture))
                    _owner.ReturnTempArray(texture);
            }
            _textures.Clear();
            _shapes.Clear();
        }
    }

    public sealed class Qwen35DecoderStep : IDisposable
    {
        public Qwen35OwnedTexture Hidden { get; private set; }
        public Qwen35DecoderState State { get; private set; }

        internal Qwen35DecoderStep(Qwen35OwnedTexture hidden, Qwen35DecoderState state)
        {
            Hidden = hidden ?? throw new ArgumentNullException(nameof(hidden));
            State = state ?? throw new ArgumentNullException(nameof(state));
        }

        public Qwen35DecoderState DetachState()
        {
            var state = State;
            State = null;
            return state;
        }

        public Qwen35OwnedTexture DetachHidden()
        {
            var hidden = Hidden;
            Hidden = null;
            return hidden;
        }

        public void Dispose()
        {
            Hidden?.Dispose();
            Hidden = null;
            State?.Dispose();
            State = null;
        }
    }

    public sealed class Qwen35SamplingConfig
    {
        public bool DoSample = true;
        public float Temperature = 0.3f;
        public float TopP = 0.8f;
        public int TopK = 50;
        public float RepetitionPenalty = 1.1f;
        public int Seed = 20260718;

        public static Qwen35SamplingConfig Greedy()
        {
            return new Qwen35SamplingConfig { DoSample = false, Temperature = 0f, TopP = 1f, TopK = 0, RepetitionPenalty = 1f };
        }
    }

    public sealed class Qwen35GenerationResult
    {
        public readonly List<int> TokenIds = new List<int>();
        public string Text = string.Empty;
        public int PromptTokenCount;
        public int ExpandedPromptTokenCount;
        public int VisionTokenCount;
        public int FinalPosition;
        public bool StoppedOnEndOfTurn;
        public long DecoderStepCount;
        public int FinalCacheTextureCount;
        public int ContextTextureCapacity;
    }

    public sealed class Qwen35DecoderSession : IDisposable
    {
        public const int HiddenSize = 1024;
        public const int AttentionCacheCount = 6;
        public const int ConvCacheCount = 18;
        public const int GdrCacheCount = 18;
        public const int RopeHalfDimension = 32;
        public const float RopeTheta = 10000000f;

        private readonly NcnnOps _ops;
        private Qwen35SharedTokenEmbeddingWeights _sharedWeights;
        private readonly NcnnGraphSession _embed;
        private readonly NcnnGraphSession _decoder;
        private readonly NcnnGraphSession _projection;
        private readonly HashSet<string> _decoderOutputs = new HashSet<string>(StringComparer.Ordinal);
        private Action<int, string, string, float[]> _debugLayerReadback;
        private ISet<string> _debugLayerReadbackBlobs;
        private int _decodeInvocation;
        private bool _disposed;

        public Qwen35ByteLevelBpeTokenizer Tokenizer { get; }
        public long SharedWeightBytes => _sharedWeights.ByteCount;
        public NcnnGraphSession.ModelLoadProfile TokenEmbeddingLoadProfile => _embed.LastLoadProfile;
        public NcnnGraphSession.ModelLoadProfile DecoderLoadProfile => _decoder.LastLoadProfile;
        public NcnnGraphSession.ModelLoadProfile ProjectionLoadProfile => _projection.LastLoadProfile;
        // Diagnostics only. Normal Qwen execution does not allocate or time layer profiles.
        public NcnnGraphSession.LayerRuntimeProfile PrefillDecoderRuntimeProfile { get; private set; }
        public NcnnGraphSession.LayerRuntimeProfile FirstDecodeRuntimeProfile { get; private set; }
        public NcnnGraphSession.LayerRuntimeProfile LastDecodeRuntimeProfile { get; private set; }

        public Qwen35DecoderSession(string modelDirectory, Qwen35ByteLevelBpeTokenizer tokenizer)
            : this(modelDirectory, tokenizer, true)
        {
        }

        private Qwen35DecoderSession(
            string modelDirectory,
            Qwen35ByteLevelBpeTokenizer tokenizer,
            bool loadSynchronously)
        {
            if (string.IsNullOrWhiteSpace(modelDirectory))
                throw new ArgumentException("Model directory is empty.", nameof(modelDirectory));
            Tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));

            _ops = new NcnnOps();
            try
            {
                _embed = CreateRepro(modelDirectory);
                _decoder = CreateRepro(modelDirectory);
                _projection = CreateRepro(modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_embed, modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_decoder, modelDirectory);
                Qwen35ModelAssetResolver.ApplyMobilePrecisionManifest(_projection, modelDirectory);
                _decoder.AttentionKvCacheTextureCapacity = Mathf.Min(4096, Mathf.Max(1, SystemInfo.maxTextureSize));
                if (loadSynchronously)
                {
                    _sharedWeights = Qwen35SharedTokenEmbeddingWeights.LoadModelAsset(modelDirectory);
                    AttachSharedWeights();
                    CollectManagedLoadGarbage();
                    Load(_embed, modelDirectory, "qwen3.5_embed_token.ncnn.param", "qwen3.5_embed_token.ncnn.bin");
                    Load(_decoder, modelDirectory, "qwen3.5_decoder.ncnn.param", "qwen3.5_decoder.ncnn.bin");
                    Load(_projection, modelDirectory, "qwen3.5_proj_out.ncnn.param", "qwen3.5_embed_token.ncnn.bin");
                }
                InitializeDecoderOutputs();
            }
            catch
            {
                Dispose();
                throw;
            }
        }

        public static async UniTask<Qwen35DecoderSession> CreateAsync(
            string modelDirectory,
            Qwen35ByteLevelBpeTokenizer tokenizer,
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            var session = new Qwen35DecoderSession(modelDirectory, tokenizer, false);
            try
            {
                onProgress?.Invoke(new Qwen35Progress("loading_decoder", "Reading shared token weights", 0f));
                session._sharedWeights = await Qwen35SharedTokenEmbeddingWeights.LoadModelAssetAsync(
                    modelDirectory,
                    cancellationToken: cancellationToken);
                cancellationToken.ThrowIfCancellationRequested();
                session.AttachSharedWeights();
                CollectManagedLoadGarbage();
                onProgress?.Invoke(new Qwen35Progress("loading_decoder", "Shared token weights ready", 0.12f));

                await LoadAsync(
                    session._embed,
                    modelDirectory,
                    "qwen3.5_embed_token.ncnn.param",
                    "qwen3.5_embed_token.ncnn.bin",
                    progress => onProgress?.Invoke(MapNetworkProgress("embed_token", progress, 0.12f, 0.16f)),
                    cancellationToken);
                await LoadAsync(
                    session._decoder,
                    modelDirectory,
                    "qwen3.5_decoder.ncnn.param",
                    "qwen3.5_decoder.ncnn.bin",
                    progress => onProgress?.Invoke(MapNetworkProgress("decoder", progress, 0.16f, 0.96f)),
                    cancellationToken);
                await LoadAsync(
                    session._projection,
                    modelDirectory,
                    "qwen3.5_proj_out.ncnn.param",
                    "qwen3.5_embed_token.ncnn.bin",
                    progress => onProgress?.Invoke(MapNetworkProgress("proj_out", progress, 0.96f, 1f)),
                    cancellationToken);
                onProgress?.Invoke(new Qwen35Progress("loading_decoder", "Decoder networks ready", 1f));
                return session;
            }
            catch
            {
                session.Dispose();
                throw;
            }
        }

        private void AttachSharedWeights()
        {
            _sharedWeights.Attach(_embed);
            _sharedWeights.Attach(_projection);
        }

        private void InitializeDecoderOutputs()
        {
            _decoderOutputs.Add("out0");
            for (var i = 0; i < AttentionCacheCount; i++)
            {
                _decoderOutputs.Add("out_cache_k" + i);
                _decoderOutputs.Add("out_cache_v" + i);
            }
            for (var i = 0; i < ConvCacheCount; i++)
            {
                _decoderOutputs.Add("out_cache_conv" + i);
                _decoderOutputs.Add("out_cache_gdr" + i);
            }
        }

        private static Qwen35Progress MapNetworkProgress(
            string network,
            NcnnGraphSession.LoadProgress progress,
            float start,
            float end)
        {
            return new Qwen35Progress(
                "loading_decoder",
                network + " " + (progress.layerName ?? progress.stage),
                Mathf.Lerp(start, end, progress.progress01),
                progress.layerIndex,
                progress.layerCount);
        }

        public Action<string> DebugLog
        {
            set
            {
                _embed.DebugLog = value;
                _decoder.DebugLog = value;
                _projection.DebugLog = value;
            }
        }

        // Explicit numerical-alignment hook. It is never populated by normal inference.
        public Action<string, float[]> DebugTextureReadback { get; set; }

        public void ConfigureDebugLayerReadback(
            ISet<string> blobNames,
            Action<int, string, string, float[]> callback)
        {
            _decoder.DebugLayerReadbackBlobs = blobNames;
            _debugLayerReadbackBlobs = blobNames;
            _debugLayerReadback = callback;
            _decoder.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback(_decodeInvocation, layerName, blobName, values);
        }

        public void ConfigureDecoderRuntimeProfiling(bool enabled, bool synchronizeGpu)
        {
            ThrowIfDisposed();
            _decoder.LayerRuntimeProfileEnabled = enabled;
            _decoder.LayerRuntimeProfileSyncGpu = enabled && synchronizeGpu;
            if (!enabled)
            {
                PrefillDecoderRuntimeProfile = null;
                FirstDecodeRuntimeProfile = null;
                LastDecodeRuntimeProfile = null;
            }
        }

        // Explicit audit-only hooks for the two small networks sharing the token weights.
        public void ConfigureDebugAuxiliaryReadback(
            ISet<string> embedBlobNames,
            ISet<string> projectionBlobNames,
            Action<string, string, string, float[]> callback)
        {
            _embed.DebugLayerReadbackBlobs = embedBlobNames;
            _embed.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback("embed_token", layerName, blobName, values);
            _projection.DebugLayerReadbackBlobs = projectionBlobNames;
            _projection.DebugLayerTextureReadback = callback == null
                ? null
                : (layerName, blobName, values) => callback("proj_out", layerName, blobName, values);
        }

        public Qwen35DecoderState CreateInitialState()
        {
            ThrowIfDisposed();
            var state = new Qwen35DecoderState(_decoder, 0, 0);
            try
            {
                for (var i = 0; i < ConvCacheCount; i++)
                {
                    state.Add(
                        "cache_conv" + i,
                        CreateZeroPack4Texture(1536, 4, 1),
                        new NcnnGraphSession.BufferShape(2, 1536, 4, 1, 1));
                    state.Add(
                        "cache_gdr" + i,
                        CreateZeroPack4Texture(32, 128, 16),
                        new NcnnGraphSession.BufferShape(3, 32, 128, 1, 16));
                }
                return state;
            }
            catch
            {
                state.Dispose();
                throw;
            }
        }

        public Qwen35OwnedTexture EmbedTokens(IReadOnlyList<int> tokenIds)
        {
            ThrowIfDisposed();
            if (tokenIds == null || tokenIds.Count == 0)
                throw new ArgumentException("Token ids are empty.", nameof(tokenIds));

            using (var indices = new NcnnTensorBuffer(tokenIds.Count, 1))
            {
                var values = new int[tokenIds.Count];
                for (var i = 0; i < values.Length; i++)
                {
                    if (tokenIds[i] < 0 || tokenIds[i] >= Tokenizer.VocabularySize)
                        throw new ArgumentOutOfRangeException(nameof(tokenIds), "Token id is outside the tokenizer vocabulary: " + tokenIds[i]);
                    values[i] = tokenIds[i];
                }
                indices.buffer.SetData(values);
                using (var result = _embed.InferWithMultiInputs(
                    null,
                    new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal) { ["in0"] = indices }))
                {
                    var shape = GetShape(result, "out0");
                    return new Qwen35OwnedTexture(_embed, result.ExtractTexture("out0"), shape);
                }
            }
        }

        public Qwen35DecoderStep Decode(Qwen35OwnedTexture embeddings, int position, Qwen35DecoderState state)
        {
            var sequenceLength = embeddings?.LogicalShape.h ?? 0;
            return DecodeCore(embeddings, position, state, null, null, checked(position + sequenceLength));
        }

        private Qwen35DecoderStep DecodeCore(
            Qwen35OwnedTexture embeddings,
            int position,
            Qwen35DecoderState state,
            float[] customCosine,
            float[] customSine,
            int nextPosition)
        {
            ThrowIfDisposed();
            if (embeddings == null || embeddings.Texture == null)
                throw new ArgumentNullException(nameof(embeddings));
            if (state == null)
                throw new ArgumentNullException(nameof(state));
            if (position < 0)
                throw new ArgumentOutOfRangeException(nameof(position));
            if (embeddings.LogicalShape.dims != 2 || embeddings.LogicalShape.w != HiddenSize || embeddings.LogicalShape.h <= 0)
                throw new ArgumentException("Decoder embeddings must have logical shape [sequence, 1024].", nameof(embeddings));
            if (state.PositionId != position)
                throw new InvalidOperationException("Decoder position/cache mismatch: position=" + position + " cache_position=" + state.PositionId + " cache_sequence=" + state.SequenceLength);
            if (nextPosition < position)
                throw new ArgumentOutOfRangeException(nameof(nextPosition));

            var sequenceLength = embeddings.LogicalShape.h;
            if (state.SequenceLength + sequenceLength > _decoder.AttentionKvCacheTextureCapacity)
                throw new NotSupportedException(
                    "Qwen3.5 context exceeds the device texture-backed KV capacity: requested="
                    + (state.SequenceLength + sequenceLength) + " capacity=" + _decoder.AttentionKvCacheTextureCapacity);
            if ((customCosine == null) != (customSine == null))
                throw new ArgumentException("Custom RoPE cosine and sine must be supplied together.");
            if (customCosine != null && (customCosine.Length != sequenceLength * RopeHalfDimension || customSine.Length != customCosine.Length))
                throw new ArgumentException("Custom RoPE cache shape must be [sequence, 32].");
            RenderTexture mask = null;
            RenderTexture cosine = null;
            RenderTexture sine = null;
            Qwen35OwnedTexture hidden = null;
            Qwen35DecoderState nextState = null;
            try
            {
                var currentDecodeInvocation = _decodeInvocation;
                mask = CreateCausalMaskTexture(sequenceLength, state.SequenceLength);
                if (customCosine == null)
                {
                    CreateRopeTextures(sequenceLength, position, out cosine, out sine);
                }
                else
                {
                    cosine = UploadScalarPack4(customCosine, RopeHalfDimension, sequenceLength, "Qwen35MropeCosUpload");
                    try
                    {
                        sine = UploadScalarPack4(customSine, RopeHalfDimension, sequenceLength, "Qwen35MropeSinUpload");
                    }
                    catch
                    {
                        _decoder.ReturnTempArray(cosine);
                        cosine = null;
                        throw;
                    }
                }

                var textureInputs = new Dictionary<string, RenderTexture>(state.Textures, StringComparer.Ordinal)
                {
                    ["in0"] = embeddings.Texture,
                    ["in1"] = mask,
                    ["in2"] = cosine,
                    ["in3"] = sine
                };
                var shapes = new Dictionary<string, NcnnGraphSession.BufferShape>(state.Shapes, StringComparer.Ordinal)
                {
                    ["in0"] = embeddings.LogicalShape,
                    ["in1"] = new NcnnGraphSession.BufferShape(2, state.SequenceLength + sequenceLength, sequenceLength, 1, 1),
                    ["in2"] = new NcnnGraphSession.BufferShape(3, RopeHalfDimension, sequenceLength, 1, 1),
                    ["in3"] = new NcnnGraphSession.BufferShape(3, RopeHalfDimension, sequenceLength, 1, 1)
                };

                using (var result = _decoder.InferWithMultiInputs(textureInputs, null, _decoderOutputs, shapes))
                {
                    var hiddenShape = GetShape(result, "out0");
                    DebugTextureReadback?.Invoke("decoder_out0", result.GetExistingTextureData("out0"));
                    DebugReadbackPackedCaches(result, currentDecodeInvocation);
                    hidden = new Qwen35OwnedTexture(_decoder, result.ExtractTexture("out0"), hiddenShape);
                    nextState = new Qwen35DecoderState(_decoder, state.SequenceLength + sequenceLength, nextPosition);
                    var transferAttentionKv = _decoder.EnableInPlaceAttentionKvCache && state.Textures.ContainsKey("cache_k0");
                    for (var i = 0; i < AttentionCacheCount; i++)
                    {
                        if (transferAttentionKv)
                        {
                            state.TransferTo(nextState, "cache_k" + i, GetShape(result, "out_cache_k" + i));
                            state.TransferTo(nextState, "cache_v" + i, GetShape(result, "out_cache_v" + i));
                        }
                        else
                        {
                            ExtractCache(result, nextState, "cache_k" + i, "out_cache_k" + i);
                            ExtractCache(result, nextState, "cache_v" + i, "out_cache_v" + i);
                        }
                    }
                    for (var i = 0; i < ConvCacheCount; i++)
                    {
                        ExtractCache(result, nextState, "cache_conv" + i, "out_cache_conv" + i);
                        ExtractCache(result, nextState, "cache_gdr" + i, "out_cache_gdr" + i);
                    }
                }
                CaptureDecoderRuntimeProfile(state.SequenceLength, sequenceLength);
                _decodeInvocation = currentDecodeInvocation + 1;
                var step = new Qwen35DecoderStep(hidden, nextState);
                hidden = null;
                nextState = null;
                return step;
            }
            finally
            {
                hidden?.Dispose();
                nextState?.Dispose();
                if (mask != null) _decoder.ReturnTempArray(mask);
                if (cosine != null) _decoder.ReturnTempArray(cosine);
                if (sine != null) _decoder.ReturnTempArray(sine);
            }
        }

        public float[] ProjectLogits(Qwen35OwnedTexture hidden)
        {
            ThrowIfDisposed();
            if (hidden == null || hidden.Texture == null)
                throw new ArgumentNullException(nameof(hidden));
            if (hidden.LogicalShape.dims != 2 || hidden.LogicalShape.w != HiddenSize || hidden.LogicalShape.h != 1)
                throw new ArgumentException("LM head requires exactly one hidden row.", nameof(hidden));
            using (var result = _projection.InferWithMultiInputs(
                new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = hidden.Texture },
                null,
                null,
                new Dictionary<string, NcnnGraphSession.BufferShape>(StringComparer.Ordinal) { ["in0"] = hidden.LogicalShape }))
            {
                var logits = result.ReadTextureDataForOutput("out0");
                DebugTextureReadback?.Invoke("logits", logits);
                return logits;
            }
        }

        private int SelectNextToken(Qwen35OwnedTexture hidden, Qwen35Sampler sampler)
        {
            if (sampler == null) throw new ArgumentNullException(nameof(sampler));
            if (!sampler.CanUseTextureArgMax || DebugTextureReadback != null)
                return sampler.Select(ProjectLogits(hidden));

            using (var result = _projection.InferWithMultiInputs(
                new Dictionary<string, RenderTexture>(StringComparer.Ordinal) { ["in0"] = hidden.Texture },
                null,
                null,
                new Dictionary<string, NcnnGraphSession.BufferShape>(StringComparer.Ordinal) { ["in0"] = hidden.LogicalShape }))
            {
                if (!result.TryGetExistingTexture("out0", out var logits)
                    || !result.TryGetExistingTextureDescriptor("out0", out var logicalShape, out var storageShape))
                {
                    throw new InvalidOperationException("LM head did not publish texture-backed logits for GPU ArgMax.");
                }
                if (logicalShape.w < Tokenizer.VocabularySize)
                    throw new InvalidOperationException("LM head texture does not cover the tokenizer vocabulary.");

                var outputShape = new NcnnGraphSession.BufferShape(logicalShape.dims, 1, 1, 1, 1);
                var outputStorageShape = new NcnnGraphSession.BufferShape(2, 1, 1, 1, 1);
                var vocabularyAxis = logicalShape.dims - 1;
                var argMax = _projection.RentTempMat(1, 1, NcnnGraphSession.ResolveLinearMatTextureFormat());
                try
                {
                    if (logits.dimension == UnityEngine.Rendering.TextureDimension.Tex2D)
                    {
                        _ops.SentisArgReduceLinearMat(
                            logits,
                            logicalShape,
                            storageShape,
                            vocabularyAxis,
                            true,
                            false,
                            true,
                            outputShape,
                            outputStorageShape,
                            argMax);
                    }
                    else if (logits.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray)
                    {
                        var packCount = (logicalShape.w + 3) / 4;
                        var tileRows = Mathf.CeilToInt(packCount / (float)Mathf.Max(1, storageShape.w));
                        if (logicalShape.dims != 2
                            || logicalShape.h != 1
                            || storageShape.dims != 3
                            || storageShape.c != 4
                            || storageShape.w != logits.width
                            || storageShape.h != logits.height
                            || storageShape.h != logicalShape.h * tileRows)
                        {
                            throw new InvalidOperationException(
                                "Unsupported pack4 LM head texture contract"
                                + " | logical=" + logicalShape
                                + " | storage=" + storageShape
                                + " | texture=" + logits.width + "x" + logits.height + "x" + logits.volumeDepth);
                        }
                        _ops.ArgMaxPack4LinearMat(logits, logicalShape.w, logicalShape.h, tileRows, argMax);
                    }
                    else
                    {
                        throw new InvalidOperationException(
                            "Unsupported LM head texture dimension for GPU ArgMax: " + logits.dimension);
                    }
                    var token = Mathf.RoundToInt(NcnnGraphSession.ReadScalarTexture(argMax));
                    if (token < 0 || token >= Tokenizer.VocabularySize)
                        throw new InvalidOperationException("GPU ArgMax returned an invalid token id: " + token);
                    return token;
                }
                finally
                {
                    _projection.ReturnTempArray(argMax);
                }
            }
        }

        public Qwen35GenerationResult GenerateMultimodal(
            IReadOnlyList<int> promptTokenIds,
            Qwen35VisionEncoding vision,
            int maxNewTokens,
            Qwen35SamplingConfig sampling = null,
            Action<int, string> onToken = null)
        {
            ThrowIfDisposed();
            if (promptTokenIds == null || promptTokenIds.Count < 2)
                throw new ArgumentException("Multimodal prompt requires at least two token ids.", nameof(promptTokenIds));
            if (vision == null || vision.Embeddings == null || vision.Embeddings.Texture == null)
                throw new ArgumentNullException(nameof(vision));
            if (maxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            sampling ??= new Qwen35SamplingConfig();

            var output = new Qwen35GenerationResult
            {
                PromptTokenCount = promptTokenIds.Count,
                VisionTokenCount = vision.EmbeddingCount
            };
            var sampler = new Qwen35Sampler(Tokenizer.VocabularySize, sampling);
            var contextCapacity = ConfigureContextCapacity(
                promptTokenIds.Count - 1 + vision.EmbeddingCount + maxNewTokens);
            var position = 0;
            long decoderRuns = 0;
            Qwen35DecoderState state = null;
            Qwen35OwnedTexture hidden = null;
            try
            {
                state = CreateInitialState();
                var prefix = new int[promptTokenIds.Count - 1];
                for (var i = 0; i < prefix.Length; i++) prefix[i] = promptTokenIds[i];

                using (var embeddings = InjectImageEmbeddings(prefix, vision, out var imagePadIndex))
                {
                    BuildVisionMrope(
                        embeddings.LogicalShape.h,
                        position,
                        imagePadIndex,
                        vision.EmbeddingCount,
                        vision.GridWidth,
                        vision.GridHeight,
                        out var cosine,
                        out var sine,
                        out var nextPosition);
                    using (var step = DecodeCore(embeddings, position, state, cosine, sine, nextPosition))
                    {
                        var oldState = state;
                        state = step.DetachState();
                        hidden = step.DetachHidden();
                        position = nextPosition;
                        oldState.Dispose();
                    }
                }
                decoderRuns++;
                hidden.Dispose();
                hidden = null;

                RunTokens(new[] { promptTokenIds[promptTokenIds.Count - 1] }, ref position, ref state, out hidden);
                decoderRuns++;
                var nextToken = SelectNextToken(hidden, sampler);
                hidden.Dispose();
                hidden = null;

                for (var stepIndex = 0; stepIndex < maxNewTokens; stepIndex++)
                {
                    output.TokenIds.Add(nextToken);
                    sampler.AddHistory(nextToken);
                    var piece = Tokenizer.Decode(new[] { nextToken }, false);
                    onToken?.Invoke(nextToken, piece);
                    if (nextToken == Tokenizer.EndOfTurnId)
                    {
                        output.StoppedOnEndOfTurn = true;
                        break;
                    }
                    if (stepIndex + 1 >= maxNewTokens)
                        break;

                    RunTokens(new[] { nextToken }, ref position, ref state, out hidden);
                    decoderRuns++;
                    nextToken = SelectNextToken(hidden, sampler);
                    hidden.Dispose();
                    hidden = null;
                }

                output.Text = Tokenizer.Decode(output.TokenIds, true);
                output.ExpandedPromptTokenCount = state.SequenceLength - Math.Max(0, output.TokenIds.Count - 1);
                output.FinalPosition = position;
                output.DecoderStepCount = decoderRuns;
                output.FinalCacheTextureCount = state.TextureCount;
                output.ContextTextureCapacity = contextCapacity;
                return output;
            }
            finally
            {
                hidden?.Dispose();
                state?.Dispose();
            }
        }

        public async UniTask<Qwen35GenerationResult> GenerateMultimodalAsync(
            IReadOnlyList<int> promptTokenIds,
            Qwen35VisionEncoding vision,
            int maxNewTokens,
            Qwen35SamplingConfig sampling = null,
            CancellationToken cancellationToken = default,
            Action<int, string> onToken = null,
            Action<int, int> onProgress = null,
            Action<Qwen35Progress> onPipelineProgress = null)
        {
            ThrowIfDisposed();
            if (promptTokenIds == null || promptTokenIds.Count < 2)
                throw new ArgumentException("Multimodal prompt requires at least two token ids.", nameof(promptTokenIds));
            if (vision == null || vision.Embeddings == null || vision.Embeddings.Texture == null)
                throw new ArgumentNullException(nameof(vision));
            if (maxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            sampling ??= new Qwen35SamplingConfig();
            cancellationToken.ThrowIfCancellationRequested();

            var output = new Qwen35GenerationResult
            {
                PromptTokenCount = promptTokenIds.Count,
                VisionTokenCount = vision.EmbeddingCount
            };
            var sampler = new Qwen35Sampler(Tokenizer.VocabularySize, sampling);
            var contextCapacity = ConfigureContextCapacity(
                promptTokenIds.Count - 1 + vision.EmbeddingCount + maxNewTokens);
            var position = 0;
            long decoderRuns = 0;
            Qwen35DecoderState state = null;
            Qwen35OwnedTexture hidden = null;
            try
            {
                onPipelineProgress?.Invoke(new Qwen35Progress("prefill", "Creating texture-backed caches", 0f));
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                state = CreateInitialState();
                onPipelineProgress?.Invoke(new Qwen35Progress("prefill", "Embedding multimodal prompt", 0.12f));
                var prefix = new int[promptTokenIds.Count - 1];
                for (var i = 0; i < prefix.Length; i++) prefix[i] = promptTokenIds[i];

                using (var embeddings = InjectImageEmbeddings(prefix, vision, out var imagePadIndex))
                {
                    onPipelineProgress?.Invoke(new Qwen35Progress("prefill", "Building multimodal RoPE", 0.24f));
                    BuildVisionMrope(
                        embeddings.LogicalShape.h,
                        position,
                        imagePadIndex,
                        vision.EmbeddingCount,
                        vision.GridWidth,
                        vision.GridHeight,
                        out var cosine,
                        out var sine,
                        out var nextPosition);
                    using (var step = DecodeCore(embeddings, position, state, cosine, sine, nextPosition))
                    {
                        var oldState = state;
                        state = step.DetachState();
                        hidden = step.DetachHidden();
                        position = nextPosition;
                        oldState.Dispose();
                    }
                }
                decoderRuns++;
                onPipelineProgress?.Invoke(new Qwen35Progress("prefill", "Image and prompt prefill complete", 0.82f));
                hidden.Dispose();
                hidden = null;

                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                RunTokens(new[] { promptTokenIds[promptTokenIds.Count - 1] }, ref position, ref state, out hidden);
                decoderRuns++;
                var nextToken = SelectNextToken(hidden, sampler);
                hidden.Dispose();
                hidden = null;
                onPipelineProgress?.Invoke(new Qwen35Progress("prefill", "First token ready", 1f));

                for (var stepIndex = 0; stepIndex < maxNewTokens; stepIndex++)
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    output.TokenIds.Add(nextToken);
                    sampler.AddHistory(nextToken);
                    var piece = Tokenizer.Decode(new[] { nextToken }, false);
                    onToken?.Invoke(nextToken, piece);
                    onProgress?.Invoke(stepIndex + 1, maxNewTokens);
                    if (nextToken == Tokenizer.EndOfTurnId)
                    {
                        output.StoppedOnEndOfTurn = true;
                        break;
                    }
                    if (stepIndex + 1 >= maxNewTokens)
                        break;

                    await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                    RunTokens(new[] { nextToken }, ref position, ref state, out hidden);
                    decoderRuns++;
                    nextToken = SelectNextToken(hidden, sampler);
                    hidden.Dispose();
                    hidden = null;
                }

                output.Text = Tokenizer.Decode(output.TokenIds, true);
                output.ExpandedPromptTokenCount = state.SequenceLength - Math.Max(0, output.TokenIds.Count - 1);
                output.FinalPosition = position;
                output.DecoderStepCount = decoderRuns;
                output.FinalCacheTextureCount = state.TextureCount;
                output.ContextTextureCapacity = contextCapacity;
                return output;
            }
            finally
            {
                hidden?.Dispose();
                state?.Dispose();
            }
        }

        public Qwen35GenerationResult Generate(
            IReadOnlyList<int> promptTokenIds,
            int maxNewTokens,
            Qwen35SamplingConfig sampling = null,
            Action<int, string> onToken = null)
        {
            ThrowIfDisposed();
            if (promptTokenIds == null || promptTokenIds.Count == 0)
                throw new ArgumentException("Prompt token ids are empty.", nameof(promptTokenIds));
            if (maxNewTokens <= 0)
                throw new ArgumentOutOfRangeException(nameof(maxNewTokens));
            sampling ??= new Qwen35SamplingConfig();

            var output = new Qwen35GenerationResult
            {
                PromptTokenCount = promptTokenIds.Count,
                ExpandedPromptTokenCount = promptTokenIds.Count
            };
            var sampler = new Qwen35Sampler(Tokenizer.VocabularySize, sampling);
            var contextCapacity = ConfigureContextCapacity(promptTokenIds.Count + maxNewTokens);
            var position = 0;
            long decoderRuns = 0;
            Qwen35DecoderState state = null;
            Qwen35OwnedTexture hidden = null;
            try
            {
                state = CreateInitialState();
                if (promptTokenIds.Count > 1)
                {
                    var prefix = new int[promptTokenIds.Count - 1];
                    for (var i = 0; i < prefix.Length; i++) prefix[i] = promptTokenIds[i];
                    RunTokens(prefix, ref position, ref state, out hidden);
                    decoderRuns++;
                    hidden.Dispose();
                    hidden = null;
                }

                RunTokens(new[] { promptTokenIds[promptTokenIds.Count - 1] }, ref position, ref state, out hidden);
                decoderRuns++;
                var nextToken = SelectNextToken(hidden, sampler);
                hidden.Dispose();
                hidden = null;

                for (var stepIndex = 0; stepIndex < maxNewTokens; stepIndex++)
                {
                    output.TokenIds.Add(nextToken);
                    sampler.AddHistory(nextToken);
                    var piece = Tokenizer.Decode(new[] { nextToken }, false);
                    onToken?.Invoke(nextToken, piece);
                    if (nextToken == Tokenizer.EndOfTurnId)
                    {
                        output.StoppedOnEndOfTurn = true;
                        break;
                    }

                    if (stepIndex + 1 >= maxNewTokens)
                        break;

                    RunTokens(new[] { nextToken }, ref position, ref state, out hidden);
                    decoderRuns++;
                    nextToken = SelectNextToken(hidden, sampler);
                    hidden.Dispose();
                    hidden = null;
                }

                output.Text = Tokenizer.Decode(output.TokenIds, true);
                output.FinalPosition = position;
                output.DecoderStepCount = decoderRuns;
                output.FinalCacheTextureCount = state.TextureCount;
                output.ContextTextureCapacity = contextCapacity;
                return output;
            }
            finally
            {
                hidden?.Dispose();
                state?.Dispose();
            }
        }

        private int ConfigureContextCapacity(int requiredTokens)
        {
            var deviceLimit = Mathf.Min(4096, Mathf.Max(1, SystemInfo.maxTextureSize));
            if (requiredTokens <= 0 || requiredTokens > deviceLimit)
                throw new NotSupportedException(
                    "Qwen3.5 context exceeds the device texture-backed KV capacity: requested="
                    + requiredTokens + " capacity=" + deviceLimit);
            // The texture kernels use the logical sequence length for bounds; no
            // allocation alignment is required. Reserve only the requested prompt
            // plus generation budget so long multimodal requests do not carry a
            // 1-63 token KV cache tail on memory-constrained devices.
            _decoder.AttentionKvCacheTextureCapacity = requiredTokens;
            return _decoder.AttentionKvCacheTextureCapacity;
        }

        private void RunTokens(
            IReadOnlyList<int> tokens,
            ref int position,
            ref Qwen35DecoderState state,
            out Qwen35OwnedTexture hidden)
        {
            using (var embeddings = EmbedTokens(tokens))
            using (var step = Decode(embeddings, position, state))
            {
                var oldState = state;
                state = step.DetachState();
                hidden = step.DetachHidden();
                position += tokens.Count;
                oldState.Dispose();
            }
        }

        private Qwen35OwnedTexture InjectImageEmbeddings(
            IReadOnlyList<int> prefixTokenIds,
            Qwen35VisionEncoding vision,
            out int imagePadIndex)
        {
            if (prefixTokenIds == null || prefixTokenIds.Count == 0)
                throw new ArgumentException("Multimodal prefix token ids are empty.", nameof(prefixTokenIds));
            if (vision.Embeddings.LogicalShape.dims != 2
                || vision.Embeddings.LogicalShape.w != HiddenSize
                || vision.Embeddings.LogicalShape.h != vision.EmbeddingCount)
                throw new ArgumentException("Vision embeddings must have logical shape [vision tokens, 1024].", nameof(vision));

            var imagePadId = Tokenizer.IdOf("<|image_pad|>");
            imagePadIndex = -1;
            for (var i = 0; i < prefixTokenIds.Count; i++)
            {
                if (prefixTokenIds[i] != imagePadId) continue;
                if (imagePadIndex >= 0)
                    throw new InvalidOperationException("Qwen3.5 currently supports exactly one image per prompt.");
                imagePadIndex = i;
            }
            if (imagePadIndex < 0)
                throw new InvalidOperationException("Multimodal prompt does not contain <|image_pad|>.");

            RenderTexture imageLinear = null;
            RenderTexture merged = null;
            using (var text = EmbedTokens(prefixTokenIds))
            {
                try
                {
                    var imageTexture = vision.Embeddings.Texture;
                    if (imageTexture.dimension == UnityEngine.Rendering.TextureDimension.Tex2D
                        && imageTexture.width == HiddenSize
                        && imageTexture.height == vision.EmbeddingCount)
                    {
                        imageLinear = imageTexture;
                    }
                    else if (imageTexture.dimension == UnityEngine.Rendering.TextureDimension.Tex2DArray
                        && imageTexture.width == HiddenSize / 4
                        && imageTexture.height == vision.EmbeddingCount
                        && imageTexture.volumeDepth == 1)
                    {
                        imageLinear = _decoder.RentTempMat(HiddenSize, vision.EmbeddingCount, NcnnGraphSession.ResolveLinearMatTextureFormat());
                        _ops.ReshapePack4ToLinearMat(
                            imageTexture,
                            HiddenSize,
                            vision.EmbeddingCount,
                            1,
                            1,
                            2,
                            imageLinear,
                            inputPack4Linear: true);
                    }
                    else
                    {
                        throw new InvalidOperationException(
                            "Unsupported Qwen3.5 vision embedding storage: "
                            + imageTexture.dimension + " " + imageTexture.width + "x" + imageTexture.height + "x" + imageTexture.volumeDepth);
                    }

                    var expandedRows = checked(prefixTokenIds.Count - 1 + vision.EmbeddingCount);
                    merged = _decoder.RentTempMat(HiddenSize, expandedRows, NcnnGraphSession.ResolveLinearMatTextureFormat());
                    if (imagePadIndex > 0)
                        CopyLinearRows(text.Texture, 0, merged, 0, imagePadIndex);
                    CopyLinearRows(imageLinear, 0, merged, imagePadIndex, vision.EmbeddingCount);
                    var suffixStart = imagePadIndex + 1;
                    var suffixRows = prefixTokenIds.Count - suffixStart;
                    if (suffixRows > 0)
                        CopyLinearRows(text.Texture, suffixStart, merged, imagePadIndex + vision.EmbeddingCount, suffixRows);

                    var result = new Qwen35OwnedTexture(
                        _decoder,
                        merged,
                        new NcnnGraphSession.BufferShape(2, HiddenSize, expandedRows, 1, 1));
                    merged = null;
                    return result;
                }
                finally
                {
                    if (imageLinear != null && imageLinear != vision.Embeddings.Texture)
                        _decoder.ReturnTempArray(imageLinear);
                    if (merged != null)
                        _decoder.ReturnTempArray(merged);
                }
            }
        }

        private static void CopyLinearRows(
            RenderTexture source,
            int sourceRow,
            RenderTexture destination,
            int destinationRow,
            int rowCount)
        {
            if (source == null || destination == null)
                throw new ArgumentNullException(source == null ? nameof(source) : nameof(destination));
            if (rowCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(rowCount));
            if (source.dimension != UnityEngine.Rendering.TextureDimension.Tex2D
                || destination.dimension != UnityEngine.Rendering.TextureDimension.Tex2D
                || source.width != HiddenSize
                || destination.width != HiddenSize)
                throw new InvalidOperationException("Qwen3.5 embedding row copy requires 1024-wide LinearMat textures.");
            Graphics.CopyTexture(
                source, 0, 0, 0, sourceRow, HiddenSize, rowCount,
                destination, 0, 0, 0, destinationRow);
        }

        private static void BuildVisionMrope(
            int sequenceLength,
            int position,
            int imagePadIndex,
            int imageEmbeddingCount,
            int gridWidth,
            int gridHeight,
            out float[] cosine,
            out float[] sine,
            out int nextPosition)
        {
            if (sequenceLength <= 0 || position < 0 || imagePadIndex < 0 || imageEmbeddingCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(sequenceLength));
            if ((gridWidth & 1) != 0 || (gridHeight & 1) != 0)
                throw new ArgumentException("Qwen3.5 vision patch grid must be divisible by spatial merge size 2.");
            var mergedWidth = gridWidth / 2;
            var mergedHeight = gridHeight / 2;
            if (imageEmbeddingCount != mergedWidth * mergedHeight)
                throw new ArgumentException("Vision embedding count does not match the merged patch grid.");
            if (imagePadIndex + imageEmbeddingCount > sequenceLength)
                throw new ArgumentException("Image embedding range exceeds the decoder sequence.");

            cosine = new float[checked(sequenceLength * RopeHalfDimension)];
            sine = new float[cosine.Length];
            const int activeHalfDimension = RopeHalfDimension / 2;
            for (var row = 0; row < sequenceLength; row++)
            for (var column = 0; column < RopeHalfDimension; column++)
            {
                var index = row * RopeHalfDimension + column;
                if (column >= activeHalfDimension)
                {
                    cosine[index] = 1f;
                    sine[index] = 0f;
                    continue;
                }

                var ropePosition = position;
                if (row < imagePadIndex)
                {
                    ropePosition += row;
                }
                else if (row >= imagePadIndex + imageEmbeddingCount)
                {
                    ropePosition += row - imageEmbeddingCount + mergedWidth;
                }
                else
                {
                    var imageIndex = row - imagePadIndex;
                    var positionAxis = 0;
                    if (column < 33 && column % 3 == 1)
                        positionAxis = 1;
                    else if (column < 30 && column % 3 == 2)
                        positionAxis = 2;

                    if (positionAxis == 0)
                        ropePosition += imagePadIndex;
                    else if (positionAxis == 1)
                        ropePosition += imagePadIndex + imageIndex / mergedWidth;
                    else
                        ropePosition += imagePadIndex + imageIndex % mergedWidth;
                }

                var inverseFrequency = 1f / Mathf.Pow(RopeTheta, column * (2f / (activeHalfDimension * 2f)));
                var angle = ropePosition * inverseFrequency;
                cosine[index] = Mathf.Cos(angle);
                sine[index] = Mathf.Sin(angle);
            }

            nextPosition = checked(position + sequenceLength - imageEmbeddingCount + mergedWidth);
        }

        private static void ExtractCache(
            NcnnGraphSession.InferResult result,
            Qwen35DecoderState state,
            string inputName,
            string outputName)
        {
            state.Add(inputName, result.ExtractTexture(outputName), GetShape(result, outputName));
        }

        private void DebugReadbackPackedCaches(NcnnGraphSession.InferResult result, int decodeInvocation)
        {
            if (_debugLayerReadback == null || _debugLayerReadbackBlobs == null)
                return;

            for (var i = 0; i < ConvCacheCount; i++)
            {
                var convName = "out_cache_conv" + i;
                if (_debugLayerReadbackBlobs.Contains(convName))
                {
                    var texture = result.GetTexture(convName);
                    var values = ReadPackedCacheTextureForDebug(texture, 1536, 4, 1, convName);
                    _debugLayerReadback(decodeInvocation, "canonical_cache_readback", convName, values);
                }

                var gdrName = "out_cache_gdr" + i;
                if (_debugLayerReadbackBlobs.Contains(gdrName))
                {
                    var texture = result.GetTexture(gdrName);
                    var values = ReadPackedCacheTextureForDebug(texture, 32, 128, 16, gdrName);
                    _debugLayerReadback(decodeInvocation, "canonical_cache_readback", gdrName, values);
                }
            }
        }

        // Explicit audit-only readback. Normal inference never enters this path.
        private static float[] ReadPackedCacheTextureForDebug(
            RenderTexture texture,
            int expectedWidth,
            int expectedHeight,
            int expectedSlices,
            string blobName)
        {
            if (texture == null
                || texture.dimension != UnityEngine.Rendering.TextureDimension.Tex2DArray
                || texture.width != expectedWidth
                || texture.height != expectedHeight
                || texture.volumeDepth != expectedSlices)
            {
                throw new InvalidOperationException(
                    "Qwen3.5 packed cache debug readback shape mismatch: blob=" + blobName
                    + " expected=" + expectedWidth + "x" + expectedHeight + "x" + expectedSlices
                    + " actual=" + (texture == null
                        ? "null"
                        : texture.width + "x" + texture.height + "x" + texture.volumeDepth + " " + texture.dimension));
            }

            var previousActive = RenderTexture.active;
            Texture2D readback = null;
            try
            {
                readback = new Texture2D(expectedWidth, expectedHeight, TextureFormat.RGBAFloat, false, true);
                var values = new float[checked(expectedWidth * expectedHeight * expectedSlices * 4)];
                var sliceValueCount = checked(expectedWidth * expectedHeight * 4);
                for (var slice = 0; slice < expectedSlices; slice++)
                {
                    Graphics.SetRenderTarget(texture, 0, CubemapFace.Unknown, slice);
                    readback.ReadPixels(new Rect(0, 0, expectedWidth, expectedHeight), 0, 0, false);
                    readback.Apply(false, false);
                    var raw = readback.GetRawTextureData<float>();
                    var destinationOffset = slice * sliceValueCount;
                    for (var index = 0; index < sliceValueCount; index++)
                        values[destinationOffset + index] = raw[index];
                }
                return values;
            }
            finally
            {
                RenderTexture.active = previousActive;
                DestroyUnityObject(readback);
            }
        }

        private static NcnnGraphSession.BufferShape GetShape(NcnnGraphSession.InferResult result, string name)
        {
            if (!result.TryGetLogicalShape(name, out var dims, out var w, out var h, out var d, out var c))
                throw new InvalidOperationException("Texture logical shape is unavailable: " + name);
            return new NcnnGraphSession.BufferShape(dims, w, h, d, c);
        }

        private RenderTexture CreateZeroPack4Texture(int width, int height, int slices)
        {
            var texture = _decoder.RentTempArray(width, height, slices, RenderTextureFormat.ARGBFloat);
            _ops.FillScalarTexture(new[] { 0f }, texture);
            return texture;
        }

        private RenderTexture CreateCausalMaskTexture(int sequenceLength, int pastLength)
        {
            var width = checked(pastLength + sequenceLength);
            var values = new float[checked(width * sequenceLength)];
            for (var row = 0; row < sequenceLength; row++)
            for (var column = pastLength + row + 1; column < width; column++)
                values[row * width + column] = -1e38f;
            return UploadScalarPack4(values, width, sequenceLength, "Qwen35CausalMaskUpload");
        }

        private void CreateRopeTextures(int sequenceLength, int position, out RenderTexture cosine, out RenderTexture sine)
        {
            cosine = null;
            sine = null;
            var count = checked(RopeHalfDimension * sequenceLength);
            var cosValues = new float[count];
            var sinValues = new float[count];
            for (var row = 0; row < sequenceLength; row++)
            for (var column = 0; column < RopeHalfDimension; column++)
            {
                var inverseFrequency = 1f / Mathf.Pow(RopeTheta, column * (2f / (RopeHalfDimension * 2f)));
                var angle = (position + row) * inverseFrequency;
                var index = row * RopeHalfDimension + column;
                cosValues[index] = Mathf.Cos(angle);
                sinValues[index] = Mathf.Sin(angle);
            }
            cosine = UploadScalarPack4(cosValues, RopeHalfDimension, sequenceLength, "Qwen35RopeCosUpload");
            try
            {
                sine = UploadScalarPack4(sinValues, RopeHalfDimension, sequenceLength, "Qwen35RopeSinUpload");
            }
            catch
            {
                _decoder.ReturnTempArray(cosine);
                cosine = null;
                throw;
            }
        }

        private void CaptureDecoderRuntimeProfile(int pastSequenceLength, int sequenceLength)
        {
            if (!_decoder.LayerRuntimeProfileEnabled)
                return;

            var profile = _decoder.LastRuntimeProfile;
            if (profile == null)
                return;

            if (pastSequenceLength == 0 && sequenceLength > 1)
            {
                PrefillDecoderRuntimeProfile = profile;
                return;
            }

            if (FirstDecodeRuntimeProfile == null)
                FirstDecodeRuntimeProfile = profile;
            LastDecodeRuntimeProfile = profile;
        }

        private RenderTexture UploadScalarPack4(float[] values, int width, int height, string uploadName)
        {
            if (values == null || values.Length != width * height)
                throw new ArgumentException("Scalar texture upload shape mismatch.", nameof(values));
            var texture = _decoder.RentTempArray(width, height, 1, RenderTextureFormat.ARGBFloat);
            var upload = new Texture2DArray(width, height, 1, TextureFormat.RGBAFloat, false, true)
            {
                filterMode = FilterMode.Point,
                wrapMode = TextureWrapMode.Clamp,
                anisoLevel = 0,
                name = uploadName
            };
            try
            {
                var pixels = new Color[values.Length];
                for (var i = 0; i < values.Length; i++)
                    pixels[i] = new Color(values[i], 0f, 0f, 0f);
                upload.SetPixels(pixels, 0, 0);
                upload.Apply(false, true);
                Graphics.CopyTexture(upload, 0, 0, texture, 0, 0);
                return texture;
            }
            catch
            {
                _decoder.ReturnTempArray(texture);
                throw;
            }
            finally
            {
                DestroyUnityObject(upload);
            }
        }

        private NcnnGraphSession CreateRepro(string modelDirectory)
        {
            var repro = new NcnnGraphSession(_ops)
            {
                ExecutionMode = NcnnInferenceExecutionMode.ProductionTextureOnly,
                DisallowInferenceTempComputeBuffers = true,
                DisallowBufferToTextureMaterialization = true,
                DisallowBufferOutputs = true,
                EnableAttentionMatMulPack4Specializations = true,
                EnableInPlaceAttentionKvCache = !string.Equals(
                    Environment.GetEnvironmentVariable("AIIMAGE_QWEN35_INPLACE_KV"),
                    "0",
                    StringComparison.Ordinal),
                EnableConv1x1TextureConvolution = true,
                EnableDepthWiseTextureConvolution = true,
                TensorTextureFormat = RenderTextureFormat.ARGBFloat,
                ManagedLoadGarbageCollectionIntervalBytes = Qwen35RuntimeTuning.ResolveManagedLoadGarbageCollectionIntervalBytes(modelDirectory)
            };
            return repro;
        }

        private static void CollectManagedLoadGarbage()
        {
            GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, false);
        }

        private static void Load(NcnnGraphSession repro, string directory, string paramName, string binName)
        {
            using (var stream = Qwen35ModelAssetResolver.OpenBin(directory, binName))
            using (var reader = new NcnnBinReader(stream))
                repro.LoadModel(File.ReadAllText(Path.Combine(directory, paramName)), reader);
        }

        private static async UniTask LoadAsync(
            NcnnGraphSession repro,
            string directory,
            string paramName,
            string binName,
            Action<NcnnGraphSession.LoadProgress> onProgress,
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

        private static void DestroyUnityObject(UnityEngine.Object value)
        {
            if (value == null) return;
            if (Application.isPlaying) UnityEngine.Object.Destroy(value);
            else UnityEngine.Object.DestroyImmediate(value);
        }

        private void ThrowIfDisposed()
        {
            if (_disposed) throw new ObjectDisposedException(nameof(Qwen35DecoderSession));
        }

        public void Dispose()
        {
            if (_disposed) return;
            _disposed = true;
            try { _projection?.Dispose(); } catch { }
            try { _decoder?.Dispose(); } catch { }
            try { _embed?.Dispose(); } catch { }
            try { _sharedWeights?.Dispose(); } catch { }
            try { _ops?.Dispose(); } catch { }
        }
    }

    internal sealed class Qwen35Sampler
    {
        private readonly int _vocabularySize;
        private readonly Qwen35SamplingConfig _config;
        private readonly HashSet<int> _history = new HashSet<int>();
        private readonly System.Random _random;

        public Qwen35Sampler(int vocabularySize, Qwen35SamplingConfig config)
        {
            _vocabularySize = vocabularySize > 0 ? vocabularySize : throw new ArgumentOutOfRangeException(nameof(vocabularySize));
            _config = config ?? throw new ArgumentNullException(nameof(config));
            _random = new System.Random(config.Seed);
        }

        public bool CanUseTextureArgMax =>
            (!_config.DoSample || _config.Temperature <= 0f)
            && Mathf.Approximately(_config.RepetitionPenalty, 1f);

        public void AddHistory(int tokenId)
        {
            if (tokenId >= 0 && tokenId < _vocabularySize)
                _history.Add(tokenId);
        }

        public int Select(float[] logits)
        {
            if (logits == null || logits.Length < _vocabularySize)
                throw new ArgumentException("LM head logits do not cover the tokenizer vocabulary.", nameof(logits));
            var scores = new float[_vocabularySize];
            Array.Copy(logits, scores, scores.Length);
            foreach (var token in _history)
                scores[token] = scores[token] < 0f ? scores[token] * _config.RepetitionPenalty : scores[token] / _config.RepetitionPenalty;

            if (!_config.DoSample || _config.Temperature <= 0f)
                return ArgMax(scores);

            var max = scores[ArgMax(scores)];
            var probabilities = new float[scores.Length];
            double sum = 0.0;
            for (var i = 0; i < scores.Length; i++)
            {
                var value = Mathf.Exp((scores[i] - max) / _config.Temperature);
                probabilities[i] = value;
                sum += value;
            }
            if (!double.IsFinite(sum) || sum <= 0.0)
                return ArgMax(scores);
            for (var i = 0; i < probabilities.Length; i++) probabilities[i] = (float)(probabilities[i] / sum);

            var order = new int[probabilities.Length];
            for (var i = 0; i < order.Length; i++) order[i] = i;
            Array.Sort(order, (a, b) => probabilities[b].CompareTo(probabilities[a]));
            var keep = _config.TopK > 0 ? Math.Min(_config.TopK, order.Length) : order.Length;
            if (_config.TopP < 1f)
            {
                double cumulative = 0.0;
                var nucleus = 0;
                while (nucleus < keep)
                {
                    cumulative += probabilities[order[nucleus++]];
                    if (cumulative >= _config.TopP) break;
                }
                keep = Math.Max(1, nucleus);
            }

            double keptSum = 0.0;
            for (var i = 0; i < keep; i++) keptSum += probabilities[order[i]];
            if (!double.IsFinite(keptSum) || keptSum <= 0.0)
                return ArgMax(scores);
            var sample = _random.NextDouble() * keptSum;
            double cursor = 0.0;
            for (var i = 0; i < keep; i++)
            {
                cursor += probabilities[order[i]];
                if (sample <= cursor) return order[i];
            }
            return order[keep - 1];
        }

        private static int ArgMax(float[] values)
        {
            var best = 0;
            for (var i = 1; i < values.Length; i++)
                if (values[i] > values[best]) best = i;
            return best;
        }
    }
}
