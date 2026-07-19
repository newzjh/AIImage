using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using Cysharp.Threading.Tasks;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace AIImage.Qwen35
{
    public readonly struct Qwen35Progress
    {
        public readonly string Stage;
        public readonly string Detail;
        public readonly float Progress01;
        public readonly long Completed;
        public readonly long Total;

        public Qwen35Progress(
            string stage,
            string detail,
            float progress01,
            long completed = 0,
            long total = 0)
        {
            Stage = stage ?? string.Empty;
            Detail = detail ?? string.Empty;
            Progress01 = Mathf.Clamp01(progress01);
            Completed = completed;
            Total = total;
        }

        public Qwen35Progress Map(float start, float end, string stage = null)
        {
            return new Qwen35Progress(
                stage ?? Stage,
                Detail,
                Mathf.Lerp(start, end, Progress01),
                Completed,
                Total);
        }
    }

    public sealed class Qwen35Runner : IDisposable
    {
        public readonly Qwen35ModelContract Contract;
        public readonly Qwen35ByteLevelBpeTokenizer Tokenizer;
        public readonly Qwen35DeviceCompatibility DeviceCompatibility;
        public readonly int MaxNewTokens;
        public bool IsReady => Contract != null && Contract.IsValid && Tokenizer != null;

        public Qwen35Runner(string modelDirectory, int maxNewTokens = 32, bool requireWeights = true)
        {
            Contract = Qwen35ModelContract.Validate(modelDirectory, requireWeights);
            if (!Contract.IsValid) throw new InvalidOperationException("Qwen3.5 model contract failed:\n" + string.Join("\n", Contract.Errors));
            Tokenizer = LoadTokenizer(modelDirectory);
            MaxNewTokens = Mathf.Clamp(maxNewTokens, 1, 4096);
            DeviceCompatibility = Qwen35DeviceCompatibility.Evaluate(Contract);
            DeviceCompatibility.ThrowIfUnsupported();
        }

        private Qwen35Runner(
            Qwen35ModelContract contract,
            Qwen35ByteLevelBpeTokenizer tokenizer,
            int maxNewTokens)
        {
            Contract = contract ?? throw new ArgumentNullException(nameof(contract));
            if (!Contract.IsValid) throw new InvalidOperationException("Qwen3.5 model contract failed:\n" + string.Join("\n", Contract.Errors));
            Tokenizer = tokenizer ?? throw new ArgumentNullException(nameof(tokenizer));
            MaxNewTokens = Mathf.Clamp(maxNewTokens, 1, 4096);
            DeviceCompatibility = Qwen35DeviceCompatibility.Evaluate(Contract);
            DeviceCompatibility.ThrowIfUnsupported();
        }

        public static async UniTask<Qwen35Runner> CreateAsync(
            string modelDirectory,
            int maxNewTokens = 32,
            bool requireWeights = true,
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var progressQueue = new ConcurrentQueue<Qwen35Progress>();
            var validationTask = UniTask.RunOnThreadPool(
                () => Qwen35ModelContract.Validate(
                    modelDirectory,
                    requireWeights,
                    value => progressQueue.Enqueue(value.Map(0f, 0.86f)),
                    cancellationToken),
                cancellationToken: cancellationToken);
            while (validationTask.Status == UniTaskStatus.Pending)
            {
                while (progressQueue.TryDequeue(out var value))
                    onProgress?.Invoke(value);
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
            }
            var contract = await validationTask;
            while (progressQueue.TryDequeue(out var value))
                onProgress?.Invoke(value);
            cancellationToken.ThrowIfCancellationRequested();
            if (!contract.IsValid)
                throw new InvalidOperationException("Qwen3.5 model contract failed:\n" + string.Join("\n", contract.Errors));
            onProgress?.Invoke(new Qwen35Progress("loading_tokenizer", "Parsing BBPE vocabulary", 0.88f));
            var tokenizer = await UniTask.RunOnThreadPool(
                () => LoadTokenizer(modelDirectory),
                cancellationToken: cancellationToken);
            cancellationToken.ThrowIfCancellationRequested();
            onProgress?.Invoke(new Qwen35Progress("initialization_complete", "Qwen3.5 runtime ready", 1f));
            return new Qwen35Runner(contract, tokenizer, maxNewTokens);
        }

        private static Qwen35ByteLevelBpeTokenizer LoadTokenizer(string modelDirectory)
        {
            var modelJson = JObject.Parse(File.ReadAllText(Path.Combine(modelDirectory, "model.json")));
            var specials = new List<string>();
            if (modelJson["tokenizer"]?["additional_special_tokens"] is JArray configuredSpecials)
            {
                for (var i = 0; i < configuredSpecials.Count; i++)
                {
                    var token = (string)configuredSpecials[i];
                    if (!string.IsNullOrEmpty(token))
                        specials.Add(token);
                }
            }
            return new Qwen35ByteLevelBpeTokenizer(
                Path.Combine(modelDirectory, "vocab.txt"),
                Path.Combine(modelDirectory, "merges.txt"),
                specials);
        }

        public string BuildImagePrompt(string userText)
        {
            return "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>" + (userText ?? string.Empty) + "<|im_end|>\n<|im_start|>assistant\n";
        }

        public List<int> EncodeImagePrompt(string userText) => Tokenizer.Encode(BuildImagePrompt(userText));

        public int SampleGreedy(float[] logits)
        {
            if (logits == null || logits.Length == 0) throw new ArgumentException("logits are empty");
            var best = 0; var value = logits[0];
            for (var i = 1; i < logits.Length; i++) if (logits[i] > value) { value = logits[i]; best = i; }
            return best;
        }

        public Qwen35DecoderSession CreateDecoderSession()
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return new Qwen35DecoderSession(Contract.ModelDirectory, Tokenizer);
        }

        public UniTask<Qwen35DecoderSession> CreateDecoderSessionAsync(
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return Qwen35DecoderSession.CreateAsync(
                Contract.ModelDirectory,
                Tokenizer,
                cancellationToken,
                onProgress);
        }

        public Qwen35VisionEncoderSession CreateVisionEncoderSession()
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return new Qwen35VisionEncoderSession(Contract.ModelDirectory);
        }

        public UniTask<Qwen35VisionEncoderSession> CreateVisionEncoderSessionAsync(
            CancellationToken cancellationToken = default,
            Action<Qwen35Progress> onProgress = null)
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return Qwen35VisionEncoderSession.CreateAsync(
                Contract.ModelDirectory,
                cancellationToken,
                onProgress);
        }

        public Qwen35GenerationResult GenerateImage(
            Texture2D image,
            string userText,
            Qwen35SamplingConfig sampling = null,
            Action<int, string> onToken = null)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            using (var visionSession = CreateVisionEncoderSession())
            using (var vision = visionSession.Encode(image))
            using (var decoder = CreateDecoderSession())
                return decoder.GenerateMultimodal(
                    EncodeImagePrompt(userText),
                    vision,
                    MaxNewTokens,
                    sampling ?? Qwen35SamplingConfig.Greedy(),
                    onToken);
        }

        public async UniTask<Qwen35GenerationResult> GenerateImageAsync(
            Texture2D image,
            string userText,
            Qwen35SamplingConfig sampling = null,
            CancellationToken cancellationToken = default,
            Action<int, string> onToken = null,
            Action<int, int> onProgress = null,
            Action<string> onStage = null,
            Action<Qwen35Progress> onPipelineProgress = null)
        {
            if (image == null) throw new ArgumentNullException(nameof(image));
            cancellationToken.ThrowIfCancellationRequested();

            onStage?.Invoke("loading_vision");
            onPipelineProgress?.Invoke(new Qwen35Progress("loading_vision", "Preparing vision networks", 0f));
            await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
            using (var visionSession = await CreateVisionEncoderSessionAsync(
                cancellationToken,
                progress => onPipelineProgress?.Invoke(progress.Map(0f, 0.2f))))
            {
                cancellationToken.ThrowIfCancellationRequested();
                onStage?.Invoke("encoding_image");
                onPipelineProgress?.Invoke(new Qwen35Progress("encoding_image", "Preprocessing image", 0.2f));
                await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                using (var vision = visionSession.Encode(image))
                {
                    cancellationToken.ThrowIfCancellationRequested();
                    onPipelineProgress?.Invoke(new Qwen35Progress("encoding_image", "Vision encoding ready", 0.4f));
                    onStage?.Invoke("loading_decoder");
                    await UniTask.Yield(PlayerLoopTiming.Update, cancellationToken);
                    using (var decoder = await CreateDecoderSessionAsync(
                        cancellationToken,
                        progress => onPipelineProgress?.Invoke(progress.Map(0.4f, 0.76f))))
                    {
                        cancellationToken.ThrowIfCancellationRequested();
                        onStage?.Invoke("generating");
                        var generated = await decoder.GenerateMultimodalAsync(
                            EncodeImagePrompt(userText),
                            vision,
                            MaxNewTokens,
                            sampling ?? Qwen35SamplingConfig.Greedy(),
                            cancellationToken,
                            onToken,
                            (completed, total) =>
                            {
                                onProgress?.Invoke(completed, total);
                                var progress = total > 0 ? completed / (float)total : 1f;
                                onPipelineProgress?.Invoke(new Qwen35Progress(
                                    "generating",
                                    "Generating token " + completed + "/" + total,
                                    Mathf.Lerp(0.84f, 1f, progress),
                                    completed,
                                    total));
                            },
                            progress => onPipelineProgress?.Invoke(progress.Map(0.76f, 0.84f)));
                        onPipelineProgress?.Invoke(new Qwen35Progress(
                            "complete",
                            "Generation complete",
                            1f,
                            generated.TokenIds.Count,
                            MaxNewTokens));
                        return generated;
                    }
                }
            }
        }

        public Qwen35GenerationResult GenerateImageFile(
            string imagePath,
            string userText,
            Qwen35SamplingConfig sampling = null,
            Action<int, string> onToken = null)
        {
            using (var visionSession = CreateVisionEncoderSession())
            using (var vision = visionSession.EncodeFile(imagePath))
            using (var decoder = CreateDecoderSession())
                return decoder.GenerateMultimodal(
                    EncodeImagePrompt(userText),
                    vision,
                    MaxNewTokens,
                    sampling ?? Qwen35SamplingConfig.Greedy(),
                    onToken);
        }

        public string RequireInferenceEntryPoint()
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return typeof(Qwen35DecoderSession).FullName;
        }

        public void ValidateDeviceForTextureInference()
        {
            (DeviceCompatibility ?? Qwen35DeviceCompatibility.Evaluate(Contract)).ThrowIfUnsupported();
        }

        public void Dispose() { }
    }
}
