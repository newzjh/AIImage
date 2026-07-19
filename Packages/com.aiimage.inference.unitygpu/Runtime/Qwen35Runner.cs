using System;
using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json.Linq;
using UnityEngine;

namespace NcnnCompute
{
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
            Tokenizer = new Qwen35ByteLevelBpeTokenizer(Path.Combine(modelDirectory, "vocab.txt"), Path.Combine(modelDirectory, "merges.txt"), specials);
            MaxNewTokens = Mathf.Clamp(maxNewTokens, 1, 4096);
            DeviceCompatibility = Qwen35DeviceCompatibility.Evaluate(Contract);
            DeviceCompatibility.ThrowIfUnsupported();
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

        public Qwen35VisionEncoderSession CreateVisionEncoderSession()
        {
            if (!IsReady)
                throw new InvalidOperationException("Qwen3.5 model contract is not ready.");
            return new Qwen35VisionEncoderSession(Contract.ModelDirectory);
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
