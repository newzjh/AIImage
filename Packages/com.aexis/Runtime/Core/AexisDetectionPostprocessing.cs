using System;
using System.Collections.Generic;

namespace Aexis
{
    [Serializable]
    public struct AexisBoundingBox
    {
        public float xMin;
        public float yMin;
        public float xMax;
        public float yMax;

        public float Width => Math.Max(0f, xMax - xMin);
        public float Height => Math.Max(0f, yMax - yMin);
        public float Area => Width * Height;
    }

    [Serializable]
    public struct AexisDetection
    {
        public int batchIndex;
        public int classIndex;
        public float score;
        public AexisBoundingBox box;
    }

    public static class AexisDetectionPostprocessing
    {
        public static AexisDetection[] DecodeDetectionOutput(
            IReadOnlyList<float> boxes,
            IReadOnlyList<float> scores,
            int proposalCount,
            int classCount,
            float scoreThreshold,
            float nmsThreshold,
            int keepTopK,
            bool includeBackground = false)
        {
            if (boxes == null || scores == null)
                throw new ArgumentNullException(boxes == null ? nameof(boxes) : nameof(scores));
            if (proposalCount <= 0 || classCount <= 0 || boxes.Count != proposalCount * 4 || scores.Count != proposalCount * classCount)
                throw new InferenceContractException("DetectionOutput tensors do not match proposalCount/classCount.");
            if (keepTopK <= 0 || scoreThreshold < 0f || nmsThreshold < 0f || nmsThreshold > 1f)
                throw new InferenceContractException("DetectionOutput thresholds are invalid.");

            var selected = new List<AexisDetection>();
            var firstClass = includeBackground ? 0 : 1;
            for (var classIndex = firstClass; classIndex < classCount; classIndex++)
            {
                var candidates = new List<AexisDetection>();
                for (var proposal = 0; proposal < proposalCount; proposal++)
                {
                    var score = scores[proposal * classCount + classIndex];
                    if (score < scoreThreshold)
                        continue;
                    candidates.Add(new AexisDetection
                    {
                        classIndex = classIndex,
                        score = score,
                        box = ReadBox(boxes, proposal * 4)
                    });
                }
                selected.AddRange(NonMaximumSuppression(candidates, nmsThreshold, keepTopK));
            }

            selected.Sort(CompareDetections);
            if (selected.Count > keepTopK)
                selected.RemoveRange(keepTopK, selected.Count - keepTopK);
            return selected.ToArray();
        }

        public static AexisDetection[] DecodeYolo(
            IReadOnlyList<float> predictions,
            int candidateCount,
            int classCount,
            float scoreThreshold,
            float nmsThreshold,
            int keepTopK,
            bool valuesAreLogits = false)
        {
            if (predictions == null)
                throw new ArgumentNullException(nameof(predictions));
            var stride = 5 + classCount;
            if (candidateCount <= 0 || classCount <= 0 || predictions.Count != candidateCount * stride)
                throw new InferenceContractException("YOLO detection tensor does not match candidateCount/classCount.");

            var candidates = new List<AexisDetection>();
            for (var candidate = 0; candidate < candidateCount; candidate++)
            {
                var offset = candidate * stride;
                var objectness = valuesAreLogits ? Sigmoid(predictions[offset + 4]) : predictions[offset + 4];
                for (var classIndex = 0; classIndex < classCount; classIndex++)
                {
                    var classScore = valuesAreLogits ? Sigmoid(predictions[offset + 5 + classIndex]) : predictions[offset + 5 + classIndex];
                    var score = objectness * classScore;
                    if (score < scoreThreshold)
                        continue;
                    var centerX = predictions[offset];
                    var centerY = predictions[offset + 1];
                    var width = Math.Max(0f, predictions[offset + 2]);
                    var height = Math.Max(0f, predictions[offset + 3]);
                    candidates.Add(new AexisDetection
                    {
                        classIndex = classIndex,
                        score = score,
                        box = new AexisBoundingBox
                        {
                            xMin = centerX - width * 0.5f,
                            yMin = centerY - height * 0.5f,
                            xMax = centerX + width * 0.5f,
                            yMax = centerY + height * 0.5f
                        }
                    });
                }
            }

            var selected = new List<AexisDetection>();
            for (var classIndex = 0; classIndex < classCount; classIndex++)
            {
                var perClass = candidates.FindAll(candidate => candidate.classIndex == classIndex);
                selected.AddRange(NonMaximumSuppression(perClass, nmsThreshold, keepTopK));
            }
            selected.Sort(CompareDetections);
            if (selected.Count > keepTopK)
                selected.RemoveRange(keepTopK, selected.Count - keepTopK);
            return selected.ToArray();
        }

        public static AexisDetection[] GenerateProposals(
            IReadOnlyList<float> anchors,
            IReadOnlyList<float> deltas,
            IReadOnlyList<float> scores,
            int proposalCount,
            int imageWidth,
            int imageHeight,
            float scoreThreshold,
            float nmsThreshold,
            int preNmsTopK,
            int postNmsTopK,
            float minimumSize = 0f)
        {
            if (anchors == null || deltas == null || scores == null)
                throw new ArgumentNullException(anchors == null ? nameof(anchors) : deltas == null ? nameof(deltas) : nameof(scores));
            if (proposalCount <= 0 || anchors.Count != proposalCount * 4 || deltas.Count != proposalCount * 4 || scores.Count != proposalCount)
                throw new InferenceContractException("Proposal tensors do not match proposalCount.");
            if (imageWidth <= 0 || imageHeight <= 0 || preNmsTopK <= 0 || postNmsTopK <= 0)
                throw new InferenceContractException("Proposal image dimensions or top-k limits are invalid.");

            var candidates = new List<AexisDetection>();
            for (var index = 0; index < proposalCount; index++)
            {
                if (scores[index] < scoreThreshold)
                    continue;
                var anchor = ReadBox(anchors, index * 4);
                var width = Math.Max(1f, anchor.xMax - anchor.xMin + 1f);
                var height = Math.Max(1f, anchor.yMax - anchor.yMin + 1f);
                var centerX = anchor.xMin + 0.5f * width;
                var centerY = anchor.yMin + 0.5f * height;
                var dx = deltas[index * 4];
                var dy = deltas[index * 4 + 1];
                var dw = Math.Min(deltas[index * 4 + 2], 4.1351666f);
                var dh = Math.Min(deltas[index * 4 + 3], 4.1351666f);
                var predictedWidth = (float)Math.Exp(dw) * width;
                var predictedHeight = (float)Math.Exp(dh) * height;
                var predictedCenterX = dx * width + centerX;
                var predictedCenterY = dy * height + centerY;
                var box = new AexisBoundingBox
                {
                    xMin = Clamp(predictedCenterX - 0.5f * predictedWidth, 0f, imageWidth - 1f),
                    yMin = Clamp(predictedCenterY - 0.5f * predictedHeight, 0f, imageHeight - 1f),
                    xMax = Clamp(predictedCenterX + 0.5f * predictedWidth, 0f, imageWidth - 1f),
                    yMax = Clamp(predictedCenterY + 0.5f * predictedHeight, 0f, imageHeight - 1f)
                };
                if (box.Width < minimumSize || box.Height < minimumSize)
                    continue;
                candidates.Add(new AexisDetection { score = scores[index], box = box });
            }

            candidates.Sort(CompareDetections);
            if (candidates.Count > preNmsTopK)
                candidates.RemoveRange(preNmsTopK, candidates.Count - preNmsTopK);
            return NonMaximumSuppression(candidates, nmsThreshold, postNmsTopK).ToArray();
        }

        public static float IntersectionOverUnion(AexisBoundingBox left, AexisBoundingBox right)
        {
            var width = Math.Max(0f, Math.Min(left.xMax, right.xMax) - Math.Max(left.xMin, right.xMin));
            var height = Math.Max(0f, Math.Min(left.yMax, right.yMax) - Math.Max(left.yMin, right.yMin));
            var intersection = width * height;
            var union = left.Area + right.Area - intersection;
            return union <= 0f ? 0f : intersection / union;
        }

        private static List<AexisDetection> NonMaximumSuppression(List<AexisDetection> candidates, float threshold, int limit)
        {
            candidates.Sort(CompareDetections);
            var selected = new List<AexisDetection>();
            foreach (var candidate in candidates)
            {
                var keep = true;
                foreach (var existing in selected)
                {
                    if (IntersectionOverUnion(candidate.box, existing.box) > threshold)
                    {
                        keep = false;
                        break;
                    }
                }
                if (!keep)
                    continue;
                selected.Add(candidate);
                if (selected.Count == limit)
                    break;
            }
            return selected;
        }

        private static int CompareDetections(AexisDetection left, AexisDetection right)
        {
            var score = right.score.CompareTo(left.score);
            if (score != 0)
                return score;
            var category = left.classIndex.CompareTo(right.classIndex);
            if (category != 0)
                return category;
            return left.box.xMin.CompareTo(right.box.xMin);
        }

        private static AexisBoundingBox ReadBox(IReadOnlyList<float> values, int offset)
        {
            return new AexisBoundingBox { xMin = values[offset], yMin = values[offset + 1], xMax = values[offset + 2], yMax = values[offset + 3] };
        }

        private static float Sigmoid(float value)
        {
            return 1f / (1f + (float)Math.Exp(-value));
        }

        private static float Clamp(float value, float minimum, float maximum)
        {
            return Math.Min(maximum, Math.Max(minimum, value));
        }
    }
}
