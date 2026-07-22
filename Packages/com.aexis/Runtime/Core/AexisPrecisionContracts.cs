using System;
using System.Collections.Generic;

namespace Aexis
{
    public enum ActivationQuantizationPacking
    {
        None = 0,
        Pack4SignedInt8 = 1,
        Pack4UnsignedInt8 = 2
    }

    public enum CalibrationMethod
    {
        MinMax = 0,
        Percentile = 1,
        Entropy = 2
    }

    [Serializable]
    public sealed class ActivationCalibrationRange
    {
        public string layerName = string.Empty;
        public string tensorName = string.Empty;
        public float minimum;
        public float maximum;
        public int sampleCount;
        public CalibrationMethod method = CalibrationMethod.MinMax;

        public float SymmetricScale
        {
            get
            {
                var magnitude = Math.Max(Math.Abs(minimum), Math.Abs(maximum));
                return magnitude == 0f ? 1f : magnitude / 127f;
            }
        }

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(layerName) && string.IsNullOrWhiteSpace(tensorName))
                throw new InferenceContractException("An activation calibration range requires layerName or tensorName.");
            if (float.IsNaN(minimum) || float.IsInfinity(minimum)
                || float.IsNaN(maximum) || float.IsInfinity(maximum)
                || minimum > maximum)
            {
                throw new InferenceContractException("Activation calibration range must contain finite ordered bounds.");
            }
            if (sampleCount <= 0)
                throw new InferenceContractException("Activation calibration ranges require sampleCount > 0.");
        }
    }

    [Serializable]
    public sealed class QuantizedActivationPlan
    {
        public string layerName = string.Empty;
        public string operatorName = string.Empty;
        public ActivationQuantizationPacking packing = ActivationQuantizationPacking.Pack4SignedInt8;
        public ActivationCalibrationRange calibration = new ActivationCalibrationRange();
        public bool dequantizeOutput = true;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(layerName))
                throw new InferenceContractException("Quantized activation plans require layerName.");
            if (packing == ActivationQuantizationPacking.None)
                throw new InferenceContractException("Quantized activation plans require an INT8 packing format.");
            if (calibration == null)
                throw new InferenceContractException("Quantized activation plans require calibration data.");
            calibration.Validate();
        }
    }

    [Serializable]
    public sealed class MixedPrecisionNodePlan
    {
        public string layerName = string.Empty;
        public string operatorName = string.Empty;
        public TensorDataType activationDataType = TensorDataType.Float16;
        public TensorDataType weightDataType = TensorDataType.Float16;
        public TensorDataType accumulationDataType = TensorDataType.Float32;
        public float maximumAbsoluteError = float.PositiveInfinity;
        public float minimumCosineSimilarity = -1f;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(layerName))
                throw new InferenceContractException("Mixed precision node plans require layerName.");
            ValidateFloating(activationDataType, "activationDataType");
            ValidateWeight(weightDataType, "weightDataType");
            ValidateFloating(accumulationDataType, "accumulationDataType");
            if (accumulationDataType == TensorDataType.BFloat16)
                throw new InferenceContractException("Mixed precision accumulationDataType must be Float16 or Float32.");
            if (float.IsNaN(maximumAbsoluteError) || maximumAbsoluteError < 0f)
                throw new InferenceContractException("Mixed precision maximumAbsoluteError must be non-negative.");
            if (float.IsNaN(minimumCosineSimilarity) || minimumCosineSimilarity < -1f || minimumCosineSimilarity > 1f)
                throw new InferenceContractException("Mixed precision minimumCosineSimilarity must be in [-1, 1].");
        }

        private static void ValidateFloating(TensorDataType value, string name)
        {
            if (value != TensorDataType.Float16 && value != TensorDataType.BFloat16 && value != TensorDataType.Float32)
                throw new InferenceContractException("Mixed precision " + name + " must be Float16, BFloat16, or Float32.");
        }

        private static void ValidateWeight(TensorDataType value, string name)
        {
            if (value != TensorDataType.Float16 && value != TensorDataType.BFloat16 && value != TensorDataType.Float32
                && value != TensorDataType.Int8 && value != TensorDataType.Int4)
            {
                throw new InferenceContractException("Mixed precision " + name + " has an unsupported dtype.");
            }
        }
    }

    [Serializable]
    public sealed class ModelMixedPrecisionContract
    {
        public string planVersion = string.Empty;
        public MixedPrecisionNodePlan[] nodePlans = Array.Empty<MixedPrecisionNodePlan>();
        public QuantizedActivationPlan[] activationPlans = Array.Empty<QuantizedActivationPlan>();

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(planVersion))
                throw new InferenceContractException("Mixed precision contracts require planVersion.");

            var names = new HashSet<string>(StringComparer.Ordinal);
            foreach (var plan in nodePlans ?? Array.Empty<MixedPrecisionNodePlan>())
            {
                if (plan == null)
                    throw new InferenceContractException("Mixed precision node plans cannot contain null.");
                plan.Validate();
                if (!names.Add(plan.layerName))
                    throw new InferenceContractException("Mixed precision node plans contain duplicate layer " + plan.layerName + ".");
            }

            var activationNames = new HashSet<string>(StringComparer.Ordinal);
            foreach (var plan in activationPlans ?? Array.Empty<QuantizedActivationPlan>())
            {
                if (plan == null)
                    throw new InferenceContractException("Quantized activation plans cannot contain null.");
                plan.Validate();
                if (!activationNames.Add(plan.layerName))
                    throw new InferenceContractException("Quantized activation plans contain duplicate layer " + plan.layerName + ".");
            }
        }
    }

    [Serializable]
    public sealed class PrecisionGateMeasurement
    {
        public string outputName = string.Empty;
        public float maximumAbsoluteError;
        public float meanAbsoluteError;
        public float cosineSimilarity = 1f;

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(outputName))
                throw new InferenceContractException("Precision gate measurements require outputName.");
            if (float.IsNaN(maximumAbsoluteError) || maximumAbsoluteError < 0f
                || float.IsNaN(meanAbsoluteError) || meanAbsoluteError < 0f
                || float.IsNaN(cosineSimilarity) || cosineSimilarity < -1f || cosineSimilarity > 1f)
            {
                throw new InferenceContractException("Precision gate measurements are invalid for " + outputName + ".");
            }
        }
    }

    [Serializable]
    public sealed class ModelPrecisionGateContract
    {
        public string gateVersion = string.Empty;
        public float maximumAbsoluteError = float.PositiveInfinity;
        public float maximumMeanAbsoluteError = float.PositiveInfinity;
        public float minimumCosineSimilarity = -1f;
        public PrecisionGateMeasurement[] baseline = Array.Empty<PrecisionGateMeasurement>();

        public void Validate()
        {
            if (string.IsNullOrWhiteSpace(gateVersion))
                throw new InferenceContractException("Precision gates require gateVersion.");
            if (float.IsNaN(maximumAbsoluteError) || maximumAbsoluteError < 0f
                || float.IsNaN(maximumMeanAbsoluteError) || maximumMeanAbsoluteError < 0f
                || float.IsNaN(minimumCosineSimilarity) || minimumCosineSimilarity < -1f || minimumCosineSimilarity > 1f)
            {
                throw new InferenceContractException("Precision gate thresholds are invalid.");
            }
            foreach (var measurement in baseline ?? Array.Empty<PrecisionGateMeasurement>())
            {
                if (measurement == null)
                    throw new InferenceContractException("Precision gate baseline cannot contain null.");
                measurement.Validate();
            }
        }

        public bool Accepts(PrecisionGateMeasurement measurement, out string reason)
        {
            Validate();
            if (measurement == null)
                throw new ArgumentNullException(nameof(measurement));
            measurement.Validate();
            if (measurement.maximumAbsoluteError > maximumAbsoluteError)
            {
                reason = "maximumAbsoluteError exceeded for " + measurement.outputName;
                return false;
            }
            if (measurement.meanAbsoluteError > maximumMeanAbsoluteError)
            {
                reason = "meanAbsoluteError exceeded for " + measurement.outputName;
                return false;
            }
            if (measurement.cosineSimilarity < minimumCosineSimilarity)
            {
                reason = "cosineSimilarity fell below the gate for " + measurement.outputName;
                return false;
            }
            reason = string.Empty;
            return true;
        }
    }

    public static class AexisImportCalibration
    {
        public static ActivationCalibrationRange Calibrate(
            string layerName,
            string tensorName,
            IEnumerable<float> samples,
            CalibrationMethod method = CalibrationMethod.MinMax)
        {
            if (samples == null)
                throw new ArgumentNullException(nameof(samples));

            var minimum = float.PositiveInfinity;
            var maximum = float.NegativeInfinity;
            var count = 0;
            foreach (var sample in samples)
            {
                if (float.IsNaN(sample) || float.IsInfinity(sample))
                    throw new InferenceContractException("Calibration samples must be finite.");
                minimum = Math.Min(minimum, sample);
                maximum = Math.Max(maximum, sample);
                count++;
            }

            var range = new ActivationCalibrationRange
            {
                layerName = layerName ?? string.Empty,
                tensorName = tensorName ?? string.Empty,
                minimum = count == 0 ? 0f : minimum,
                maximum = count == 0 ? 0f : maximum,
                sampleCount = count,
                method = method
            };
            range.Validate();
            return range;
        }
    }

    public static class AexisPrecisionGateEvaluator
    {
        public static PrecisionGateMeasurement Measure(
            string outputName,
            IReadOnlyList<float> reference,
            IReadOnlyList<float> candidate)
        {
            if (reference == null)
                throw new ArgumentNullException(nameof(reference));
            if (candidate == null)
                throw new ArgumentNullException(nameof(candidate));
            if (reference.Count != candidate.Count || reference.Count == 0)
                throw new InferenceContractException("Precision gate comparison requires equally sized, non-empty outputs.");

            var maximumAbsoluteError = 0f;
            double absoluteErrorSum = 0d;
            double dot = 0d;
            double referenceNorm = 0d;
            double candidateNorm = 0d;
            for (var index = 0; index < reference.Count; index++)
            {
                var expected = reference[index];
                var actual = candidate[index];
                if (float.IsNaN(expected) || float.IsInfinity(expected)
                    || float.IsNaN(actual) || float.IsInfinity(actual))
                {
                    throw new InferenceContractException("Precision gate outputs must be finite.");
                }
                var error = Math.Abs(expected - actual);
                maximumAbsoluteError = Math.Max(maximumAbsoluteError, error);
                absoluteErrorSum += error;
                dot += expected * actual;
                referenceNorm += expected * expected;
                candidateNorm += actual * actual;
            }

            var denominator = Math.Sqrt(referenceNorm) * Math.Sqrt(candidateNorm);
            return new PrecisionGateMeasurement
            {
                outputName = outputName ?? string.Empty,
                maximumAbsoluteError = maximumAbsoluteError,
                meanAbsoluteError = (float)(absoluteErrorSum / reference.Count),
                cosineSimilarity = denominator == 0d ? (referenceNorm == 0d && candidateNorm == 0d ? 1f : 0f) : (float)(dot / denominator)
            };
        }
    }
}
