using System;
using System.Collections.Generic;

namespace AIImage.Inference.Core
{
    public enum InferenceBackend
    {
        Unknown = 0,
        UnityGpu = 1
    }

    public enum InferenceSessionState
    {
        Created = 0,
        Ready = 1,
        Disposed = 2
    }

    public enum TensorLayout
    {
        Unknown = 0,
        Nchw = 1,
        Nhwc = 2,
        Cdhw = 3,
        Linear = 4,
        Scalar = 5,
        Packed4 = 6
    }

    public enum TensorDataType
    {
        Unknown = 0,
        Float32 = 1,
        Float16 = 2,
        Int8 = 3,
        UInt8 = 4,
        Int32 = 5,
        Int4 = 6
    }

    // D2 exposes narrowly-scoped quantized weight storage contracts. Selective
    // nodePlans may additionally request calibrated W8A8 math for specific INT8 layers;
    // intermediate tensors remain Float16/Float32 Pack4 textures.
    public enum WeightQuantizationScheme
    {
        None = 0,
        Int8WeightOnlyPerOutputChannelSymmetric = 1,
        Int4WeightOnlyPerOutputChannelSymmetric = 2
    }

    public enum QuantizedNodeMode
    {
        Float = 0,
        Int8WeightOnly = 1,
        Int8W8A8 = 2,
        Int4WeightOnly = 3
    }

    [Serializable]
    public sealed class QuantizedNodePlan
    {
        public string layerName = string.Empty;
        public string operatorName = string.Empty;
        public QuantizedNodeMode mode = QuantizedNodeMode.Float;
        public float activationScale = 1f;
        public int activationZeroPoint;

        public void Validate()
        {
            if (mode == QuantizedNodeMode.Float)
                return;
            if (string.IsNullOrWhiteSpace(layerName))
                throw new InferenceContractException("Quantized node plans require layerName.");
            if (mode == QuantizedNodeMode.Int8W8A8
                && (activationScale <= 0f || float.IsNaN(activationScale) || float.IsInfinity(activationScale)))
            {
                throw new InferenceContractException("W8A8 node " + layerName + " requires a finite positive activationScale.");
            }
            if (mode == QuantizedNodeMode.Int4WeightOnly && activationScale != 1f)
                throw new InferenceContractException("W4 node " + layerName + " does not support activation quantization.");
            if (activationZeroPoint < -128 || activationZeroPoint > 127)
                throw new InferenceContractException("Quantized node " + layerName + " activationZeroPoint must fit INT8.");
        }
    }

    // Precision is a model contract, rather than a renderer-wide switch.  The public
    // fields deliberately remain serializer-friendly so importers may use Unity's JSON
    // support without making the core package depend on Unity.
    [Serializable]
    public sealed class ModelPrecisionContract
    {
        public TensorDataType activationDataType = TensorDataType.Float32;
        public TensorDataType weightDataType = TensorDataType.Float32;
        public TensorDataType sensitiveOutputDataType = TensorDataType.Float32;
        public bool requireStrictTexturePlan = true;

        public void Validate()
        {
            ValidateFloatingType(activationDataType, nameof(activationDataType));
            ValidateWeightType(weightDataType, nameof(weightDataType));
            ValidateFloatingType(sensitiveOutputDataType, nameof(sensitiveOutputDataType));
        }

        private static void ValidateFloatingType(TensorDataType value, string name)
        {
            if (value != TensorDataType.Float16 && value != TensorDataType.Float32)
                throw new InferenceContractException("Model precision " + name + " must be Float16 or Float32; INT8 is not part of this contract.");
        }

        private static void ValidateWeightType(TensorDataType value, string name)
        {
            if (value != TensorDataType.Float16
                && value != TensorDataType.Float32
                && value != TensorDataType.Int8
                && value != TensorDataType.Int4)
            {
                throw new InferenceContractException(
                    "Model precision " + name + " must be Float16, Float32, INT8, or INT4 weight-only.");
            }
        }
    }

    // The serialized names are deliberately explicit so a released model records both
    // the mathematical policy and the calibration provenance required to reproduce it.
    [Serializable]
    public sealed class ModelQuantizationContract
    {
        public string quantizationVersion = string.Empty;
        public string calibrationVersion = string.Empty;
        public string calibrationMethod = string.Empty;
        public WeightQuantizationScheme weightScheme = WeightQuantizationScheme.None;
        public int outputChannelAxis = 0;
        public bool symmetric = true;
        public int zeroPoint = 0;
        public TensorDataType accumulationDataType = TensorDataType.Float32;
        public bool activationQuantized;
        public QuantizedNodePlan[] nodePlans = Array.Empty<QuantizedNodePlan>();
        // Selective INT8 may use quantizedOperators as the default W8 set, while
        // nodePlans override individual layers to Float or calibrated W8A8.
        // When nodePlans are present and quantizedOperators is empty, selection is
        // explicit-only so runner manifests can grow coverage from measured layers.
        public string[] quantizedOperators = Array.Empty<string>();
        public TensorDataType unquantizedWeightDataType = TensorDataType.Float32;

        public bool IsInt8WeightOnly => weightScheme == WeightQuantizationScheme.Int8WeightOnlyPerOutputChannelSymmetric;
        public bool IsInt4WeightOnly => weightScheme == WeightQuantizationScheme.Int4WeightOnlyPerOutputChannelSymmetric;
        public bool IsWeightOnlyQuantization => IsInt8WeightOnly || IsInt4WeightOnly;

        public void ValidateWeightOnly()
        {
            if (!IsWeightOnlyQuantization)
                throw new InferenceContractException("Only INT8/INT4 per-output-channel weight-only quantization is supported.");
            var label = IsInt4WeightOnly ? "INT4" : "INT8";
            if (string.IsNullOrWhiteSpace(quantizationVersion))
                throw new InferenceContractException(label + " weight-only quantization requires a quantizationVersion.");
            if (string.IsNullOrWhiteSpace(calibrationVersion))
                throw new InferenceContractException(label + " weight-only quantization requires a calibrationVersion.");
            if (string.IsNullOrWhiteSpace(calibrationMethod))
                throw new InferenceContractException(label + " weight-only quantization requires a calibrationMethod.");
            if (outputChannelAxis != 0)
                throw new InferenceContractException(label + " weight-only quantization supports only outputChannelAxis=0.");
            if (!symmetric || zeroPoint != 0)
                throw new InferenceContractException(label + " weight-only quantization is symmetric with zeroPoint=0.");
            if (accumulationDataType != TensorDataType.Float32)
                throw new InferenceContractException(label + " weight-only kernels require Float32 accumulation.");
            if (activationQuantized && (nodePlans == null || nodePlans.Length == 0))
                throw new InferenceContractException("Activation quantization requires explicit calibrated nodePlans.");
            if (unquantizedWeightDataType != TensorDataType.Float16
                && unquantizedWeightDataType != TensorDataType.Float32)
            {
                throw new InferenceContractException(label + " weight-only unquantized weights must remain Float16 or Float32.");
            }
            if (quantizedOperators != null)
            {
                for (var index = 0; index < quantizedOperators.Length; index++)
                {
                    if (string.IsNullOrWhiteSpace(quantizedOperators[index]))
                        throw new InferenceContractException(label + " weight-only quantizedOperators cannot contain an empty operator name.");
                }
            }
            var plannedLayers = new HashSet<string>(StringComparer.Ordinal);
            foreach (var plan in nodePlans ?? Array.Empty<QuantizedNodePlan>())
            {
                if (plan == null)
                    throw new InferenceContractException("Quantization nodePlans cannot contain null.");
                plan.Validate();
                if (plan.mode != QuantizedNodeMode.Float && !plannedLayers.Add(plan.layerName))
                    throw new InferenceContractException("Quantization nodePlans contain duplicate layer " + plan.layerName + ".");
                if (plan.mode == QuantizedNodeMode.Int8W8A8 && !activationQuantized)
                    throw new InferenceContractException("W8A8 node " + plan.layerName + " requires activationQuantized=true.");
                if (plan.mode == QuantizedNodeMode.Int8W8A8 && !IsInt8WeightOnly)
                    throw new InferenceContractException("W8A8 node " + plan.layerName + " requires INT8 weight quantization.");
                if (plan.mode == QuantizedNodeMode.Int4WeightOnly && !IsInt4WeightOnly)
                    throw new InferenceContractException("W4 node " + plan.layerName + " requires INT4 weight quantization.");
            }
        }

        public void ValidateInt8WeightOnly()
        {
            ValidateWeightOnly();
        }

        public bool QuantizesOperator(string operatorName)
        {
            if (quantizedOperators == null || quantizedOperators.Length == 0)
                return nodePlans == null || nodePlans.Length == 0;
            if (string.IsNullOrWhiteSpace(operatorName))
                return false;
            for (var index = 0; index < quantizedOperators.Length; index++)
            {
                if (string.Equals(quantizedOperators[index], operatorName, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

        public bool TryGetNodePlan(string layerName, string operatorName, out QuantizedNodePlan plan)
        {
            foreach (var candidate in nodePlans ?? Array.Empty<QuantizedNodePlan>())
            {
                if (candidate != null && string.Equals(candidate.layerName, layerName, StringComparison.Ordinal))
                {
                    if (!string.IsNullOrWhiteSpace(candidate.operatorName)
                        && !string.Equals(candidate.operatorName, operatorName, StringComparison.Ordinal))
                    {
                        break;
                    }
                    plan = candidate;
                    return candidate.mode != QuantizedNodeMode.Float;
                }
            }
            if (QuantizesOperator(operatorName))
            {
                plan = new QuantizedNodePlan
                {
                    layerName = layerName ?? string.Empty,
                    operatorName = operatorName ?? string.Empty,
                    mode = IsInt4WeightOnly ? QuantizedNodeMode.Int4WeightOnly : QuantizedNodeMode.Int8WeightOnly,
                    activationScale = 1f,
                    activationZeroPoint = 0
                };
                return true;
            }
            plan = null;
            return false;
        }
    }

    [Serializable]
    public sealed class ModelManifest
    {
        public const string Contract = "aiimage.model-manifest/v1";

        public string schemaVersion = Contract;
        public string modelId = string.Empty;
        public ModelPrecisionContract precision = new ModelPrecisionContract();
        public ModelQuantizationContract quantization;

        public void Validate()
        {
            if (!string.Equals(schemaVersion, Contract, StringComparison.Ordinal))
                throw new InferenceContractException("Unsupported model manifest schema: " + (schemaVersion ?? string.Empty));
            if (string.IsNullOrWhiteSpace(modelId))
                throw new InferenceContractException("Model manifest requires a modelId.");
            if (precision == null)
                throw new InferenceContractException("Model manifest requires a precision contract.");
            precision.Validate();
            if (precision.weightDataType == TensorDataType.Int8
                || precision.weightDataType == TensorDataType.Int4)
            {
                if (quantization == null)
                    throw new InferenceContractException("Quantized weights require a quantization contract.");
                quantization.ValidateWeightOnly();
            }
            else if (quantization != null && quantization.weightScheme != WeightQuantizationScheme.None)
            {
                throw new InferenceContractException("A quantization contract requires precision.weightDataType=Int8 or Int4.");
            }
        }

        public bool IsFp16Mixed
        {
            get
            {
                return precision != null
                    && (precision.activationDataType == TensorDataType.Float16
                        || precision.weightDataType == TensorDataType.Float16);
            }
        }

        public bool IsInt8WeightOnly
        {
            get
            {
                return precision != null
                    && precision.weightDataType == TensorDataType.Int8
                    && quantization != null
                    && quantization.IsInt8WeightOnly;
            }
        }

        public bool IsInt4WeightOnly
        {
            get
            {
                return precision != null
                    && precision.weightDataType == TensorDataType.Int4
                    && quantization != null
                    && quantization.IsInt4WeightOnly;
            }
        }

        public bool IsWeightOnlyQuantized => IsInt8WeightOnly || IsInt4WeightOnly;

        public bool UsesInt8WeightOnlyForOperator(string operatorName)
        {
            return IsInt8WeightOnly && quantization.QuantizesOperator(operatorName);
        }

        public bool UsesInt4WeightOnlyForOperator(string operatorName)
        {
            return IsInt4WeightOnly && quantization.QuantizesOperator(operatorName);
        }

        public bool TryGetQuantizedNodePlan(string layerName, string operatorName, out QuantizedNodePlan plan)
        {
            plan = null;
            return IsWeightOnlyQuantized && quantization != null && quantization.TryGetNodePlan(layerName, operatorName, out plan);
        }
    }

    public sealed class TensorDescriptor
    {
        private readonly int[] _logicalShape;
        private readonly int[] _storageShape;

        public TensorDescriptor(
            IEnumerable<int> logicalShape,
            IEnumerable<int> storageShape,
            TensorLayout layout,
            TensorDataType dataType,
            string debugName = null)
        {
            _logicalShape = CopyAndValidate(logicalShape, nameof(logicalShape));
            _storageShape = CopyAndValidate(storageShape, nameof(storageShape));
            Layout = layout;
            DataType = dataType;
            DebugName = debugName ?? string.Empty;
        }

        public IReadOnlyList<int> LogicalShape => _logicalShape;
        public IReadOnlyList<int> StorageShape => _storageShape;
        public TensorLayout Layout { get; }
        public TensorDataType DataType { get; }
        public string DebugName { get; }

        private static int[] CopyAndValidate(IEnumerable<int> values, string parameterName)
        {
            if (values == null)
                throw new ArgumentNullException(parameterName);

            var result = values as int[];
            if (result == null)
            {
                var list = new List<int>(values);
                result = list.ToArray();
            }
            else
            {
                result = (int[])result.Clone();
            }

            if (result.Length == 0)
                throw new ArgumentException("A tensor shape must have at least one dimension.", parameterName);
            for (var i = 0; i < result.Length; i++)
            {
                if (result[i] <= 0)
                    throw new ArgumentOutOfRangeException(parameterName, "Tensor dimensions must be positive.");
            }
            return result;
        }
    }

    // This intentionally exposes no UnityEngine or buffer API. Backend packages own native resources.
    public interface IInferenceTensor
    {
        TensorDescriptor Descriptor { get; }
        ulong ResourceId { get; }
    }

    public interface IInferenceSession : IDisposable
    {
        string SessionId { get; }
        InferenceBackend Backend { get; }
        InferenceSessionState State { get; }
    }

    public sealed class InferenceContractException : InvalidOperationException
    {
        public InferenceContractException(string message)
            : base(message)
        {
        }
    }
}
