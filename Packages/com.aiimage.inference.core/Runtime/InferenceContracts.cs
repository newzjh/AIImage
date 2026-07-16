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
        Int32 = 5
    }

    // D2 intentionally exposes one narrowly-scoped quantized storage contract.  It is
    // not an activation quantization API: activations and intermediate tensors remain
    // Float16/Float32 Pack4 textures while only immutable weights are stored as INT8.
    public enum WeightQuantizationScheme
    {
        None = 0,
        Int8WeightOnlyPerOutputChannelSymmetric = 1
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
                && value != TensorDataType.Int8)
            {
                throw new InferenceContractException(
                    "Model precision " + name + " must be Float16, Float32, or INT8 weight-only.");
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
        // D2 permits a model to quantize only the nodes for which a packed texture
        // kernel exists.  An empty list preserves the v1 all-weight legacy behavior
        // (and therefore its strict rejection of every unsupported weighted op).
        public string[] quantizedOperators = Array.Empty<string>();
        public TensorDataType unquantizedWeightDataType = TensorDataType.Float32;

        public bool IsInt8WeightOnly => weightScheme == WeightQuantizationScheme.Int8WeightOnlyPerOutputChannelSymmetric;

        public void ValidateInt8WeightOnly()
        {
            if (!IsInt8WeightOnly)
                throw new InferenceContractException("Only INT8 per-output-channel weight-only quantization is supported.");
            if (string.IsNullOrWhiteSpace(quantizationVersion))
                throw new InferenceContractException("INT8 weight-only quantization requires a quantizationVersion.");
            if (string.IsNullOrWhiteSpace(calibrationVersion))
                throw new InferenceContractException("INT8 weight-only quantization requires a calibrationVersion.");
            if (string.IsNullOrWhiteSpace(calibrationMethod))
                throw new InferenceContractException("INT8 weight-only quantization requires a calibrationMethod.");
            if (outputChannelAxis != 0)
                throw new InferenceContractException("INT8 weight-only quantization supports only outputChannelAxis=0.");
            if (!symmetric || zeroPoint != 0)
                throw new InferenceContractException("INT8 weight-only quantization is symmetric with zeroPoint=0.");
            if (accumulationDataType != TensorDataType.Float32)
                throw new InferenceContractException("INT8 weight-only kernels require Float32 accumulation.");
            if (activationQuantized)
                throw new InferenceContractException("INT8 weight-only does not quantize activations; W8A8 is not supported by this contract.");
            if (unquantizedWeightDataType != TensorDataType.Float16
                && unquantizedWeightDataType != TensorDataType.Float32)
            {
                throw new InferenceContractException("INT8 weight-only unquantized weights must remain Float16 or Float32.");
            }
            if (quantizedOperators != null)
            {
                for (var index = 0; index < quantizedOperators.Length; index++)
                {
                    if (string.IsNullOrWhiteSpace(quantizedOperators[index]))
                        throw new InferenceContractException("INT8 weight-only quantizedOperators cannot contain an empty operator name.");
                }
            }
        }

        public bool QuantizesOperator(string operatorName)
        {
            // Manifests produced before selective D2 plans have no list.  Keep their
            // fail-closed semantics: every immutable-weight node must prove an INT8
            // kernel instead of silently remaining float.
            if (quantizedOperators == null || quantizedOperators.Length == 0)
                return true;
            if (string.IsNullOrWhiteSpace(operatorName))
                return false;
            for (var index = 0; index < quantizedOperators.Length; index++)
            {
                if (string.Equals(quantizedOperators[index], operatorName, StringComparison.Ordinal))
                    return true;
            }
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
            if (precision.weightDataType == TensorDataType.Int8)
            {
                if (quantization == null)
                    throw new InferenceContractException("INT8 weights require a quantization contract.");
                quantization.ValidateInt8WeightOnly();
            }
            else if (quantization != null && quantization.weightScheme != WeightQuantizationScheme.None)
            {
                throw new InferenceContractException("A quantization contract requires precision.weightDataType=Int8.");
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

        public bool UsesInt8WeightOnlyForOperator(string operatorName)
        {
            return IsInt8WeightOnly && quantization.QuantizesOperator(operatorName);
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
