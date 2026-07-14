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
            ValidateFloatingType(weightDataType, nameof(weightDataType));
            ValidateFloatingType(sensitiveOutputDataType, nameof(sensitiveOutputDataType));
        }

        private static void ValidateFloatingType(TensorDataType value, string name)
        {
            if (value != TensorDataType.Float16 && value != TensorDataType.Float32)
                throw new InferenceContractException("Model precision " + name + " must be Float16 or Float32; INT8 is not part of this contract.");
        }
    }

    [Serializable]
    public sealed class ModelManifest
    {
        public const string Contract = "aiimage.model-manifest/v1";

        public string schemaVersion = Contract;
        public string modelId = string.Empty;
        public ModelPrecisionContract precision = new ModelPrecisionContract();

        public void Validate()
        {
            if (!string.Equals(schemaVersion, Contract, StringComparison.Ordinal))
                throw new InferenceContractException("Unsupported model manifest schema: " + (schemaVersion ?? string.Empty));
            if (string.IsNullOrWhiteSpace(modelId))
                throw new InferenceContractException("Model manifest requires a modelId.");
            if (precision == null)
                throw new InferenceContractException("Model manifest requires a precision contract.");
            precision.Validate();
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
