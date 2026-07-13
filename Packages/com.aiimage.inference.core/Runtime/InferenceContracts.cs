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
