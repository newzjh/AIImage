using System;

namespace Aexis.Execution
{
    // Format adapters supply decoded weights while execution stays format-neutral.
    public abstract class AexisWeightReader : IDisposable
    {
        public abstract long Position { get; }

        public abstract float[] ReadFloat32Array(int count);

        public abstract float[] ReadTensorAsFloat32(int w, int h, int d, int c, int loadType);

        public abstract void SkipTensor(int w, int h, int d, int c, int loadType);

        public abstract bool TryReadQuantizedTensor(int count, int expectedBlockSize, out AexisQuantizedTensor packed);

        // Readers that carry native W4 records override this. Keeping it separate
        // from the legacy W8 probe prevents a W4 archive from being decoded to FP32
        // before the immutable GPU upload.
        public virtual bool TryReadInt4QuantizedTensor(int count, int expectedBlockSize, out AexisQuantizedTensor packed)
        {
            packed = null;
            return false;
        }

        public abstract void Dispose();
    }

    public class AexisQuantizedTensor
    {
        public AexisQuantizedTensor(int elementCount, int blockSize, uint[] packedValues, float[] scales)
        {
            ElementCount = elementCount;
            BlockSize = blockSize;
            PackedValues = packedValues ?? throw new ArgumentNullException(nameof(packedValues));
            Scales = scales ?? throw new ArgumentNullException(nameof(scales));
        }

        public int ElementCount { get; }
        public int BlockSize { get; }
        public uint[] PackedValues { get; }
        public float[] Scales { get; }
    }
}
