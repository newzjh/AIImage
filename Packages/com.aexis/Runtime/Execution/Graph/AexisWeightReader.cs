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
