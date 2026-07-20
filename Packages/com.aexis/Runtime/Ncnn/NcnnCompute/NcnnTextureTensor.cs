using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Ncnn
{
    public enum NcnnTextureTensorLayoutKind
    {
        LinearMat = 0,
        Pack4Image = 1
    }

    public enum InferenceTensorDataType
    {
        Unknown = 0,
        Float32 = 1,
        Float16 = 2,
        Int8 = 3
    }

    public enum InferenceTensorLifetime
    {
        ExternalInput = 0,
        GraphOwned = 1,
        SharedAlias = 2,
        ExtractedOutput = 3
    }

    public readonly struct TensorPacking
    {
        public TensorPacking(int packSize, int packCount)
        {
            PackSize = Mathf.Max(1, packSize);
            PackCount = Mathf.Max(1, packCount);
        }

        public int PackSize { get; }
        public int PackCount { get; }
        public bool IsPack4 => PackSize == 4;
    }

    public readonly struct TensorQuantizationMetadata
    {
        public TensorQuantizationMetadata(string scheme, float scale, int zeroPoint, int channelAxis = -1)
        {
            Scheme = scheme ?? string.Empty;
            Scale = scale;
            ZeroPoint = zeroPoint;
            ChannelAxis = channelAxis;
        }

        public string Scheme { get; }
        public float Scale { get; }
        public int ZeroPoint { get; }
        public int ChannelAxis { get; }
        public bool IsQuantized => !string.IsNullOrEmpty(Scheme);
        public static TensorQuantizationMetadata None => default;
    }

    public readonly struct TensorProvenance
    {
        public TensorProvenance(string producer, string nodeName, string blobName, string debugName)
        {
            Producer = producer ?? string.Empty;
            NodeName = nodeName ?? string.Empty;
            BlobName = blobName ?? string.Empty;
            DebugName = debugName ?? string.Empty;
        }

        public string Producer { get; }
        public string NodeName { get; }
        public string BlobName { get; }
        public string DebugName { get; }
    }

    public interface IInferenceTensor
    {
        TensorDescriptor Descriptor { get; }
        bool IsDescriptorPublished { get; }
    }

    public sealed class TensorDescriptor
    {
        public TensorDescriptor(
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape,
            NcnnTextureTensorLayoutKind layout,
            TensorPacking packing,
            InferenceTensorDataType dataType,
            TensorQuantizationMetadata quantization,
            string aliasGroup,
            InferenceTensorLifetime lifetime,
            IInferenceTensor owner,
            TensorProvenance provenance,
            NcnnTextureTensor nativeTensor)
        {
            if (logicalShape.dims <= 0)
                throw new ArgumentOutOfRangeException(nameof(logicalShape));
            if (storageShape.dims <= 0)
                throw new ArgumentOutOfRangeException(nameof(storageShape));
            if (packing.PackSize <= 0 || packing.PackCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(packing));

            LogicalShape = logicalShape;
            StorageShape = storageShape;
            Layout = layout;
            Packing = packing;
            DataType = dataType;
            Quantization = quantization;
            AliasGroup = string.IsNullOrWhiteSpace(aliasGroup) ? Guid.NewGuid().ToString("N") : aliasGroup;
            Lifetime = lifetime;
            Owner = owner;
            Provenance = provenance;
            NativeTensor = nativeTensor;
        }

        public NcnnGraphSession.BufferShape LogicalShape { get; }
        public NcnnGraphSession.BufferShape StorageShape { get; }
        public NcnnTextureTensorLayoutKind Layout { get; }
        public TensorPacking Packing { get; }
        public InferenceTensorDataType DataType { get; }
        public TensorQuantizationMetadata Quantization { get; }
        public string AliasGroup { get; }
        public InferenceTensorLifetime Lifetime { get; }
        public IInferenceTensor Owner { get; }
        public TensorProvenance Provenance { get; }
        public NcnnTextureTensor NativeTensor { get; }

        public bool IsStorageLayoutCompatibleWith(
            NcnnGraphSession.BufferShape targetStorageShape,
            NcnnTextureTensorLayoutKind targetLayout,
            TensorPacking targetPacking,
            InferenceTensorDataType targetDataType)
        {
            return ShapeEquals(StorageShape, targetStorageShape)
                && Layout == targetLayout
                && Packing.PackSize == targetPacking.PackSize
                && Packing.PackCount == targetPacking.PackCount
                && DataType == targetDataType;
        }

        public override string ToString()
        {
            return "logical=" + FormatShape(LogicalShape)
                + " storage=" + FormatShape(StorageShape)
                + " layout=" + Layout
                + " pack=" + Packing.PackSize + "x" + Packing.PackCount
                + " dtype=" + DataType
                + " alias_group=" + AliasGroup;
        }

        internal static bool ShapeEquals(NcnnGraphSession.BufferShape a, NcnnGraphSession.BufferShape b)
        {
            return a.dims == b.dims
                && a.w == b.w
                && a.h == b.h
                && a.d == b.d
                && a.c == b.c;
        }

        internal static string FormatShape(NcnnGraphSession.BufferShape shape)
        {
            return "dims=" + shape.dims
                + " w=" + shape.w
                + " h=" + shape.h
                + " d=" + shape.d
                + " c=" + shape.c;
        }
    }

    public sealed class TensorAliasTransformRequiredException : InvalidOperationException
    {
        public TensorAliasTransformRequiredException(TensorDescriptor source, NcnnGraphSession.BufferShape targetLogicalShape, NcnnGraphSession.BufferShape targetStorageShape)
            : base(
                "texture alias requires a real texture transform; buffer fallback is prohibited"
                + " | source_logical=" + TensorDescriptor.FormatShape(source.LogicalShape)
                + " | source_storage=" + TensorDescriptor.FormatShape(source.StorageShape)
                + " | target_logical=" + TensorDescriptor.FormatShape(targetLogicalShape)
                + " | target_storage=" + TensorDescriptor.FormatShape(targetStorageShape)
                + " | requires_texture_transform=true")
        {
            SourceDescriptor = source;
            TargetLogicalShape = targetLogicalShape;
            TargetStorageShape = targetStorageShape;
        }

        public TensorDescriptor SourceDescriptor { get; }
        public NcnnGraphSession.BufferShape TargetLogicalShape { get; }
        public NcnnGraphSession.BufferShape TargetStorageShape { get; }
    }

    public abstract class NcnnTextureTensor
    {
        protected NcnnTextureTensor(
            NcnnTextureTensorLayoutKind layoutKind,
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape)
        {
            LayoutKind = layoutKind;
            LogicalShape = logicalShape;
            StorageShape = storageShape;
        }

        public NcnnTextureTensorLayoutKind LayoutKind { get; }
        public NcnnGraphSession.BufferShape LogicalShape { get; }
        public NcnnGraphSession.BufferShape StorageShape { get; }
        public abstract RenderTexture RenderTexture { get; }
        public abstract ComputeTexture ComputeTexture { get; }
        public abstract TextureDimension Dimension { get; }
        public abstract int Width { get; }
        public abstract int Height { get; }
        public abstract int Depth { get; }
        public abstract int Packs { get; }

        public bool IsRenderTextureBacked => RenderTexture != null;
        public bool IsCommandTextureBacked => ComputeTexture != null;
        public bool UsesTexture2DPhysicalBacking => Dimension == TextureDimension.Tex2D;
        public bool UsesTexture2DArrayPhysicalBacking => Dimension == TextureDimension.Tex2DArray;
    }

    public sealed class NcnnTextureMat : NcnnTextureTensor
    {
        private readonly RenderTexture _renderTexture;
        private readonly ComputeTexture _computeTexture;
        private readonly int _width;
        private readonly int _height;
        private readonly int _depth;
        private readonly int _packs;

        public NcnnTextureMat(
            RenderTexture texture,
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape,
            int packs = 1)
            : base(NcnnTextureTensorLayoutKind.LinearMat, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _renderTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = 1;
            _packs = Mathf.Max(1, packs);
        }

        public NcnnTextureMat(
            ComputeTexture texture,
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape,
            int packs = 1)
            : base(NcnnTextureTensorLayoutKind.LinearMat, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _computeTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = 1;
            _packs = Mathf.Max(1, packs);
        }

        public override RenderTexture RenderTexture => _renderTexture;
        public override ComputeTexture ComputeTexture => _computeTexture;
        public override TextureDimension Dimension => _renderTexture != null
            ? _renderTexture.dimension
            : (_computeTexture != null ? _computeTexture.dimension : TextureDimension.Unknown);
        public override int Width => _width;
        public override int Height => _height;
        public override int Depth => _depth;
        public override int Packs => _packs;
        public bool IsStrictTexture2D => Dimension == TextureDimension.Tex2D;
    }

    public sealed class NcnnTextureImageMat : NcnnTextureTensor
    {
        private readonly RenderTexture _renderTexture;
        private readonly ComputeTexture _computeTexture;
        private readonly int _width;
        private readonly int _height;
        private readonly int _depth;
        private readonly int _packs;

        public NcnnTextureImageMat(
            RenderTexture texture,
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape,
            int packs,
            int depth)
            : base(NcnnTextureTensorLayoutKind.Pack4Image, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _renderTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = Mathf.Max(1, depth);
            _packs = Mathf.Max(1, packs);
        }

        public NcnnTextureImageMat(
            ComputeTexture texture,
            NcnnGraphSession.BufferShape logicalShape,
            NcnnGraphSession.BufferShape storageShape,
            int packs,
            int depth)
            : base(NcnnTextureTensorLayoutKind.Pack4Image, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _computeTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = Mathf.Max(1, depth);
            _packs = Mathf.Max(1, packs);
        }

        public override RenderTexture RenderTexture => _renderTexture;
        public override ComputeTexture ComputeTexture => _computeTexture;
        public override TextureDimension Dimension => _renderTexture != null
            ? _renderTexture.dimension
            : (_computeTexture != null ? _computeTexture.dimension : TextureDimension.Unknown);
        public override int Width => _width;
        public override int Height => _height;
        public override int Depth => _depth;
        public override int Packs => _packs;
    }
}
