using System;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public enum RepoVkTensorLayoutKind
    {
        LinearMat = 0,
        Pack4Image = 1
    }

    public abstract class RepoVkTensor
    {
        protected RepoVkTensor(
            RepoVkTensorLayoutKind layoutKind,
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape)
        {
            LayoutKind = layoutKind;
            LogicalShape = logicalShape;
            StorageShape = storageShape;
        }

        public RepoVkTensorLayoutKind LayoutKind { get; }
        public NcnnRepro.BufferShape LogicalShape { get; }
        public NcnnRepro.BufferShape StorageShape { get; }
        public abstract RenderTexture RenderTexture { get; }
        public abstract ComputeTexture ComputeTexture { get; }
        public abstract int Width { get; }
        public abstract int Height { get; }
        public abstract int Depth { get; }
        public abstract int Packs { get; }

        public bool IsRenderTextureBacked => RenderTexture != null;
        public bool IsCommandTextureBacked => ComputeTexture != null;
        public bool UsesTexture2DPhysicalBacking => RenderTexture != null && RenderTexture.dimension == TextureDimension.Tex2D;
        public bool UsesTexture2DArrayPhysicalBacking => RenderTexture != null && RenderTexture.dimension == TextureDimension.Tex2DArray;
    }

    public sealed class RepoVkMat : RepoVkTensor
    {
        private readonly RenderTexture _renderTexture;
        private readonly ComputeTexture _computeTexture;
        private readonly int _width;
        private readonly int _height;
        private readonly int _depth;
        private readonly int _packs;

        public RepoVkMat(
            RenderTexture texture,
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape,
            int packs = 1)
            : base(RepoVkTensorLayoutKind.LinearMat, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _renderTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = 1;
            _packs = Mathf.Max(1, packs);
        }

        public RepoVkMat(
            ComputeTexture texture,
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape,
            int packs = 1)
            : base(RepoVkTensorLayoutKind.LinearMat, logicalShape, storageShape)
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
        public override int Width => _width;
        public override int Height => _height;
        public override int Depth => _depth;
        public override int Packs => _packs;
        public bool IsStrictTexture2D => _renderTexture != null && _renderTexture.dimension == TextureDimension.Tex2D;
    }

    public sealed class RepoVkImageMat : RepoVkTensor
    {
        private readonly RenderTexture _renderTexture;
        private readonly ComputeTexture _computeTexture;
        private readonly int _width;
        private readonly int _height;
        private readonly int _depth;
        private readonly int _packs;

        public RepoVkImageMat(
            RenderTexture texture,
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape,
            int packs,
            int depth)
            : base(RepoVkTensorLayoutKind.Pack4Image, logicalShape, storageShape)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            _renderTexture = texture;
            _width = Mathf.Max(1, texture.width);
            _height = Mathf.Max(1, texture.height);
            _depth = Mathf.Max(1, depth);
            _packs = Mathf.Max(1, packs);
        }

        public RepoVkImageMat(
            ComputeTexture texture,
            NcnnRepro.BufferShape logicalShape,
            NcnnRepro.BufferShape storageShape,
            int packs,
            int depth)
            : base(RepoVkTensorLayoutKind.Pack4Image, logicalShape, storageShape)
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
        public override int Width => _width;
        public override int Height => _height;
        public override int Depth => _depth;
        public override int Packs => _packs;
    }
}
