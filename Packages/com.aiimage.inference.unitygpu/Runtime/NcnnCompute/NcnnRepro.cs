using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections;
using AIImage.Inference.Core;
using Cysharp.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace NcnnCompute
{
    public enum NcnnInferenceExecutionMode
    {
        ProductionTextureOnly = 0,
        DebugOracle = 1
    }

    public partial class NcnnRepro : IInferenceSession
    {
        internal static readonly HashSet<string> CodeFormerSftMulLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Mul_581",
            "Mul_687",
            "Mul_794",
            "Mul_900"
        };

        internal static readonly HashSet<string> CodeFormerSftAddLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_582",
            "Add_688",
            "Add_795",
            "Add_901"
        };

        internal static readonly HashSet<string> CodeFormerSftResidualLayers = new HashSet<string>(StringComparer.Ordinal)
        {
            "Add_585",
            "Add_691",
            "Add_798",
            "Add_904"
        };

        public readonly struct BufferShape
        {
            public readonly int dims;
            public readonly int w;
            public readonly int h;
            public readonly int d;
            public readonly int c;

            public BufferShape(int dims, int w, int h, int d, int c)
            {
                this.dims = dims;
                this.w = w;
                this.h = h;
                this.d = d;
                this.c = c;
            }

            public override string ToString()
            {
                return TensorDescriptor.FormatShape(this);
            }
        }

        public readonly struct RepoVkTensorContract
        {
            public RepoVkTensorContract(IInferenceTensor tensor)
            {
                Descriptor = tensor != null
                    ? tensor.Descriptor
                    : throw new ArgumentNullException(nameof(tensor));
                if (Descriptor == null)
                    throw new InvalidOperationException("tensor descriptor has not been published");
            }

            public TensorDescriptor Descriptor { get; }
            public RepoVkTensor Tensor => Descriptor != null ? Descriptor.NativeTensor : null;
            public BufferShape LogicalShape => Descriptor != null ? Descriptor.LogicalShape : default;
            public BufferShape StorageShape => Descriptor != null ? Descriptor.StorageShape : default;
            public RepoVkTensorLayoutKind LayoutKind => Descriptor != null ? Descriptor.Layout : default;
            public int Width => Tensor != null ? Tensor.Width : 0;
            public int Height => Tensor != null ? Tensor.Height : 0;
            public int Depth => Tensor != null ? Tensor.Depth : 0;
            public int Packs => Tensor != null ? Tensor.Packs : 0;
            public bool IsLinearMat => LayoutKind == RepoVkTensorLayoutKind.LinearMat;
            public bool IsPack4Image => LayoutKind == RepoVkTensorLayoutKind.Pack4Image;
        }

        public sealed class TensorRef : IInferenceTensor
        {
            public RenderTexture texture;
            public int width;
            public int height;
            public int packs;
            public int refs;
            public bool owned;
            public bool hasLogicalShape;
            public BufferShape logicalShape;
            public bool hasStorageShape;
            public BufferShape storageShape;
            public RepoVkTensor repoTensor;
            public RepoVkTensorLayoutKind layoutKind;
            public TensorRef sharedTextureOwner;
            private TensorDescriptor _descriptor;

            public TensorDescriptor Descriptor => _descriptor;
            public bool IsDescriptorPublished => _descriptor != null;

            internal void PublishDescriptor(TensorDescriptor descriptor)
            {
                if (descriptor == null)
                    throw new ArgumentNullException(nameof(descriptor));
                if (_descriptor != null)
                    return;

                _descriptor = descriptor;
                repoTensor = descriptor.NativeTensor;
                layoutKind = descriptor.Layout;
                logicalShape = descriptor.LogicalShape;
                storageShape = descriptor.StorageShape;
                hasLogicalShape = true;
                hasStorageShape = true;
                width = descriptor.NativeTensor != null ? descriptor.NativeTensor.Width : width;
                height = descriptor.NativeTensor != null ? descriptor.NativeTensor.Height : height;
                packs = descriptor.Packing.PackCount;
            }

            public void ClearTexture()
            {
                texture = null;
                repoTensor = null;
                sharedTextureOwner = null;
            }
        }

        public sealed class IndexRef
        {
            public RenderTexture texture;
            public ComputeBuffer buffer;
            public NcnnTensorBuffer view;
            public int width;
            public int height;
            public int packs;
            public int sourceWidth;
            public int sourceHeight;
            public int kernelW;
            public int kernelH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padTop;
            public int refs;
            public bool owned;

            public void ClearTexture()
            {
                texture = null;
            }
        }

        public sealed class CmdTensorRef : IInferenceTensor
        {
            public ComputeTexture texture;
            public int width;
            public int height;
            public int packs;
            public int sourceWidth;
            public int sourceHeight;
            public int kernelW;
            public int kernelH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padTop;
            public int refs;
            public bool owned;
            public bool hasLogicalShape;
            public BufferShape logicalShape;
            public bool hasStorageShape;
            public BufferShape storageShape;
            public RepoVkTensor repoTensor;
            public RepoVkTensorLayoutKind layoutKind;
            public CmdTensorRef sharedTextureOwner;
            private TensorDescriptor _descriptor;

            public TensorDescriptor Descriptor => _descriptor;
            public bool IsDescriptorPublished => _descriptor != null;

            internal void PublishDescriptor(TensorDescriptor descriptor)
            {
                if (descriptor == null)
                    throw new ArgumentNullException(nameof(descriptor));
                if (_descriptor != null)
                    return;

                _descriptor = descriptor;
                repoTensor = descriptor.NativeTensor;
                layoutKind = descriptor.Layout;
                logicalShape = descriptor.LogicalShape;
                storageShape = descriptor.StorageShape;
                hasLogicalShape = true;
                hasStorageShape = true;
                width = descriptor.NativeTensor != null ? descriptor.NativeTensor.Width : width;
                height = descriptor.NativeTensor != null ? descriptor.NativeTensor.Height : height;
                packs = descriptor.Packing.PackCount;
            }

            public void ClearTexture()
            {
                texture = null;
                repoTensor = null;
                sharedTextureOwner = null;
            }
        }

        public sealed class BufferRef
        {
            public ComputeBuffer buffer;
            public int refs;
            public bool owned;
        }

        public sealed class ConvPack : IDisposable
        {
            public int outC;
            public int inC;
            public int group;
            public int outPacks;
            public int inPacks;
            public int kernelW;
            public int kernelH;
            public int kernelD;
            public int dilationW;
            public int dilationH;
            public int dilationD;
            public int strideW;
            public int strideH;
            public int strideD;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public int padFront;
            public int padBehind;
            public int biasTerm;
            public int weightSize;
            public int activationType;
            public float activationSlope;
            public bool useBufferPath;
            public bool useWinograd23;
            public bool isDepthWise;
            public ComputeBuffer packedWeight4;
            public ComputeBuffer packedBias4;
            public ComputeBuffer packedWeightTm23;
            public ComputeBuffer packedDepthWiseWeight4;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedWeight4, "NcnnRepro.ConvPack.Dispose"); packedWeight4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedBias4, "NcnnRepro.ConvPack.Dispose"); packedBias4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedWeightTm23, "NcnnRepro.ConvPack.Dispose"); packedWeightTm23?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedDepthWiseWeight4, "NcnnRepro.ConvPack.Dispose"); packedDepthWiseWeight4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(rawWeight, "NcnnRepro.ConvPack.Dispose"); rawWeight?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(rawBias, "NcnnRepro.ConvPack.Dispose"); rawBias?.Dispose(); } catch { }
            }
        }

        public sealed class InnerProductPack : IDisposable
        {
            public int inFeatures;
            public int outFeatures;
            public int biasTerm;
            public int weightSize;
            public ComputeBuffer w;
            public ComputeBuffer b;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(w, "NcnnRepro.InnerProductPack.Dispose"); w?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(b, "NcnnRepro.InnerProductPack.Dispose"); b?.Dispose(); } catch { }
            }
        }

        public sealed class DeconvPack : IDisposable
        {
            public int outC;
            public int inC;
            public int group;
            public int outPacks;
            public int inPacks;
            public int kernelW;
            public int kernelH;
            public int kernelD;
            public int dilationW;
            public int dilationH;
            public int dilationD;
            public int strideW;
            public int strideH;
            public int strideD;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public int padFront;
            public int padBehind;
            public int outputPadRight;
            public int outputPadBottom;
            public int outputPadBehind;
            public int biasTerm;
            public int weightSize;
            public int activationType;
            public float activationSlope;
            public ComputeBuffer packedWeight4;
            public ComputeBuffer packedBias4;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedWeight4, "NcnnRepro.DeconvPack.Dispose"); packedWeight4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(packedBias4, "NcnnRepro.DeconvPack.Dispose"); packedBias4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(rawWeight, "NcnnRepro.DeconvPack.Dispose"); rawWeight?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(rawBias, "NcnnRepro.DeconvPack.Dispose"); rawBias?.Dispose(); } catch { }
            }
        }

        public sealed class GemmPack : IDisposable
        {
            public float alpha;
            public float beta;
            public bool transA;
            public bool transB;
            public bool constantA;
            public bool constantB;
            public bool constantC;
            public int constantM;
            public int constantN;
            public int constantK;
            public int broadcastTypeC;
            public ComputeBuffer bData;
            public ComputeBuffer cData;
            public float[] bDataCpu;
            public float[] cDataCpu;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(bData, "NcnnRepro.GemmPack.Dispose"); bData?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(cData, "NcnnRepro.GemmPack.Dispose"); cData?.Dispose(); } catch { }
            }
        }

        public sealed class MemoryDataPack : IDisposable
        {
            public ComputeBuffer data;
            public Texture2D channelVectorTexture;
            public int dims;
            public int w;
            public int h;
            public int d;
            public int c;
            public float[] cpuData;
            public RenderTexture pack4Rt;
            public int pack4RtChannels;
            public int pack4RtDepth;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(data, "NcnnRepro.MemoryDataPack.Dispose"); data?.Dispose(); } catch { }
                try { if (channelVectorTexture != null) UnityEngine.Object.DestroyImmediate(channelVectorTexture); } catch { }
                try
                {
                    if (pack4Rt != null)
                    {
                        NcnnGpuResourceTracker.ReleaseTexture(pack4Rt, "NcnnRepro.MemoryDataPack.Dispose");
                        pack4Rt.Release();
                        UnityEngine.Object.DestroyImmediate(pack4Rt);
                    }
                }
                catch { }
            }
        }

        public sealed class EmbedPack : IDisposable
        {
            public int numOutput;
            public int inputDim;
            public int biasTerm;
            public int weightSize;
            public ComputeBuffer w;
            public ComputeBuffer b;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(w, "NcnnRepro.EmbedPack.Dispose"); w?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(b, "NcnnRepro.EmbedPack.Dispose"); b?.Dispose(); } catch { }
            }
        }

        public sealed class LayerNormPack : IDisposable
        {
            public int affineSize;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public ComputeBuffer beta;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(gamma, "NcnnRepro.LayerNormPack.Dispose"); gamma?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(beta, "NcnnRepro.LayerNormPack.Dispose"); beta?.Dispose(); } catch { }
            }
        }

        public sealed class GroupNormPack : IDisposable
        {
            public int group;
            public int channels;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public ComputeBuffer beta;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(gamma, "NcnnRepro.GroupNormPack.Dispose"); gamma?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(beta, "NcnnRepro.GroupNormPack.Dispose"); beta?.Dispose(); } catch { }
            }
        }

        public sealed class BatchNormPack : IDisposable
        {
            public int channels;
            public ComputeBuffer biasA;
            public ComputeBuffer scaleB;
            public ComputeBuffer biasA4;
            public ComputeBuffer scaleB4;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(biasA, "NcnnRepro.BatchNormPack.Dispose"); biasA?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(scaleB, "NcnnRepro.BatchNormPack.Dispose"); scaleB?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(biasA4, "NcnnRepro.BatchNormPack.Dispose"); biasA4?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(scaleB4, "NcnnRepro.BatchNormPack.Dispose"); scaleB4?.Dispose(); } catch { }
            }
        }

        public sealed class MultiHeadAttentionPack : IDisposable
        {
            public int embedDim;
            public int numHeads;
            public int weightDataSize;
            public int kdim;
            public int vdim;
            public int qdim;
            public bool attnMask;
            public bool kvCache;
            public float scale;
            public ComputeBuffer qW;
            public ComputeBuffer qB;
            public ComputeBuffer kW;
            public ComputeBuffer kB;
            public ComputeBuffer vW;
            public ComputeBuffer vB;
            public ComputeBuffer oW;
            public ComputeBuffer oB;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(qW, "NcnnRepro.MultiHeadAttentionPack.Dispose"); qW?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(qB, "NcnnRepro.MultiHeadAttentionPack.Dispose"); qB?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(kW, "NcnnRepro.MultiHeadAttentionPack.Dispose"); kW?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(kB, "NcnnRepro.MultiHeadAttentionPack.Dispose"); kB?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(vW, "NcnnRepro.MultiHeadAttentionPack.Dispose"); vW?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(vB, "NcnnRepro.MultiHeadAttentionPack.Dispose"); vB?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(oW, "NcnnRepro.MultiHeadAttentionPack.Dispose"); oW?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(oB, "NcnnRepro.MultiHeadAttentionPack.Dispose"); oB?.Dispose(); } catch { }
            }
        }

        public sealed class ScalePack : IDisposable
        {
            public int scaleDataSize;
            public bool biasTerm;
            public bool dynamic;
            public ComputeBuffer scale;
            public ComputeBuffer bias;
            public ComputeBuffer packedScale4;
            public ComputeBuffer packedBias4;
            public float[] scaleCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
                try { packedScale4?.Dispose(); } catch { }
                try { packedBias4?.Dispose(); } catch { }
            }
        }

        public sealed class PReluPack : IDisposable
        {
            public int numSlope;
            public ComputeBuffer slope;
            public float[] slopeCpu;

            public void Dispose()
            {
                try { slope?.Dispose(); } catch { }
            }
        }

        public sealed class QuantizePack : IDisposable
        {
            public int scaleDataSize;
            public ComputeBuffer scale;
            public float[] scaleCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
            }
        }

        public sealed class DequantizePack : IDisposable
        {
            public int scaleDataSize;
            public int biasDataSize;
            public ComputeBuffer scale;
            public ComputeBuffer bias;
            public float[] scaleCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
            }
        }

        public sealed class RequantizePack : IDisposable
        {
            public int scaleInDataSize;
            public int scaleOutDataSize;
            public int biasDataSize;
            public int activationType;
            public float activationParam0;
            public float activationParam1;
            public ComputeBuffer scaleIn;
            public ComputeBuffer scaleOut;
            public ComputeBuffer bias;
            public float[] scaleInCpu;
            public float[] scaleOutCpu;
            public float[] biasCpu;

            public void Dispose()
            {
                try { scaleIn?.Dispose(); } catch { }
                try { scaleOut?.Dispose(); } catch { }
                try { bias?.Dispose(); } catch { }
            }
        }

        public sealed class NormalizePack : IDisposable
        {
            public bool acrossSpatial;
            public bool acrossChannel;
            public bool channelShared;
            public float eps;
            public int epsMode;
            public int scaleDataSize;
            public ComputeBuffer scale;
            public float[] scaleCpu;

            public void Dispose()
            {
                try { scale?.Dispose(); } catch { }
            }
        }

        public sealed class LrnPack : IDisposable
        {
            public int regionType;
            public int localSize;
            public float alpha;
            public float beta;
            public float bias;

            public void Dispose()
            {
            }
        }

        public sealed class RmsNormPack : IDisposable
        {
            public int affineSize;
            public float eps;
            public bool affine;
            public ComputeBuffer gamma;
            public float[] gammaCpu;

            public void Dispose()
            {
                try { gamma?.Dispose(); } catch { }
            }
        }

        public sealed class RotaryEmbedPack : IDisposable
        {
            public bool interleaved;

            public void Dispose()
            {
            }
        }

        public sealed class SdpaPack : IDisposable
        {
            public bool attnMask;
            public float scale;
            public bool kvCache;
            public bool int8ScaleTerm;

            public void Dispose()
            {
            }
        }

        public sealed class UnfoldPack : IDisposable
        {
            public int kernelW;
            public int kernelH;
            public int dilationW;
            public int dilationH;
            public int strideW;
            public int strideH;
            public int padLeft;
            public int padRight;
            public int padTop;
            public int padBottom;
            public float padValue;

            public void Dispose()
            {
            }
        }

        public sealed class PriorBoxPack : IDisposable
        {
            public float[] minSizes;
            public float[] maxSizes;
            public float[] aspectRatios;
            public float[] variances;
            public bool flip;
            public bool clip;
            public int imageWidth;
            public int imageHeight;
            public float stepWidth;
            public float stepHeight;
            public float offset;
            public bool stepMmdetection;
            public bool centerMmdetection;
            public ComputeBuffer minSizeBuffer;
            public ComputeBuffer maxSizeBuffer;
            public ComputeBuffer aspectRatioBuffer;
            public ComputeBuffer varianceBuffer;

            public void Dispose()
            {
                try { NcnnGpuResourceTracker.ReleaseBuffer(minSizeBuffer, "NcnnRepro.PriorBoxPack.Dispose"); minSizeBuffer?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(maxSizeBuffer, "NcnnRepro.PriorBoxPack.Dispose"); maxSizeBuffer?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(aspectRatioBuffer, "NcnnRepro.PriorBoxPack.Dispose"); aspectRatioBuffer?.Dispose(); } catch { }
                try { NcnnGpuResourceTracker.ReleaseBuffer(varianceBuffer, "NcnnRepro.PriorBoxPack.Dispose"); varianceBuffer?.Dispose(); } catch { }
            }
        }

        public sealed class InferResult : IDisposable
        {
            private readonly Dictionary<string, TensorRef> _textureBlobs;
            private readonly Dictionary<string, BufferShape> _textureShapes;
            private readonly Dictionary<string, ComputeBuffer> _bufferBlobs;
            private readonly Dictionary<string, BufferRef> _bufferRefs;
            private readonly Dictionary<string, NcnnTensorBuffer> _bufferViews;
            private readonly List<IDisposable> _tempOwned;
            private readonly NcnnRepro _owner;
            private readonly bool _disallowTextureToBufferFallback;
            private readonly HashSet<RenderTexture> _visitedTextures = new HashSet<RenderTexture>();
            private readonly HashSet<ComputeBuffer> _visitedBuffers = new HashSet<ComputeBuffer>();

            internal InferResult(
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, BufferRef> bufferRefs,
                Dictionary<string, NcnnTensorBuffer> bufferViews,
                List<IDisposable> tempOwned,
                NcnnRepro owner,
                bool disallowTextureToBufferFallback)
            {
                _textureBlobs = textureBlobs;
                _textureShapes = textureShapes;
                _bufferBlobs = bufferBlobs;
                _bufferRefs = bufferRefs;
                _bufferViews = bufferViews;
                _tempOwned = tempOwned;
                _owner = owner;
                _disallowTextureToBufferFallback = disallowTextureToBufferFallback;
            }

            private ComputeBuffer GetOrMaterializeBuffer(string name)
            {
                if (!_owner.IsDebugOracleExecution)
                {
                    throw _owner.CreateDisallowedBufferPathException(
                        "production texture-only contract rejects InferResult buffer readback",
                        name,
                        "rejected_fallback=InferResult.GetBuffer/GetBufferView");
                }

                if (_bufferBlobs.TryGetValue(name, out var existing) && existing != null)
                    return existing;

                if (_disallowTextureToBufferFallback
                    && _textureBlobs.TryGetValue(name, out var tr)
                    && tr != null
                    && tr.texture != null)
                {
                    throw _owner.CreateDisallowedBufferPathException(
                        "pack4-only guard: texture-to-buffer materialization disallowed",
                        name);
                }

                var materialized = _owner.GetOrConvertToBuffer(name, _textureBlobs, _bufferBlobs, _textureShapes, _bufferViews, _tempOwned);
                if (materialized == null)
                    throw new InvalidOperationException("buffer blob not found: " + name);
                return materialized;
            }

            private void DetachTempOwnedBuffer(ComputeBuffer buffer)
            {
                if (buffer == null || _tempOwned == null)
                    return;

                for (var i = _tempOwned.Count - 1; i >= 0; i--)
                {
                    if (ReferenceEquals(_tempOwned[i], buffer))
                    {
                        _tempOwned.RemoveAt(i);
                        return;
                    }
                }
            }

            public RenderTexture GetTexture(string name)
            {
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    var existingLogicalShape = _textureShapes.TryGetValue(name, out var shape)
                        ? shape
                        : GetTextureShape(_textureShapes, tr, name);
                    EnsureRepoVkTensor(tr, existingLogicalShape, GetTextureStorageShape(tr, existingLogicalShape));
                    return tr.texture;
                }

                var materialized = _owner.MaterializeTextureFromBuffer(name, _bufferBlobs, _bufferViews);
                if (materialized == null)
                    throw new InvalidOperationException("blob not found: " + name);

                if (_bufferViews.TryGetValue(name, out var view) && view != null)
                    _textureShapes[name] = new BufferShape(view.dims, view.w, view.h, view.d, view.c);

                var logicalShape = _textureShapes.TryGetValue(name, out var existingShape)
                    ? existingShape
                    : new BufferShape(3, materialized.width, materialized.height, 1, Mathf.Max(1, materialized.volumeDepth > 0 ? materialized.volumeDepth : 1) * 4);
                _textureBlobs[name] = CreateTextureRef(materialized, logicalShape, logicalShape, owned: true, blobName: name);
                return materialized;
            }

            public RenderTexture ExtractTexture(string name)
            {
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    var rt = tr.texture;
                    DetachExtractedTextureOwnership(rt);
                    tr.ClearTexture();
                    return rt;
                }

                var materialized = _owner.MaterializeTextureFromBuffer(name, _bufferBlobs, _bufferViews);
                if (materialized == null)
                    throw new InvalidOperationException("blob not found: " + name);
                return materialized;
            }

            private void DetachExtractedTextureOwnership(RenderTexture texture)
            {
                if (texture == null)
                    return;

                foreach (var kv in _textureBlobs)
                {
                    var candidate = kv.Value;
                    if (candidate == null || !ReferenceEquals(candidate.texture, texture))
                        continue;

                    DetachTextureOwnership(candidate);
                    candidate.sharedTextureOwner = null;
                }
            }

            public bool TryGetExistingTexture(string name, out RenderTexture texture)
            {
                texture = null;
                if (_textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                {
                    texture = tr.texture;
                    return true;
                }

                return false;
            }

            public bool TryGetExistingTextureData(string name, out float[] data)
            {
                data = null;
                if (!TryGetExistingTextureContract(_textureBlobs, _textureShapes, name, out var textureRef, out var contract))
                    return false;

                data = ReadExistingTextureData(textureRef.texture, contract.LogicalShape, contract.StorageShape, contract.LayoutKind);
                return true;
            }

            public float[] GetExistingTextureData(string name)
            {
                if (TryGetExistingTextureData(name, out var data) && data != null)
                    return data;
                throw new InvalidOperationException("texture blob not found: " + name);
            }

            // Production output readback only. This never creates a ComputeBuffer.
            public float[] ReadTextureDataForOutput(string name)
            {
                return GetExistingTextureData(name);
            }

            private static float[] ReadExistingTextureData(
                RenderTexture texture,
                BufferShape logicalShape,
                BufferShape storageShape,
                RepoVkTensorLayoutKind layoutKind)
            {
                if (texture == null)
                    return null;

                var logicalCount = GetShapeElementCount(logicalShape);
                if (logicalCount <= 0)
                    return Array.Empty<float>();

                var previousActive = RenderTexture.active;
                Texture2D readback = null;
                try
                {
                    readback = new Texture2D(
                        Mathf.Max(1, texture.width),
                        Mathf.Max(1, texture.height),
                        TextureFormat.RGBAFloat,
                        false,
                        true);

                    if (layoutKind == RepoVkTensorLayoutKind.LinearMat || texture.dimension == TextureDimension.Tex2D)
                    {
                        ReadRenderTextureSlice(texture, readback, 0);
                        var raw = readback.GetRawTextureData<float>();
                        var linearPhysicalCount = Mathf.Max(1, texture.width) * Mathf.Max(1, texture.height);
                        if (logicalCount > linearPhysicalCount)
                            throw new InvalidOperationException("linear texture logical shape mismatch | physical=" + linearPhysicalCount + " logical=" + logicalCount);

                        var values = new float[logicalCount];
                        for (var i = 0; i < logicalCount; i++)
                            values[i] = raw[i * 4];
                        return values;
                    }

                    if (TryReadPack4LinearMatData(texture, readback, logicalShape, storageShape, out var pack4LinearValues))
                        return pack4LinearValues;

                    if (TryReadAttentionPackedLinearData(texture, readback, logicalShape, storageShape, out var attentionValues))
                        return attentionValues;

                    var physicalShape = storageShape.dims > 0 ? storageShape : logicalShape;
                    var physicalW = Mathf.Max(1, physicalShape.w > 0 ? physicalShape.w : texture.width);
                    var physicalH = Mathf.Max(1, physicalShape.dims >= 2 && physicalShape.h > 0 ? physicalShape.h : texture.height);
                    var physicalD = physicalShape.dims == 4 ? Mathf.Max(1, physicalShape.d) : 1;
                    var physicalC = physicalShape.dims >= 3
                        ? Mathf.Max(1, physicalShape.c)
                        : Mathf.Max(1, logicalShape.c);
                    var texturePhysicalCount = checked(physicalW * physicalH * physicalD * physicalC);
                    if (logicalCount > texturePhysicalCount)
                        throw new InvalidOperationException("texture logical shape mismatch | physical=" + texturePhysicalCount + " logical=" + logicalCount);

                    var physical = logicalCount == texturePhysicalCount ? null : new float[texturePhysicalCount];
                    var valuesOut = physical ?? new float[logicalCount];
                    var packCount = Mathf.Max(1, Mathf.CeilToInt(physicalC / 4f));
                    var availableSlices = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
                    var plane = physicalW * physicalH;

                    for (var z = 0; z < physicalD; z++)
                    {
                        for (var pack = 0; pack < packCount; pack++)
                        {
                            var slice = physicalShape.dims == 4 ? z * packCount + pack : pack;
                            if (slice >= availableSlices)
                                continue;

                            ReadRenderTextureSlice(texture, readback, slice);
                            var raw = readback.GetRawTextureData<float>();
                            for (var i = 0; i < plane; i++)
                            {
                                for (var lane = 0; lane < 4; lane++)
                                {
                                    var c = pack * 4 + lane;
                                    if (c >= physicalC)
                                        break;

                                    var dst = physicalShape.dims == 4
                                        ? ((c * physicalD + z) * plane) + i
                                        : (c * plane) + i;
                                    if ((uint)dst < (uint)valuesOut.Length)
                                        valuesOut[dst] = raw[i * 4 + lane];
                                }
                            }
                        }
                    }

                    if (physical == null)
                        return valuesOut;

                    var logicalValues = new float[logicalCount];
                    Array.Copy(physical, logicalValues, logicalCount);
                    return logicalValues;
                }
                finally
                {
                    RenderTexture.active = previousActive;
                    if (readback != null)
                        UnityEngine.Object.DestroyImmediate(readback);
                }
            }

            private static void ReadRenderTextureSlice(RenderTexture texture, Texture2D readback, int slice)
            {
                Graphics.SetRenderTarget(texture, 0, CubemapFace.Unknown, Mathf.Max(0, slice));
                readback.ReadPixels(new Rect(0, 0, readback.width, readback.height), 0, 0, false);
                readback.Apply(false, false);
            }

            private static bool TryReadAttentionPackedLinearData(
                RenderTexture texture,
                Texture2D readback,
                BufferShape logicalShape,
                BufferShape storageShape,
                out float[] values)
            {
                values = null;
                if (logicalShape.dims != 2
                    || storageShape.dims != 3
                    || storageShape.d != 1
                    || storageShape.w <= 0
                    || storageShape.h != logicalShape.h
                    || storageShape.c <= 4
                    || logicalShape.w != storageShape.w * storageShape.c
                    || texture.dimension != TextureDimension.Tex2DArray
                    || texture.width != storageShape.w
                    || texture.height != storageShape.h)
                {
                    return false;
                }

                var headDim = storageShape.w;
                var tokens = storageShape.h;
                var heads = storageShape.c;
                var packs = Mathf.CeilToInt(heads / 4f);
                if (texture.volumeDepth < packs)
                    return false;

                values = new float[logicalShape.w * logicalShape.h];
                for (var headPack = 0; headPack < packs; headPack++)
                {
                    ReadRenderTextureSlice(texture, readback, headPack);
                    var raw = readback.GetRawTextureData<float>();
                    for (var token = 0; token < tokens; token++)
                    {
                        for (var dim = 0; dim < headDim; dim++)
                        {
                            var srcBase = (token * headDim + dim) * 4;
                            for (var lane = 0; lane < 4; lane++)
                            {
                                var head = headPack * 4 + lane;
                                if (head >= heads)
                                    break;
                                values[token * logicalShape.w + head * headDim + dim] = raw[srcBase + lane];
                            }
                        }
                    }
                }

                return true;
            }

            // Debug/replay readback only. Pack4 linear matrices store four adjacent logical
            // columns in one RGBA texel rather than as four channel-major image planes.
            private static bool TryReadPack4LinearMatData(
                RenderTexture texture,
                Texture2D readback,
                BufferShape logicalShape,
                BufferShape storageShape,
                out float[] values)
            {
                values = null;
                if (logicalShape.dims != 2
                    || storageShape.dims != 3
                    || storageShape.w != Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f)
                    || storageShape.h != Mathf.Max(1, logicalShape.h)
                    || storageShape.d != 1
                    || storageShape.c != 4
                    || texture.dimension != TextureDimension.Tex2DArray
                    || texture.width != storageShape.w
                    || texture.height != storageShape.h
                    || texture.volumeDepth < 1)
                {
                    return false;
                }

                values = new float[logicalShape.w * logicalShape.h];
                ReadRenderTextureSlice(texture, readback, 0);
                var raw = readback.GetRawTextureData<float>();
                for (var row = 0; row < logicalShape.h; row++)
                {
                    for (var packX = 0; packX < storageShape.w; packX++)
                    {
                        var srcBase = (row * storageShape.w + packX) * 4;
                        var dstBase = row * logicalShape.w + packX * 4;
                        for (var lane = 0; lane < 4 && dstBase + lane < values.Length && packX * 4 + lane < logicalShape.w; lane++)
                            values[dstBase + lane] = raw[srcBase + lane];
                    }
                }

                return true;
            }

            private static int GetShapeElementCount(BufferShape shape)
            {
                return checked(
                    Mathf.Max(1, shape.w)
                    * Mathf.Max(1, shape.h)
                    * Mathf.Max(1, shape.d)
                    * Mathf.Max(1, shape.c));
            }

#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE
            public NcnnTensorBuffer GetBufferView(string name)
            {
                if (_bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                    return view;

                var buf = GetOrMaterializeBuffer(name);
                if (_bufferViews.TryGetValue(name, out view) && view != null && view.buffer != null)
                    return view;

                if (_textureShapes.TryGetValue(name, out var shape))
                {
                    view = new NcnnTensorBuffer(buf, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                    _bufferViews[name] = view;
                    return view;
                }

                view = new NcnnTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
                _bufferViews[name] = view;
                return view;
            }

            public bool TryGetExistingBufferView(string name, out NcnnTensorBuffer view)
            {
                if (_bufferViews.TryGetValue(name, out view) && view != null && view.buffer != null)
                    return true;

                view = null;
                return false;
            }

            public bool TryGetLogicalShape(string name, out int dims, out int w, out int h, out int d, out int c)
            {
                if (_textureBlobs.TryGetValue(name, out var textureRef)
                    && textureRef != null
                    && textureRef.texture != null
                    && _textureShapes.TryGetValue(name, out var textureShape))
                {
                    dims = textureShape.dims;
                    w = textureShape.w;
                    h = textureShape.h;
                    d = textureShape.d;
                    c = textureShape.c;
                    return true;
                }

                if (_bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                {
                    dims = view.dims;
                    w = view.w;
                    h = view.h;
                    d = view.d;
                    c = view.c;
                    return true;
                }

                if (_textureShapes.TryGetValue(name, out var shape))
                {
                    dims = shape.dims;
                    w = shape.w;
                    h = shape.h;
                    d = shape.d;
                    c = shape.c;
                    return true;
                }

                dims = 0;
                w = 0;
                h = 0;
                d = 0;
                c = 0;
                return false;
            }

            public ComputeBuffer ExtractBuffer(string name)
            {
                var buf = GetOrMaterializeBuffer(name);
                _bufferBlobs.Remove(name);
                _bufferRefs.Remove(name);
                _bufferViews.Remove(name);
                DetachTempOwnedBuffer(buf);
                return buf;
            }
#endif

            public void Dispose()
            {
                var visited = new HashSet<TensorRef>();
                foreach (var kv in _textureBlobs)
                {
                    var tr = kv.Value;
                    if (tr == null || tr.texture == null || !visited.Add(tr))
                        continue;
                    _owner.ReleaseTextureRef(tr);
                }

                foreach (var owned in _tempOwned)
                {
                    try
                    {
                        if (owned is ComputeBuffer tempBuffer)
                            _owner.ReturnTempBuffer(tempBuffer);
                        else
                            owned?.Dispose();
                    }
                    catch
                    {
                    }
                }

                foreach (var kv in _bufferRefs)
                {
                    var br = kv.Value;
                    if (br == null || !br.owned || br.buffer == null || !_visitedBuffers.Add(br.buffer))
                        continue;
                    try { _owner.ReturnTempBuffer(br.buffer); } catch { }
                }

                _bufferBlobs.Clear();
                _bufferRefs.Clear();
                _bufferViews.Clear();
                _tempOwned.Clear();
            }
        }

        public string SessionId { get; } = Guid.NewGuid().ToString("N");
        public InferenceBackend Backend => InferenceBackend.UnityGpu;
        public InferenceSessionState State => _isDisposed
            ? InferenceSessionState.Disposed
            : Model == null
                ? InferenceSessionState.Created
                : InferenceSessionState.Ready;

        public NcnnParamModel Model { get; private set; }
        public IReadOnlyList<NcnnBaseLayerRepro> LayerRepros { get; private set; }
        public ModelLoadProfile LastLoadProfile { get; private set; }
        public bool ForceBufferBinaryOpAll { get; set; }
        public bool ForceCpuGemmAll { get; set; }
        public bool ForceBufferGeluAll { get; set; }
        // Default false: some runners intentionally use the buffer GELU fallback as a GPU sync point via SetData.
        public bool EnableGpuGeluBufferPath { get; set; }
        public bool EnableDepthWiseTextureConvolution { get; set; } = true;
        public bool EnableConv1x1TextureConvolution { get; set; } = true;
        public bool EnableGeneralTextureConvolution { get; set; }
        public bool EnableGroupNormTexturePath { get; set; }
        public bool ForceBufferConvolution { get; set; }
        public bool UseTextureMaxPoolingInd { get; set; }
        public bool UseNcnnStyleGroupNorm { get; set; }

        internal readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, DeconvPack> _deconv = new Dictionary<string, DeconvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, InnerProductPack> _innerProduct = new Dictionary<string, InnerProductPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GemmPack> _gemm = new Dictionary<string, GemmPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, MemoryDataPack> _memoryData = new Dictionary<string, MemoryDataPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, EmbedPack> _embed = new Dictionary<string, EmbedPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, LayerNormPack> _layerNorm = new Dictionary<string, LayerNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GroupNormPack> _groupNorm = new Dictionary<string, GroupNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, BatchNormPack> _batchNorm = new Dictionary<string, BatchNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, MultiHeadAttentionPack> _multiHeadAttention = new Dictionary<string, MultiHeadAttentionPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, IDisposable> _extraPacks = new Dictionary<string, IDisposable>(StringComparer.Ordinal);
        internal Dictionary<string, int> _blobUseCount;
        private readonly float[] _gpuSyncScratch = new float[1];
        private int _runtimeProfileInferenceIndex;
        private int _tempBufferRentCount;
        private long _tempBufferRentBytes;
        private int _tempBufferLiveCount;
        private long _tempBufferLiveBytes;
        private int _tempBufferPeakLiveCount;
        private long _tempBufferPeakLiveBytes;
        private int _tempRtRentCount;
        private int _tempRtLiveCount;
        private int _tempRtPeakLiveCount;
        private bool _trackInferenceTempResources;
        private bool _isDisposed;

        private readonly NcnnOps _ops;

        public const bool EnableWinograd23 = false;
        public bool PreferTexturePathForFaceDetector { get; set; }
        public bool ForceBufferConvolutionAll { get; set; }
        public bool KeepRawConvWeightsForTexturePath { get; set; } = true;
        public bool EnableMhaParallelSoftmax { get; set; }
        public bool EnableMhaQkvFusion { get; set; }
        public bool EnableAttentionMatMulPack4Specializations { get; set; }
        public bool EnableVistaTailPack4Specializations { get; set; }
        public RenderTextureFormat TensorTextureFormat { get; set; } = RenderTextureFormat.ARGBHalf;
        public long TemporaryTextureBudgetBytes { get; set; }
        public NcnnInferenceExecutionMode ExecutionMode { get; set; } = NcnnInferenceExecutionMode.ProductionTextureOnly;
        // Production CommandBuffer execution is always planned strictly. DebugOracle is the
        // only explicit relaxation path and remains unavailable in non-debug builds.
        public bool StrictTextureInference => !IsExplicitDebugOracleExecution;
        public string StrictTextureTargetDtype { get; set; } = "FP16";
        public string StrictTextureTargetLayout { get; set; } = NcnnTexturePlanLayout.Packed4;
        public NcnnTextureExecutionPlan LastTextureExecutionPlan { get; private set; }
        public bool DisallowBufferAccess { get; set; }
        public bool DisallowBufferOutputs { get; set; }
        public bool DisallowBufferToTextureMaterialization { get; set; }
        public bool ForceBufferOutputsForDims4 { get; set; }
        public ISet<string> ForceBufferLayerTypes { get; set; }
        public ISet<string> ForceBufferLayerNames { get; set; }
        public ISet<string> DebugCompareTextureLayers { get; set; }
        public ISet<string> DebugCompareTextureConvLayers { get; set; }
        public ISet<string> DebugCompareMaxPoolingLayers { get; set; }
        public Action<string> DebugLog { get; set; }
        public bool DebugLogAllLayerOutputs { get; set; }
        public bool DebugLogAllLayerHeartbeats { get; set; }
        public bool DebugLogAllBufferMaterialize { get; set; }
        public bool DisallowInferenceTempComputeBuffers { get; set; }
        public bool DebugBreakOnFirstNonFiniteLayerOutput { get; set; }
        public float CodeFormerSftMulScale { get; set; } = 1f;
        public float CodeFormerSftAddScale { get; set; } = 1f;
        public bool CodeFormerBypassSftMul { get; set; }
        public string CodeFormerTargetSftMulLayer { get; set; }
        public string CodeFormerTargetSftAddLayer { get; set; }
        public string CodeFormerTargetSftResidualLayer { get; set; }
        public NcnnOps Ops => _ops;
        public bool gpuLayerProfileEnabled = false;
        public bool useExperimentalIteratePath = false;
        public bool LayerRuntimeProfileEnabled { get; set; }
        public bool LayerRuntimeProfileSyncGpu { get; set; }
        public string LayerRuntimeProfilePathKindOverride { get; set; }
        public LayerRuntimeProfile LastRuntimeProfile { get; private set; }
        public string TimingSplitSyncAfterTopName { get; set; }
        public Action<string, double> OnTimingSplitSyncPoint { get; set; }
        public event Action<string, string, int, int, int, int, double> OnConvComplete;
        private const int FallbackMaxTextureArraySlices = 2048;
        private const int FallbackMaxTextureSize = 16384;
        private string _currentExecutingLayerName;
        private string _currentExecutingLayerTypeName;
        private NcnnLayerBufferContext _currentBufferContext;

        public static bool IsDebugOracleBuild
        {
            get
            {
#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE
                return true;
#else
                return false;
#endif
            }
        }

        // Legacy buffer/debug controls may still be used by immediate diagnostic runners.
        // CommandBuffer planning is relaxed only by the explicit execution mode below.
        public bool IsDebugOracleExecution => IsDebugOracleBuild
            && (ExecutionMode == NcnnInferenceExecutionMode.DebugOracle || HasDebugOracleIntent());

        private bool IsExplicitDebugOracleExecution => IsDebugOracleBuild
            && ExecutionMode == NcnnInferenceExecutionMode.DebugOracle;

        private void EnsureCommandBufferTextureExecutionPlan(
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes)
        {
            if (Model == null)
                throw new InvalidOperationException("model not loaded");

            var inputs = new List<NcnnTexturePlanTensorDescriptor>();
            foreach (var kv in textureInputs ?? new Dictionary<string, ComputeTexture>(StringComparer.Ordinal))
            {
                var texture = kv.Value;
                if (texture == null)
                    throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");

                var depth = Mathf.Max(1, texture.depth);
                var fallbackChannels = string.Equals(kv.Key, "data", StringComparison.OrdinalIgnoreCase) ? 3 : depth * 4;
                var logicalShape = textureInputShapes != null && textureInputShapes.TryGetValue(kv.Key, out var suppliedShape)
                    ? suppliedShape
                    : new BufferShape(3, texture.width, texture.height, 1, ResolveInputLogicalChannels(kv.Key, fallbackChannels));
                inputs.Add(new NcnnTexturePlanTensorDescriptor
                {
                    blob = kv.Key,
                    logicalShape = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c },
                    storageShape = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c },
                    layout = StrictTextureTargetLayout,
                    dtype = ResolveTexturePlanDtype(texture.format),
                    aliasGroup = "input:" + kv.Key,
                    textureBacked = true
                });
            }

            CompleteTextureExecutionPlan(inputs, IsExplicitDebugOracleExecution);
        }

        private void CompleteTextureExecutionPlan(
            List<NcnnTexturePlanTensorDescriptor> inputs,
            bool debugOracleRelaxed)
        {
            LastTextureExecutionPlan = NcnnTextureExecutionPlanner.Analyze(Model, new NcnnTextureExecutionPlanRequest
            {
                modelName = LastLoadProfile?.modelMagic ?? string.Empty,
                targetBackend = NcnnOperatorCapabilityBackend.CommandBuffer,
                targetDtype = StrictTextureTargetDtype,
                targetLayout = StrictTextureTargetLayout,
                strict = StrictTextureInference,
                debugOracleRelaxed = debugOracleRelaxed,
                inputs = inputs.ToArray(),
                nodeVerifier = VerifyStrictCommandBufferPack4Node
            });
            NcnnTextureExecutionPlanner.ThrowIfDispatchRejected(LastTextureExecutionPlan);
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPack4Node(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!string.Equals(request?.targetBackend, NcnnOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                || (!string.Equals(request?.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(request?.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase))
                || !string.Equals(request?.targetLayout, NcnnTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase))
            {
                return RejectStrictCommandBufferPack4Node("The loaded runtime profile supports only FP16/FP32 Packed4 CommandBuffer branches.");
            }
            if (string.Equals(request.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase)
                && TensorTextureFormat != RenderTextureFormat.ARGBFloat)
                return RejectStrictCommandBufferPack4Node("FP32 Pack4 requires TensorTextureFormat=ARGBFloat for texture-native intermediate/output storage.");

            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            switch (operatorName)
            {
                case "Convolution":
                    return VerifyStrictCommandBufferConvolution(layer, inputs, request);
                case "ConvolutionDepthWise":
                    return VerifyStrictCommandBufferDepthWiseConvolution(layer, inputs, request);
                case "Convolution3D":
                    return VerifyStrictCommandBufferConvolution3D(layer, inputs, request);
                case "Deconvolution":
                    return VerifyStrictCommandBufferDeconvolution(layer, inputs, request, depthWiseLayer: false);
                case "DeconvolutionDepthWise":
                    return VerifyStrictCommandBufferDeconvolution(layer, inputs, request, depthWiseLayer: true);
                case "Deconvolution3D":
                    return VerifyStrictCommandBufferDeconvolution3D(layer, inputs, request);
                case "Eltwise":
                    return VerifyStrictCommandBufferEltwise(layer, inputs, request);
                case "Concat":
                    return VerifyStrictCommandBufferConcat(layer, inputs, request);
                case "BinaryOp":
                    return VerifyStrictCommandBufferBinaryOp(layer, inputs, request);
                case "Interp":
                    return VerifyStrictCommandBufferInterp3DOr2D(layer, inputs, request);
                case "PixelShuffle":
                    return VerifyStrictCommandBufferPixelShuffle(layer, inputs, request);
                case "UnaryOp":
                    return VerifyStrictCommandBufferUnaryOp(layer, inputs, request);
                case "GELU":
                    return VerifyStrictCommandBufferGelu(layer, inputs, request);
                case "MemoryData":
                    return VerifyStrictCommandBufferMemoryData(layer, request);
                case "InnerProduct":
                    return VerifyStrictCommandBufferInnerProduct(layer, inputs, request);
                case "Pooling":
                    return VerifyStrictCommandBufferPooling(layer, inputs, request);
                case "Pooling3D":
                    return VerifyStrictCommandBufferPooling3D(layer, inputs, request);
                case "Reduction":
                    return VerifyStrictCommandBufferReduction(layer, inputs, request);
                case "BatchNorm":
                    return VerifyStrictCommandBufferBatchNorm(layer, inputs, request);
                case "Reshape":
                    return VerifyStrictCommandBufferReshape(layer, inputs, request);
                case "Permute":
                    return VerifyStrictCommandBufferPermute(layer, inputs, request);
                case "Gemm":
                    return VerifyStrictCommandBufferGemm(layer, inputs, request);
                case "Slice":
                    return VerifyStrictCommandBufferSlice(layer, inputs, request);
                case "MatMul":
                    return VerifyStrictCommandBufferMatMul(layer, inputs, request);
                case "Softmax":
                    return VerifyStrictCommandBufferSoftmax(layer, inputs, request);
                default:
                    return RejectStrictCommandBufferPack4Node("No loaded-runtime Pack4 proof exists for operator " + (operatorName ?? string.Empty) + ".");
            }
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolution3D(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictCdhwPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_conv.TryGetValue(layer.name, out var conv) || conv == null)
                return RejectStrictCommandBufferPack4Node("Packed 3D convolution weights were not loaded for this layer.");
            if (input.c != conv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded 3D convolution profile.");
            if (!TryValidateCommandBuffer3dConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer 3D convolution profile rejected: " + profileReason);

            var output = new BufferShape(
                4,
                ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight),
                ComputeConvOut(input.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom),
                ComputeConvOut(input.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind),
                conv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                return RejectStrictCommandBufferPack4Node("The 3D convolution profile resolves a non-positive CDHW output extent.");
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:convolution3d-cdhw");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolution3D(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictCdhwPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                return RejectStrictCommandBufferPack4Node("Packed 3D deconvolution weights were not loaded for this layer.");
            if (input.c != deconv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded 3D deconvolution profile.");
            if (!TryValidateCommandBuffer3dDeconvProfile(deconv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer 3D deconvolution profile rejected: " + profileReason);

            var output = new BufferShape(
                4,
                ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight),
                ComputeDeconvOut(input.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom),
                ComputeDeconvOut(input.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind),
                deconv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                return RejectStrictCommandBufferPack4Node("The 3D deconvolution profile resolves a non-positive CDHW output extent.");
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:deconvolution3d-cdhw");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPooling3D(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictCdhwPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!TryResolveStrictCommandBufferPooling3dOutput(layer, input, out var output, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer 3D pooling profile rejected: " + profileReason);
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:pooling3d-cdhw");
        }

        private static bool TryValidateCommandBuffer3dConvProfile(ConvPack conv, out string reason)
        {
            reason = null;
            if (conv == null || conv.packedWeight4 == null || conv.packedBias4 == null)
                reason = "immutable Pack4 O4I4K3 weights/bias are unavailable";
            else if (conv.group != 1)
                reason = "only group=1 is implemented by the CDHW Pack4 kernel";
            else if (conv.inC <= 0 || conv.outC <= 0)
                reason = "input and output channels must be positive";
            else if (conv.kernelW <= 0 || conv.kernelH <= 0 || conv.kernelD <= 0
                || conv.strideW <= 0 || conv.strideH <= 0 || conv.strideD <= 0
                || conv.dilationW <= 0 || conv.dilationH <= 0 || conv.dilationD <= 0)
                reason = "kernel, stride, and dilation must be positive on W/H/D";
            else if (conv.padLeft < 0 || conv.padRight < 0 || conv.padTop < 0 || conv.padBottom < 0 || conv.padFront < 0 || conv.padBehind < 0)
                reason = "negative or auto padding is unsupported";
            else if (!IsCommandBufferConvActivationSupported(conv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (conv.weightSize != conv.outC * conv.inC * conv.kernelW * conv.kernelH * conv.kernelD)
                reason = "weight_data_size does not match the group=1 OIDHW profile";
            return reason == null;
        }

        private static bool TryValidateCommandBuffer3dDeconvProfile(DeconvPack deconv, out string reason)
        {
            reason = null;
            if (deconv == null || deconv.packedWeight4 == null || deconv.packedBias4 == null)
                reason = "immutable Pack4 O4I4K3 weights/bias are unavailable";
            else if (deconv.group != 1)
                reason = "only group=1 is implemented by the CDHW Pack4 kernel";
            else if (deconv.inC <= 0 || deconv.outC <= 0)
                reason = "input and output channels must be positive";
            else if (deconv.kernelW <= 0 || deconv.kernelH <= 0 || deconv.kernelD <= 0
                || deconv.strideW <= 0 || deconv.strideH <= 0 || deconv.strideD <= 0
                || deconv.dilationW <= 0 || deconv.dilationH <= 0 || deconv.dilationD <= 0)
                reason = "kernel, stride, and dilation must be positive on W/H/D";
            else if (deconv.padLeft < 0 || deconv.padRight < 0 || deconv.padTop < 0 || deconv.padBottom < 0 || deconv.padFront < 0 || deconv.padBehind < 0)
                reason = "negative or auto padding is unsupported";
            else if (deconv.outputPadRight != 0 || deconv.outputPadBottom != 0 || deconv.outputPadBehind != 0)
                reason = "non-zero output padding has no verified CDHW Pack4 kernel profile";
            else if (!IsCommandBufferConvActivationSupported(deconv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (deconv.weightSize != deconv.outC * deconv.inC * deconv.kernelW * deconv.kernelH * deconv.kernelD)
                reason = "weight_data_size does not match the group=1 OIDHW profile";
            return reason == null;
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolution(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer convolution requires a 2D Pack4 activation.");
            if (!_conv.TryGetValue(layer.name, out var conv) || conv == null)
                return RejectStrictCommandBufferPack4Node("Packed convolution weights were not loaded for this layer.");
            if (input.c != conv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded convolution profile.");
            if (!TryValidateCommandBuffer2dConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer convolution profile rejected: " + profileReason);

            var output = new BufferShape(
                3,
                ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight),
                ComputeConvOut(input.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom),
                1,
                conv.outC);
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:convolution");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDepthWiseConvolution(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer depthwise convolution requires a 2D Pack4 activation.");
            if (!_conv.TryGetValue(layer.name, out var conv) || conv == null)
                return RejectStrictCommandBufferPack4Node("Packed depthwise convolution weights were not loaded for this layer.");
            if (input.c != conv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded depthwise convolution profile.");
            if (!conv.isDepthWise)
                return RejectStrictCommandBufferPack4Node("The loaded profile is not ConvolutionDepthWise.");
            if (!TryValidateCommandBuffer2dConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer depthwise convolution profile rejected: " + profileReason);

            var output = new BufferShape(
                3,
                ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight),
                ComputeConvOut(input.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom),
                1,
                conv.outC);
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:convolution-depthwise");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolution(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request,
            bool depthWiseLayer)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer deconvolution requires a 2D Pack4 activation.");
            if (!_deconv.TryGetValue(layer.name, out var deconv) || deconv == null)
                return RejectStrictCommandBufferPack4Node("Immutable deconvolution weights were not loaded for this layer.");
            if (input.c != deconv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded deconvolution profile.");
            if (!TryValidateCommandBuffer2dDeconvProfile(deconv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer deconvolution profile rejected: " + profileReason);

            var output = new BufferShape(
                3,
                ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight),
                ComputeDeconvOut(input.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom),
                1,
                deconv.outC);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                request,
                depthWiseLayer ? "command-buffer-pack4:deconvolution-depthwise" : "command-buffer-pack4:deconvolution");
        }

        private static bool TryValidateCommandBuffer2dConvProfile(ConvPack conv, out string reason)
        {
            reason = null;
            if (conv == null || conv.rawWeight == null || conv.rawBias == null)
                reason = "immutable scalar weights/bias are unavailable";
            else if (conv.inC <= 0 || conv.outC <= 0 || conv.group <= 0 || conv.inC % conv.group != 0 || conv.outC % conv.group != 0)
                reason = "group must divide positive input/output channels";
            else if (conv.kernelW <= 0 || conv.kernelH <= 0 || conv.strideW <= 0 || conv.strideH <= 0 || conv.dilationW <= 0 || conv.dilationH <= 0)
                reason = "kernel/stride/dilation must be positive";
            else if (conv.padLeft < 0 || conv.padRight < 0 || conv.padTop < 0 || conv.padBottom < 0)
                reason = "negative/auto padding is unsupported";
            else if (!IsCommandBufferConvActivationSupported(conv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (conv.weightSize != conv.outC * (conv.inC / conv.group) * conv.kernelW * conv.kernelH)
                reason = "weight_data_size does not match grouped OIHW";
            return reason == null;
        }

        private static bool TryValidateCommandBuffer2dDeconvProfile(DeconvPack deconv, out string reason)
        {
            reason = null;
            if (deconv == null || deconv.rawWeight == null || deconv.rawBias == null)
                reason = "immutable scalar weights/bias are unavailable";
            else if (deconv.inC <= 0 || deconv.outC <= 0 || deconv.group <= 0 || deconv.inC % deconv.group != 0 || deconv.outC % deconv.group != 0)
                reason = "group must divide positive input/output channels";
            else if (deconv.kernelW <= 0 || deconv.kernelH <= 0 || deconv.strideW <= 0 || deconv.strideH <= 0 || deconv.dilationW <= 0 || deconv.dilationH <= 0)
                reason = "kernel/stride/dilation must be positive";
            else if (deconv.padLeft < 0 || deconv.padRight < 0 || deconv.padTop < 0 || deconv.padBottom < 0 || deconv.outputPadRight < 0 || deconv.outputPadBottom < 0)
                reason = "negative/auto padding or output padding is unsupported";
            else if (!IsCommandBufferConvActivationSupported(deconv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (deconv.weightSize != deconv.outC * (deconv.inC / deconv.group) * deconv.kernelW * deconv.kernelH)
                reason = "weight_data_size does not match grouped OIHW";
            return reason == null;
        }

        private static bool IsCommandBufferConvActivationSupported(int activationType)
        {
            return activationType == 0 || activationType == 1 || activationType == 2 || activationType == 4;
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPooling(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Pooling profile requires a 2D Pack4 activation.");
            if (layer.GetInt(4, 0) != 1)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Pooling profile supports global pooling only.");
            if (layer.GetInt(7, 0) != 0)
                return RejectStrictCommandBufferPack4Node("Adaptive pooling does not have a verified CommandBuffer Pack4 path.");
            if (layer.GetInt(0, 0) != 1)
                return RejectStrictCommandBufferPack4Node("The verified global Pooling profile supports average pooling only.");

            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, 1, 1, 1, input.c),
                request,
                "command-buffer-pack4:pooling-global");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferEltwise(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length == 0 || shapes.Any(shape => !StrictPlanShapesEqual(shape, shapes[0])))
                return RejectStrictCommandBufferPack4Node("Eltwise CommandBuffer Pack4 requires equal descriptor-backed input shapes.");
            var operation = layer.GetInt(0, 1);
            if (operation != 0 && operation != 1 && operation != 2)
                return RejectStrictCommandBufferPack4Node("The Eltwise operation is outside the verified CommandBuffer Pack4 subset.");
            return AcceptStrictCommandBufferPack4Node(layer, shapes[0], request, "command-buffer-pack4:eltwise");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConcat(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length < 2 || (shapes[0].dims != 3 && shapes[0].dims != 4))
                return RejectStrictCommandBufferPack4Node("Concat CommandBuffer Pack4 requires two or more 3D/4D inputs.");

            var axis = layer.GetInt(0, 0);
            if (axis < 0)
                axis += shapes[0].dims;
            if (axis < 0 || axis >= shapes[0].dims)
                return RejectStrictCommandBufferPack4Node("Concat axis is outside the input rank.");
            var tensorAxis = MapNcnnAxisToTensorAxis(shapes[0].dims, axis);
            var channelAxis = shapes[0].dims == 4 ? 3 : 2;
            if (tensorAxis != channelAxis)
                return RejectStrictCommandBufferPack4Node("Only channel-axis Concat has a verified CommandBuffer Pack4 path.");

            var outputChannels = 0;
            for (var index = 0; index < shapes.Length; index++)
            {
                var shape = shapes[index];
                if (shape.dims != shapes[0].dims
                    || shape.w != shapes[0].w
                    || shape.h != shapes[0].h
                    || shape.d != shapes[0].d)
                {
                    return RejectStrictCommandBufferPack4Node("Concat inputs do not preserve the required Pack4 spatial descriptor.");
                }
                outputChannels += shape.c;
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(shapes[0].dims, shapes[0].w, shapes[0].h, shapes[0].d, outputChannels),
                request,
                "command-buffer-pack4:concat-channel");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferBinaryOp(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            var operation = layer.GetInt(0, 0);
            if (operation < 0 || operation > 9)
                return RejectStrictCommandBufferPack4Node("The BinaryOp code is outside the verified CommandBuffer Pack4 kernel range.");

            if (layer.GetInt(1, 0) != 0)
            {
                if (!TryGetSingleStrictPlanShape(inputs, out var scalarInput, out var scalarReason))
                    return RejectStrictCommandBufferPack4Node(scalarReason);
                if (scalarInput.dims < 3 || scalarInput.dims > 4)
                    return RejectStrictCommandBufferPack4Node("The verified scalar BinaryOp profile requires a 3D or 4D Pack4 texture.");
                var storage = inputs[0].storageShape;
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    scalarInput,
                    new BufferShape(storage[0], storage[1], storage[2], storage[3], storage[4]),
                    request,
                    "command-buffer-pack4:binary-scalar");
            }

            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length != 2)
                return RejectStrictCommandBufferPack4Node("BinaryOp CommandBuffer Pack4 requires exactly two descriptor-backed inputs.");

            if ((shapes[0].dims == 3 || shapes[0].dims == 4)
                && StrictPlanShapesEqual(shapes[0], shapes[1]))
            {
                return AcceptStrictCommandBufferPack4Node(layer, shapes[0], request, "command-buffer-pack4:binary-exact");
            }

            if (TryResolveStrictCommandBufferChannelVector(inputs[0], shapes[0], inputs[1], shapes[1], out var channelVectorOutput))
                return AcceptStrictCommandBufferPack4Node(layer, channelVectorOutput, request, "command-buffer-pack4:binary-channel-vector");

            if (TryResolveStrictCommandBufferSpatialBroadcast(shapes[0], shapes[1], out var spatialBroadcastOutput))
                return AcceptStrictCommandBufferPack4Node(layer, spatialBroadcastOutput, request, "command-buffer-pack4:binary-spatial-broadcast");

            return RejectStrictCommandBufferPack4Node("BinaryOp does not match an exact or channel-vector CommandBuffer Pack4 descriptor profile.");
        }

        private static bool TryResolveStrictCommandBufferSpatialBroadcast(
            BufferShape first,
            BufferShape second,
            out BufferShape output)
        {
            output = default;
            if ((first.dims != 3 && first.dims != 4)
                || first.dims != second.dims
                || first.c != second.c)
            {
                return false;
            }

            var firstIsScalarSpatial = first.w == 1 && first.h == 1 && first.d == 1;
            var secondIsScalarSpatial = second.w == 1 && second.h == 1 && second.d == 1;
            if (firstIsScalarSpatial && !secondIsScalarSpatial)
            {
                output = second;
                return true;
            }

            if (secondIsScalarSpatial && !firstIsScalarSpatial)
            {
                output = first;
                return true;
            }

            return false;
        }

        private static bool TryResolveStrictCommandBufferChannelVector(
            NcnnTexturePlanTensorDescriptor firstDescriptor,
            BufferShape first,
            NcnnTexturePlanTensorDescriptor secondDescriptor,
            BufferShape second,
            out BufferShape output)
        {
            output = default;
            if ((first.dims == 3 || first.dims == 4)
                && IsStrictCommandBufferChannelVector(secondDescriptor, second, first.c))
            {
                output = first;
                return true;
            }

            if ((second.dims == 3 || second.dims == 4)
                && IsStrictCommandBufferChannelVector(firstDescriptor, first, second.c))
            {
                output = second;
                return true;
            }

            return false;
        }

        private static bool IsStrictCommandBufferChannelVector(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape,
            int expectedChannels)
        {
            var storage = descriptor?.storageShape;
            return descriptor != null
                && (logicalShape.dims == 1 || (logicalShape.dims == 2 && logicalShape.h == 1))
                && logicalShape.w == expectedChannels
                && storage != null
                && storage.Length == 5
                && storage[0] == 3
                && storage[1] == expectedChannels
                && storage[2] == 1
                && storage[3] == 1
                && storage[4] == 1;
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp3DOr2D(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            return input.dims == 4
                ? VerifyStrictCommandBufferInterp3D(layer, inputs, input, request)
                : VerifyStrictCommandBufferInterp(layer, inputs, request);
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp3D(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            BufferShape input,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!HasStrictCdhwPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("CommandBuffer Interp requires a TensorDescriptor-backed CDHW Pack4 Texture2DArray input.");
            if (!string.IsNullOrWhiteSpace(layer.GetString(9, null)) || layer.GetInt(5, 0) != 0)
                return RejectStrictCommandBufferPack4Node("Dynamic Interp size expressions are not proven by the static CDHW CommandBuffer Pack4 profile.");

            var resizeType = layer.GetInt(0, 0);
            if (resizeType != 1 && resizeType != 2)
                return RejectStrictCommandBufferPack4Node("The CDHW Interp profile supports only nearest (1) and trilinear (2) modes.");

            var outputWidth = layer.GetInt(4, 0);
            var outputHeight = layer.GetInt(3, 0);
            var outputDepth = layer.GetInt(8, 0);
            var scaleX = layer.GetFloat(2, 1f);
            var scaleY = layer.GetFloat(1, 1f);
            var scaleZ = layer.GetFloat(7, 0f);
            if (scaleZ <= 0f)
                scaleZ = scaleY;
            if (outputWidth == 0 && scaleX <= 0f || outputHeight == 0 && scaleY <= 0f || outputDepth == 0 && scaleZ <= 0f)
                return RejectStrictCommandBufferPack4Node("The CDHW Interp profile requires positive static sizes or scale factors on W/H/D.");
            if (outputWidth == 0)
                outputWidth = Mathf.Max(1, (int)(input.w * scaleX));
            if (outputHeight == 0)
                outputHeight = Mathf.Max(1, (int)(input.h * scaleY));
            if (outputDepth == 0)
                outputDepth = Mathf.Max(1, (int)(input.d * scaleZ));
            if (outputWidth <= 0 || outputHeight <= 0 || outputDepth <= 0)
                return RejectStrictCommandBufferPack4Node("Interp resolved a non-positive CDHW output extent.");

            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(4, outputWidth, outputHeight, outputDepth, input.c),
                request,
                "command-buffer-pack4:interp-cdhw");
        }

        private static bool TryResolveStrictCommandBufferPooling3dOutput(
            NcnnParamModel.Layer layer,
            BufferShape input,
            out BufferShape output,
            out string reason)
        {
            output = default;
            reason = null;
            var poolType = layer.GetInt(0, 0);
            if (poolType != 0 && poolType != 1)
            {
                reason = "pooling supports only max (0) and average (1)";
                return false;
            }

            var global = layer.GetInt(4, 0) != 0;
            var adaptive = layer.GetInt(7, 0) != 0;
            if (global && adaptive)
            {
                reason = "global and adaptive pooling cannot be combined";
                return false;
            }
            if (global)
            {
                output = new BufferShape(4, 1, 1, 1, input.c);
                return true;
            }

            var kernelW = Mathf.Max(1, layer.GetInt(1, 0));
            var kernelH = Mathf.Max(1, layer.GetInt(11, kernelW));
            var kernelD = Mathf.Max(1, layer.GetInt(21, kernelW));
            var strideW = Mathf.Max(1, layer.GetInt(2, 1));
            var strideH = Mathf.Max(1, layer.GetInt(12, strideW));
            var strideD = Mathf.Max(1, layer.GetInt(22, strideW));
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var padFront = layer.GetInt(23, padLeft);
            var padBehind = layer.GetInt(16, padFront);
            var padMode = layer.GetInt(5, 0);
            if (padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0 || padFront < 0 || padBehind < 0)
            {
                reason = "negative or auto padding is unsupported";
                return false;
            }
            if (padMode < 0 || padMode > 3)
            {
                reason = "pad mode is outside the explicit/full/SAME_UPPER/SAME_LOWER CDHW subset";
                return false;
            }

            if (adaptive)
            {
                var outW = layer.GetInt(8, 0);
                var outH = layer.GetInt(18, outW);
                var outD = layer.GetInt(28, outW);
                output = new BufferShape(
                    4,
                    outW == -233 || outW <= 0 ? input.w : outW,
                    outH == -233 || outH <= 0 ? input.h : outH,
                    outD == -233 || outD <= 0 ? input.d : outD,
                    input.c);
                return true;
            }

            if (input.w + padLeft + padRight < kernelW || input.h + padTop + padBottom < kernelH || input.d + padFront + padBehind < kernelD)
            {
                reason = "kernel exceeds the padded CDHW input extent";
                return false;
            }

            var totalPadLeft = padLeft;
            var totalPadRight = padRight;
            var totalPadTop = padTop;
            var totalPadBottom = padBottom;
            var totalPadFront = padFront;
            var totalPadBehind = padBehind;
            if (padMode == 0)
            {
                var wtail = (input.w + padLeft + padRight - kernelW) % strideW;
                var htail = (input.h + padTop + padBottom - kernelH) % strideH;
                var dtail = (input.d + padFront + padBehind - kernelD) % strideD;
                if (wtail != 0) totalPadRight += strideW - wtail;
                if (htail != 0) totalPadBottom += strideH - htail;
                if (dtail != 0) totalPadBehind += strideD - dtail;
            }
            else if (padMode == 2 || padMode == 3)
            {
                var wpad = kernelW + (input.w - 1) / strideW * strideW - input.w;
                var hpad = kernelH + (input.h - 1) / strideH * strideH - input.h;
                var dpad = kernelD + (input.d - 1) / strideD * strideD - input.d;
                if (padMode == 2)
                {
                    totalPadLeft = wpad / 2; totalPadRight = wpad - totalPadLeft;
                    totalPadTop = hpad / 2; totalPadBottom = hpad - totalPadTop;
                    totalPadFront = dpad / 2; totalPadBehind = dpad - totalPadFront;
                }
                else
                {
                    totalPadLeft = wpad - wpad / 2; totalPadRight = wpad / 2;
                    totalPadTop = hpad - hpad / 2; totalPadBottom = hpad / 2;
                    totalPadFront = dpad - dpad / 2; totalPadBehind = dpad / 2;
                }
            }

            output = new BufferShape(
                4,
                Mathf.Max(1, (input.w + totalPadLeft + totalPadRight - kernelW) / strideW + 1),
                Mathf.Max(1, (input.h + totalPadTop + totalPadBottom - kernelH) / strideH + 1),
                Mathf.Max(1, (input.d + totalPadFront + totalPadBehind - kernelD) / strideD + 1),
                input.c);
            return true;
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer Interp requires a 2D Pack4 activation.");
            if (!string.IsNullOrWhiteSpace(layer.GetString(9, null)))
                return RejectStrictCommandBufferPack4Node("Dynamic Interp size expressions are not proven by the static CommandBuffer Pack4 profile.");

            var resizeType = layer.GetInt(0, 0);
            if (resizeType != 0 && resizeType != 1 && resizeType != 3)
                return RejectStrictCommandBufferPack4Node("The Interp mode is outside the verified CommandBuffer Pack4 subset.");
            var scaleX = layer.GetFloat(2, 1f);
            var scaleY = layer.GetFloat(1, 1f);
            var outputWidth = layer.GetInt(4, 0);
            var outputHeight = layer.GetInt(3, 0);
            if (outputWidth == 0)
                outputWidth = Mathf.Max(1, (int)(input.w * Mathf.Max(0f, scaleX)));
            if (outputHeight == 0)
                outputHeight = Mathf.Max(1, (int)(input.h * Mathf.Max(0f, scaleY)));
            if (outputWidth <= 0 || outputHeight <= 0)
                return RejectStrictCommandBufferPack4Node("Interp resolved a non-positive output extent.");

            if (outputWidth == input.w && outputHeight == input.h)
                return AcceptStrictCommandBufferPack4NoopAlias(layer, inputs[0], request);

            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, outputWidth, outputHeight, 1, input.c),
                request,
                "command-buffer-pack4:interp");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPixelShuffle(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            var factor = layer.GetInt(0, 1);
            var divisor = factor * factor;
            if (input.dims != 3 || input.d != 1 || factor <= 0 || input.c % divisor != 0)
                return RejectStrictCommandBufferPack4Node("PixelShuffle does not meet the CommandBuffer Pack4 shape contract.");
            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, input.w * factor, input.h * factor, 1, input.c / divisor),
                request,
                "command-buffer-pack4:pixel-shuffle");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferUnaryOp(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            var operation = layer.GetInt(0, 0);
            if ((operation < 0 || operation > 11) && operation != 15 && operation != 16)
                return RejectStrictCommandBufferPack4Node("The UnaryOp code is outside the verified CommandBuffer Pack4 kernel range.");
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:unary");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGelu(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 1 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("GELU has no verified CommandBuffer Pack4 path for this rank.");
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:gelu");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMemoryData(
            NcnnParamModel.Layer layer,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!_memoryData.TryGetValue(layer.name, out var memory) || memory == null)
                return RejectStrictCommandBufferPack4Node("MemoryData was not loaded for this node.");
            if (memory.channelVectorTexture == null
                || memory.dims != 3
                || memory.w != 1
                || memory.h != 1
                || memory.d != 1
                || memory.c <= 0)
            {
                return RejectStrictCommandBufferPack4Node("Only loaded 1x1 channel-vector MemoryData has a texture-native CommandBuffer path.");
            }

            var logicalShape = new BufferShape(1, memory.c, 1, 1, 1);
            var storageShape = new BufferShape(3, memory.c, 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                logicalShape,
                storageShape,
                request,
                "command-buffer-pack4:memory-data-channel-vector");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInnerProduct(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 1 && input.dims != 2)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer InnerProduct profile requires a vector or matrix input.");
            if (!_innerProduct.TryGetValue(layer.name, out var innerProduct)
                || innerProduct == null
                || innerProduct.w == null
                || innerProduct.b == null
                || innerProduct.inFeatures <= 0
                || innerProduct.outFeatures <= 0)
            {
                return RejectStrictCommandBufferPack4Node("The loaded InnerProduct weights or bias are unavailable.");
            }
            if (input.w != innerProduct.inFeatures)
                return RejectStrictCommandBufferPack4Node("InnerProduct input width does not match the loaded weight profile.");

            var expectedInputStorage = ResolveLinearMatStorageShape(input);
            var inputStorage = inputs[0]?.storageShape;
            if (inputStorage == null
                || inputStorage.Length != 5
                || inputStorage[0] != expectedInputStorage.dims
                || inputStorage[1] != expectedInputStorage.w
                || inputStorage[2] != expectedInputStorage.h
                || inputStorage[3] != expectedInputStorage.d
                || inputStorage[4] != expectedInputStorage.c)
            {
                return RejectStrictCommandBufferPack4Node("InnerProduct input storage does not prove the required LinearMat CommandBuffer texture layout.");
            }

            var output = input.dims == 2
                ? new BufferShape(2, innerProduct.outFeatures, input.h, 1, 1)
                : new BufferShape(1, innerProduct.outFeatures, 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                ResolveLinearMatStorageShape(output),
                request,
                "command-buffer-pack4:inner-product-linear-mat");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferReduction(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("The verified Reduction profile requires a 2D Pack4 activation.");

            var operation = layer.GetInt(0, 0);
            if (operation != 0 && operation != 3)
                return RejectStrictCommandBufferPack4Node("The Reduction operation is outside the verified SUM/MEAN CommandBuffer Pack4 subset.");
            if (layer.GetInt(1, 1) != 0)
                return RejectStrictCommandBufferPack4Node("Reduction over channels is not part of the verified CommandBuffer Pack4 profile.");

            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length != 2)
                return RejectStrictCommandBufferPack4Node("The verified Reduction profile requires spatial axes H and W.");
            var axis0 = axes[0] < 0 ? axes[0] + input.dims : axes[0];
            var axis1 = axes[1] < 0 ? axes[1] + input.dims : axes[1];
            if (!((axis0 == 1 && axis1 == 2) || (axis0 == 2 && axis1 == 1)))
                return RejectStrictCommandBufferPack4Node("The verified Reduction profile requires spatial axes H and W.");

            if (layer.GetInt(4, 0) != 0)
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    new BufferShape(3, 1, 1, 1, input.c),
                    request,
                    "command-buffer-pack4:reduction-spatial");

            var logicalShape = new BufferShape(1, input.c, 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                logicalShape,
                ResolveLinearMatStorageShape(logicalShape),
                request,
                "command-buffer-pack4:reduction-spatial-linear");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferBatchNorm(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("The verified BatchNorm profile requires a 2D Pack4 activation.");
            if (!_batchNorm.TryGetValue(layer.name, out var batchNorm)
                || batchNorm == null
                || batchNorm.channels != input.c
                || batchNorm.biasA4 == null
                || batchNorm.scaleB4 == null)
            {
                return RejectStrictCommandBufferPack4Node("The loaded BatchNorm Pack4 constants do not match the input descriptor.");
            }
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:batch-norm");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferReshape(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            BufferShape output;
            try
            {
                output = ResolveReshapeShape(input, layer);
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node("Reshape output shape resolution failed: " + exception.Message);
            }

            if (GetStrictPlanElementCount(input) != GetStrictPlanElementCount(output))
                return RejectStrictCommandBufferPack4Node("Reshape changes the logical element count.");

            if (input.dims >= 3 && input.dims <= 4 && output.dims >= 1 && output.dims <= 2
                && CanUseStrictPack4ToLinearMatReshape(layer))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    ResolveLinearMatStorageShape(output),
                    request,
                    "command-buffer-pack4:reshape-pack4-to-linear-mat");
            }

            if (input.dims >= 3 && output.dims >= 3 && output.dims <= 4 && !StrictPlanShapesEqual(input, output))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    request,
                    "command-buffer-pack4:reshape-pack4-to-pack4");
            }

            if (input.dims == 2 && output.dims >= 3 && output.dims <= 4
                && HasStrictPack4LinearMatStorage(inputs[0], input))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    request,
                    "command-buffer-pack4:reshape-pack4-linear-to-pack4");
            }

            if (input.dims == 2 && output.dims >= 3 && output.dims <= 4
                && HasStrictLinearMatStorage(inputs[0], input))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    request,
                    "command-buffer-pack4:reshape-linear-mat-to-pack4");
            }

            if (input.dims == 2 && output.dims >= 3 && output.dims <= 4
                && HasStrictScalar2DPack4Storage(inputs[0], input))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    request,
                    "command-buffer-pack4:reshape-scalar-2d-to-pack4");
            }

            return RejectStrictCommandBufferPack4Node("No loaded CommandBuffer Pack4 reshape transform matches this logical/storage profile.");
        }

        private static bool HasStrictPack4LinearMatStorage(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            if (descriptor?.storageShape == null || descriptor.storageShape.Length != 5 || logicalShape.dims != 2)
                return false;
            var storage = descriptor.storageShape;
            return storage[0] == 3
                && storage[1] == Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f)
                && storage[2] == Mathf.Max(1, logicalShape.h)
                && storage[3] == 1
                && storage[4] == 4;
        }

        private static bool HasStrictLinearMatStorage(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            if (descriptor?.storageShape == null || descriptor.storageShape.Length != 5 || logicalShape.dims != 2)
                return false;
            var expected = ResolveLinearMatStorageShape(logicalShape);
            var storage = descriptor.storageShape;
            return storage[0] == expected.dims
                && storage[1] == expected.w
                && storage[2] == expected.h
                && storage[3] == expected.d
                && storage[4] == expected.c;
        }

        private static bool HasStrictScalar2DPack4Storage(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            return logicalShape.dims == 2
                && storage != null
                && storage.Length == 5
                && storage[0] == 3
                && storage[1] == logicalShape.w
                && storage[2] == logicalShape.h
                && storage[3] == 1
                && storage[4] == 1;
        }

        private bool CanUseStrictPack4ToLinearMatReshape(NcnnParamModel.Layer layer)
        {
            if (Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var outputBlob = layer.topNames[0];
            var consumerCount = 0;
            NcnnParamModel.Layer consumer = null;
            foreach (var candidate in Model.layers)
            {
                if (candidate?.bottomNames == null || !candidate.bottomNames.Contains(outputBlob))
                    continue;
                consumer = candidate;
                consumerCount++;
            }

            return consumerCount == 1
                && (consumer.type == NcnnLayerTypes.Permute
                    || consumer.type == NcnnLayerTypes.Gemm
                    || consumer.type == NcnnLayerTypes.InnerProduct);
        }

        private static long GetStrictPlanElementCount(BufferShape shape)
        {
            return (long)Mathf.Max(1, shape.w)
                * Mathf.Max(1, shape.h)
                * Mathf.Max(1, shape.d)
                * Mathf.Max(1, shape.c);
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPermute(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            var orderType = layer.GetInt(0, 0);
            BufferShape output;
            try
            {
                if (input.dims == 2)
                {
                    if (orderType != 1)
                        return RejectStrictCommandBufferPack4Node("The verified LinearMat CommandBuffer permute profile requires a 2D transpose (order=1).");
                    output = new BufferShape(2, input.h, input.w, 1, 1);
                    if (HasStrictPack4LinearMatStorage(inputs[0], input)
                        || HasStrictScalar2DPack4Storage(inputs[0], input))
                    {
                        return AcceptStrictCommandBufferPack4Node(
                            layer,
                            output,
                            new BufferShape(3, output.w, output.h, 1, 1),
                            request,
                            "command-buffer-pack4:permute-pack4-scalar-2d");
                    }
                    if (!HasStrictLinearMatStorage(inputs[0], input))
                        return RejectStrictCommandBufferPack4Node("The 2D Permute input lacks a verified LinearMat or Pack4-Linear storage descriptor.");
                    return AcceptStrictCommandBufferPack4Node(
                        layer,
                        output,
                        ResolveLinearMatStorageShape(output),
                        request,
                        "command-buffer-pack4:permute-linear-mat-2d");
                }

                if (input.dims == 3 && input.d == 1)
                {
                    if (orderType == 0)
                        return RejectStrictCommandBufferPack4Node("Identity Permute must be represented by descriptor alias evidence, not a computation profile.");
                    output = ResolvePermuteShape(input, 3, ResolvePermuteAxes(3, orderType, layer.name));
                    return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:permute-pack4");
                }

                if (input.dims == 4)
                {
                    if (orderType == 0)
                        return RejectStrictCommandBufferPack4Node("Identity Permute must be represented by descriptor alias evidence, not a computation profile.");
                    output = ResolvePermuteShape(input, 4, ResolvePermuteAxes(4, orderType, layer.name));
                    return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:permute-pack4-cdhw");
                }
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node("Permute profile resolution failed: " + exception.Message);
            }

            return RejectStrictCommandBufferPack4Node("No loaded CommandBuffer Pack4 permute transform matches this logical/storage profile.");
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGemm(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_gemm.TryGetValue(layer.name, out var gemm) || gemm == null || gemm.bData == null)
                return RejectStrictCommandBufferPack4Node("The loaded Gemm constant-B weights are unavailable.");
            if (input.dims != 2 || gemm.transA || !gemm.constantB || gemm.constantK <= 0 || gemm.constantN <= 0)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Gemm profile requires a 2D non-transposed A with loaded constant B, K, and N.");
            if (input.w != gemm.constantK)
                return RejectStrictCommandBufferPack4Node("Gemm K does not match the descriptor-backed LinearMat width.");

            var bRows = gemm.transB ? gemm.constantN : gemm.constantK;
            var bColumns = gemm.transB ? gemm.constantK : gemm.constantN;
            var kFromB = gemm.transB ? bColumns : bRows;
            var outputColumns = gemm.transB ? bRows : bColumns;
            if (kFromB != input.w || outputColumns <= 0)
                return RejectStrictCommandBufferPack4Node("The loaded Gemm B matrix does not match its K/N profile.");
            if (gemm.constantC && gemm.broadcastTypeC != -1 && gemm.cData == null)
                return RejectStrictCommandBufferPack4Node("The loaded Gemm bias matrix is unavailable.");

            var output = new BufferShape(2, outputColumns, input.h, 1, 1);
            var storage = outputColumns % 4 == 0
                ? ResolvePack4LinearMatStorageShape(output)
                : ResolveLinearMatStorageShape(output);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                storage,
                request,
                outputColumns % 4 == 0
                    ? "command-buffer-pack4:gemm-linear-to-pack4-linear"
                    : "command-buffer-pack4:gemm-linear-mat");
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSlice(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (layer?.topNames == null || layer.topNames.Length == 0)
                return RejectStrictCommandBufferPack4Node("Slice has no output blobs.");
            if (input.dims < 3 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("The verified Slice profile requires a Pack4 3D or 4D texture.");

            var ncnnAxis = layer.GetInt(1, 0);
            if (ncnnAxis < 0)
                ncnnAxis += input.dims;
            if (ncnnAxis < 0 || ncnnAxis >= input.dims)
                return RejectStrictCommandBufferPack4Node("Slice axis is outside the input rank.");

            var axis = MapNcnnAxisToTensorAxis(input.dims, ncnnAxis);
            var axisSize = GetAxisSize(input.dims, input.w, input.h, input.d, input.c, axis);
            var sliceParams = layer.GetInts(-23300, null);
            var indices = layer.GetInts(-23302, null);
            if ((sliceParams == null || sliceParams.Length == 0) && (indices == null || indices.Length == 0))
                return RejectStrictCommandBufferPack4Node("Slice requires static split sizes or indices.");

            var outputs = new NcnnTexturePlanTensorDescriptor[layer.topNames.Length];
            var begin = 0;
            for (var index = 0; index < layer.topNames.Length; index++)
            {
                var size = ResolveStrictSliceSize(sliceParams, indices, index, layer.topNames.Length, axisSize, begin);
                if (size <= 0 || begin + size > axisSize)
                    return RejectStrictCommandBufferPack4Node("Slice output size is invalid for the descriptor-backed input axis.");

                var outputW = axis == 0 ? size : input.w;
                var outputH = axis == 1 ? size : input.h;
                var outputD = axis == 2 && input.dims == 4 ? size : input.d;
                var outputC = (axis == 2 && input.dims != 4) || axis == 3 ? size : input.c;
                var output = new BufferShape(input.dims, outputW, outputH, outputD, outputC);

                outputs[index] = new NcnnTexturePlanTensorDescriptor
                {
                    blob = layer.topNames[index],
                    logicalShape = new[] { output.dims, output.w, output.h, output.d, output.c },
                    storageShape = new[] { output.dims, output.w, output.h, output.d, output.c },
                    layout = request.targetLayout,
                    dtype = request.targetDtype,
                    aliasGroup = "computed:" + (layer.name ?? layer.typeName ?? "slice") + ":" + index,
                    textureBacked = true
                };
                begin += size;
            }

            if (begin != axisSize)
                return RejectStrictCommandBufferPack4Node("Slice sizes do not cover the descriptor-backed input axis.");

            return new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = input.dims == 4
                    ? "command-buffer-pack4:slice-pack4-cdhw"
                    : "command-buffer-pack4:slice-pack4",
                outputs = outputs
            };
        }

        private static int ResolveStrictSliceSize(
            int[] sliceParams,
            int[] indices,
            int outputIndex,
            int outputCount,
            int axisSize,
            int begin)
        {
            if (indices != null && indices.Length > 0)
            {
                if (outputIndex == outputCount - 1)
                    return axisSize - begin;
                var end = indices[Mathf.Min(outputIndex, indices.Length - 1)];
                if (end < 0)
                    end += axisSize;
                return end - begin;
            }

            var size = sliceParams[Mathf.Min(outputIndex, sliceParams.Length - 1)];
            return size == -233 ? (axisSize - begin) / Mathf.Max(1, outputCount - outputIndex) : size;
        }

        private NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMatMul(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!EnableAttentionMatMulPack4Specializations)
                return RejectStrictCommandBufferPack4Node("CommandBuffer MatMul Pack4 specializations are not enabled for this runtime profile.");
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length != 2)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer MatMul profile requires exactly two inputs.");
            if (!IsStrictAttentionMatMulInput(inputs[0], shapes[0]) || !IsStrictAttentionMatMulInput(inputs[1], shapes[1]))
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer MatMul profile requires Pack4 3D or 4D texture descriptors.");

            var aRows = shapes[0].h;
            var aColumns = shapes[0].w;
            var bRows = shapes[1].h;
            var bColumns = shapes[1].w;
            var transposeB = layer.GetInt(0, 0) != 0;
            var kFromB = transposeB ? bColumns : bRows;
            var outputColumns = transposeB ? bRows : bColumns;
            if (aRows <= 0 || aColumns <= 0 || outputColumns <= 0 || aColumns != kFromB)
                return RejectStrictCommandBufferPack4Node("MatMul dimensions do not match the verified Pack4 attention matrix profile.");

            var outputDepth = Mathf.Max(shapes[0].d, shapes[1].d);
            var outputChannels = Mathf.Max(shapes[0].c, shapes[1].c);
            if ((shapes[0].d != 1 && shapes[0].d != outputDepth)
                || (shapes[1].d != 1 && shapes[1].d != outputDepth)
                || (shapes[0].c != 1 && shapes[0].c != outputChannels)
                || (shapes[1].c != 1 && shapes[1].c != outputChannels))
            {
                return RejectStrictCommandBufferPack4Node("MatMul batch dimensions are outside the verified Pack4 attention broadcast profile.");
            }

            var output = outputDepth == 1 && outputChannels == 1
                ? new BufferShape(2, outputColumns, aRows, 1, 1)
                : Mathf.Max(shapes[0].dims, shapes[1].dims) >= 4 || outputDepth > 1
                    ? new BufferShape(4, outputColumns, aRows, outputDepth, outputChannels)
                    : new BufferShape(3, outputColumns, aRows, 1, outputChannels);
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:matmul-attention");
        }

        private static bool IsStrictAttentionMatMulInput(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape shape)
        {
            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h > 0
                && shape.d > 0
                && shape.c > 0
                && (shape.dims != 3 || shape.d == 1)
                && descriptor?.storageShape != null
                && descriptor.storageShape.Length == 5
                && descriptor.storageShape[0] == shape.dims
                && descriptor.storageShape[1] == shape.w
                && descriptor.storageShape[2] == shape.h
                && descriptor.storageShape[3] == shape.d
                && descriptor.storageShape[4] == shape.c;
        }

        private static NcnnTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSoftmax(
            NcnnParamModel.Layer layer,
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            NcnnTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 && input.dims != 4)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Softmax profile requires a Pack4 3D or 4D texture.");
            if (input.dims == 3 && input.d != 1)
                return RejectStrictCommandBufferPack4Node("The verified 3D CommandBuffer Softmax profile requires a unit depth dimension.");
            if (!IsStrictAttentionMatMulInput(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Softmax input storage does not prove the required Pack4 physical mapping.");

            var ncnnAxis = layer.GetInt(0, 0);
            if (ncnnAxis < 0)
                ncnnAxis += input.dims;
            if (ncnnAxis < 0 || ncnnAxis >= input.dims || MapNcnnAxisToTensorAxis(input.dims, ncnnAxis) != 0)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Softmax profile supports only the tensor width axis.");

            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:softmax-width");
        }

        private static bool TryGetSingleStrictCdhwPlanShape(
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            out BufferShape shape,
            out string reason)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out shape, out reason))
                return false;
            if (shape.dims != 4)
            {
                reason = "This CommandBuffer Pack4 path requires a CDHW logical tensor (dims=4).";
                return false;
            }
            if (!HasStrictCdhwPack4Storage(inputs[0], shape))
            {
                reason = "The CDHW TensorDescriptor must map logical [dims,w,h,d,c] to an exact Pack4 Texture2DArray storage profile.";
                return false;
            }
            return true;
        }

        private static bool HasStrictCdhwPack4Storage(
            NcnnTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, NcnnTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase)
                && logicalShape.dims == 4
                && storage != null
                && storage.Length == 5
                && storage[0] == 4
                && storage[1] == logicalShape.w
                && storage[2] == logicalShape.h
                && storage[3] == logicalShape.d
                && storage[4] == logicalShape.c;
        }

        private static bool TryGetSingleStrictPlanShape(
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            out BufferShape shape,
            out string reason)
        {
            shape = default;
            reason = null;
            if (inputs == null || inputs.Count != 1)
            {
                reason = "This CommandBuffer Pack4 path requires exactly one descriptor-backed input.";
                return false;
            }
            return TryGetStrictPlanShape(inputs[0], out shape, out reason);
        }

        private static bool TryGetStrictPlanShapes(
            IReadOnlyList<NcnnTexturePlanTensorDescriptor> inputs,
            out BufferShape[] shapes,
            out string reason)
        {
            shapes = Array.Empty<BufferShape>();
            reason = null;
            if (inputs == null || inputs.Count == 0)
            {
                reason = "This CommandBuffer Pack4 path requires descriptor-backed inputs.";
                return false;
            }

            shapes = new BufferShape[inputs.Count];
            for (var index = 0; index < inputs.Count; index++)
            {
                if (!TryGetStrictPlanShape(inputs[index], out shapes[index], out reason))
                    return false;
            }
            return true;
        }

        private static bool TryGetStrictPlanShape(
            NcnnTexturePlanTensorDescriptor descriptor,
            out BufferShape shape,
            out string reason)
        {
            shape = default;
            reason = null;
            var logical = descriptor?.logicalShape;
            var storage = descriptor?.storageShape;
            if (descriptor == null || !descriptor.textureBacked || logical == null || storage == null
                || logical.Length != 5 || storage.Length != 5
                || logical[0] < 1 || logical[0] > 4
                || logical[1] <= 0 || logical[2] <= 0 || logical[3] <= 0 || logical[4] <= 0
                || storage[1] <= 0 || storage[2] <= 0 || storage[3] <= 0 || storage[4] <= 0)
            {
                reason = "The input lacks a valid texture-backed logical/storage descriptor.";
                return false;
            }

            shape = new BufferShape(logical[0], logical[1], logical[2], logical[3], logical[4]);
            return true;
        }

        private static bool StrictPlanShapesEqual(BufferShape left, BufferShape right)
        {
            return left.dims == right.dims
                && left.w == right.w
                && left.h == right.h
                && left.d == right.d
                && left.c == right.c;
        }

        private static NcnnTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Node(
            NcnnParamModel.Layer layer,
            BufferShape outputShape,
            NcnnTextureExecutionPlanRequest request,
            string executionPath)
        {
            return AcceptStrictCommandBufferPack4Node(layer, outputShape, outputShape, request, executionPath);
        }

        private static NcnnTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Node(
            NcnnParamModel.Layer layer,
            BufferShape logicalShape,
            BufferShape storageShape,
            NcnnTextureExecutionPlanRequest request,
            string executionPath)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            var logical = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c };
            var storage = new[] { storageShape.dims, storageShape.w, storageShape.h, storageShape.d, storageShape.c };
            return new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = outputNames.Length > 0,
                executionPath = executionPath,
                reason = outputNames.Length > 0 ? null : "The node has no output blobs.",
                outputs = outputNames.Select((name, index) => new NcnnTexturePlanTensorDescriptor
                {
                    blob = name,
                    logicalShape = (int[])logical.Clone(),
                    storageShape = (int[])storage.Clone(),
                    layout = request.targetLayout,
                    dtype = request.targetDtype,
                    aliasGroup = "computed:" + (layer?.name ?? layer?.typeName ?? "layer") + ":" + index,
                    textureBacked = true
                }).ToArray()
            };
        }

        private static NcnnTextureExecutionPlanNodeVerification RejectStrictCommandBufferPack4Node(string reason)
        {
            return new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = false,
                reason = reason ?? "The loaded runtime profile rejected this CommandBuffer Pack4 node."
            };
        }

        private static NcnnTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4NoopAlias(
            NcnnParamModel.Layer layer,
            NcnnTexturePlanTensorDescriptor source,
            NcnnTextureExecutionPlanRequest request)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            return new NcnnTextureExecutionPlanNodeVerification
            {
                accepted = outputNames.Length > 0,
                usesDescriptorAlias = outputNames.Length > 0,
                executionPath = "descriptor-alias",
                reason = outputNames.Length > 0 ? null : "The node has no output blobs.",
                outputs = outputNames.Select(name => new NcnnTexturePlanTensorDescriptor
                {
                    blob = name,
                    logicalShape = source.logicalShape == null ? Array.Empty<int>() : (int[])source.logicalShape.Clone(),
                    storageShape = source.storageShape == null ? Array.Empty<int>() : (int[])source.storageShape.Clone(),
                    layout = request.targetLayout,
                    dtype = request.targetDtype,
                    aliasGroup = source.aliasGroup,
                    textureBacked = source.textureBacked
                }).ToArray()
            };
        }

        private bool HasDebugOracleIntent()
        {
            return ForceBufferBinaryOpAll
                || ForceCpuGemmAll
                || ForceBufferGeluAll
                || ForceBufferConvolution
                || ForceBufferConvolutionAll
                || ForceBufferOutputsForDims4
                || (ForceBufferLayerTypes != null && ForceBufferLayerTypes.Count > 0)
                || (ForceBufferLayerNames != null && ForceBufferLayerNames.Count > 0)
                || (DebugCompareTextureLayers != null && DebugCompareTextureLayers.Count > 0)
                || (DebugCompareTextureConvLayers != null && DebugCompareTextureConvLayers.Count > 0)
                || (DebugCompareMaxPoolingLayers != null && DebugCompareMaxPoolingLayers.Count > 0);
        }

        private static string ResolveTexturePlanDtype(RenderTextureFormat format)
        {
            switch (ResolveInferenceTensorDataType(format))
            {
                case InferenceTensorDataType.Float16:
                    return "FP16";
                case InferenceTensorDataType.Float32:
                    return "FP32";
                case InferenceTensorDataType.Int8:
                    return "INT8";
                default:
                    return "Unknown";
            }
        }

        internal void SetCurrentBufferExecutionContext(NcnnLayerBufferContext context)
        {
            _currentBufferContext = context;
        }

        internal void ClearCurrentBufferExecutionContext()
        {
            _currentBufferContext = null;
        }

        internal void NotifyConvComplete(string layerName, string mode, int srcW, int srcH, int inPacks, int outPacks, double gpuMs)
        {
            try { OnConvComplete?.Invoke(layerName, mode, srcW, srcH, inPacks, outPacks, gpuMs); } catch { }
        }

        internal void SetCurrentExecutingLayer(NcnnParamModel.Layer layer)
        {
            _currentExecutingLayerName = layer?.name;
            _currentExecutingLayerTypeName = layer?.typeName;
        }

        internal void ClearCurrentExecutingLayer()
        {
            _currentExecutingLayerName = null;
            _currentExecutingLayerTypeName = null;
        }

        public sealed class CmdInferResult : IDisposable
        {
            private readonly Dictionary<string, CmdTensorRef> _blobs;
            private readonly Dictionary<string, BufferShape> _shapes;
            private readonly NcnnRepro _owner;
            private readonly RenderTexture _readbackTexture;

            internal CmdInferResult(
                Dictionary<string, CmdTensorRef> blobs,
                Dictionary<string, BufferShape> shapes,
                NcnnRepro owner,
                RenderTexture readbackTexture)
            {
                _blobs = blobs;
                _shapes = shapes;
                _owner = owner;
                _readbackTexture = readbackTexture;
            }

            public RenderTexture GetReadbackTexture()
            {
                return _readbackTexture;
            }

            public bool TryGetLogicalShape(string name, out int dims, out int w, out int h, out int d, out int c)
            {
                if (_shapes != null && _shapes.TryGetValue(name, out var shape))
                {
                    dims = shape.dims;
                    w = shape.w;
                    h = shape.h;
                    d = shape.d;
                    c = shape.c;
                    return true;
                }

                dims = 0;
                w = 0;
                h = 0;
                d = 0;
                c = 0;
                return false;
            }

            public void Dispose()
            {
                if (_readbackTexture != null)
                    _owner?.ReturnTempArray(_readbackTexture);

                if (_blobs == null)
                    return;

                var visited = new HashSet<CmdTensorRef>();
                foreach (var kv in _blobs)
                {
                    var tr = kv.Value;
                    if (tr == null || !visited.Add(tr))
                        continue;

                    tr.sharedTextureOwner = null;
                    tr.ClearTexture();
                    tr.owned = false;
                }
            }
        }

        internal void NotifyTimingSplitSyncPoint(string topName, double elapsedMs)
        {
            try { OnTimingSplitSyncPoint?.Invoke(topName, elapsedMs); } catch { }
        }

        public void ResetInferenceTempResourceStats()
        {
            _tempBufferRentCount = 0;
            _tempBufferRentBytes = 0L;
            _tempBufferLiveCount = 0;
            _tempBufferLiveBytes = 0L;
            _tempBufferPeakLiveCount = 0;
            _tempBufferPeakLiveBytes = 0L;
            _tempRtRentCount = 0;
            _tempRtLiveCount = 0;
            _tempRtPeakLiveCount = 0;
        }

        public void BeginInferenceTempResourceTracking()
        {
            ResetInferenceTempResourceStats();
            _trackInferenceTempResources = true;
            if (!IsDebugOracleExecution)
                DebugLog?.Invoke("[InferencePathAudit] mode=ProductionTextureOnly | activation_storage=Pack4Texture | buffer_materialization=forbidden");
        }

        public void EndInferenceTempResourceTracking()
        {
            _trackInferenceTempResources = false;
            if (!IsDebugOracleExecution)
                DebugLog?.Invoke("[InferencePathAudit] completed | intermediate_buffer_materializations=0");
        }

        public TempResourceStatsSnapshot GetInferenceTempResourceStats()
        {
            return new TempResourceStatsSnapshot
            {
                tempBufferRentCount = _tempBufferRentCount,
                tempBufferRentBytes = _tempBufferRentBytes,
                tempBufferLiveCount = _tempBufferLiveCount,
                tempBufferLiveBytes = _tempBufferLiveBytes,
                tempBufferPeakLiveCount = _tempBufferPeakLiveCount,
                tempBufferPeakLiveBytes = _tempBufferPeakLiveBytes,
                tempRtRentCount = _tempRtRentCount,
                tempRtLiveCount = _tempRtLiveCount,
                tempRtPeakLiveCount = _tempRtPeakLiveCount
            };
        }

        internal bool ShouldForceCurrentLayerBufferPath()
        {
            if (MatchesForceBufferToken(ForceBufferLayerNames, _currentExecutingLayerName))
                return true;
            if (MatchesForceBufferToken(ForceBufferLayerTypes, _currentExecutingLayerTypeName))
                return true;
            return false;
        }

        internal bool ShouldAllowCurrentLayerBufferGuardBypass()
        {
            return ShouldForceCurrentLayerBufferPath();
        }

        internal bool ShouldBlockPack4BufferFallback()
        {
            if (!IsDebugOracleExecution)
                return true;
            if (ShouldAllowCurrentLayerBufferGuardBypass())
                return false;
            return DisallowBufferAccess
                || DisallowBufferOutputs
                || DisallowBufferToTextureMaterialization
                || DisallowInferenceTempComputeBuffers;
        }

        private static bool MatchesForceBufferToken(ISet<string> set, string value)
        {
            if (set == null || set.Count == 0 || string.IsNullOrWhiteSpace(value))
                return false;
            return set.Contains(value) || set.Contains("*");
        }

        private string DescribeCurrentExecutionSite()
        {
            if (!string.IsNullOrWhiteSpace(_currentExecutingLayerTypeName) || !string.IsNullOrWhiteSpace(_currentExecutingLayerName))
                return (_currentExecutingLayerTypeName ?? "Unknown") + ":" + (_currentExecutingLayerName ?? "Unknown");
            return "outside-layer";
        }

        private static string DescribeShape(BufferShape shape)
        {
            return "d" + shape.dims + ":" + shape.w + "x" + shape.h + "x" + shape.d + "x" + shape.c;
        }

        private static string DescribeTextureDtype(RenderTexture texture)
        {
            if (texture == null)
                return "unknown";
            return texture.format == RenderTextureFormat.ARGBFloat ? "FP32" : "FP16";
        }

        private string DescribeCurrentBlobContract(string blobName)
        {
            const string unknown = "unknown";
            if (string.IsNullOrWhiteSpace(blobName) || _currentBufferContext == null)
                return "logical_shape=" + unknown + " | storage_shape=" + unknown + " | layout=" + unknown + " | dtype=" + unknown;

            var textureBlobs = _currentBufferContext.textureBlobs;
            var textureShapes = _currentBufferContext.textureShapes;
            if (textureBlobs != null
                && textureBlobs.TryGetValue(blobName, out var textureRef)
                && textureRef != null
                && textureRef.texture != null)
            {
                var logical = GetTextureShape(textureShapes, textureRef, blobName);
                var storage = GetTextureStorageShape(textureRef, logical);
                return "logical_shape=" + DescribeShape(logical)
                    + " | storage_shape=" + DescribeShape(storage)
                    + " | layout=" + textureRef.layoutKind
                    + " | dtype=" + DescribeTextureDtype(textureRef.texture);
            }

            var bufferViews = _currentBufferContext.bufferViews;
            if (bufferViews != null
                && bufferViews.TryGetValue(blobName, out var bufferView)
                && bufferView != null)
            {
                var logical = new BufferShape(bufferView.dims, bufferView.w, bufferView.h, bufferView.d, bufferView.c);
                return "logical_shape=" + DescribeShape(logical)
                    + " | storage_shape=" + DescribeShape(logical)
                    + " | layout=Linear"
                    + " | dtype=FP32";
            }

            return "logical_shape=" + unknown + " | storage_shape=" + unknown + " | layout=" + unknown + " | dtype=" + unknown;
        }

        private static long EstimateTempBufferBytes(int count, int stride)
        {
            return Math.Max(0L, (long)Mathf.Max(1, count) * Math.Max(1, stride));
        }

        private void TrackTempBufferRent(int count, int stride)
        {
            if (!_trackInferenceTempResources)
                return;

            var bytes = EstimateTempBufferBytes(count, stride);
            _tempBufferRentCount++;
            _tempBufferRentBytes += bytes;
            _tempBufferLiveCount++;
            _tempBufferLiveBytes += bytes;
            if (_tempBufferLiveCount > _tempBufferPeakLiveCount)
                _tempBufferPeakLiveCount = _tempBufferLiveCount;
            if (_tempBufferLiveBytes > _tempBufferPeakLiveBytes)
                _tempBufferPeakLiveBytes = _tempBufferLiveBytes;
        }

        private void TrackTempBufferReturn(ComputeBuffer buffer)
        {
            if (!_trackInferenceTempResources || buffer == null)
                return;

            try
            {
                _tempBufferLiveCount = Math.Max(0, _tempBufferLiveCount - 1);
                _tempBufferLiveBytes = Math.Max(0L, _tempBufferLiveBytes - EstimateTempBufferBytes(buffer.count, buffer.stride));
            }
            catch
            {
            }
        }

        private void TrackTempRtRent()
        {
            if (!_trackInferenceTempResources)
                return;

            _tempRtRentCount++;
            _tempRtLiveCount++;
            if (_tempRtLiveCount > _tempRtPeakLiveCount)
                _tempRtPeakLiveCount = _tempRtLiveCount;
        }

        private void TrackTempRtReturn()
        {
            if (!_trackInferenceTempResources)
                return;

            _tempRtLiveCount = Math.Max(0, _tempRtLiveCount - 1);
        }

        private void ValidateTempBufferAllowed(int count, int stride)
        {
            if (IsDebugOracleExecution && !DisallowInferenceTempComputeBuffers)
                return;

            var detail =
                "count=" + Mathf.Max(1, count).ToString(CultureInfo.InvariantCulture)
                + " stride=" + Math.Max(1, stride).ToString(CultureInfo.InvariantCulture)
                + " bytes=" + EstimateTempBufferBytes(count, stride).ToString(CultureInfo.InvariantCulture);
            throw CreateDisallowedBufferPathException(
                "pack4-only guard: temporary compute buffer allocation disallowed during inference",
                null,
                detail);
        }

        private static int GetMaxTextureArraySlices()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureArraySlices);
            }
            catch
            {
                return FallbackMaxTextureArraySlices;
            }
        }

        private static int GetMaxTextureSize()
        {
            try
            {
                return Mathf.Max(1, SystemInfo.maxTextureSize);
            }
            catch
            {
                return FallbackMaxTextureSize;
            }
        }

        private bool WouldExceedTextureArraySliceLimit(int depth)
        {
            return Mathf.Max(1, depth) > GetMaxTextureArraySlices();
        }

        private bool WouldExceedTextureSizeLimit(int width, int height)
        {
            var max = GetMaxTextureSize();
            return Mathf.Max(1, width) > max || Mathf.Max(1, height) > max;
        }

        private bool TryResolveTensorTextureMaterialization(
            NcnnTensorBuffer view,
            out int texW,
            out int texH,
            out int channels,
            out int sliceCount,
            out RenderTextureFormat format,
            bool logSkipReason = true)
        {
            texW = 0;
            texH = 0;
            channels = 0;
            sliceCount = 0;
            format = default;
            if (view == null)
                return false;

            if (view.dims == 1)
            {
                texW = view.w;
                texH = 1;
                channels = 1;
            }
            else if (view.dims == 2)
            {
                texW = view.w;
                texH = view.h;
                channels = 1;
            }
            else if (view.dims == 3)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else if (view.dims == 4)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else
            {
                return false;
            }

            var channelPacks = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            sliceCount = view.dims == 4
                ? Mathf.Max(1, view.d) * channelPacks
                : channelPacks;
            format = ResolveTensorTextureFormat(view.dims);

            if (WouldExceedTextureSizeLimit(texW, texH))
            {
                if (logSkipReason)
                {
                    DebugLog?.Invoke(
                        "[BufferMaterialize] skip-texture-size-limit"
                        + " | site=" + DescribeCurrentExecutionSite()
                        + " | dims=" + view.dims
                        + " | shape=" + view.w + "x" + view.h + "x" + view.d + "x" + view.c
                        + " | requested_size=" + texW.ToString(CultureInfo.InvariantCulture) + "x" + texH.ToString(CultureInfo.InvariantCulture)
                        + " | max_size=" + GetMaxTextureSize().ToString(CultureInfo.InvariantCulture));
                }
                return false;
            }

            if (WouldExceedTextureArraySliceLimit(sliceCount))
            {
                if (logSkipReason)
                {
                    DebugLog?.Invoke(
                        "[BufferMaterialize] skip-texture-slice-limit"
                        + " | site=" + DescribeCurrentExecutionSite()
                        + " | dims=" + view.dims
                        + " | shape=" + view.w + "x" + view.h + "x" + view.d + "x" + view.c
                        + " | slices=" + sliceCount.ToString(CultureInfo.InvariantCulture)
                        + " | max_slices=" + GetMaxTextureArraySlices().ToString(CultureInfo.InvariantCulture));
                }
                return false;
            }

            return true;
        }

        private InvalidOperationException CreateDisallowedBufferPathException(string reason, string blobName, string detail = null)
        {
            var sb = new StringBuilder(256);
            sb.Append(reason);
            sb.Append(" | site=");
            sb.Append(DescribeCurrentExecutionSite());
            sb.Append(" | blob=");
            sb.Append(string.IsNullOrWhiteSpace(blobName) ? "unknown" : blobName);
            if (!string.IsNullOrWhiteSpace(detail))
            {
                sb.Append(" | ");
                sb.Append(detail);
            }
            sb.Append(" | ");
            sb.Append(DescribeCurrentBlobContract(blobName));
            sb.Append(" | rejected_fallback=");
            sb.Append(reason);
            return new InvalidOperationException(sb.ToString());
        }

        public NcnnRepro(NcnnOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void LoadModel(string paramText, NcnnBinReader br, Action<LoadProgress> onProgress = null)
        {
            try
            {
                foreach (var progress in EnumerateModelLoad(paramText, br))
                    onProgress?.Invoke(progress);
            }
            catch
            {
                try { Release(); } catch { }
                throw;
            }
        }

        public int MergePnnxStringParams(string pnnxParamText, bool overwriteExisting = false)
        {
            if (string.IsNullOrWhiteSpace(pnnxParamText) || Model == null)
                return 0;

            var pnnxModel = NcnnParamParser.Parse(pnnxParamText);
            return NcnnParamParser.MergeStringParamsByLayerName(Model, pnnxModel, overwriteExisting);
        }

        public async UniTask LoadModelAsync(
            string paramText,
            NcnnBinReader br,
            Action<LoadProgress> onProgress = null,
            CancellationToken ct = default,
            int yieldEveryLayers = 6)
        {
            yieldEveryLayers = Mathf.Max(1, yieldEveryLayers);
            var nextYieldLayer = yieldEveryLayers;

            try
            {
                foreach (var progress in EnumerateModelLoad(paramText, br))
                {
                    onProgress?.Invoke(progress);

                    var shouldYield = false;
                    if (!string.Equals(progress.stage, "layer", StringComparison.Ordinal))
                    {
                        shouldYield = true;
                    }
                    else if (progress.layerIndex >= nextYieldLayer || progress.layerIndex >= progress.layerCount)
                    {
                        shouldYield = true;
                        nextYieldLayer = progress.layerIndex + yieldEveryLayers;
                    }

                    ct.ThrowIfCancellationRequested();
                    if (shouldYield)
                        await UniTask.Yield();
                }
            }
            catch
            {
                try { Release(); } catch { }
                throw;
            }
        }

        private IEnumerable<LoadProgress> EnumerateModelLoad(string paramText, NcnnBinReader br)
        {
            if (paramText == null)
                throw new ArgumentNullException(nameof(paramText));
            if (br == null)
                throw new ArgumentNullException(nameof(br));

            var stageSw = Stopwatch.StartNew();
            var profile = new ModelLoadProfile();
            LastLoadProfile = profile;
            long totalLoadMs = 0;

            Release();
            stageSw.Stop();
            profile.releaseMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.releaseMs;
            yield return new LoadProgress("release", 0, 0, null, null, 0.01f);

            stageSw.Restart();
            Model = NcnnParamParser.Parse(paramText);
            LayerRepros = NcnnLayerFactoryRepro.CreateModelLayers(Model?.layers);
            stageSw.Stop();
            profile.parseParamMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.parseParamMs;
            profile.modelMagic = Model?.magic;
            profile.layerCount = Model?.layers?.Count ?? 0;
            yield return new LoadProgress("parse", 0, profile.layerCount, null, null, 0.03f);

            stageSw.Restart();
            _blobUseCount = BuildBlobUseCount(Model);
            stageSw.Stop();
            profile.buildBlobUseCountMs = stageSw.ElapsedMilliseconds;
            totalLoadMs += profile.buildBlobUseCountMs;
            yield return new LoadProgress("build-blobs", 0, profile.layerCount, null, null, 0.05f);

            var totalLayers = Model?.layers?.Count ?? 0;
            for (var i = 0; i < totalLayers; i++)
            {
                var layer = Model.layers[i];
                var layerSw = Stopwatch.StartNew();
                var layerRepro = LayerRepros != null && i < LayerRepros.Count ? LayerRepros[i] : null;
                var metrics = layerRepro != null ? layerRepro.LoadLayer(this, layer, br) : LoadLayer(layer, br);
                layerSw.Stop();
                totalLoadMs += layerSw.ElapsedMilliseconds;

                AccumulateLayerProfile(profile, layer?.typeName, metrics, layerSw.ElapsedMilliseconds);

                var progress01 = totalLayers > 0
                    ? 0.05f + 0.94f * ((float)(i + 1) / totalLayers)
                    : 0.99f;
                yield return new LoadProgress("layer", i + 1, totalLayers, layer?.name, layer?.typeName, progress01);
            }

            profile.totalMs = totalLoadMs;
            yield return new LoadProgress("complete", totalLayers, totalLayers, null, null, 1f);
        }

        private LayerLoadMetrics LoadLayer(NcnnParamModel.Layer layer, NcnnBinReader br)
        {
            if (layer == null)
                return default;

            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            if (layer.type == NcnnLayerTypes.Convolution
                || layer.type == NcnnLayerTypes.ConvolutionDepthWise)
            {
                var pack = new ConvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.group = Mathf.Max(1, layer.GetInt(7, 1));
                pack.kernelW = layer.GetInt(1, 0);
                pack.kernelH = layer.GetInt(11, pack.kernelW);
                pack.dilationW = layer.GetInt(2, 1);
                pack.dilationH = layer.GetInt(12, pack.dilationW);
                pack.strideW = layer.GetInt(3, 1);
                pack.strideH = layer.GetInt(13, pack.strideW);
                pack.padLeft = layer.GetInt(4, 0);
                pack.padRight = layer.GetInt(15, pack.padLeft);
                pack.padTop = layer.GetInt(14, pack.padLeft);
                pack.padBottom = layer.GetInt(16, pack.padTop);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);
                pack.isDepthWise = layer.type == NcnnLayerTypes.ConvolutionDepthWise;

                var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                if (pack.isDepthWise)
                {
                    pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                    pack.useBufferPath = true;
                }
                else
                {
                    pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));
                    pack.useBufferPath = pack.strideW != 1
                                         || pack.strideH != 1
                                         || pack.kernelW != 1 && pack.kernelW != 3
                                         || pack.kernelH != pack.kernelW
                                         || pack.dilationW != 1
                                         || pack.dilationH != 1
                                         || pack.padLeft != pack.padRight
                                         || pack.padTop != pack.padBottom
                                         || pack.kernelW != 3 && pack.kernelW != 1;
                }
                pack.inPacks = (pack.inC + 3) / 4;
                pack.outPacks = (pack.outC + 3) / 4;

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                var needGeneralTexturePack = !ForceBufferConvolutionAll
                                             && !pack.useBufferPath
                                             && !pack.isDepthWise
                                             && pack.group == 1
                                             && !(pack.kernelW == 1 && pack.kernelH == 1 && !EnableConv1x1TextureConvolution);
                var needDepthWiseTexturePack = !ForceBufferConvolutionAll
                                               && EnableDepthWiseTextureConvolution
                                               && pack.isDepthWise
                                               && pack.group == pack.inC
                                               && pack.outC == pack.inC
                                               && pack.kernelW > 0
                                               && pack.kernelH > 0
                                               && pack.strideW > 0
                                               && pack.strideH > 0
                                               && pack.dilationW > 0
                                               && pack.dilationH > 0;

                // Legacy ForwardPack4 and layer-repro Cmd paths share immutable scalar
                // OIHW uploads for group/tail-safe Pack4 convolution dispatch.
                phaseSw.Restart();
                UploadRawConvWeights(pack, w, b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                if (needGeneralTexturePack)
                {
                    phaseSw.Restart();
                    var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(pack.packedWeight4, w4.Length, sizeof(float) * 4, "NcnnRepro.ConvPackedWeight4:" + layer.name);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "NcnnRepro.ConvPackedBias4:" + layer.name);
                    pack.packedWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);

                    if (EnableWinograd23
                        && pack.kernelW == 3
                        && pack.kernelH == 3
                        && pack.strideW == 1
                        && pack.strideH == 1
                        && pack.padLeft == 1
                        && pack.padRight == 1
                        && pack.padTop == 1
                        && pack.padBottom == 1
                        && NcnnWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                    {
                        pack.useWinograd23 = true;
                        var wTm = NcnnWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                        pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        NcnnGpuResourceTracker.RegisterBuffer(pack.packedWeightTm23, wTm.Length, sizeof(float) * 4, "NcnnRepro.ConvPackedWeightTm23:" + layer.name);
                        pack.packedWeightTm23.SetData(wTm);
                    }
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }
                else if (needDepthWiseTexturePack)
                {
                    phaseSw.Restart();
                    var w4 = PackDepthWiseWeightsToP4KhKw(w, pack.outC, pack.kernelW, pack.kernelH, pack.outPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(pack.packedDepthWiseWeight4, w4.Length, sizeof(float) * 4, "NcnnRepro.ConvPackedDepthWiseWeight4:" + layer.name);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "NcnnRepro.ConvPackedBias4:" + layer.name);
                    pack.packedDepthWiseWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }

                _conv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Deconvolution)
            {
                var pack = new DeconvPack();
                pack.outC = layer.GetInt(0, 0);
                pack.group = Mathf.Max(1, layer.GetInt(7, 1));
                pack.kernelW = layer.GetInt(1, 0);
                pack.kernelH = layer.GetInt(11, pack.kernelW);
                pack.dilationW = layer.GetInt(2, 1);
                pack.dilationH = layer.GetInt(12, pack.dilationW);
                pack.strideW = layer.GetInt(3, 1);
                pack.strideH = layer.GetInt(13, pack.strideW);
                pack.padLeft = layer.GetInt(4, 0);
                pack.padRight = layer.GetInt(15, pack.padLeft);
                pack.padTop = layer.GetInt(14, pack.padLeft);
                pack.padBottom = layer.GetInt(16, pack.padTop);
                pack.outputPadRight = layer.GetInt(18, 0);
                pack.outputPadBottom = layer.GetInt(19, pack.outputPadRight);
                pack.biasTerm = layer.GetInt(5, 0);
                pack.weightSize = layer.GetInt(6, 0);
                pack.activationType = layer.GetInt(9, 0);
                pack.activationSlope = ParseLeakySlope(layer);

                var kernelArea = Mathf.Max(1, pack.kernelW * pack.kernelH);
                pack.inC = Mathf.Max(1, (pack.weightSize * pack.group) / Mathf.Max(1, pack.outC * kernelArea));

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, pack.weightSize, layer.name);
                var b = pack.biasTerm != 0 ? br.ReadFloat32Array(pack.outC) : new float[pack.outC];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                pack.rawWeight = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(pack.rawWeight, w.Length, sizeof(float), "NcnnRepro.DeconvRawWeight:" + layer.name);
                pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(pack.rawBias, b.Length, sizeof(float), "NcnnRepro.DeconvRawBias:" + layer.name);
                pack.rawWeight.SetData(w);
                pack.rawBias.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _deconv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.InnerProduct)
            {
                var ip = new InnerProductPack();
                ip.outFeatures = layer.GetInt(0, 0);
                ip.biasTerm = layer.GetInt(1, 0);
                ip.weightSize = layer.GetInt(2, 0);
                ip.inFeatures = ip.outFeatures > 0 ? ip.weightSize / ip.outFeatures : 0;

                phaseSw.Restart();
                var w = ReadPackedOrRawWeightArray(br, ip.weightSize, layer.name);
                var b = ip.biasTerm != 0 ? br.ReadFloat32Array(ip.outFeatures) : new float[ip.outFeatures];
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(ip.w, w.Length, sizeof(float), "NcnnRepro.InnerProductWeight:" + layer.name);
                ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(ip.b, b.Length, sizeof(float), "NcnnRepro.InnerProductBias:" + layer.name);
                ip.w.SetData(w);
                ip.b.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _innerProduct[layer.name] = ip;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Gemm)
            {
                var gp = new GemmPack
                {
                    alpha = layer.GetFloat(0, 1f),
                    beta = layer.GetFloat(1, 1f),
                    transA = layer.GetInt(2, 0) != 0,
                    transB = layer.GetInt(3, 0) != 0,
                    constantA = layer.GetInt(4, 0) != 0,
                    constantB = layer.GetInt(5, 0) != 0,
                    constantC = layer.GetInt(6, 0) != 0,
                    constantM = layer.GetInt(7, 0),
                    constantN = layer.GetInt(8, 0),
                    constantK = layer.GetInt(9, 0),
                    broadcastTypeC = layer.GetInt(10, 0)
                };

                if (gp.constantA)
                    throw new InvalidOperationException("Gemm constantA is not supported in NcnnRepro: " + layer.name);
                if (!gp.constantB)
                    throw new InvalidOperationException("Gemm currently expects constantB=1: " + layer.name);

                var bw = gp.transB ? gp.constantK : gp.constantN;
                var bh = gp.transB ? gp.constantN : gp.constantK;

                phaseSw.Restart();
                var b = ReadClipMatAsFloat32(br, bw, bh, 0, 0, 0);
                gp.bDataCpu = b;
                if (gp.constantC && gp.broadcastTypeC != -1)
                {
                    int cw;
                    int ch;
                    switch (gp.broadcastTypeC)
                    {
                        case 0: cw = 1; ch = 0; break;
                        case 1: cw = gp.constantM; ch = 0; break;
                        case 2: cw = 1; ch = gp.constantM; break;
                        case 3: cw = gp.constantN; ch = gp.constantM; break;
                        case 4: cw = gp.constantN; ch = 1; break;
                        default:
                            throw new InvalidOperationException("Gemm broadcast_type_C unsupported: " + gp.broadcastTypeC + " | " + layer.name);
                    }

                    var c = ReadClipMatAsFloat32(br, cw, ch, 0, 0, 0);
                    gp.cDataCpu = c;
                }
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                gp.bData = NewBuffer(gp.bDataCpu);
                if (gp.cDataCpu != null)
                    gp.cData = NewBuffer(gp.cDataCpu);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _gemm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.MemoryData)
            {
                var w = layer.GetInt(0, 0);
                var h = layer.GetInt(1, 0);
                var d = layer.GetInt(11, 0);
                var c = layer.GetInt(2, 0);
                var loadType = layer.GetInt(21, 1);

                phaseSw.Restart();
                var a = ReadClipMatAsFloat32(br, w, h, d, c, loadType);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                var buf = new ComputeBuffer(a.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(buf, a.Length, sizeof(float), "NcnnRepro.MemoryData:" + layer.name);
                buf.SetData(a);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                var dims = 1;
                if (h > 0) dims = 2;
                if (c > 0) dims = d > 0 ? 4 : 3;
                _memoryData[layer.name] = new MemoryDataPack
                {
                    data = buf,
                    dims = dims,
                    w = Mathf.Max(1, w),
                    h = Mathf.Max(1, h),
                    d = Mathf.Max(1, d),
                    c = Mathf.Max(1, c)
                };
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.Embed)
            {
                var ep = new EmbedPack();
                ep.numOutput = layer.GetInt(0, 0);
                ep.inputDim = layer.GetInt(1, 0);
                ep.biasTerm = layer.GetInt(2, 0);
                ep.weightSize = layer.GetInt(3, 0);

                phaseSw.Restart();
                var w = ReadClipArrayAsFloat32(br, ep.weightSize, 0);
                float[] b = null;
                if (ep.biasTerm != 0)
                    b = br.ReadNcnnMatAsFloat32(ep.numOutput, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(ep.w, w.Length, sizeof(float), "NcnnRepro.EmbedWeight:" + layer.name);
                ep.w.SetData(w);
                if (b != null)
                {
                    ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(ep.b, b.Length, sizeof(float), "NcnnRepro.EmbedBias:" + layer.name);
                    ep.b.SetData(b);
                }
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _embed[layer.name] = ep;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.LayerNorm)
            {
                var lp = new LayerNormPack();
                lp.affineSize = layer.GetInt(0, 0);
                lp.eps = layer.GetFloat(1, 1e-5f);
                lp.affine = layer.GetInt(2, 1) != 0;

                float[] gamma = null;
                float[] beta = null;
                if (lp.affine && lp.affineSize > 0)
                {
                    phaseSw.Restart();
                    gamma = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    beta = br.ReadNcnnMatAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(lp.gamma, gamma.Length, sizeof(float), "NcnnRepro.LayerNormGamma:" + layer.name);
                    lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(lp.beta, beta.Length, sizeof(float), "NcnnRepro.LayerNormBeta:" + layer.name);
                    lp.gamma.SetData(gamma);
                    lp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _layerNorm[layer.name] = lp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.GroupNorm)
            {
                var gp = new GroupNormPack();
                gp.group = layer.GetInt(0, 1);
                gp.channels = layer.GetInt(1, 0);
                gp.eps = layer.GetFloat(2, 1e-5f);
                gp.affine = layer.GetInt(3, 1) != 0;

                float[] gamma = null;
                float[] beta = null;
                if (gp.affine && gp.channels > 0)
                {
                    phaseSw.Restart();
                    gamma = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                    beta = br.ReadNcnnMatAsFloat32(gp.channels, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(gp.gamma, gamma.Length, sizeof(float), "NcnnRepro.GroupNormGamma:" + layer.name);
                    gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    NcnnGpuResourceTracker.RegisterBuffer(gp.beta, beta.Length, sizeof(float), "NcnnRepro.GroupNormBeta:" + layer.name);
                    gp.gamma.SetData(gamma);
                    gp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _groupNorm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.BatchNorm)
            {
                var bp = new BatchNormPack();
                bp.channels = layer.GetInt(0, 0);

                phaseSw.Restart();
                var slope = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var mean = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var variance = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                var bias = br.ReadNcnnMatAsFloat32(bp.channels, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                var eps = layer.GetFloat(1, 0f);
                var a = new float[bp.channels];
                var b = new float[bp.channels];
                for (var i = 0; i < bp.channels; i++)
                {
                    var sqrtVar = Mathf.Sqrt(variance[i] + eps);
                    if (Mathf.Abs(sqrtVar) < 1e-8f)
                        sqrtVar = 1e-4f;
                    b[i] = slope[i] / sqrtVar;
                    a[i] = bias[i] - slope[i] * mean[i] / sqrtVar;
                }

                phaseSw.Restart();
                var packs = (bp.channels + 3) / 4;
                var a4 = PackBiasToO4(a, bp.channels, packs);
                var b4 = PackBiasToO4(b, bp.channels, packs);
                bp.biasA4 = new ComputeBuffer(a4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(bp.biasA4, a4.Length, sizeof(float) * 4, "NcnnRepro.BatchNormBiasA4:" + layer.name);
                bp.scaleB4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                NcnnGpuResourceTracker.RegisterBuffer(bp.scaleB4, b4.Length, sizeof(float) * 4, "NcnnRepro.BatchNormScaleB4:" + layer.name);
                bp.biasA4.SetData(a4);
                bp.scaleB4.SetData(b4);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
                packMs += phaseSw.ElapsedMilliseconds;

                _batchNorm[layer.name] = bp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == NcnnLayerTypes.MultiHeadAttention)
            {
                var mp = new MultiHeadAttentionPack();
                mp.embedDim = layer.GetInt(0, 0);
                mp.numHeads = layer.GetInt(1, 1);
                mp.weightDataSize = layer.GetInt(2, 0);
                mp.kdim = layer.GetInt(3, mp.embedDim);
                mp.vdim = layer.GetInt(4, mp.embedDim);
                mp.scale = layer.GetFloat(6, 1f / Mathf.Sqrt(Mathf.Max(1, mp.embedDim / Mathf.Max(1, mp.numHeads))));
                mp.qdim = mp.embedDim > 0 ? mp.weightDataSize / Mathf.Max(1, mp.embedDim) : 0;

                phaseSw.Restart();
                var qW = ReadClipMatAsFloat32(br, mp.embedDim * mp.qdim, 0, 0, 0, 0);
                var qB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var kW = ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                var kB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var vW = ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                var vB = br.ReadNcnnMatAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var oW = ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                var oB = br.ReadNcnnMatAsFloat32(mp.qdim, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                mp.qW = NewBuffer(qW);
                mp.qB = NewBuffer(qB);
                mp.kW = NewBuffer(kW);
                mp.kB = NewBuffer(kB);
                mp.vW = NewBuffer(vW);
                mp.vB = NewBuffer(vB);
                mp.oW = NewBuffer(oW);
                mp.oB = NewBuffer(oB);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _multiHeadAttention[layer.name] = mp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
        }

        private static void AccumulateLayerProfile(ModelLoadProfile profile, string layerType, LayerLoadMetrics metrics, long totalMs)
        {
            if (profile == null)
                return;

            profile.totalBytesRead += metrics.bytesRead;
            layerType = string.IsNullOrWhiteSpace(layerType) ? "Unknown" : layerType;
            if (!profile.layerTypes.TryGetValue(layerType, out var typeProfile) || typeProfile == null)
            {
                typeProfile = new LayerTypeLoadProfile();
                profile.layerTypes[layerType] = typeProfile;
            }

            typeProfile.count++;
            typeProfile.totalMs += totalMs;
            typeProfile.bytesRead += metrics.bytesRead;
            typeProfile.readMs += metrics.readMs;
            typeProfile.uploadMs += metrics.uploadMs;
            typeProfile.packMs += metrics.packMs;
        }

        public InferResult Infer(RenderTexture inputPack4, int inputPacks, string inputBlobName = "input", ICollection<string> pinnedNames = null)
        {
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");

            var textureInputs = new Dictionary<string, RenderTexture>(StringComparer.Ordinal)
            {
                { inputBlobName, inputPack4 }
            };
            return InferWithMultiInputs(textureInputs, null, pinnedNames);
        }

        public RenderTexture ForwardPack4(RenderTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            using var infer = Infer(inputPack4, inputPacks, inputBlobName, pinnedNames);
            return infer.ExtractTexture(ResolveDefaultOutputBlobName());
        }

        public InferResult InferFromBuffers(Dictionary<string, ComputeBuffer> inputBuffers, string stopAfterTopName = null)
        {
            if (inputBuffers == null || inputBuffers.Count == 0)
                throw new ArgumentNullException(nameof(inputBuffers));

            var bufferInputs = new Dictionary<string, NcnnTensorBuffer>(StringComparer.Ordinal);
            foreach (var kv in inputBuffers)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("inputBuffers[\"" + kv.Key + "\"]");
                bufferInputs[kv.Key] = new NcnnTensorBuffer(kv.Value, 1, kv.Value.count, 1, 1, 1, false);
            }

            return InferWithMultiInputs(null, bufferInputs, null, null, stopAfterTopName);
        }

        public InferResult InferWithMultiInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, NcnnTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null)
        {
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if ((textureInputs == null || textureInputs.Count == 0) && (bufferInputs == null || bufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
                return InferWithMultiInputsByLayerRepros(textureInputs, bufferInputs, pinnedNames, textureInputShapes, stopAfterTopName, startAtTopName);

            return null;
        }

        private string ResolveDefaultOutputBlobName()
        {
            if (Model != null && Model.layers != null)
            {
                for (var i = Model.layers.Count - 1; i >= 0; i--)
                {
                    var topNames = Model.layers[i]?.topNames;
                    if (topNames != null && topNames.Length > 0 && !string.IsNullOrWhiteSpace(topNames[0]))
                        return topNames[0];
                }
            }

            return "output";
        }

        public ComputeTexture ForwardPack4(CommandBuffer cmd, ComputeTexture inputPack4, int inputPacks, string inputBlobName = "data", ICollection<string> pinnedNames = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
            {
                var fallbackChannels = string.Equals(inputBlobName, "data", StringComparison.OrdinalIgnoreCase) ? 3 : inputPacks * 4;
                var inputLogicalShape = new BufferShape(3, inputPack4.width, inputPack4.height, 1, ResolveInputLogicalChannels(inputBlobName, fallbackChannels));
                return ForwardPack4(cmd, inputPack4, inputLogicalShape, out _, inputBlobName, pinnedNames);
            }

            var remaining = new Dictionary<string, int>(_blobUseCount, StringComparer.Ordinal);
            var legacyFallbackChannels = string.Equals(inputBlobName, "data", StringComparison.OrdinalIgnoreCase) ? 3 : inputPacks * 4;
            var legacyInputLogicalShape = new BufferShape(3, inputPack4.width, inputPack4.height, 1, ResolveInputLogicalChannels(inputBlobName, legacyFallbackChannels));
            var blobs = new Dictionary<string, CmdTensorRef>(StringComparer.Ordinal)
            {
                [inputBlobName] = new CmdTensorRef
                {
                    texture = inputPack4,
                    width = inputPack4.width,
                    height = inputPack4.height,
                    packs = inputPacks,
                    refs = 1,
                    owned = false,
                    hasLogicalShape = true,
                    logicalShape = legacyInputLogicalShape,
                    hasStorageShape = true,
                    storageShape = legacyInputLogicalShape
                }
            };

            for (var li = 0; li < Model.layers.Count; li++)
            {
                var l = Model.layers[li];
                if (l.type == NcnnLayerTypes.Input)
                    continue;

                if (l.type == NcnnLayerTypes.Split)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    for (var i = 0; i < l.topNames.Length; i++)
                    {
                        blobs[l.topNames[i]] = src;
                        src.refs++;
                    }
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Concat)
                {
                    var parts = new CmdTensorRef[l.bottomNames.Length];
                    var sumP = 0;
                    var w = 0;
                    var h = 0;
                    for (var i = 0; i < l.bottomNames.Length; i++)
                    {
                        var tr = GetCmdTensor(blobs, l.bottomNames[i]);
                        parts[i] = tr;
                        w = tr.width;
                        h = tr.height;
                        sumP += tr.packs;
                    }

                    var outArr = RentTempArray(cmd, w, h, sumP, RenderTextureFormat.ARGBHalf);
                    var off = 0;
                    for (var i = 0; i < parts.Length; i++)
                    {
                        _ops.CopyPack4(cmd, parts[i].texture, 0, outArr, off, parts[i].packs);
                        off += parts[i].packs;
                    }

                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = w, height = h, packs = sumP, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Reshape)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    blobs[l.topNames[0]] = src;
                    src.refs++;
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Padding)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var top = l.GetInt(0, 0);
                    var bottom = l.GetInt(1, 0);
                    var left = l.GetInt(2, 0);
                    var right = l.GetInt(3, 0);
                    var type = l.GetInt(4, 0);
                    var value = l.GetFloat(5, 0f);

                    var outW = src.width + left + right;
                    var outH = src.height + top + bottom;
                    if (outW <= 0 || outH <= 0)
                        throw new InvalidOperationException("Padding invalid out size: " + outW + "x" + outH);

                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PaddingPack4(cmd, src.texture, src.packs, left, right, top, bottom, type, new Vector4(value, value, value, value), outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Pooling)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var poolingType = l.GetInt(0, 0);
                    var kernelW = l.GetInt(1, 0);
                    var kernelH = l.GetInt(11, kernelW);
                    var strideW = l.GetInt(2, 1);
                    var strideH = l.GetInt(12, strideW);
                    var padLeft = l.GetInt(3, 0);
                    var padTop = l.GetInt(13, padLeft);
                    var globalPooling = l.GetInt(4, 0);
                    var adaptivePooling = l.GetInt(7, 0);
                    if (globalPooling != 0 || adaptivePooling != 0)
                        throw new InvalidOperationException("Pooling(global/adaptive) not supported");

                    var outW = (src.width + padLeft * 2 - kernelW) / strideW + 1;
                    var outH = (src.height + padTop * 2 - kernelH) / strideH + 1;
                    outW = Mathf.Max(1, outW);
                    outH = Mathf.Max(1, outH);
                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Softmax)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var axis = l.GetInt(0, 0);
                    if (axis != 0)
                        throw new InvalidOperationException("Softmax axis not supported: " + axis);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SoftmaxChannelPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Convolution || l.type == NcnnLayerTypes.ConvolutionDepthWise)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    if (!_conv.TryGetValue(l.name, out var conv))
                        throw new InvalidOperationException("Convolution not found: " + l.name);
                    if (src.packs != conv.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + conv.inPacks);
                    if (!TryValidateCommandBuffer2dConvProfile(conv, out var profileReason))
                        throw new InvalidOperationException("CommandBuffer convolution profile rejected: " + l.name + " | " + profileReason);

                    var outW = ComputeConvOut(src.width, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
                    var outH = ComputeConvOut(src.height, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom);
                    var outArr = RentTempArray(cmd, outW, outH, conv.outPacks, RenderTextureFormat.ARGBHalf);
                    _ops.Conv2dGroupPack4(
                        cmd, src.texture, conv.rawWeight, conv.rawBias, conv.inC, conv.outC, conv.group,
                        conv.kernelW, conv.kernelH, conv.strideW, conv.strideH, conv.padLeft, conv.padTop,
                        conv.dilationW, conv.dilationH, conv.activationType, conv.activationSlope, outArr);

                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = conv.outPacks, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Deconvolution || l.type == NcnnLayerTypes.DeconvolutionDepthWise)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    if (!_deconv.TryGetValue(l.name, out var deconv))
                        throw new InvalidOperationException("Deconvolution not found: " + l.name);
                    if (src.packs != deconv.inPacks)
                        throw new InvalidOperationException("unexpected in packs for " + l.name + ": " + src.packs + " vs " + deconv.inPacks);
                    if (!TryValidateCommandBuffer2dDeconvProfile(deconv, out var profileReason))
                        throw new InvalidOperationException("CommandBuffer deconvolution profile rejected: " + l.name + " | " + profileReason);

                    var outW = ComputeDeconvOut(src.width, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
                    var outH = ComputeDeconvOut(src.height, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom);
                    var outArr = RentTempArray(cmd, outW, outH, deconv.outPacks, RenderTextureFormat.ARGBHalf);
                    _ops.Deconvolution2dGroupPack4(
                        cmd, src.texture, deconv.rawWeight, deconv.rawBias, deconv.inC, deconv.outC, deconv.group,
                        deconv.kernelW, deconv.kernelH, deconv.strideW, deconv.strideH, deconv.padLeft, deconv.padTop,
                        deconv.dilationW, deconv.dilationH, deconv.activationType, deconv.activationSlope, outArr);

                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = deconv.outPacks, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Eltwise)
                {
                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                    var b = GetCmdTensor(blobs, l.bottomNames[1]);
                    var coeff = ParseEltwiseCoeff(l);
                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    _ops.AddPack4(cmd, a.texture, b.texture, coeff.coeffA, coeff.coeffB, a.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.BinaryOp)
                {
                    var opType = l.GetInt(0, 0);
                    var withScalar = l.GetInt(1, 0);
                    var scalarB = l.GetFloat(2, 0f);
                    var a = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, a.width, a.height, a.packs, RenderTextureFormat.ARGBHalf);
                    if (withScalar != 0)
                    {
                        _ops.BinaryOpScalarPack4(cmd, a.texture, scalarB, a.packs, opType, outArr);
                    }
                    else
                    {
                        var b = GetCmdTensor(blobs, l.bottomNames[1]);
                        if (a.width != b.width || a.height != b.height || a.packs != b.packs)
                            throw new InvalidOperationException("BinaryOp broadcast not supported: " + l.name);
                        _ops.BinaryOpPack4(cmd, a.texture, b.texture, a.packs, opType, outArr);
                    }
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = a.width, height = a.height, packs = a.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.UnaryOp)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var opType = l.GetInt(0, 0);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.UnaryOpPack4(cmd, src.texture, src.packs, opType, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Swish)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SwishPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Sigmoid)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SigmoidPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.GELU)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var fast = l.GetInt(0, 0) != 0;
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.GeluPack4(cmd, src.texture, src.packs, fast, outArr);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width, height = src.height, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == NcnnLayerTypes.Interp)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var resizeType = l.GetInt(0, 2);
                    var sx = l.GetFloat(1, 1f);
                    var sy = l.GetFloat(2, 1f);

                    if (Mathf.Abs(sx - 2f) < 1e-3f && Mathf.Abs(sy - 2f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.width * 2, src.height * 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.Interp2xNearestPack4(cmd, src.texture, src.packs, outArr);
                        else
                            _ops.Interp2xPack4(cmd, src.texture, src.packs, outArr);
                        blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width * 2, height = src.height * 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    if (Mathf.Abs(sx - 0.5f) < 1e-3f && Mathf.Abs(sy - 0.5f) < 1e-3f)
                    {
                        var outArr = RentTempArray(cmd, src.width / 2, src.height / 2, src.packs, RenderTextureFormat.ARGBHalf);
                        if (resizeType == 1)
                            _ops.InterpDown2NearestPack4(cmd, src.texture, src.packs, outArr);
                        else
                            _ops.InterpDown2Pack4(cmd, src.texture, src.packs, outArr);
                        blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = src.width / 2, height = src.height / 2, packs = src.packs, refs = 1, owned = true };
                        ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                        continue;
                    }

                    throw new InvalidOperationException("unsupported interp scale: " + sx.ToString("0.###", CultureInfo.InvariantCulture) + "," + sy.ToString("0.###", CultureInfo.InvariantCulture));
                }

                throw new InvalidOperationException("unsupported layer type in CommandBuffer path: " + (l.typeName ?? l.type.ToString()));
            }

            var outBlobName = ResolveDefaultOutputBlobName();
            var outRef = GetCmdTensor(blobs, outBlobName);
            var keep = outRef.texture;
            DetachReturnedCmdTextureOwnership(blobs, keep);

            var visited = new HashSet<CmdTensorRef>();
            foreach (var kv in blobs)
            {
                var tr = kv.Value;
                if (tr == null || !visited.Add(tr))
                    continue;
                if (tr.owned && tr.texture != null)
                    ReturnTempArray(cmd, tr.texture);
            }

            return keep;
        }

        public ComputeTexture ForwardPack4(
            CommandBuffer cmd,
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            out BufferShape outputLogicalShape,
            ICollection<string> pinnedNames = null,
            string outputBlobName = null,
            string stopAfterTopName = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (textureInputs == null || textureInputs.Count == 0)
                throw new ArgumentNullException(nameof(textureInputs));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
            {
                return ForwardPack4ByLayerRepros(
                    cmd,
                    textureInputs,
                    textureInputShapes,
                    out outputLogicalShape,
                    pinnedNames,
                    outputBlobName,
                    stopAfterTopName);
            }

            outputLogicalShape = default;
            return null;
        }

        public ComputeTexture ForwardPack4(
            CommandBuffer cmd,
            ComputeTexture inputPack4,
            BufferShape inputLogicalShape,
            out BufferShape outputLogicalShape,
            string inputBlobName = "data",
            ICollection<string> pinnedNames = null,
            string stopAfterTopName = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (inputPack4 == null)
                throw new ArgumentNullException(nameof(inputPack4));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            var textureInputs = new Dictionary<string, ComputeTexture>(StringComparer.Ordinal)
            {
                [inputBlobName] = inputPack4
            };
            var textureInputShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal)
            {
                [inputBlobName] = inputLogicalShape
            };
            return ForwardPack4(
                cmd,
                textureInputs,
                textureInputShapes,
                out outputLogicalShape,
                pinnedNames,
                null,
                stopAfterTopName);
        }

        internal static CmdTensorRef GetCmdTensor(Dictionary<string, CmdTensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null)
                throw new InvalidOperationException("blob not found: " + name);
            EnsureRepoVkTensor(tr);
            return tr;
        }

        internal static BufferShape InferCmdShape(CmdTensorRef tensor)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));
            return GetCmdTensorContract(tensor).LogicalShape;
        }

        internal static bool TryGetCmdShape(
            Dictionary<string, BufferShape> shapes,
            Dictionary<string, CmdTensorRef> blobs,
            string name,
            out BufferShape shape)
        {
            if (shapes != null && shapes.TryGetValue(name, out shape))
                return true;

            if (blobs != null && blobs.TryGetValue(name, out var tensor) && tensor != null)
            {
                EnsureRepoVkTensor(tensor);
                shape = InferCmdShape(tensor);
                return true;
            }

            shape = default;
            return false;
        }

        internal static BufferShape GetCmdShape(
            Dictionary<string, BufferShape> shapes,
            Dictionary<string, CmdTensorRef> blobs,
            string name)
        {
            if (TryGetCmdShape(shapes, blobs, name, out var shape))
                return shape;
            throw new InvalidOperationException("cmd blob shape not found: " + name);
        }

        internal void ConsumeCmd(
            CommandBuffer cmd,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames,
            Dictionary<string, BufferShape> shapes = null)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var b = bottomNames[i];
                if (!remaining.TryGetValue(b, out var c))
                    continue;
                c--;
                remaining[b] = c;
                if (c > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(b))
                    continue;

                if (blobs.TryGetValue(b, out var tr) && tr != null)
                {
                    ReleaseCmdTensorRef(cmd, tr);
                }
                blobs.Remove(b);
                shapes?.Remove(b);
            }
        }

        private static string GetTempBufferLabel(string member, int line)
        {
            return "NcnnRepro.RentTempBuffer(" + (member ?? "?") + ":" + line.ToString(CultureInfo.InvariantCulture) + ")";
        }

        internal ComputeBuffer RentTempBuffer(
            int count,
            int stride,
            ComputeBufferType type = ComputeBufferType.Structured,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            ValidateTempBufferAllowed(count, stride);
            TrackTempBufferRent(count, stride);
            var buffer = new ComputeBuffer(Mathf.Max(1, count), Math.Max(1, stride), type);
            NcnnGpuResourceTracker.RegisterBuffer(buffer, Mathf.Max(1, count), Math.Max(1, stride), GetTempBufferLabel(callerMember, callerLine));
            return buffer;
        }

        internal void ReturnTempBuffer(ComputeBuffer buffer)
        {
            TrackTempBufferReturn(buffer);
            if (buffer == null)
                return;
            NcnnGpuResourceTracker.ReleaseBuffer(buffer, "NcnnRepro.ReturnTempBuffer");
            try { buffer.Dispose(); } catch { }
        }

        internal NcnnTensorBuffer RentTempTensorBuffer(
            int dims,
            int w,
            int h = 1,
            int d = 1,
            int c = 1,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            var count = checked(w * h * d * c);
            var buffer = RentTempBuffer(count, sizeof(float), ComputeBufferType.Structured, callerMember, callerLine);
            return new NcnnTensorBuffer(buffer, dims, w, h, d, c, true, ReturnTempBuffer);
        }

        public RenderTexture RentTempArray(
            int w,
            int h,
            int depth,
            RenderTextureFormat format,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);
            if (format == RenderTextureFormat.ARGBHalf)
                format = TensorTextureFormat;
            if (WouldExceedTextureArraySliceLimit(depth))
            {
                throw new InvalidOperationException(
                    "Texture2DArray slice limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_depth=" + depth.ToString(CultureInfo.InvariantCulture)
                    + " | max_slices=" + GetMaxTextureArraySlices().ToString(CultureInfo.InvariantCulture)
                    + " | size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture));
            }

            var allocLabel = "NcnnRepro.RentTempArray(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                msaaSamples = 1,
            };
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(3, w, h, 1, depth * 4),
                new BufferShape(3, w, h, 1, depth * 4),
                desc,
                allocLabel);
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            var allocated = RenderTexture.GetTemporary(desc);
            NcnnGpuResourceTracker.RegisterTemporaryTexture(allocated, temporaryDescriptor);
            _ops?.FillScalarTexture(null, allocated);
            TrackTempRtRent();
            return allocated;

        }

        public RenderTexture RentTempMat(
            int w,
            int h,
            RenderTextureFormat format,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            format = format == RenderTextureFormat.ARGBHalf ? ResolveLinearMatTextureFormat() : format;
            if (WouldExceedTextureSizeLimit(w, h))
            {
                throw new InvalidOperationException(
                    "Texture2D size limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture)
                    + " | max_size=" + GetMaxTextureSize().ToString(CultureInfo.InvariantCulture));
            }

            var allocLabel = "NcnnRepro.RentTempMat(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";
            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1,
            };
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(2, w, h, 1, 1),
                new BufferShape(2, w, h, 1, 1),
                desc,
                allocLabel);
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            var allocated = RenderTexture.GetTemporary(desc);
            NcnnGpuResourceTracker.RegisterTemporaryTexture(allocated, temporaryDescriptor);
            _ops?.FillScalarTexture(null, allocated);
            TrackTempRtRent();
            return allocated;
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;
            TrackTempRtReturn();
            NcnnGpuResourceTracker.ReleaseTexture(rt, "NcnnRepro.ReturnTempArray");
            RenderTexture.ReleaseTemporary(rt);
        }

        public ComputeTexture RentTempArray(CommandBuffer cmd, int w, int h, int depth, RenderTextureFormat format)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));

            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);
            if (format == RenderTextureFormat.ARGBHalf)
                format = TensorTextureFormat;
            if (WouldExceedTextureArraySliceLimit(depth))
            {
                throw new InvalidOperationException(
                    "Texture2DArray slice limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_depth=" + depth.ToString(CultureInfo.InvariantCulture)
                    + " | max_slices=" + GetMaxTextureArraySlices().ToString(CultureInfo.InvariantCulture)
                    + " | size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture));
            }

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2DArray,
                volumeDepth = depth,
                enableRandomWrite = true,
                msaaSamples = 1,
            };

            var id = Shader.PropertyToID(Guid.NewGuid().ToString());
            var allocLabel = "NcnnRepro.RentTempArrayCmd(" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture) + "x" + depth.ToString(CultureInfo.InvariantCulture) + ")";
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(3, w, h, 1, depth * 4),
                new BufferShape(3, w, h, 1, depth * 4),
                desc,
                allocLabel);
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            cmd.GetTemporaryRT(id, desc);
            NcnnGpuResourceTracker.RegisterTemporaryTextureHandle(id, temporaryDescriptor);
            var t = new ComputeTexture
            {
                nameID = id,
                width = w,
                height = h,
                depth = depth,
                dimension = TextureDimension.Tex2DArray,
                format = format,
                trackerLabel = allocLabel,
                isTemporary = true,
                temporaryDescriptor = temporaryDescriptor
            };
            _ops?.FillScalarTexture(cmd, null, t);
            TrackTempRtRent();
            return t;
        }

        public ComputeTexture RentTempMat(CommandBuffer cmd, int w, int h, RenderTextureFormat format)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));

            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            format = format == RenderTextureFormat.ARGBHalf ? ResolveLinearMatTextureFormat() : format;
            if (WouldExceedTextureSizeLimit(w, h))
            {
                throw new InvalidOperationException(
                    "Texture2D size limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture)
                    + " | max_size=" + GetMaxTextureSize().ToString(CultureInfo.InvariantCulture));
            }

            var desc = new RenderTextureDescriptor(w, h, format, 0)
            {
                dimension = TextureDimension.Tex2D,
                volumeDepth = 1,
                enableRandomWrite = true,
                msaaSamples = 1,
            };

            var id = Shader.PropertyToID(Guid.NewGuid().ToString());
            var allocLabel = "NcnnRepro.RentTempMatCmd(" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture) + ")";
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(2, w, h, 1, 1),
                new BufferShape(2, w, h, 1, 1),
                desc,
                allocLabel);
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            cmd.GetTemporaryRT(id, desc);
            NcnnGpuResourceTracker.RegisterTemporaryTextureHandle(id, temporaryDescriptor);
            var t = new ComputeTexture
            {
                nameID = id,
                width = w,
                height = h,
                depth = 1,
                dimension = TextureDimension.Tex2D,
                format = format,
                trackerLabel = allocLabel,
                isTemporary = true,
                temporaryDescriptor = temporaryDescriptor
            };
            _ops?.FillScalarTexture(cmd, null, t);
            TrackTempRtRent();
            return t;
        }

        public void ReturnTempArray(CommandBuffer cmd, ComputeTexture t)
        {
            if (cmd == null || t == null || !t.isTemporary || t.isReleased)
                return;
            t.isReleased = true;
            TrackTempRtReturn();
            NcnnGpuResourceTracker.ReleaseTextureHandle(t.nameID, t.trackerLabel ?? "NcnnRepro.ReturnTempArrayCmd");
            cmd.ReleaseTemporaryRT(t.nameID);
        }

        internal NcnnTemporaryRtDescriptor CreateTemporaryRtDescriptor(
            TensorDescriptor tensorDescriptor,
            RenderTextureDescriptor renderTextureDescriptor,
            string label)
        {
            if (tensorDescriptor == null)
                throw new ArgumentNullException(nameof(tensorDescriptor));

            return CreateTemporaryRtDescriptor(
                tensorDescriptor.LogicalShape,
                tensorDescriptor.StorageShape,
                renderTextureDescriptor,
                label);
        }

        private NcnnTemporaryRtDescriptor CreateTemporaryRtDescriptor(
            BufferShape logicalShape,
            BufferShape storageShape,
            RenderTextureDescriptor renderTextureDescriptor,
            string label)
        {
            return new NcnnTemporaryRtDescriptor(
                logicalShape,
                storageShape,
                renderTextureDescriptor,
                SessionId,
                DescribeCurrentExecutionSite(),
                label);
        }

        private void EnsureTemporaryTextureBudget(NcnnTemporaryRtDescriptor descriptor)
        {
            NcnnGpuResourceTracker.EnsureTemporaryTextureBudget(
                NcnnGpuResourceTracker.EstimateTemporaryTextureBytes(descriptor),
                TemporaryTextureBudgetBytes,
                descriptor.Node);
        }

        public void Release()
        {
            foreach (var kv in _conv) kv.Value?.Dispose();
            foreach (var kv in _deconv) kv.Value?.Dispose();
            foreach (var kv in _innerProduct) kv.Value?.Dispose();
            foreach (var kv in _gemm) kv.Value?.Dispose();
            foreach (var kv in _memoryData) kv.Value?.Dispose();
            foreach (var kv in _embed) kv.Value?.Dispose();
            foreach (var kv in _layerNorm) kv.Value?.Dispose();
            foreach (var kv in _groupNorm) kv.Value?.Dispose();
            foreach (var kv in _batchNorm) kv.Value?.Dispose();
            foreach (var kv in _multiHeadAttention) kv.Value?.Dispose();
            foreach (var kv in _extraPacks) kv.Value?.Dispose();

            _conv.Clear();
            _deconv.Clear();
            _innerProduct.Clear();
            _gemm.Clear();
            _memoryData.Clear();
            _embed.Clear();
            _layerNorm.Clear();
            _groupNorm.Clear();
            _batchNorm.Clear();
            _multiHeadAttention.Clear();
            _extraPacks.Clear();
            Model = null;
            LayerRepros = null;
            _blobUseCount = null;
        }

        public void Dispose()
        {
            if (_isDisposed)
                return;
            _isDisposed = true;
            Release();
        }

        internal static RenderTextureFormat ResolveTensorTextureFormat(int dims)
        {
            return dims >= 4 ? RenderTextureFormat.ARGBFloat : RenderTextureFormat.ARGBHalf;
        }

        internal static RenderTextureFormat ResolveLinearMatTextureFormat()
        {
            return RenderTextureFormat.RFloat;
        }

        internal static BufferShape ResolveLinearMatStorageShape(BufferShape logicalShape)
        {
            var logicalWidth = Mathf.Max(1, logicalShape.w);
            var logicalHeight = logicalShape.dims >= 2 ? Mathf.Max(1, logicalShape.h) : 1;
            if (logicalShape.dims <= 1)
                return new BufferShape(2, logicalWidth, 1, 1, 1);
            if (logicalShape.dims == 2)
                return new BufferShape(2, logicalWidth, logicalHeight, 1, 1);

            var logicalCount = Mathf.Max(1, logicalShape.w)
                * Mathf.Max(1, logicalShape.h)
                * Mathf.Max(1, logicalShape.d)
                * Mathf.Max(1, logicalShape.c);
            var widthTiled = logicalCount <= 4096
                ? logicalCount
                : AlignUp(Mathf.Min(SystemInfo.maxTextureSize, Mathf.CeilToInt(Mathf.Sqrt(logicalCount))), 16);
            widthTiled = Mathf.Clamp(Mathf.Max(1, widthTiled), 1, Mathf.Max(1, SystemInfo.maxTextureSize));
            var heightTiled = Mathf.CeilToInt(logicalCount / (float)widthTiled);
            if (heightTiled <= Mathf.Max(1, SystemInfo.maxTextureSize))
                return new BufferShape(2, widthTiled, Mathf.Max(1, heightTiled), 1, 1);
            return new BufferShape(2, logicalWidth, logicalHeight, 1, 1);
        }

        internal static BufferShape ResolvePack4LinearMatStorageShape(BufferShape logicalShape)
        {
            var logicalWidth = Mathf.Max(1, logicalShape.w);
            var logicalHeight = logicalShape.dims >= 2 ? Mathf.Max(1, logicalShape.h) : 1;
            return new BufferShape(3, Mathf.CeilToInt(logicalWidth / 4f), logicalHeight, 1, 4);
        }

        internal static bool IsStrictLinearMatTexture(TensorRef tensor)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.texture.dimension == TextureDimension.Tex2D
                && tensor.layoutKind == RepoVkTensorLayoutKind.LinearMat;
        }

        internal static bool IsPack4LinearMatTexture(TensorRef tensor, BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || logicalShape.dims != 2)
                return false;
            if (tensor.texture.dimension != TextureDimension.Tex2DArray || tensor.packs != 1)
                return false;
            var storageShape = GetTextureStorageShape(tensor, logicalShape);
            return storageShape.dims == 3
                && storageShape.w == Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f)
                && storageShape.h == Mathf.Max(1, logicalShape.h)
                && storageShape.d == 1
                && storageShape.c == 4
                && tensor.width == storageShape.w
                && tensor.height == storageShape.h;
        }

        internal static bool IsStrictLinearMatTexture(CmdTensorRef tensor)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.texture.dimension == TextureDimension.Tex2D
                && tensor.layoutKind == RepoVkTensorLayoutKind.LinearMat;
        }

        internal static bool IsPack4LinearMatTexture(CmdTensorRef tensor, BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || logicalShape.dims != 2)
                return false;
            if (tensor.texture.dimension != TextureDimension.Tex2DArray || tensor.packs != 1)
                return false;
            var storageShape = GetCmdStorageShape(tensor, logicalShape);
            return storageShape.dims == 3
                && storageShape.w == Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f)
                && storageShape.h == Mathf.Max(1, logicalShape.h)
                && storageShape.d == 1
                && storageShape.c == 4
                && tensor.width == storageShape.w
                && tensor.height == storageShape.h;
        }

        private static int AlignUp(int value, int alignment)
        {
            if (alignment <= 1)
                return Mathf.Max(1, value);
            var safeValue = Mathf.Max(1, value);
            return ((safeValue + alignment - 1) / alignment) * alignment;
        }

        private static RenderTextureFormat ResolveTensorTextureFormatWithOverride(int dims, RenderTextureFormat? formatOverride)
        {
            return formatOverride ?? ResolveTensorTextureFormat(dims);
        }

        internal static RepoVkTensorLayoutKind ResolveRepoVkLayoutKind(BufferShape logicalShape, BufferShape storageShape, int packs)
        {
            var effectivePacks = Mathf.Max(1, packs);
            if (storageShape.dims <= 2)
                return RepoVkTensorLayoutKind.LinearMat;
            if (storageShape.dims == 3 && Mathf.Max(1, storageShape.c) <= 1 && effectivePacks <= 1)
                return RepoVkTensorLayoutKind.LinearMat;
            if (logicalShape.dims <= 2 && effectivePacks <= 1 && Mathf.Max(1, storageShape.d) <= 1 && Mathf.Max(1, storageShape.c) <= 1)
                return RepoVkTensorLayoutKind.LinearMat;
            return RepoVkTensorLayoutKind.Pack4Image;
        }

        internal static RepoVkTensor CreateRepoVkTensor(
            RenderTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            int packs)
        {
            if (texture == null)
                return null;

            var layoutKind = texture.dimension == TextureDimension.Tex2D
                ? RepoVkTensorLayoutKind.LinearMat
                : ResolveRepoVkLayoutKind(logicalShape, storageShape, packs);

            if (layoutKind == RepoVkTensorLayoutKind.LinearMat)
                return new RepoVkMat(texture, logicalShape, storageShape, packs);

            var depth = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
            return new RepoVkImageMat(texture, logicalShape, storageShape, packs, depth);
        }

        internal static RepoVkTensor CreateRepoVkTensor(
            ComputeTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            int packs)
        {
            if (texture == null)
                return null;

            var layoutKind = texture.dimension == TextureDimension.Tex2D
                ? RepoVkTensorLayoutKind.LinearMat
                : ResolveRepoVkLayoutKind(logicalShape, storageShape, packs);
            if (layoutKind == RepoVkTensorLayoutKind.LinearMat)
                return new RepoVkMat(texture, logicalShape, storageShape, packs);

            var depth = Mathf.Max(1, texture.depth);
            return new RepoVkImageMat(texture, logicalShape, storageShape, packs, depth);
        }

        internal static bool BufferShapeEquals(BufferShape a, BufferShape b)
        {
            return a.dims == b.dims
                && a.w == b.w
                && a.h == b.h
                && a.d == b.d
                && a.c == b.c;
        }

        internal static int GetCmdTexturePackCount(BufferShape storageShape, ComputeTexture texture)
        {
            if (texture == null)
                return 1;

            if (texture.dimension == TextureDimension.Tex2D)
                return 1;
            if (storageShape.dims == 4)
                return Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));

            return Mathf.Max(1, texture.depth);
        }

        private static InferenceTensorDataType ResolveInferenceTensorDataType(RenderTextureFormat format)
        {
            switch (format)
            {
                case RenderTextureFormat.RHalf:
                case RenderTextureFormat.RGHalf:
                case RenderTextureFormat.ARGBHalf:
                    return InferenceTensorDataType.Float16;
                case RenderTextureFormat.RFloat:
                case RenderTextureFormat.RGFloat:
                case RenderTextureFormat.ARGBFloat:
                    return InferenceTensorDataType.Float32;
                default:
                    return InferenceTensorDataType.Unknown;
            }
        }

        private static TensorProvenance CreateTensorProvenance(string blobName, string debugName)
        {
            return new TensorProvenance("NcnnRepro", debugName, blobName, debugName);
        }

        private static TensorPacking ResolveTensorPacking(BufferShape storageShape, RepoVkTensorLayoutKind layout, int packCount)
        {
            var packSize = layout == RepoVkTensorLayoutKind.Pack4Image
                || (storageShape.dims >= 3 && Mathf.Max(1, storageShape.c) > 1)
                ? 4
                : 1;
            return new TensorPacking(packSize, packCount);
        }

        internal static void SyncTextureContractMetadata(
            TensorRef tensor,
            BufferShape logicalShape,
            BufferShape storageShape,
            TensorProvenance? provenance = null)
        {
            if (tensor == null || tensor.texture == null)
                return;

            if (tensor.IsDescriptorPublished)
                return;

            var packs = GetTexturePackCount(storageShape, tensor.texture);
            var layout = ResolveRepoVkLayoutKind(logicalShape, storageShape, packs);
            var nativeTensor = CreateRepoVkTensor(tensor.texture, logicalShape, storageShape, packs);
            var owner = ResolveTextureLifetimeOwner(tensor) ?? tensor;
            var ownerDescriptor = owner != null ? owner.Descriptor : null;
            var descriptor = new TensorDescriptor(
                logicalShape,
                storageShape,
                layout,
                ResolveTensorPacking(storageShape, layout, packs),
                ResolveInferenceTensorDataType(tensor.texture.format),
                TensorQuantizationMetadata.None,
                ownerDescriptor != null ? ownerDescriptor.AliasGroup : null,
                ReferenceEquals(owner, tensor)
                    ? (tensor.owned ? InferenceTensorLifetime.GraphOwned : InferenceTensorLifetime.ExternalInput)
                    : InferenceTensorLifetime.SharedAlias,
                owner,
                provenance ?? CreateTensorProvenance(string.Empty, tensor.texture.name),
                nativeTensor);
            tensor.PublishDescriptor(descriptor);
        }

        internal static void SyncCmdTensorContractMetadata(
            CmdTensorRef tensor,
            BufferShape logicalShape,
            BufferShape storageShape,
            TensorProvenance? provenance = null)
        {
            if (tensor == null || tensor.texture == null)
                return;

            if (tensor.IsDescriptorPublished)
                return;

            var packs = GetCmdTexturePackCount(storageShape, tensor.texture);
            var layout = ResolveRepoVkLayoutKind(logicalShape, storageShape, packs);
            var nativeTensor = CreateRepoVkTensor(tensor.texture, logicalShape, storageShape, packs);
            var owner = ResolveCmdTextureLifetimeOwner(tensor) ?? tensor;
            var ownerDescriptor = owner != null ? owner.Descriptor : null;
            var descriptor = new TensorDescriptor(
                logicalShape,
                storageShape,
                layout,
                ResolveTensorPacking(storageShape, layout, packs),
                ResolveInferenceTensorDataType(tensor.texture.format),
                TensorQuantizationMetadata.None,
                ownerDescriptor != null ? ownerDescriptor.AliasGroup : null,
                ReferenceEquals(owner, tensor)
                    ? (tensor.owned ? InferenceTensorLifetime.GraphOwned : InferenceTensorLifetime.ExternalInput)
                    : InferenceTensorLifetime.SharedAlias,
                owner,
                provenance ?? CreateTensorProvenance(string.Empty, tensor.texture.trackerLabel),
                nativeTensor);
            tensor.PublishDescriptor(descriptor);
        }

        public static TensorRef CreateTextureRef(
            RenderTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            bool owned,
            int refs = 1,
            TensorRef sharedTextureOwner = null,
            string blobName = null)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            var tensor = new TensorRef
            {
                texture = texture,
                refs = refs,
                owned = owned,
                sharedTextureOwner = sharedTextureOwner
            };
            SyncTextureContractMetadata(tensor, logicalShape, storageShape, CreateTensorProvenance(blobName, texture.name));
            return tensor;
        }

        public static CmdTensorRef CreateCmdTensorRef(
            ComputeTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            bool owned,
            int refs = 1,
            CmdTensorRef sharedTextureOwner = null,
            string blobName = null)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));

            var tensor = new CmdTensorRef
            {
                texture = texture,
                refs = refs,
                owned = owned,
                sharedTextureOwner = sharedTextureOwner
            };
            SyncCmdTensorContractMetadata(tensor, logicalShape, storageShape, CreateTensorProvenance(blobName, texture.trackerLabel));
            return tensor;
        }

        internal static TensorRef ResolveTextureLifetimeOwner(TensorRef tensor)
        {
            var current = tensor;
            while (current != null && current.sharedTextureOwner != null)
                current = current.sharedTextureOwner;
            return current;
        }

        internal static CmdTensorRef ResolveCmdTextureLifetimeOwner(CmdTensorRef tensor)
        {
            var current = tensor;
            while (current != null && current.sharedTextureOwner != null)
                current = current.sharedTextureOwner;
            return current;
        }

        public static TensorRef CreateTextureAlias(
            TensorRef source,
            BufferShape logicalShape,
            BufferShape storageShape)
        {
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            EnsureRepoVkTensor(source);
            var sourceDescriptor = source.Descriptor;
            var targetPacks = GetTexturePackCount(storageShape, source.texture);
            var targetLayout = ResolveRepoVkLayoutKind(logicalShape, storageShape, targetPacks);
            var targetPacking = ResolveTensorPacking(storageShape, targetLayout, targetPacks);
            var targetDataType = ResolveInferenceTensorDataType(source.texture.format);
            if (!sourceDescriptor.IsStorageLayoutCompatibleWith(storageShape, targetLayout, targetPacking, targetDataType))
                throw new TensorAliasTransformRequiredException(sourceDescriptor, logicalShape, storageShape);

            var lifetimeOwner = ResolveTextureLifetimeOwner(source) ?? source;
            lifetimeOwner.refs++;
            return CreateTextureRef(
                source.texture,
                logicalShape,
                storageShape,
                owned: false,
                refs: 1,
                sharedTextureOwner: lifetimeOwner,
                blobName: sourceDescriptor.Provenance.BlobName);
        }

        public static CmdTensorRef CreateCmdTensorAlias(
            CmdTensorRef source,
            BufferShape logicalShape,
            BufferShape storageShape)
        {
            if (source == null || source.texture == null)
                throw new ArgumentNullException(nameof(source));

            EnsureRepoVkTensor(source);
            var sourceDescriptor = source.Descriptor;
            var targetPacks = GetCmdTexturePackCount(storageShape, source.texture);
            var targetLayout = ResolveRepoVkLayoutKind(logicalShape, storageShape, targetPacks);
            var targetPacking = ResolveTensorPacking(storageShape, targetLayout, targetPacks);
            var targetDataType = ResolveInferenceTensorDataType(source.texture.format);
            if (!sourceDescriptor.IsStorageLayoutCompatibleWith(storageShape, targetLayout, targetPacking, targetDataType))
                throw new TensorAliasTransformRequiredException(sourceDescriptor, logicalShape, storageShape);

            var lifetimeOwner = ResolveCmdTextureLifetimeOwner(source) ?? source;
            lifetimeOwner.refs++;
            return CreateCmdTensorRef(
                source.texture,
                logicalShape,
                storageShape,
                owned: false,
                refs: 1,
                sharedTextureOwner: lifetimeOwner,
                blobName: sourceDescriptor.Provenance.BlobName);
        }

        internal static void DetachTextureOwnership(TensorRef tensor)
        {
            if (tensor == null)
                return;

            tensor.owned = false;
            var lifetimeOwner = ResolveTextureLifetimeOwner(tensor);
            if (lifetimeOwner != null)
                lifetimeOwner.owned = false;
        }

        internal static void DetachCmdTextureOwnership(CmdTensorRef tensor)
        {
            if (tensor == null)
                return;

            tensor.owned = false;
            var lifetimeOwner = ResolveCmdTextureLifetimeOwner(tensor);
            if (lifetimeOwner != null)
                lifetimeOwner.owned = false;
        }

        internal static void DetachReturnedCmdTextureOwnership(Dictionary<string, CmdTensorRef> blobs, ComputeTexture texture)
        {
            if (blobs == null || texture == null)
                return;

            foreach (var kv in blobs)
            {
                var candidate = kv.Value;
                if (candidate == null || candidate.texture == null || candidate.texture.nameID != texture.nameID)
                    continue;

                DetachCmdTextureOwnership(candidate);
                candidate.ClearTexture();
            }
        }

        internal static RepoVkTensorContract GetTextureContract(
            Dictionary<string, BufferShape> textureShapes,
            TensorRef tensor,
            string name)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("texture contract unavailable: " + name);

            // The per-name dictionary remains for legacy execution bookkeeping only.
            // Published tensor descriptors are the sole source of contract metadata.
            EnsureRepoVkTensor(tensor);
            return new RepoVkTensorContract(tensor);
        }

        internal static RepoVkTensorContract GetCmdTensorContract(
            CmdTensorRef tensor,
            BufferShape? fallbackLogicalShape = null,
            BufferShape? fallbackStorageShape = null)
        {
            if (tensor == null || tensor.texture == null)
                throw new ArgumentNullException(nameof(tensor));

            EnsureRepoVkTensor(tensor);
            return new RepoVkTensorContract(tensor);
        }

        internal static bool TryGetExistingTextureContract(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            out TensorRef texture,
            out RepoVkTensorContract contract)
        {
            texture = null;
            contract = default;
            if (textureBlobs == null || string.IsNullOrWhiteSpace(name))
                return false;
            if (!textureBlobs.TryGetValue(name, out texture) || texture == null || texture.texture == null)
            {
                texture = null;
                return false;
            }

            contract = GetTextureContract(null, texture, name);
            return true;
        }

        internal static bool TryGetExistingCmdTextureContract(
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes,
            string name,
            out CmdTensorRef texture,
            out RepoVkTensorContract contract)
        {
            texture = null;
            contract = default;
            if (blobs == null || string.IsNullOrWhiteSpace(name))
                return false;
            if (!blobs.TryGetValue(name, out texture) || texture == null || texture.texture == null)
            {
                texture = null;
                return false;
            }

            contract = GetCmdTensorContract(texture);
            return true;
        }

        internal static void EnsureRepoVkTensor(
            TensorRef tensor,
            BufferShape? fallbackLogicalShape = null,
            BufferShape? fallbackStorageShape = null)
        {
            if (tensor == null || tensor.texture == null)
                return;

            if (tensor.IsDescriptorPublished)
                return;

            var logicalShape = tensor.hasLogicalShape
                ? tensor.logicalShape
                : new BufferShape(3, Mathf.Max(1, tensor.width), Mathf.Max(1, tensor.height), 1, Mathf.Max(1, tensor.packs * 4));
            var storageShape = tensor.hasStorageShape
                ? tensor.storageShape
                : logicalShape;
            SyncTextureContractMetadata(tensor, logicalShape, storageShape);
        }

        internal static void EnsureRepoVkTensor(
            CmdTensorRef tensor,
            BufferShape? fallbackLogicalShape = null,
            BufferShape? fallbackStorageShape = null)
        {
            if (tensor == null || tensor.texture == null)
                return;

            if (tensor.IsDescriptorPublished)
                return;

            var logicalShape = tensor.hasLogicalShape
                ? tensor.logicalShape
                : new BufferShape(3, Mathf.Max(1, tensor.width), Mathf.Max(1, tensor.height), 1, Mathf.Max(1, tensor.packs * 4));
            var storageShape = tensor.hasStorageShape
                ? tensor.storageShape
                : logicalShape;
            SyncCmdTensorContractMetadata(tensor, logicalShape, storageShape);
        }

        internal static int GetTexturePackCount(BufferShape shape, RenderTexture texture)
        {
            if (texture == null)
                return 1;

            var volumeDepth = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
            if (shape.dims == 4)
                return Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
            return volumeDepth;
        }

        internal static int GetTextureSliceCount(BufferShape shape, RenderTexture texture)
        {
            if (texture == null)
                return 1;

            var volumeDepth = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
            if (shape.dims == 4)
            {
                var packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
                return Mathf.Max(1, volumeDepth / packs);
            }
            return volumeDepth;
        }

        private RenderTexture MaterializeTextureFromBufferViewCore(ComputeBuffer buffer, NcnnTensorBuffer view, bool ignoreGuard, RenderTextureFormat? formatOverride = null)
        {
            if (buffer == null || view == null)
                return null;
            if (!ignoreGuard && ShouldBlockPack4BufferFallback())
            {
                throw CreateDisallowedBufferPathException(
                    "pack4-only guard: buffer-to-texture materialization disallowed",
                    null,
                    "dims=" + view.dims + " w=" + view.w + " h=" + view.h + " d=" + view.d + " c=" + view.c);
            }

            if (!TryResolveTensorTextureMaterialization(view, out var texW, out var texH, out var channels, out var sliceCount, out var format))
                return null;

            if (view.dims <= 2)
            {
                var linear = RentTempMat(texW, texH, ResolveLinearMatTextureFormat());
                _ops.FillLinearMatFromBuffer(buffer, texW, texH, linear);
                return linear;
            }

            var rt = RentTempArray(texW, texH, sliceCount, ResolveTensorTextureFormatWithOverride(view.dims, formatOverride));
            if (view.dims == 4)
                _ops.FillPack4FromBufferCDHW(buffer, texW, texH, view.d, channels, rt);
            else
                _ops.FillPack4FromBufferCHW(buffer, texW, texH, channels, rt);
            return rt;
        }

        internal RenderTexture MaterializeTextureFromBufferView(ComputeBuffer buffer, NcnnTensorBuffer view, RenderTextureFormat? formatOverride = null)
        {
            return MaterializeTextureFromBufferViewCore(buffer, view, ignoreGuard: false, formatOverride);
        }

        internal RenderTexture MaterializeScratchTextureFromBufferView(ComputeBuffer buffer, NcnnTensorBuffer view, RenderTextureFormat? formatOverride = null)
        {
            return MaterializeTextureFromBufferViewCore(buffer, view, ignoreGuard: true, formatOverride);
        }

        internal ComputeTexture MaterializeCmdTextureFromBufferView(CommandBuffer cmd, ComputeBuffer buffer, NcnnTensorBuffer view)
        {
            if (cmd == null || buffer == null || view == null)
                return null;
            if (ShouldBlockPack4BufferFallback())
            {
                throw CreateDisallowedBufferPathException(
                    "production texture-only contract rejects command-buffer buffer-to-texture materialization",
                    null,
                    "dims=" + view.dims + " w=" + view.w + " h=" + view.h + " d=" + view.d + " c=" + view.c);
            }

            int texW;
            int texH;
            int channels;
            if (view.dims == 1)
            {
                texW = view.w;
                texH = 1;
                channels = 1;
            }
            else if (view.dims == 2)
            {
                texW = view.w;
                texH = view.h;
                channels = 1;
            }
            else if (view.dims == 3)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else if (view.dims == 4)
            {
                texW = view.w;
                texH = view.h;
                channels = view.c;
            }
            else
            {
                return null;
            }

            if (view.dims <= 2)
            {
                var linear = RentTempMat(cmd, texW, texH, ResolveLinearMatTextureFormat());
                _ops.FillLinearMatFromBuffer(cmd, buffer, texW, texH, linear);
                return linear;
            }

            var channelPacks = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            var sliceCount = view.dims == 4
                ? Mathf.Max(1, view.d) * channelPacks
                : channelPacks;
            if (WouldExceedTextureArraySliceLimit(sliceCount))
            {
                DebugLog?.Invoke(
                    "[BufferMaterialize] skip-cmd-texture-slice-limit"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | dims=" + view.dims
                    + " | shape=" + view.w + "x" + view.h + "x" + view.d + "x" + view.c
                    + " | slices=" + sliceCount.ToString(CultureInfo.InvariantCulture)
                    + " | max_slices=" + GetMaxTextureArraySlices().ToString(CultureInfo.InvariantCulture));
                return null;
            }
            var rt = RentTempArray(cmd, texW, texH, sliceCount, ResolveTensorTextureFormat(view.dims));
            if (view.dims == 4)
                _ops.FillPack4FromBufferCDHW(cmd, buffer, texW, texH, view.d, channels, rt);
            else
                _ops.FillPack4FromBufferCHW(cmd, buffer, texW, texH, channels, rt);
            return rt;
        }

        internal RenderTexture MaterializeTextureFromBuffer(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (!bufferBlobs.TryGetValue(name, out var buffer) || buffer == null)
                return null;
            if (!bufferViews.TryGetValue(name, out var view) || view == null)
                return null;
            if (ShouldBlockPack4BufferFallback())
                throw CreateDisallowedBufferPathException("pack4-only guard: buffer-to-texture materialization disallowed", name);
            return MaterializeTextureFromBufferView(buffer, view);
        }

        internal TensorRef GetOrMaterializeTexture(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (textureBlobs.TryGetValue(name, out var tr) && tr != null && tr.texture != null)
                return tr;

            var materialized = MaterializeTextureFromBuffer(name, bufferBlobs, bufferViews);
            if (materialized == null)
            {
                if (bufferViews.TryGetValue(name, out var materializeView) && materializeView != null)
                {
                    var channelPacks = Mathf.Max(1, Mathf.CeilToInt(materializeView.c / 4f));
                    var sliceCount = materializeView.dims == 4
                        ? Mathf.Max(1, materializeView.d) * channelPacks
                        : channelPacks;
                    if (WouldExceedTextureArraySliceLimit(sliceCount))
                        throw new InvalidOperationException("blob texture materialization skipped by slice limit: " + name);
                }
                throw new InvalidOperationException("blob not found: " + name);
            }

            var shape = bufferViews.TryGetValue(name, out var view) && view != null
                ? new BufferShape(view.dims, view.w, view.h, view.d, view.c)
                : new BufferShape(3, materialized.width, materialized.height, 1, Mathf.Max(1, materialized.volumeDepth > 0 ? materialized.volumeDepth : 1) * 4);
            var packs = GetTexturePackCount(shape, materialized);

            tr = CreateTextureRef(materialized, shape, shape, owned: true, blobName: name);
            textureBlobs[name] = tr;
            textureShapes[name] = shape;
            return tr;
        }

        internal static bool TryGetExistingTexture(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            out TensorRef texture,
            out BufferShape shape)
        {
            if (TryGetExistingTextureContract(textureBlobs, textureShapes, name, out texture, out var contract))
            {
                shape = contract.LogicalShape;
                return true;
            }

            texture = null;
            shape = default;
            return false;
        }

        internal NcnnTensorBuffer RentScratchTensorFromTexture(
            TensorRef texture,
            BufferShape shape,
            string blobName = null,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            if (texture == null || texture.texture == null)
                throw new ArgumentNullException(nameof(texture));
            if (ShouldBlockPack4BufferFallback())
            {
                throw CreateDisallowedBufferPathException(
                    "pack4-only guard: texture-to-buffer scratch materialization disallowed",
                    blobName,
                    "dims=" + shape.dims + " w=" + shape.w + " h=" + shape.h + " d=" + shape.d + " c=" + shape.c);
            }

            var sliceCount = GetTextureSliceCount(shape, texture.texture);
            var physicalChannels = Mathf.Max(1, texture.packs * 4);
            var physicalCount = Mathf.Max(1, texture.width * texture.height * sliceCount * physicalChannels);
            var logicalCount = Mathf.Max(1, shape.w) * Mathf.Max(1, shape.h) * Mathf.Max(1, shape.d) * Mathf.Max(1, shape.c);
            if (logicalCount > physicalCount)
                throw new InvalidOperationException("texture logical shape exceeds physical storage");

            var converted = RentTempBuffer(logicalCount, sizeof(float), ComputeBufferType.Structured, callerMember, callerLine);
            if (physicalCount == logicalCount)
            {
                if (shape.dims == 4)
                    _ops.Pack4ToBufferCDHW(texture.texture, texture.width, texture.height, shape.d, shape.c, converted);
                else
                    _ops.Pack4ToBufferCHW(texture.texture, texture.width, texture.height, physicalChannels, converted);
                return new NcnnTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, true, ReturnTempBuffer);
            }

            var physical = RentTempBuffer(physicalCount, sizeof(float), ComputeBufferType.Structured, callerMember, callerLine);
            try
            {
                if (shape.dims == 4)
                    _ops.Pack4ToBufferCDHW(texture.texture, texture.width, texture.height, shape.d, shape.c, physical);
                else
                    _ops.Pack4ToBufferCHW(texture.texture, texture.width, texture.height, physicalChannels, physical);
                _ops.CopyBufPartial(physical, 0, converted, logicalCount);
            }
            finally
            {
                ReturnTempBuffer(physical);
            }

            return new NcnnTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, true, ReturnTempBuffer);
        }

        internal NcnnTensorBuffer GetReadableTensorInput(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned = null,
            [CallerMemberName] string callerMember = null,
            [CallerLineNumber] int callerLine = 0)
        {
            if (TryGetBufferView(name, bufferBlobs, bufferViews) is { } existingView)
                return existingView;

            if (TryGetExistingTexture(textureBlobs, textureShapes, name, out var texture, out var shape))
                return RentScratchTensorFromTexture(texture, shape, name, callerMember, callerLine);

            var buffer = GetOrConvertToBuffer(name, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
            if (buffer == null)
                return null;
            return TryGetBufferView(name, bufferBlobs, bufferViews);
        }

        internal void PublishScratchTextureOutput(
            string topName,
            NcnnTensorBuffer tensor,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            RenderTextureFormat? textureFormatOverride = null)
        {
            if (string.IsNullOrWhiteSpace(topName))
                throw new ArgumentNullException(nameof(topName));
            if (tensor == null || tensor.buffer == null)
                throw new ArgumentNullException(nameof(tensor));
            if (tensor.dims > 4)
                throw new InvalidOperationException("scratch texture outputs currently require dims<=4: " + topName);

            var rt = MaterializeScratchTextureFromBufferView(tensor.buffer, tensor, textureFormatOverride);
            if (rt == null)
                throw new InvalidOperationException("failed to materialize scratch texture output: " + topName);

            SetTextureBlob(
                textureBlobs,
                textureShapes,
                topName,
                rt,
                new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c));
        }

        internal void PublishScratchTextureOutput(
            string topName,
            ComputeBuffer buffer,
            BufferShape logicalShape,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            RenderTextureFormat? textureFormatOverride = null)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            var view = new NcnnTensorBuffer(buffer, logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c, false);
            var rt = MaterializeScratchTextureFromBufferView(buffer, view, textureFormatOverride);
            if (rt == null)
                throw new InvalidOperationException("failed to materialize scratch texture output: " + topName);
            SetTextureBlob(textureBlobs, textureShapes, topName, rt, logicalShape);
        }

        internal void RegisterTextureInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes)
        {
            if (textureInputs == null)
                return;

            foreach (var kv in textureInputs)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");

                var rt = kv.Value;
                var packs = rt.volumeDepth > 0 ? rt.volumeDepth : 1;
                var useCount = _blobUseCount.TryGetValue(kv.Key, out var c) ? c : 1;
                var fallbackChannels = string.Equals(kv.Key, "data", StringComparison.OrdinalIgnoreCase) ? 3 : packs * 4;
                var logicalShape = textureInputShapes != null && textureInputShapes.TryGetValue(kv.Key, out var suppliedShape)
                    ? suppliedShape
                    : new BufferShape(3, rt.width, rt.height, 1, ResolveInputLogicalChannels(kv.Key, fallbackChannels));
                packs = GetTexturePackCount(logicalShape, rt);

                var sliceCount = GetTextureSliceCount(logicalShape, rt);
                var physicalCount = rt.width * rt.height * sliceCount * packs * 4;
                var logicalCount = Mathf.Max(1, logicalShape.w) * Mathf.Max(1, logicalShape.h) * Mathf.Max(1, logicalShape.d) * Mathf.Max(1, logicalShape.c);
                if (logicalCount > physicalCount)
                    throw new InvalidOperationException("texture input logical shape exceeds physical storage: " + kv.Key);

                textureBlobs[kv.Key] = CreateTextureRef(rt, logicalShape, logicalShape, owned: false, refs: useCount, blobName: kv.Key);
                textureShapes[kv.Key] = logicalShape;
            }
        }

        internal void RegisterCmdTextureInputs(
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes)
        {
            if (textureInputs == null)
                return;

            foreach (var kv in textureInputs)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");

                var texture = kv.Value;
                var depth = Mathf.Max(1, texture.depth);
                var useCount = _blobUseCount.TryGetValue(kv.Key, out var c) ? c : 1;
                var fallbackChannels = string.Equals(kv.Key, "data", StringComparison.OrdinalIgnoreCase) ? 3 : depth * 4;
                var logicalShape = textureInputShapes != null && textureInputShapes.TryGetValue(kv.Key, out var suppliedShape)
                    ? suppliedShape
                    : new BufferShape(3, texture.width, texture.height, 1, ResolveInputLogicalChannels(kv.Key, fallbackChannels));
                var packs = logicalShape.dims == 4
                    ? Mathf.Max(1, Mathf.CeilToInt(logicalShape.c / 4f))
                    : depth;
                var sliceCount = logicalShape.dims == 4
                    ? Mathf.Max(1, depth / Mathf.Max(1, packs))
                    : depth;

                var physicalCount = texture.width * texture.height * sliceCount * packs * 4;
                var logicalCount = Mathf.Max(1, logicalShape.w) * Mathf.Max(1, logicalShape.h) * Mathf.Max(1, logicalShape.d) * Mathf.Max(1, logicalShape.c);
                if (logicalCount > physicalCount)
                    throw new InvalidOperationException("command-buffer texture input logical shape exceeds physical storage: " + kv.Key);

                blobs[kv.Key] = CreateCmdTensorRef(texture, logicalShape, logicalShape, owned: false, refs: useCount, blobName: kv.Key);
                shapes[kv.Key] = logicalShape;
            }
        }

        internal bool TryGetPack4Texture(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            out TensorRef texture,
            out BufferShape shape)
        {
            texture = null;
            shape = default;

            if (ShouldForceCurrentLayerBufferPath())
                return false;

            try
            {
                texture = GetOrMaterializeTexture(name, textureBlobs, textureShapes, bufferBlobs, bufferViews);
                if (texture == null || texture.texture == null)
                    return false;

                shape = GetTextureShape(textureShapes, texture, name);
                return shape.dims >= 1 && shape.dims <= 4;
            }
            catch
            {
                texture = null;
                shape = default;
                return false;
            }
        }

        internal static bool AreAllLayerTopsAlreadyAvailable(
            NcnnParamModel.Layer layer,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, IndexRef> indexBlobs)
        {
            var topNames = layer?.topNames;
            if (topNames == null || topNames.Length == 0)
                return false;

            for (var i = 0; i < topNames.Length; i++)
            {
                var name = topNames[i];
                if (string.IsNullOrEmpty(name))
                    return false;

                var hasTexture = textureBlobs != null
                    && textureBlobs.TryGetValue(name, out var tr)
                    && tr != null
                    && tr.texture != null;
                var hasBuffer = bufferBlobs != null
                    && bufferBlobs.TryGetValue(name, out var buf)
                    && buf != null;
                var hasIndex = indexBlobs != null
                    && indexBlobs.TryGetValue(name, out var ir)
                    && ir != null
                    && (ir.texture != null || ir.buffer != null);

                if (!hasTexture && !hasBuffer && !hasIndex)
                    return false;
            }

            return true;
        }

        internal LayerRuntimeProfile BeginLayerRuntimeProfile(string pathKind)
        {
            if (!LayerRuntimeProfileEnabled)
            {
                LastRuntimeProfile = null;
                return null;
            }

            var profile = new LayerRuntimeProfile
            {
                inferenceIndex = ++_runtimeProfileInferenceIndex,
                pathKind = string.IsNullOrWhiteSpace(LayerRuntimeProfilePathKindOverride)
                    ? (string.IsNullOrWhiteSpace(pathKind) ? "buffer" : pathKind)
                    : LayerRuntimeProfilePathKindOverride,
                syncGpu = LayerRuntimeProfileSyncGpu
            };
            LastRuntimeProfile = profile;
            return profile;
        }

        internal static void FinishLayerRuntimeProfile(LayerRuntimeProfile profile)
        {
            if (profile == null)
                return;
            profile.totalMs = TicksToMilliseconds(profile.totalTicks);
        }

        internal static void RecordLayerRuntime(
            LayerRuntimeProfile profile,
            int layerIndex,
            NcnnParamModel.Layer layer,
            string path,
            long elapsedTicks)
        {
            if (profile == null || layer == null)
                return;

            if (elapsedTicks < 0)
                elapsedTicks = 0;

            var typeName = string.IsNullOrWhiteSpace(layer.typeName) ? "Unknown" : layer.typeName;
            var record = new LayerRuntimeRecord
            {
                layerIndex = layerIndex,
                layerName = layer.name ?? string.Empty,
                layerType = typeName,
                path = path ?? string.Empty,
                elapsedTicks = elapsedTicks,
                elapsedMs = TicksToMilliseconds(elapsedTicks)
            };
            profile.layers.Add(record);
            profile.totalTicks += elapsedTicks;

            if (!profile.layerTypes.TryGetValue(typeName, out var typeProfile) || typeProfile == null)
            {
                typeProfile = new LayerRuntimeTypeProfile { layerType = typeName };
                profile.layerTypes[typeName] = typeProfile;
            }

            typeProfile.count++;
            typeProfile.totalTicks += elapsedTicks;
            typeProfile.totalMs = TicksToMilliseconds(typeProfile.totalTicks);
            typeProfile.avgMs = typeProfile.count > 0 ? typeProfile.totalMs / typeProfile.count : 0d;
        }

        internal static string DescribeLayerOutputPath(
            NcnnParamModel.Layer layer,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            Dictionary<string, IndexRef> indexBlobs)
        {
            var topNames = layer?.topNames;
            if (topNames == null || topNames.Length == 0)
                return string.Empty;

            var sb = new StringBuilder();
            for (var i = 0; i < topNames.Length; i++)
            {
                if (i > 0)
                    sb.Append(';');

                var name = topNames[i] ?? string.Empty;
                sb.Append(name);
                sb.Append('=');

                if (textureBlobs != null
                    && textureBlobs.TryGetValue(name, out var tex)
                    && tex != null
                    && tex.texture != null)
                {
                    sb.Append("tex:");
                    sb.Append(tex.width.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.height.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.packs.ToString(CultureInfo.InvariantCulture));
                    sb.Append('p');
                    if (textureShapes != null && textureShapes.TryGetValue(name, out var shape))
                    {
                        sb.Append(":c");
                        sb.Append(shape.c.ToString(CultureInfo.InvariantCulture));
                        if (IsPack4LinearMatTexture(tex, shape))
                            sb.Append(":p4lin");
                    }
                    continue;
                }

                if (bufferBlobs != null
                    && bufferBlobs.TryGetValue(name, out var buffer)
                    && buffer != null)
                {
                    sb.Append("buf:");
                    if (bufferViews != null && bufferViews.TryGetValue(name, out var view) && view != null)
                    {
                        sb.Append('d');
                        sb.Append(view.dims.ToString(CultureInfo.InvariantCulture));
                        sb.Append(':');
                        sb.Append(view.w.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(view.h.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(view.c.ToString(CultureInfo.InvariantCulture));
                    }
                    else
                    {
                        sb.Append(buffer.count.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                if (indexBlobs != null
                    && indexBlobs.TryGetValue(name, out var index)
                    && index != null
                    && (index.texture != null || index.buffer != null))
                {
                    sb.Append("idx:");
                    sb.Append(index.width.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(index.height.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    if (index.texture != null)
                    {
                        sb.Append(index.packs.ToString(CultureInfo.InvariantCulture));
                        sb.Append('p');
                    }
                    else if (index.view != null)
                    {
                        sb.Append("buf:");
                        sb.Append(index.view.w.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(index.view.h.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(index.view.c.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                sb.Append("missing");
            }

            return sb.ToString();
        }

        internal static string DescribeCmdLayerOutputPath(
            NcnnParamModel.Layer layer,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes)
        {
            var topNames = layer?.topNames;
            if (topNames == null || topNames.Length == 0)
                return string.Empty;

            var sb = new StringBuilder();
            for (var i = 0; i < topNames.Length; i++)
            {
                if (i > 0)
                    sb.Append(';');

                var name = topNames[i] ?? string.Empty;
                sb.Append(name);
                sb.Append('=');

                if (blobs != null
                    && blobs.TryGetValue(name, out var tex)
                    && tex != null
                    && tex.texture != null)
                {
                    sb.Append("tex:");
                    sb.Append(tex.width.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.height.ToString(CultureInfo.InvariantCulture));
                    sb.Append('x');
                    sb.Append(tex.packs.ToString(CultureInfo.InvariantCulture));
                    sb.Append('p');
                    if (shapes != null && shapes.TryGetValue(name, out var shape))
                    {
                        sb.Append(":d");
                        sb.Append(shape.dims.ToString(CultureInfo.InvariantCulture));
                        sb.Append(':');
                        sb.Append(shape.w.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(shape.h.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(shape.d.ToString(CultureInfo.InvariantCulture));
                        sb.Append('x');
                        sb.Append(shape.c.ToString(CultureInfo.InvariantCulture));
                    }
                    continue;
                }

                sb.Append("missing");
            }

            return sb.ToString();
        }

        public string FormatLastLayerRuntimeProfile(int topN = 40)
        {
            return FormatLayerRuntimeProfile(LastRuntimeProfile, topN);
        }

        public static string FormatLayerRuntimeProfile(LayerRuntimeProfile profile, int topN = 40)
        {
            if (profile == null)
                return string.Empty;

            topN = Mathf.Max(1, topN);
            var sb = new StringBuilder(4096);
            sb.AppendLine("section\tinference\tpath_kind\tsync_gpu\tlayer_index\tname\ttype\tpath\tcount\tms\tavg_ms");
            sb.Append("summary\t");
            sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
            sb.Append('\t');
            sb.Append(profile.pathKind ?? string.Empty);
            sb.Append('\t');
            sb.Append(profile.syncGpu ? "1" : "0");
            sb.Append("\t\t\t\t");
            sb.Append(profile.layers.Count.ToString(CultureInfo.InvariantCulture));
            sb.Append('\t');
            sb.Append(profile.totalMs.ToString("0.###", CultureInfo.InvariantCulture));
            sb.AppendLine("\t");

            var typeProfiles = new List<LayerRuntimeTypeProfile>(profile.layerTypes.Values);
            typeProfiles.Sort((a, b) => b.totalTicks.CompareTo(a.totalTicks));
            for (var i = 0; i < typeProfiles.Count; i++)
            {
                var item = typeProfiles[i];
                sb.Append("type\t");
                sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(profile.pathKind ?? string.Empty);
                sb.Append('\t');
                sb.Append(profile.syncGpu ? "1" : "0");
                sb.Append("\t\t\t");
                sb.Append(item.layerType ?? string.Empty);
                sb.Append("\t\t");
                sb.Append(item.count.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(item.totalMs.ToString("0.###", CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.AppendLine(item.avgMs.ToString("0.###", CultureInfo.InvariantCulture));
            }

            var records = new List<LayerRuntimeRecord>(profile.layers);
            records.Sort((a, b) => b.elapsedTicks.CompareTo(a.elapsedTicks));
            var count = Mathf.Min(topN, records.Count);
            for (var i = 0; i < count; i++)
            {
                var item = records[i];
                sb.Append("layer\t");
                sb.Append(profile.inferenceIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(profile.pathKind ?? string.Empty);
                sb.Append('\t');
                sb.Append(profile.syncGpu ? "1" : "0");
                sb.Append('\t');
                sb.Append(item.layerIndex.ToString(CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.Append(item.layerName ?? string.Empty);
                sb.Append('\t');
                sb.Append(item.layerType ?? string.Empty);
                sb.Append('\t');
                sb.Append(item.path ?? string.Empty);
                sb.Append("\t1\t");
                sb.Append(item.elapsedMs.ToString("0.###", CultureInfo.InvariantCulture));
                sb.Append('\t');
                sb.AppendLine(item.elapsedMs.ToString("0.###", CultureInfo.InvariantCulture));
            }

            return sb.ToString();
        }

        private static double TicksToMilliseconds(long ticks)
        {
            return ticks <= 0 ? 0d : (double)ticks * 1000d / Stopwatch.Frequency;
        }

        public sealed class LayerRuntimeRecord
        {
            public int layerIndex;
            public string layerName;
            public string layerType;
            public string path;
            public long elapsedTicks;
            public double elapsedMs;
        }

        public sealed class LayerRuntimeTypeProfile
        {
            public string layerType;
            public int count;
            public long totalTicks;
            public double totalMs;
            public double avgMs;
        }

        public sealed class LayerRuntimeProfile
        {
            public int inferenceIndex;
            public string pathKind;
            public bool syncGpu;
            public long totalTicks;
            public double totalMs;
            public readonly List<LayerRuntimeRecord> layers = new List<LayerRuntimeRecord>();
            public readonly Dictionary<string, LayerRuntimeTypeProfile> layerTypes = new Dictionary<string, LayerRuntimeTypeProfile>(StringComparer.Ordinal);
        }

        public sealed class TempResourceStatsSnapshot
        {
            public int tempBufferRentCount;
            public long tempBufferRentBytes;
            public int tempBufferLiveCount;
            public long tempBufferLiveBytes;
            public int tempBufferPeakLiveCount;
            public long tempBufferPeakLiveBytes;
            public int tempRtRentCount;
            public int tempRtLiveCount;
            public int tempRtPeakLiveCount;
        }

        public sealed class LayerTypeLoadProfile
        {
            public int count;
            public long totalMs;
            public long bytesRead;
            public long readMs;
            public long uploadMs;
            public long packMs;
        }

        public sealed class ModelLoadProfile
        {
            public string modelMagic;
            public int layerCount;
            public long releaseMs;
            public long parseParamMs;
            public long buildBlobUseCountMs;
            public long totalMs;
            public long totalBytesRead;
            public readonly Dictionary<string, LayerTypeLoadProfile> layerTypes = new Dictionary<string, LayerTypeLoadProfile>(StringComparer.Ordinal);
        }

        public readonly struct LoadProgress
        {
            public readonly string stage;
            public readonly int layerIndex;
            public readonly int layerCount;
            public readonly string layerName;
            public readonly string layerType;
            public readonly float progress01;

            public LoadProgress(string stage, int layerIndex, int layerCount, string layerName, string layerType, float progress01)
            {
                this.stage = stage;
                this.layerIndex = layerIndex;
                this.layerCount = layerCount;
                this.layerName = layerName;
                this.layerType = layerType;
                this.progress01 = progress01;
            }
        }

        public readonly struct LayerLoadMetrics
        {
            public readonly long bytesRead;
            public readonly long readMs;
            public readonly long uploadMs;
            public readonly long packMs;

            public LayerLoadMetrics(long bytesRead, long readMs, long uploadMs, long packMs)
            {
                this.bytesRead = bytesRead;
                this.readMs = readMs;
                this.uploadMs = uploadMs;
                this.packMs = packMs;
            }
        }

        internal static void SetTextureBlob(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            RenderTexture texture,
            BufferShape logicalShape)
        {
            textureBlobs[name] = CreateTextureRef(texture, logicalShape, logicalShape, owned: true, blobName: name);
            textureShapes[name] = logicalShape;
        }

        internal static void SetTextureBlob(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            RenderTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape)
        {
            textureBlobs[name] = CreateTextureRef(texture, logicalShape, storageShape, owned: true, blobName: name);
            textureShapes[name] = logicalShape;
        }

        internal void PublishTensorBufferOutput(
            string topName,
            NcnnTensorBuffer tensor,
            bool preferTexture,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferRef> bufferRefs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned,
            RenderTextureFormat? textureFormatOverride = null)
        {
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (tensor == null || tensor.buffer == null)
                throw new ArgumentNullException(nameof(tensor));
            var canRepresentAsTexture = preferTexture
                && tensor.dims <= 4
                && TryResolveTensorTextureMaterialization(
                    tensor,
                    out _,
                    out _,
                    out _,
                    out _,
                    out _,
                    logSkipReason: false);
            if (ShouldBlockPack4BufferFallback())
            {
                if (!tensor.ownsBuffer && preferTexture && tensor.dims <= 4 && canRepresentAsTexture)
                {
                    PublishScratchTextureOutput(topName, tensor, textureBlobs, textureShapes, textureFormatOverride);
                    return;
                }

                throw CreateDisallowedBufferPathException(
                    "pack4-only guard: buffer output disallowed",
                    topName,
                    "dims=" + tensor.dims
                    + " w=" + tensor.w
                    + " h=" + tensor.h
                    + " d=" + tensor.d
                    + " c=" + tensor.c
                    + " preferTexture=" + preferTexture
                    + " ownsBuffer=" + tensor.ownsBuffer
                    + " canRepresentAsTexture=" + canRepresentAsTexture);
            }

            var logicalShape = new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c);
            bufferBlobs[topName] = tensor.buffer;
            bufferRefs[topName] = NewBufferRef(tensor.buffer, tensor.ownsBuffer);
            bufferViews[topName] = new NcnnTensorBuffer(tensor.buffer, tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c, false);

            if (ForceBufferOutputsForDims4 && tensor.dims == 4)
                return;

            if (preferTexture && tensor.dims <= 4 && canRepresentAsTexture)
            {
                var rt = MaterializeTextureFromBufferView(tensor.buffer, tensor, textureFormatOverride);
                if (rt != null)
                    SetTextureBlob(textureBlobs, textureShapes, topName, rt, logicalShape);
            }
        }

        internal void PublishCmdTensorBufferOutput(
            CommandBuffer cmd,
            string topName,
            NcnnTensorBuffer tensor,
            bool preferTexture,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (tensor == null || tensor.buffer == null)
                throw new ArgumentNullException(nameof(tensor));

            if (!preferTexture || tensor.dims > 4)
                throw new InvalidOperationException("CommandBuffer outputs currently require dims<=4 materialized texture: " + topName);
            if (ShouldBlockPack4BufferFallback() && tensor.ownsBuffer)
            {
                throw CreateDisallowedBufferPathException(
                    "pack4-only guard: command-buffer buffer output disallowed",
                    topName,
                    "dims=" + tensor.dims + " w=" + tensor.w + " h=" + tensor.h + " d=" + tensor.d + " c=" + tensor.c);
            }

            var rt = MaterializeCmdTextureFromBufferView(cmd, tensor.buffer, tensor);
            if (rt == null)
                throw new InvalidOperationException("Failed to materialize CommandBuffer tensor: " + topName);

            var logicalShape = new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c);
            blobs[topName] = CreateCmdTensorRef(rt, logicalShape, logicalShape, owned: true, blobName: topName);
            if (shapes != null)
                shapes[topName] = logicalShape;
        }

        internal void PublishCmdTensorLikeInput(
            CommandBuffer cmd,
            string topName,
            int width,
            int height,
            int packs,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null,
            BufferShape? logicalShape = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (string.IsNullOrEmpty(topName))
                throw new ArgumentNullException(nameof(topName));
            if (blobs == null)
                throw new ArgumentNullException(nameof(blobs));

            var outArr = RentTempArray(cmd, width, height, packs, RenderTextureFormat.ARGBHalf);
            var storageShapeValue = logicalShape ?? default;
            blobs[topName] = logicalShape.HasValue
                ? CreateCmdTensorRef(outArr, logicalShape.Value, storageShapeValue, owned: true, blobName: topName)
                : new CmdTensorRef
                {
                    texture = outArr,
                    width = width,
                    height = height,
                    packs = packs,
                    refs = 1,
                    owned = true
                };
            if (shapes != null)
                shapes[topName] = logicalShape ?? new BufferShape(3, Mathf.Max(1, width), Mathf.Max(1, height), 1, Mathf.Max(1, packs * 4));
        }

        internal static void ResolveCmdTextureLayout(NcnnTensorBuffer tensor, out int width, out int height, out int packs)
        {
            if (tensor == null)
                throw new ArgumentNullException(nameof(tensor));

            width = Mathf.Max(1, tensor.w);
            height = 1;
            packs = 1;

            if (tensor.dims == 2)
            {
                height = Mathf.Max(1, tensor.h);
                return;
            }

            if (tensor.dims == 3)
            {
                height = Mathf.Max(1, tensor.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(tensor.c / 4f));
                return;
            }

            if (tensor.dims >= 4)
            {
                height = Mathf.Max(1, tensor.h * Mathf.Max(1, tensor.d));
                packs = Mathf.Max(1, Mathf.CeilToInt(tensor.c / 4f));
            }
        }

        internal static void ResolveCmdTextureLayout(BufferShape shape, out int width, out int height, out int packs)
        {
            width = Mathf.Max(1, shape.w);
            height = 1;
            packs = 1;

            if (shape.dims == 2)
            {
                height = Mathf.Max(1, shape.h);
                return;
            }

            if (shape.dims == 3)
            {
                height = Mathf.Max(1, shape.h);
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
                return;
            }

            if (shape.dims >= 4)
            {
                height = Mathf.Max(1, shape.h * Mathf.Max(1, shape.d));
                packs = Mathf.Max(1, Mathf.CeilToInt(shape.c / 4f));
            }
        }

        internal void PublishCmdPlaceholderFromTensorView(
            CommandBuffer cmd,
            string topName,
            NcnnTensorBuffer tensor,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            ResolveCmdTextureLayout(tensor, out var width, out var height, out var packs);
            PublishCmdTensorLikeInput(cmd, topName, width, height, packs, blobs, shapes, new BufferShape(tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c));
        }

        internal void PublishCmdPlaceholder(
            CommandBuffer cmd,
            string topName,
            BufferShape shape,
            Dictionary<string, CmdTensorRef> blobs,
            Dictionary<string, BufferShape> shapes = null)
        {
            ResolveCmdTextureLayout(shape, out var width, out var height, out var packs);
            PublishCmdTensorLikeInput(cmd, topName, width, height, packs, blobs, shapes, shape);
        }

        internal void CopyCmdTensor(
            CommandBuffer cmd,
            CmdTensorRef src,
            string topName,
            Dictionary<string, CmdTensorRef> blobs,
            int width = -1,
            int height = -1,
            int packs = -1,
            Dictionary<string, BufferShape> shapes = null,
            BufferShape? logicalShape = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            if (blobs == null)
                throw new ArgumentNullException(nameof(blobs));

            var outWidth = width > 0 ? width : src.width;
            var outHeight = height > 0 ? height : src.height;
            var outPacks = packs > 0 ? packs : src.packs;
            var outArr = RentTempArray(cmd, outWidth, outHeight, outPacks, RenderTextureFormat.ARGBHalf);
            if (outWidth == src.width && outHeight == src.height)
            {
                Ops.CopyPack4(cmd, src.texture, 0, outArr, 0, Mathf.Min(src.packs, outPacks));
            }

            blobs[topName] = logicalShape.HasValue
                ? CreateCmdTensorRef(outArr, logicalShape.Value, logicalShape.Value, owned: true, blobName: topName)
                : new CmdTensorRef
                {
                    texture = outArr,
                    width = outWidth,
                    height = outHeight,
                    packs = outPacks,
                    refs = 1,
                    owned = true
                };
            if (shapes != null)
                shapes[topName] = logicalShape ?? new BufferShape(3, Mathf.Max(1, outWidth), Mathf.Max(1, outHeight), 1, Mathf.Max(1, outPacks * 4));
        }

        internal static bool CanUseExactPack4BinaryPath(TensorRef a, BufferShape aShape, TensorRef b, BufferShape bShape)
        {
            return a != null
                && b != null
                && a.texture != null
                && b.texture != null
                && (aShape.dims == 3 || aShape.dims == 4)
                && aShape.dims == bShape.dims
                && aShape.w == bShape.w
                && aShape.h == bShape.h
                && aShape.d == bShape.d
                && aShape.c == bShape.c
                && MatchesPack4TextureStorage(a, aShape)
                && MatchesPack4TextureStorage(b, bShape)
                && a.width == b.width
                && a.height == b.height
                && a.packs == b.packs;
        }

        internal static bool CanUseGroupNormPack4Path(TensorRef src, BufferShape shape, GroupNormPack gp)
        {
            var logicalDepth = shape.dims == 4 ? Mathf.Max(1, shape.d) : 1;
            var expectedVolumeDepth = logicalDepth * Mathf.Max(1, src?.packs ?? 0);
            return src != null
                && src.texture != null
                && gp != null
                && gp.affine
                && gp.gamma != null
                && gp.beta != null
                && (shape.dims == 3 || shape.dims == 4)
                && shape.w == src.width
                && shape.h == src.height
                && (shape.dims != 4 || Mathf.Max(1, src.texture.volumeDepth) == expectedVolumeDepth)
                && shape.c == gp.channels
                && gp.channels > 0
                && gp.group > 0
                && gp.channels % gp.group == 0
                && src.packs == Mathf.CeilToInt(gp.channels / 4f);
        }

        internal static int ComputeConvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            return Mathf.Max(1, (inSize + padBefore + padAfter - kernelExtent) / Mathf.Max(1, stride) + 1);
        }

        internal static int ComputeConv3dWeightCount(int outC, int inC, int kernelW, int kernelH, int kernelD)
        {
            return Mathf.Max(1, outC) * Mathf.Max(1, inC) * Mathf.Max(1, kernelW) * Mathf.Max(1, kernelH) * Mathf.Max(1, kernelD);
        }

        internal static int ComputeDeconvOut(int inSize, int kernel, int dilation, int stride, int padBefore, int padAfter, int outputPadAfter)
        {
            var kernelExtent = dilation * (kernel - 1) + 1;
            var bordered = (inSize - 1) * Mathf.Max(1, stride) + kernelExtent + Mathf.Max(0, outputPadAfter);
            return Mathf.Max(1, bordered - Mathf.Max(0, padBefore) - Mathf.Max(0, padAfter));
        }

        internal static ComputeBuffer NewBuffer(float[] data)
        {
            var buf = new ComputeBuffer(data.Length, sizeof(float), ComputeBufferType.Structured);
            NcnnGpuResourceTracker.RegisterBuffer(buf, data.Length, sizeof(float), "NcnnRepro.NewBuffer");
            buf.SetData(data);
            return buf;
        }

        internal BufferRef NewBufferRef(ComputeBuffer buffer, bool owned)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            return new BufferRef
            {
                buffer = buffer,
                refs = 1,
                owned = owned
            };
        }

        internal BufferRef NewOwnedBufferRef(string name, ComputeBuffer buffer)
        {
            return NewBufferRef(buffer, true);
        }

        internal void ReleaseTextureRef(TensorRef tensor)
        {
            if (tensor == null)
                return;

            tensor.refs--;
            if (tensor.refs > 0)
                return;

            var lifetimeOwner = tensor.sharedTextureOwner;
            tensor.sharedTextureOwner = null;
            if (lifetimeOwner != null)
                ReleaseTextureRef(lifetimeOwner);

            if (tensor.owned && tensor.texture != null)
            {
                try { ReturnTempArray(tensor.texture); } catch { }
            }

            tensor.ClearTexture();
            tensor.owned = false;
        }

        internal void ReleaseCmdTensorRef(CommandBuffer cmd, CmdTensorRef tensor)
        {
            if (tensor == null)
                return;

            tensor.refs--;
            if (tensor.refs > 0)
                return;

            var lifetimeOwner = tensor.sharedTextureOwner;
            tensor.sharedTextureOwner = null;
            if (lifetimeOwner != null)
                ReleaseCmdTensorRef(cmd, lifetimeOwner);

            if (tensor.owned && tensor.texture != null)
            {
                try { ReturnTempArray(cmd, tensor.texture); } catch { }
            }

            tensor.ClearTexture();
            tensor.owned = false;
        }

        internal void ReleaseAllCmdTemporaryTensors(CommandBuffer cmd, Dictionary<string, CmdTensorRef> blobs)
        {
            if (cmd == null || blobs == null)
                return;

            var visited = new HashSet<CmdTensorRef>();
            foreach (var tensor in blobs.Values)
            {
                if (tensor == null || !visited.Add(tensor))
                    continue;
                if (tensor.owned && tensor.texture != null)
                    ReturnTempArray(cmd, tensor.texture);
                tensor.sharedTextureOwner = null;
                tensor.ClearTexture();
                tensor.owned = false;
            }
        }

        internal static TensorRef GetTexture(Dictionary<string, TensorRef> blobs, string name)
        {
            if (!blobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
                throw new InvalidOperationException("blob not found: " + name);
            EnsureRepoVkTensor(tr);
            return tr;
        }

        internal static NcnnTensorBuffer TryGetBufferView(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, NcnnTensorBuffer> bufferViews)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                return view;
            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
            {
                view = new NcnnTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
                bufferViews[name] = view;
                return view;
            }
            return null;
        }

        internal ComputeBuffer GetOrConvertToBuffer(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            if (ShouldBlockPack4BufferFallback())
            {
                var detail = bufferBlobs != null && bufferBlobs.TryGetValue(name, out var existing) && existing != null
                    ? "mode=existing-buffer"
                    : "mode=materialize-from-texture";
                throw CreateDisallowedBufferPathException("pack4-only guard: buffer access disallowed", name, detail);
            }

            var emitMaterializeLog = DebugLog != null
                && !string.IsNullOrEmpty(name)
                && (DebugLogAllBufferMaterialize || name.StartsWith("stride_", StringComparison.Ordinal));
            var site = emitMaterializeLog ? DescribeCurrentExecutionSite() : null;
            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
            {
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] reuse | site=" + site + " | name=" + name + " | count=" + buf.count);
                return buf;
            }
            if (!textureBlobs.TryGetValue(name, out var tr) || tr == null || tr.texture == null)
            {
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] missing-source | site=" + site + " | name=" + name + " | hasShape=" + textureShapes.ContainsKey(name));
                return null;
            }

            var shape = GetTextureShape(textureShapes, tr, name);
            if (IsStrictLinearMatTexture(tr))
            {
                var physicalCountLinear = Mathf.Max(1, tr.width) * Mathf.Max(1, tr.height);
                var logicalCountLinear = shape.w * shape.h * shape.d * shape.c;
                if (emitMaterializeLog)
                {
                    DebugLog("[BufferMaterialize] convert-start | site=" + site + " | name=" + name
                        + " | layout=linear-mat"
                        + " | size=" + tr.width + "x" + tr.height
                        + " | physical=" + physicalCountLinear
                        + " | logical=" + logicalCountLinear
                        + " | dims=" + shape.dims
                        + " | shape=" + shape.w + "x" + shape.h + "x" + shape.c);
                }

                if (logicalCountLinear <= 0 || logicalCountLinear > physicalCountLinear)
                    throw new InvalidOperationException("linear texture logical shape mismatch: " + name + " | physical=" + physicalCountLinear + " logical=" + logicalCountLinear);

                var physicalLinearBuffer = RentTempBuffer(physicalCountLinear, sizeof(float));
                _ops.LinearMatToBuffer(tr.texture, tr.width, tr.height, physicalLinearBuffer);
                tempOwned.Add(physicalLinearBuffer);

                ComputeBuffer convertedLinear;
                if (logicalCountLinear == physicalCountLinear)
                {
                    convertedLinear = physicalLinearBuffer;
                }
                else
                {
                    convertedLinear = RentTempBuffer(logicalCountLinear, sizeof(float));
                    _ops.CopyBufPartial(physicalLinearBuffer, 0, convertedLinear, logicalCountLinear);
                    tempOwned.Add(convertedLinear);
                }

                bufferBlobs[name] = convertedLinear;
                bufferViews[name] = new NcnnTensorBuffer(convertedLinear, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] convert-done | site=" + site + " | name=" + name + " | mode=linear-mat | count=" + convertedLinear.count);
                return convertedLinear;
            }

            var sliceCount = GetTextureSliceCount(shape, tr.texture);
            var physicalChannels = tr.packs * 4;
            var physicalCount = tr.width * tr.height * sliceCount * physicalChannels;
            var logicalCount = shape.w * shape.h * shape.d * shape.c;
            if (emitMaterializeLog)
            {
                DebugLog("[BufferMaterialize] convert-start | site=" + site + " | name=" + name
                    + " | size=" + tr.width + "x" + tr.height
                    + " | packs=" + tr.packs
                    + " | slices=" + sliceCount
                    + " | physical=" + physicalCount
                    + " | logical=" + logicalCount
                    + " | dims=" + shape.dims
                    + " | shape=" + shape.w + "x" + shape.h + "x" + shape.c);
            }
            if (physicalCount == logicalCount)
            {
                var convertedExact = RentTempBuffer(logicalCount, sizeof(float));
                if (shape.dims == 4)
                    _ops.Pack4ToBufferCDHW(tr.texture, tr.width, tr.height, shape.d, shape.c, convertedExact);
                else
                    _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, convertedExact);
                bufferBlobs[name] = convertedExact;
                bufferViews[name] = new NcnnTensorBuffer(convertedExact, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(convertedExact);
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] convert-done | site=" + site + " | name=" + name + " | mode=exact | count=" + convertedExact.count);
                return convertedExact;
            }

            if (logicalCount > 0 && logicalCount < physicalCount)
            {
                var physicalBuffer = RentTempBuffer(physicalCount, sizeof(float));
                if (shape.dims == 4)
                    _ops.Pack4ToBufferCDHW(tr.texture, tr.width, tr.height, shape.d, shape.c, physicalBuffer);
                else
                    _ops.Pack4ToBufferCHW(tr.texture, tr.width, tr.height, physicalChannels, physicalBuffer);
                tempOwned.Add(physicalBuffer);

                var converted = RentTempBuffer(logicalCount, sizeof(float));
                _ops.CopyBufPartial(physicalBuffer, 0, converted, logicalCount);

                bufferBlobs[name] = converted;
                bufferViews[name] = new NcnnTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(converted);
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] convert-done | site=" + site + " | name=" + name + " | mode=partial | count=" + converted.count);
                return converted;
            }

            throw new InvalidOperationException("texture logical shape mismatch: " + name + " | physical=" + physicalCount + " logical=" + logicalCount);
        }

        internal static float[] ReadClipArrayAsFloat32(NcnnBinReader br, int count, int loadType)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();
            return br.ReadNcnnMatAsFloat32(count, 0, 0, 0, loadType);
        }

        internal static float[] ReadClipMatAsFloat32(NcnnBinReader br, int w, int h, int d, int c, int loadType)
        {
            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;
            return ReadClipArrayAsFloat32(br, count, loadType);
        }

        internal static float[] ReadPackedOrRawWeightArray(NcnnBinReader br, int count, string layerName)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();

            try
            {
                return ReadClipArrayAsFloat32(br, count, 0);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("failed to read weights for " + layerName + " at " + br.Position + ": " + e.Message, e);
            }
        }

        internal static float[] ReadFloatBuffer(ComputeBuffer buffer)
        {
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            var data = new float[buffer.count];
            buffer.GetData(data);
            return data;
        }

        internal static float[] RunGemmCpu(ComputeBuffer aBuf, NcnnTensorBuffer aView, GemmPack gp, float[] cDataOverride = null)
        {
            if (aBuf == null)
                throw new ArgumentNullException(nameof(aBuf));
            if (aView == null)
                throw new ArgumentNullException(nameof(aView));
            if (gp == null || gp.bDataCpu == null)
                throw new ArgumentNullException(nameof(gp));

            var m = aView.dims == 1 ? 1 : aView.h;
            var k = aView.w;
            var n = gp.constantN;
            var a = ReadFloatBuffer(aBuf);
            var b = gp.bDataCpu;
            var c = cDataOverride ?? gp.cDataCpu;
            var output = new float[m * n];

            for (var row = 0; row < m; row++)
            {
                var aBase = row * k;
                for (var col = 0; col < n; col++)
                {
                    double sum = 0.0;
                    for (var kk = 0; kk < k; kk++)
                    {
                        var aValue = a[aBase + kk];
                        var bValue = gp.transB
                            ? b[col * k + kk]
                            : b[kk * n + col];
                        sum += (double)aValue * bValue;
                    }

                    var value = gp.alpha * (float)sum;
                    if (gp.constantC && c != null && gp.broadcastTypeC != -1)
                    {
                        float cValue;
                        switch (gp.broadcastTypeC)
                        {
                            case 0: cValue = c[0]; break;
                            case 1: cValue = c[row]; break;
                            case 2: cValue = c[row]; break;
                            case 3: cValue = c[row * n + col]; break;
                            case 4: cValue = c[col]; break;
                            default: cValue = 0f; break;
                        }
                        value += gp.beta * cValue;
                    }

                    output[row * n + col] = value;
                }
            }

            return output;
        }

        internal NcnnTensorBuffer RunMatMulLayer(ComputeBuffer aBuf, NcnnTensorBuffer aView, ComputeBuffer bBuf, NcnnTensorBuffer bView, bool transB)
        {
            static bool ExceedsDispatchLimit(int width, int height, int tileSize)
            {
                const int maxGroups = 65535;
                return Mathf.CeilToInt(width / (float)tileSize) > maxGroups
                    || Mathf.CeilToInt(height / (float)tileSize) > maxGroups;
            }

            static void RunMatMulCpu(ComputeBuffer aBufCpu, ComputeBuffer bBufCpu, int rows, int cols, int shared, bool transBCpu, ComputeBuffer outputCpu)
            {
                var aData = ReadFloatBuffer(aBufCpu);
                var bData = ReadFloatBuffer(bBufCpu);
                var outputData = new float[rows * cols];
                for (var row = 0; row < rows; row++)
                {
                    var aBase = row * shared;
                    var outBase = row * cols;
                    for (var col = 0; col < cols; col++)
                    {
                        double sum = 0d;
                        for (var kk = 0; kk < shared; kk++)
                        {
                            var aValue = aData[aBase + kk];
                            var bValue = transBCpu
                                ? bData[col * shared + kk]
                                : bData[kk * cols + col];
                            sum += aValue * bValue;
                        }

                        outputData[outBase + col] = (float)sum;
                    }
                }

                outputCpu.SetData(outputData);
            }

            static void GetMatrixShape(NcnnTensorBuffer view, out int rows, out int cols)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                if (view.dims == 1)
                {
                    rows = 1;
                    cols = view.w;
                    return;
                }
                if (view.dims == 2 || view.dims == 3 || view.dims == 4)
                {
                    rows = view.h;
                    cols = view.w;
                    return;
                }
                throw new InvalidOperationException("MatMul currently supports dims 1/2/3/4 only");
            }

            static int GetBatchDepth(NcnnTensorBuffer view)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                return view.dims == 4 ? view.d : 1;
            }

            static int GetBatchChannels(NcnnTensorBuffer view)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                if (view.dims == 4 || view.dims == 3)
                    return view.c;
                return 1;
            }

            static int GetMatrixOffset(NcnnTensorBuffer view, int depthIndex, int channelIndex)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));

                var matrixCount = view.w * view.h;
                if (view.dims == 4)
                    return ((channelIndex * view.d) + depthIndex) * matrixCount;
                if (view.dims == 3)
                    return channelIndex * matrixCount;
                return 0;
            }

            GetMatrixShape(aView, out var aRows, out var aCols);
            GetMatrixShape(bView, out var bRows, out var bCols);

            var k = aCols;
            var kFromB = transB ? bCols : bRows;
            var n = transB ? bRows : bCols;
            if (k != kFromB)
                throw new InvalidOperationException("MatMul K mismatch: " + k + " vs " + kFromB);

            var batchDepthA = GetBatchDepth(aView);
            var batchDepthB = GetBatchDepth(bView);
            var batchChannelsA = GetBatchChannels(aView);
            var batchChannelsB = GetBatchChannels(bView);
            var batchDepth = Mathf.Max(batchDepthA, batchDepthB);
            var batchChannels = Mathf.Max(batchChannelsA, batchChannelsB);

            if (batchDepthA != 1 && batchDepthA != batchDepth)
                throw new InvalidOperationException("MatMul batchDepthA mismatch: " + batchDepthA + " vs " + batchDepth);
            if (batchDepthB != 1 && batchDepthB != batchDepth)
                throw new InvalidOperationException("MatMul batchDepthB mismatch: " + batchDepthB + " vs " + batchDepth);
            if (batchChannelsA != 1 && batchChannelsA != batchChannels)
                throw new InvalidOperationException("MatMul batchA mismatch: " + batchChannelsA + " vs " + batchChannels);
            if (batchChannelsB != 1 && batchChannelsB != batchChannels)
                throw new InvalidOperationException("MatMul batchB mismatch: " + batchChannelsB + " vs " + batchChannels);

            var aCount = aRows * aCols;
            var bCount = bRows * bCols;
            var outCount = aRows * n;
            var maxDims = Mathf.Max(aView.dims, bView.dims);
            var useCpuFallback = ExceedsDispatchLimit(n, aRows, 8);
            if (useCpuFallback)
            {
                DebugLog?.Invoke(
                    "[MatMul] cpu-fallback-dispatch-limit"
                    + " | rows=" + aRows.ToString(CultureInfo.InvariantCulture)
                    + " | cols=" + n.ToString(CultureInfo.InvariantCulture)
                    + " | k=" + k.ToString(CultureInfo.InvariantCulture)
                    + " | transB=" + transB.ToString());
            }

            if (batchDepth == 1 && batchChannels == 1)
            {
                var outTensor2D = RentTempTensorBuffer(2, n, aRows);
                if (useCpuFallback)
                    RunMatMulCpu(aBuf, bBuf, aRows, n, k, transB, outTensor2D.buffer);
                else
                    Ops.MatMul2D(aBuf, bBuf, aRows, n, k, transB, outTensor2D.buffer);
                return outTensor2D;
            }

            var outTensor = maxDims >= 4
                ? RentTempTensorBuffer(4, n, aRows, batchDepth, batchChannels)
                : RentTempTensorBuffer(3, n, aRows, 1, batchChannels);
            var tempA = (batchDepthA == 1 && batchChannelsA == 1) ? null : RentTempBuffer(aCount, sizeof(float));
            var tempB = (batchDepthB == 1 && batchChannelsB == 1) ? null : RentTempBuffer(bCount, sizeof(float));
            var tempOut = RentTempBuffer(outCount, sizeof(float));
            try
            {
                for (var pc = 0; pc < batchChannels; pc++)
                {
                    var aChannelIndex = batchChannelsA == 1 ? 0 : pc;
                    var bChannelIndex = batchChannelsB == 1 ? 0 : pc;
                    for (var pd = 0; pd < batchDepth; pd++)
                    {
                        var aDepthIndex = batchDepthA == 1 ? 0 : pd;
                        var bDepthIndex = batchDepthB == 1 ? 0 : pd;

                        var aSrc = aBuf;
                        var bSrc = bBuf;
                        if (tempA != null)
                        {
                            Ops.CopyBufPartial(aBuf, GetMatrixOffset(aView, aDepthIndex, aChannelIndex), tempA, aCount);
                            aSrc = tempA;
                        }

                        if (tempB != null)
                        {
                            Ops.CopyBufPartial(bBuf, GetMatrixOffset(bView, bDepthIndex, bChannelIndex), tempB, bCount);
                            bSrc = tempB;
                        }

                        if (useCpuFallback)
                            RunMatMulCpu(aSrc, bSrc, aRows, n, k, transB, tempOut);
                        else
                            Ops.MatMul2D(aSrc, bSrc, aRows, n, k, transB, tempOut);
                        var outOffset = maxDims >= 4
                            ? ((pc * batchDepth) + pd) * outCount
                            : pc * outCount;
                        Ops.CopyBufPartial(tempOut, 0, outTensor.buffer, outCount, outOffset);
                    }
                }
            }
            finally
            {
                if (tempA != null)
                    ReturnTempBuffer(tempA);
                if (tempB != null)
                    ReturnTempBuffer(tempB);
                ReturnTempBuffer(tempOut);
            }

            return outTensor;
        }

        internal void Consume(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null)
                {
                    ReleaseTextureRef(tr);
                }
                textureBlobs.Remove(name);
            }
        }

        internal void Consume(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferRef> bufferRefs,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (textureBlobs.TryGetValue(name, out var tr) && tr != null)
                {
                    ReleaseTextureRef(tr);
                }

                if (bufferRefs != null && bufferRefs.TryGetValue(name, out var br) && br != null)
                {
                    br.refs--;
                    if (br.refs <= 0 && br.owned && br.buffer != null)
                    {
                        try { ReturnTempBuffer(br.buffer); } catch { }
                        br.buffer = null;
                    }
                    bufferRefs.Remove(name);
                }

                textureBlobs.Remove(name);
                bufferViews?.Remove(name);
                bufferBlobs?.Remove(name);
            }
        }

        internal void ConsumeIndex(
            Dictionary<string, IndexRef> indexBlobs,
            Dictionary<string, int> remaining,
            string[] bottomNames,
            ICollection<string> pinnedNames)
        {
            if (indexBlobs == null || bottomNames == null)
                return;

            for (var i = 0; i < bottomNames.Length; i++)
            {
                var name = bottomNames[i];
                if (!remaining.TryGetValue(name, out var count))
                    continue;

                count--;
                remaining[name] = count;
                if (count > 0)
                    continue;
                if (pinnedNames != null && pinnedNames.Contains(name))
                    continue;

                if (indexBlobs.TryGetValue(name, out var ir) && ir != null)
                {
                    ir.refs--;
                    if (ir.refs <= 0 && ir.owned)
                    {
                        if (ir.texture != null)
                        {
                            try { ReturnTempArray(ir.texture); } catch { }
                            ir.texture = null;
                        }
                        if (ir.buffer != null)
                        {
                            try { ReturnTempBuffer(ir.buffer); } catch { }
                            ir.buffer = null;
                        }
                        ir.view = null;
                    }
                }
                indexBlobs.Remove(name);
            }
        }

        internal static NcnnTensorBuffer ResolveReshapeTensor(NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            return ResolveReshapeTensor(src, layer, null);
        }

        internal static NcnnTensorBuffer ResolveReshapeTensor(NcnnTensorBuffer src, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
            {
                var exprShape = EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes ?? new[] { new BufferShape(src.dims, src.w, src.h, src.d, src.c) }, layer);
                return src.Reshape(exprShape.dims, exprShape.w, exprShape.h, exprShape.d, exprShape.c);
            }

            var outw = layer.GetInt(0, -233);
            var outh = layer.GetInt(1, -233);
            var outd = layer.GetInt(11, -233);
            var outc = layer.GetInt(2, -233);
            if (outw == -233 && outh == -233 && outd == -233 && outc == -233)
            {
                // Some pnnx-lowered graphs emit param-less Reshape nodes as logical rank markers.
                // Native ncnn does not preserve >4D tensor metadata here, so keep the current view
                // and let the surrounding lowered layers consume the existing buffer layout.
                return src;
            }
            var ndim = 4;
            if (outd == -233) ndim = 3;
            if (outc == -233) ndim = 2;
            if (outh == -233) ndim = 1;

            var total = src.elementCount;

            static int SafeDiv(int a, int b, string reason)
            {
                if (b == 0 || (a % b) != 0)
                    throw new InvalidOperationException(reason + " | " + a + " / " + b);
                return a / b;
            }

            if (ndim == 1)
            {
                if (outw == 0) outw = src.w;
                if (outw == -1) outw = total;
                return src.Reshape(1, outw);
            }

            if (ndim == 2)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outw == -1) outw = SafeDiv(total, outh, "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, outw, "Reshape outh");
                return src.Reshape(2, outw, outh);
            }

            if (ndim == 3)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outc == 0) outc = src.c;
                if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outh), "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw), "Reshape outh");
                if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
                return src.Reshape(3, outw, outh, 1, outc);
            }

            if (outw == 0) outw = src.w;
            if (outh == 0) outh = src.h;
            if (outd == 0) outd = src.d;
            if (outc == 0) outc = src.c;
            if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outh), "Reshape outw");
            if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outw), "Reshape outh");
            if (outd == -1) outd = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outd");
            if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outd) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
            return src.Reshape(4, outw, outh, outd, outc);
        }

        internal static BufferShape ResolveReshapeShape(BufferShape src, NcnnParamModel.Layer layer)
        {
            return ResolveReshapeShape(src, layer, null);
        }

        internal static BufferShape ResolveReshapeShape(BufferShape src, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            if (!string.IsNullOrWhiteSpace(layer.GetString(6, null)))
                return EvaluateReshapeShapeExpression(layer.GetString(6, null), bottomShapes ?? new[] { src }, layer);

            var outw = layer.GetInt(0, -233);
            var outh = layer.GetInt(1, -233);
            var outd = layer.GetInt(11, -233);
            var outc = layer.GetInt(2, -233);
            if (outw == -233 && outh == -233 && outd == -233 && outc == -233)
                return src;
            var ndim = 4;
            if (outd == -233) ndim = 3;
            if (outc == -233) ndim = 2;
            if (outh == -233) ndim = 1;

            var total = src.w * src.h * src.d * src.c;

            static int SafeDiv(int a, int b, string reason)
            {
                if (b == 0 || (a % b) != 0)
                    throw new InvalidOperationException(reason + " | " + a + " / " + b);
                return a / b;
            }

            if (ndim == 1)
            {
                if (outw == 0) outw = src.w;
                if (outw == -1) outw = total;
                return new BufferShape(1, outw, 1, 1, 1);
            }

            if (ndim == 2)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outw == -1) outw = SafeDiv(total, outh, "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, outw, "Reshape outh");
                return new BufferShape(2, outw, outh, 1, 1);
            }

            if (ndim == 3)
            {
                if (outw == 0) outw = src.w;
                if (outh == 0) outh = src.h;
                if (outc == 0) outc = src.c;
                if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outh), "Reshape outw");
                if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw), "Reshape outh");
                if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
                return new BufferShape(3, outw, outh, 1, outc);
            }

            if (outw == 0) outw = src.w;
            if (outh == 0) outh = src.h;
            if (outd == 0) outd = src.d;
            if (outc == 0) outc = src.c;
            if (outw == -1) outw = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outh), "Reshape outw");
            if (outh == -1) outh = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outd) * Mathf.Max(1, outw), "Reshape outh");
            if (outd == -1) outd = SafeDiv(total, Mathf.Max(1, outc) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outd");
            if (outc == -1) outc = SafeDiv(total, Mathf.Max(1, outd) * Mathf.Max(1, outw) * Mathf.Max(1, outh), "Reshape outc");
            return new BufferShape(4, outw, outh, outd, outc);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            return EvaluateReshapeShapeExpression(expr, new[] { new BufferShape(src.dims, src.w, src.h, src.d, src.c) }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, BufferShape src, NcnnParamModel.Layer layer)
        {
            return EvaluateReshapeShapeExpression(expr, new[] { src }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                throw new ArgumentException("shape expr is empty", nameof(expr));
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));
            if (bottomShapes == null || bottomShapes.Count == 0)
                throw new ArgumentException("bottomShapes is empty", nameof(bottomShapes));

            var values = EvaluateExpressionList(expr, bottomShapes, layer);
            if (values.Count <= 0 || values.Count > 4)
                throw new InvalidOperationException("Unsupported reshape shape_expr rank: " + values.Count + " | " + layer.name);

            if (values.Count == 1)
                return new BufferShape(1, values[0], 1, 1, 1);
            if (values.Count == 2)
                return new BufferShape(2, values[0], values[1], 1, 1);
            if (values.Count == 3)
                return new BufferShape(3, values[0], values[1], 1, values[2]);
            return new BufferShape(4, values[0], values[1], values[2], values[3]);
        }

        internal static IReadOnlyList<int> EvaluateExpressionList(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                throw new ArgumentException("expression is empty", nameof(expr));
            if (bottomShapes == null || bottomShapes.Count == 0)
                throw new ArgumentException("bottomShapes is empty", nameof(bottomShapes));

            var trimmed = expr.Trim();
            if (trimmed.Length >= 2 && trimmed[0] == '"' && trimmed[trimmed.Length - 1] == '"')
                trimmed = trimmed.Substring(1, trimmed.Length - 2);

            var tokens = new List<string>();
            var token = new StringBuilder();
            for (var i = 0; i < trimmed.Length; i++)
            {
                var ch = trimmed[i];
                if (ch == '(' || ch == ')' || ch == ',')
                {
                    if (token.Length > 0)
                    {
                        tokens.Add(token.ToString());
                        token.Clear();
                    }
                }
                else if (!char.IsWhiteSpace(ch))
                {
                    token.Append(ch);
                }
            }
            if (token.Length > 0)
                tokens.Add(token.ToString());

            var stack = new Stack<ExprValue>();
            for (var i = tokens.Count - 1; i >= 0; i--)
            {
                var t = tokens[i];
                if (IsShapeRefToken(t))
                {
                    stack.Push(new ExprValue(GetShapeRefValue(t, bottomShapes, layer)));
                    continue;
                }

                if (t == "+" || t == "-" || t == "*" || t == "//" || t == "max" || t == "min")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryIntPref(t, a, b, layer));
                    continue;
                }

                if (t == "abs" || t == "neg" || t == "sign" || t == "square")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnarySimple(t, a));
                    continue;
                }

                if (t == "trunc" || t == "ceil" || t == "floor" || t == "round")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnaryRound(t, a));
                    continue;
                }

                if (t == "acos" || t == "acosh" || t == "asin" || t == "asinh" || t == "atan" || t == "atanh"
                    || t == "cos" || t == "cosh" || t == "erf" || t == "exp" || t == "log" || t == "log10"
                    || t == "reciprocal" || t == "rsqrt" || t == "sin" || t == "sinh" || t == "sqrt" || t == "tan" || t == "tanh")
                {
                    var a = PopExpr(stack, layer);
                    stack.Push(ApplyUnaryFloat(t, a));
                    continue;
                }

                if (t == "/" || t == "atan2" || t == "fmod" || t == "pow" || t == "remainder" || t == "logaddexp")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryFloat(t, a, b));
                    continue;
                }

                if (t == "and" || t == "or" || t == "xor" || t == "lshift" || t == "rshift")
                {
                    var a = PopExpr(stack, layer);
                    var b = PopExpr(stack, layer);
                    stack.Push(ApplyBinaryBitwise(t, a, b));
                    continue;
                }

                if (TryParseExprLiteral(t, out var literal))
                {
                    stack.Push(literal);
                    continue;
                }

                throw new InvalidOperationException("Malformed expression token " + t + " | " + layer?.name);
            }

            var values = new List<int>(stack.Count);
            while (stack.Count > 0)
                values.Add(stack.Pop().ToInt());
            return values;
        }

        internal static int[] EvaluateExpressionListOrNull(string expr, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            if (string.IsNullOrWhiteSpace(expr))
                return null;
            var values = EvaluateExpressionList(expr, bottomShapes, layer);
            if (values == null || values.Count == 0)
                return Array.Empty<int>();
            var arr = new int[values.Count];
            for (var i = 0; i < values.Count; i++)
                arr[i] = values[i];
            return arr;
        }

        private readonly struct ExprValue
        {
            public readonly bool isFloat;
            public readonly int i;
            public readonly float f;

            public ExprValue(int value)
            {
                isFloat = false;
                i = value;
                f = value;
            }

            public ExprValue(float value)
            {
                isFloat = true;
                i = (int)value;
                f = value;
            }

            public int ToInt() => isFloat ? (int)f : i;
            public float ToFloat() => isFloat ? f : i;
        }

        private static ExprValue PopExpr(Stack<ExprValue> stack, NcnnParamModel.Layer layer)
        {
            if (stack == null || stack.Count == 0)
                throw new InvalidOperationException("Malformed expression stack underflow: " + layer?.name);
            return stack.Pop();
        }

        private static bool IsShapeRefToken(string token)
        {
            return token != null
                && token.Length == 2
                && token[0] >= '0' && token[0] <= '9'
                && (token[1] == 'w' || token[1] == 'h' || token[1] == 'd' || token[1] == 'c');
        }

        private static int GetShapeRefValue(string token, IReadOnlyList<BufferShape> bottomShapes, NcnnParamModel.Layer layer)
        {
            var index = token[0] - '0';
            if (index < 0 || index >= bottomShapes.Count)
                throw new InvalidOperationException("shape expression blob index out of range: " + token + " | " + layer?.name);

            var shape = bottomShapes[index];
            return token[1] switch
            {
                'w' => shape.w,
                'h' => shape.h,
                'd' => shape.d,
                'c' => shape.c,
                _ => throw new InvalidOperationException("invalid shape ref token: " + token)
            };
        }

        private static bool TryParseExprLiteral(string token, out ExprValue value)
        {
            if (int.TryParse(token, NumberStyles.Integer, CultureInfo.InvariantCulture, out var i)
                && float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var fi)
                && Mathf.Approximately(i, fi))
            {
                value = new ExprValue(i);
                return true;
            }

            if (float.TryParse(token, NumberStyles.Float, CultureInfo.InvariantCulture, out var f))
            {
                value = new ExprValue(f);
                return true;
            }

            value = default;
            return false;
        }

        private static ExprValue ApplyBinaryIntPref(string op, ExprValue a, ExprValue b, NcnnParamModel.Layer layer)
        {
            if (!a.isFloat && !b.isFloat)
            {
                return op switch
                {
                    "+" => new ExprValue(a.i + b.i),
                    "-" => new ExprValue(a.i - b.i),
                    "*" => new ExprValue(a.i * b.i),
                    "//" => b.i != 0 ? new ExprValue(a.i / b.i) : throw new InvalidOperationException("expr divide by zero | " + layer?.name),
                    "max" => new ExprValue(Mathf.Max(a.i, b.i)),
                    "min" => new ExprValue(Mathf.Min(a.i, b.i)),
                    _ => throw new InvalidOperationException("unsupported int-pref op: " + op)
                };
            }

            var af = a.ToFloat();
            var bf = b.ToFloat();
            return op switch
            {
                "+" => new ExprValue(af + bf),
                "-" => new ExprValue(af - bf),
                "*" => new ExprValue(af * bf),
                "//" => new ExprValue(Mathf.Floor(af / bf)),
                "max" => new ExprValue(Mathf.Max(af, bf)),
                "min" => new ExprValue(Mathf.Min(af, bf)),
                _ => throw new InvalidOperationException("unsupported float-pref op: " + op)
            };
        }

        private static ExprValue ApplyUnarySimple(string op, ExprValue a)
        {
            if (!a.isFloat)
            {
                return op switch
                {
                    "abs" => new ExprValue(Mathf.Abs(a.i)),
                    "neg" => new ExprValue(-a.i),
                    "sign" => new ExprValue(a.i > 0 ? 1 : (a.i == 0 ? 0 : -1)),
                    "square" => new ExprValue(a.i * a.i),
                    _ => throw new InvalidOperationException("unsupported unary op: " + op)
                };
            }

            var af = a.f;
            return op switch
            {
                "abs" => new ExprValue(Mathf.Abs(af)),
                "neg" => new ExprValue(-af),
                "sign" => new ExprValue(af > 0f ? 1f : (af == 0f ? 0f : -1f)),
                "square" => new ExprValue(af * af),
                _ => throw new InvalidOperationException("unsupported unary op: " + op)
            };
        }

        private static ExprValue ApplyUnaryRound(string op, ExprValue a)
        {
            if (!a.isFloat)
                return new ExprValue(a.i);

            return op switch
            {
                "trunc" => new ExprValue((int)a.f),
                "ceil" => new ExprValue((int)Math.Ceiling(a.f)),
                "floor" => new ExprValue((int)Math.Floor(a.f)),
                "round" => new ExprValue((int)Math.Round(a.f, MidpointRounding.AwayFromZero)),
                _ => throw new InvalidOperationException("unsupported round op: " + op)
            };
        }

        private static ExprValue ApplyUnaryFloat(string op, ExprValue a)
        {
            var af = a.ToFloat();
            return op switch
            {
                "acos" => new ExprValue(Mathf.Acos(af)),
                "acosh" => new ExprValue((float)Acosh(af)),
                "asin" => new ExprValue(Mathf.Asin(af)),
                "asinh" => new ExprValue((float)Asinh(af)),
                "atan" => new ExprValue(Mathf.Atan(af)),
                "atanh" => new ExprValue((float)Atanh(af)),
                "cos" => new ExprValue(Mathf.Cos(af)),
                "cosh" => new ExprValue((float)Math.Cosh(af)),
                "erf" => new ExprValue((float)Erf(af)),
                "exp" => new ExprValue(Mathf.Exp(af)),
                "log" => new ExprValue(Mathf.Log(af)),
                "log10" => new ExprValue(Mathf.Log10(af)),
                "reciprocal" => new ExprValue(1f / af),
                "rsqrt" => new ExprValue(1f / Mathf.Sqrt(af)),
                "sin" => new ExprValue(Mathf.Sin(af)),
                "sinh" => new ExprValue((float)Math.Sinh(af)),
                "sqrt" => new ExprValue(Mathf.Sqrt(af)),
                "tan" => new ExprValue(Mathf.Tan(af)),
                "tanh" => new ExprValue((float)Math.Tanh(af)),
                _ => throw new InvalidOperationException("unsupported float unary op: " + op)
            };
        }

        private static ExprValue ApplyBinaryFloat(string op, ExprValue a, ExprValue b)
        {
            var af = a.ToFloat();
            var bf = b.ToFloat();
            return op switch
            {
                "/" => new ExprValue(af / bf),
                "atan2" => new ExprValue(Mathf.Atan2(af, bf)),
                "fmod" => new ExprValue(af % bf),
                "pow" => new ExprValue(Mathf.Pow(af, bf)),
                "remainder" => new ExprValue(RepeatRemainder(af, bf)),
                "logaddexp" => new ExprValue(Mathf.Log(Mathf.Exp(af) + Mathf.Exp(bf))),
                _ => throw new InvalidOperationException("unsupported float binary op: " + op)
            };
        }

        private static ExprValue ApplyBinaryBitwise(string op, ExprValue a, ExprValue b)
        {
            var ai = a.ToInt();
            var bi = b.ToInt();
            return op switch
            {
                "and" => new ExprValue(ai & bi),
                "or" => new ExprValue(ai | bi),
                "xor" => new ExprValue(ai ^ bi),
                "lshift" => new ExprValue(ai << bi),
                "rshift" => new ExprValue(ai >> bi),
                _ => throw new InvalidOperationException("unsupported bitwise op: " + op)
            };
        }

        private static float RepeatRemainder(float a, float b)
        {
            var r = a % b;
            if (a * b < 0f)
                r += b;
            return r;
        }

        private static double Erf(double x)
        {
            var sign = x < 0 ? -1d : 1d;
            x = Math.Abs(x);

            var a1 = 0.254829592d;
            var a2 = -0.284496736d;
            var a3 = 1.421413741d;
            var a4 = -1.453152027d;
            var a5 = 1.061405429d;
            var p = 0.3275911d;
            var t = 1d / (1d + p * x);
            var y = 1d - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.Exp(-x * x);
            return sign * y;
        }

        private static double Acosh(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x - 1d));
        }

        private static double Asinh(double x)
        {
            return Math.Log(x + Math.Sqrt(x * x + 1d));
        }

        private static double Atanh(double x)
        {
            return 0.5d * Math.Log((1d + x) / (1d - x));
        }

        internal static BufferShape GetTextureShape(Dictionary<string, BufferShape> textureShapes, TensorRef tr, string name)
        {
            return GetTextureContract(textureShapes, tr, name).LogicalShape;
        }

        internal static BufferShape GetTextureStorageShape(TensorRef tr, BufferShape fallbackLogicalShape)
        {
            if (tr == null || tr.texture == null)
                return fallbackLogicalShape;
            return GetTextureContract(null, tr, string.Empty).StorageShape;
        }

        internal static BufferShape GetCmdStorageShape(CmdTensorRef tr, BufferShape fallbackLogicalShape)
        {
            if (tr == null || tr.texture == null)
                return fallbackLogicalShape;
            return GetCmdTensorContract(tr).StorageShape;
        }

        internal static bool MatchesPack4TextureStorage(TensorRef tr, BufferShape logicalShape)
        {
            if (tr == null || tr.texture == null)
                return false;

            var storageShape = GetTextureStorageShape(tr, logicalShape);
            if (storageShape.w != tr.width || storageShape.h != tr.height)
                return false;

            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));
            if (storageShape.dims == 4)
            {
                if (tr.packs != expectedPacks)
                    return false;
                var expectedVolumeDepth = Mathf.Max(1, storageShape.d) * expectedPacks;
                return Mathf.Max(1, tr.texture.volumeDepth) == expectedVolumeDepth;
            }

            return tr.packs == Mathf.Max(1, tr.texture.volumeDepth);
        }

        internal static bool MatchesPack4TextureStorage(CmdTensorRef tr, BufferShape logicalShape)
        {
            if (tr == null || tr.texture == null)
                return false;

            var storageShape = GetCmdStorageShape(tr, logicalShape);
            if (storageShape.w != tr.width || storageShape.h != tr.height)
                return false;

            var expectedPacks = Mathf.Max(1, Mathf.CeilToInt(Mathf.Max(1, storageShape.c) / 4f));
            if (storageShape.dims == 4)
            {
                if (tr.packs != expectedPacks)
                    return false;
                var expectedDepth = Mathf.Max(1, storageShape.d) * expectedPacks;
                return Mathf.Max(1, tr.texture.depth) == expectedDepth;
            }

            return tr.packs == Mathf.Max(1, tr.texture.depth);
        }

        internal int ResolveInputLogicalChannels(string inputBlobName, int fallbackChannels)
        {
            if (Model?.layers == null || string.IsNullOrWhiteSpace(inputBlobName))
                return fallbackChannels;

            for (var i = 0; i < Model.layers.Count; i++)
            {
                var layer = Model.layers[i];
                if (layer?.bottomNames == null || layer.bottomNames.Length == 0)
                    continue;
                if (!string.Equals(layer.bottomNames[0], inputBlobName, StringComparison.Ordinal))
                    continue;
                if (layer.type != NcnnLayerTypes.Convolution
                    && layer.type != NcnnLayerTypes.ConvolutionDepthWise)
                    continue;
                if (_conv.TryGetValue(layer.name, out var conv) && conv != null && conv.inC > 0)
                    return conv.inC;
            }

            return fallbackChannels;
        }

        internal static Vector4Int ResolvePermuteAxes(int dims, int orderType, string layerName)
        {
            if (dims == 2)
            {
                return orderType switch
                {
                    0 => new Vector4Int(0, 1, 0, 0),
                    1 => new Vector4Int(1, 0, 0, 0),
                    _ => throw new InvalidOperationException("unsupported permute dims=2 orderType: " + orderType + " | " + layerName)
                };
            }

            if (dims == 3)
            {
                return orderType switch
                {
                    0 => new Vector4Int(0, 1, 2, 0),
                    1 => new Vector4Int(1, 0, 2, 0),
                    2 => new Vector4Int(0, 2, 1, 0),
                    3 => new Vector4Int(2, 0, 1, 0),
                    4 => new Vector4Int(1, 2, 0, 0),
                    5 => new Vector4Int(2, 1, 0, 0),
                    _ => throw new InvalidOperationException("unsupported permute dims=3 orderType: " + orderType + " | " + layerName)
                };
            }

            return orderType switch
            {
                0 => new Vector4Int(0, 1, 2, 3),
                1 => new Vector4Int(1, 0, 2, 3),
                2 => new Vector4Int(0, 2, 1, 3),
                3 => new Vector4Int(2, 0, 1, 3),
                4 => new Vector4Int(1, 2, 0, 3),
                5 => new Vector4Int(2, 1, 0, 3),
                6 => new Vector4Int(0, 1, 3, 2),
                7 => new Vector4Int(1, 0, 3, 2),
                8 => new Vector4Int(0, 3, 1, 2),
                9 => new Vector4Int(3, 0, 1, 2),
                10 => new Vector4Int(1, 3, 0, 2),
                11 => new Vector4Int(3, 1, 0, 2),
                12 => new Vector4Int(0, 2, 3, 1),
                13 => new Vector4Int(2, 0, 3, 1),
                14 => new Vector4Int(0, 3, 2, 1),
                15 => new Vector4Int(3, 0, 2, 1),
                16 => new Vector4Int(2, 3, 0, 1),
                17 => new Vector4Int(3, 2, 0, 1),
                18 => new Vector4Int(1, 2, 3, 0),
                19 => new Vector4Int(2, 1, 3, 0),
                20 => new Vector4Int(1, 3, 2, 0),
                21 => new Vector4Int(3, 1, 2, 0),
                22 => new Vector4Int(2, 3, 1, 0),
                23 => new Vector4Int(3, 2, 1, 0),
                _ => throw new InvalidOperationException("unsupported permute dims=4 orderType: " + orderType + " | " + layerName)
            };
        }

        internal static BufferShape ResolvePermuteShape(BufferShape src, int dims, Vector4Int axes)
        {
            int GetAxisSize(int axis)
            {
                if (axis == 0) return src.w;
                if (axis == 1) return src.h;
                if (axis == 2) return dims == 4 ? src.d : src.c;
                if (axis == 3) return src.c;
                throw new InvalidOperationException("invalid axis: " + axis);
            }

            var outW = GetAxisSize(axes.x);
            var outH = GetAxisSize(axes.y);
            var outD = dims == 4 ? GetAxisSize(axes.z) : 1;
            var outC = dims == 2 ? 1 : GetAxisSize(dims == 4 ? axes.w : axes.z);
            return new BufferShape(dims, outW, outH, outD, outC);
        }

        internal static BufferShape ResolvePermuteShape(NcnnTensorBuffer src, int dims, Vector4Int axes)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            return ResolvePermuteShape(new BufferShape(src.dims, src.w, src.h, src.d, src.c), dims, axes);
        }

        internal static int MapNcnnAxisToTensorAxis(int dims, int axis)
        {
            if (dims == 1)
                return 0;
            if (dims == 2)
                return axis == 0 ? 1 : 0;
            if (dims == 3)
            {
                if (axis == 0) return 2;
                if (axis == 1) return 1;
                return 0;
            }

            if (axis == 0) return 3;
            if (axis == 1) return 2;
            if (axis == 2) return 1;
            return 0;
        }

        internal static int GetAxisSize(int dims, int w, int h, int d, int c, int axis)
        {
            if (axis == 0) return w;
            if (axis == 1) return h;
            if (axis == 2) return dims == 4 ? d : c;
            if (axis == 3) return c;
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        internal readonly struct CropRoi
        {
            public readonly int woffset;
            public readonly int hoffset;
            public readonly int doffset;
            public readonly int coffset;
            public readonly int outw;
            public readonly int outh;
            public readonly int outd;
            public readonly int outc;

            public CropRoi(int woffset, int hoffset, int doffset, int coffset, int outw, int outh, int outd, int outc)
            {
                this.woffset = woffset;
                this.hoffset = hoffset;
                this.doffset = doffset;
                this.coffset = coffset;
                this.outw = outw;
                this.outh = outh;
                this.outd = outd;
                this.outc = outc;
            }
        }

        internal static BufferShape GetShapeOf(NcnnTensorBuffer view)
        {
            if (view == null)
                throw new ArgumentNullException(nameof(view));
            return new BufferShape(view.dims, view.w, view.h, view.d, view.c);
        }

        internal static BufferShape ResolveSqueezeShape(BufferShape src, NcnnParamModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            bool squeezeW = false;
            bool squeezeH = false;
            bool squeezeD = false;
            bool squeezeC = false;

            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, null);

            if (axes == null || axes.Length == 0)
            {
                squeezeW = src.w == 1 && layer.GetInt(0, 0) != 0;
                squeezeH = src.h == 1 && layer.GetInt(1, 0) != 0;
                squeezeD = src.d == 1 && layer.GetInt(11, 0) != 0;
                squeezeC = src.c == 1 && layer.GetInt(2, 0) != 0;
            }
            else
            {
                for (var i = 0; i < axes.Length; i++)
                {
                    var axis = axes[i];
                    if (axis < 0)
                        axis += src.dims;

                    if (src.dims == 1 && axis == 0) squeezeW = src.w == 1;
                    if (src.dims == 2 && axis == 0) squeezeH = src.h == 1;
                    if (src.dims == 2 && axis == 1) squeezeW = src.w == 1;
                    if (src.dims == 3 && axis == 0) squeezeC = src.c == 1;
                    if (src.dims == 3 && axis == 1) squeezeH = src.h == 1;
                    if (src.dims == 3 && axis == 2) squeezeW = src.w == 1;
                    if (src.dims == 4 && axis == 0) squeezeC = src.c == 1;
                    if (src.dims == 4 && axis == 1) squeezeD = src.d == 1;
                    if (src.dims == 4 && axis == 2) squeezeH = src.h == 1;
                    if (src.dims == 4 && axis == 3) squeezeW = src.w == 1;
                }
            }

            if (src.dims == 1)
            {
                if (squeezeW)
                    return new BufferShape(1, 1, 1, 1, 1);
                return src;
            }

            if (src.dims == 2)
            {
                if (squeezeW && squeezeH) return new BufferShape(1, 1, 1, 1, 1);
                if (squeezeW) return new BufferShape(1, src.h, 1, 1, 1);
                if (squeezeH) return new BufferShape(1, src.w, 1, 1, 1);
                return src;
            }

            if (src.dims == 3)
            {
                if (squeezeW && squeezeH && squeezeC) return new BufferShape(1, 1, 1, 1, 1);
                if (squeezeW && squeezeH) return new BufferShape(1, src.c, 1, 1, 1);
                if (squeezeH && squeezeC) return new BufferShape(1, src.w, 1, 1, 1);
                if (squeezeW && squeezeC) return new BufferShape(1, src.h, 1, 1, 1);
                if (squeezeW) return new BufferShape(2, src.h, src.c, 1, 1);
                if (squeezeH) return new BufferShape(2, src.w, src.c, 1, 1);
                if (squeezeC) return new BufferShape(2, src.w, src.h, 1, 1);
                return src;
            }

            if (squeezeW && squeezeH && squeezeD && squeezeC) return new BufferShape(1, 1, 1, 1, 1);
            if (squeezeW && squeezeH && squeezeD) return new BufferShape(1, src.c, 1, 1, 1);
            if (squeezeH && squeezeD && squeezeC) return new BufferShape(1, src.w, 1, 1, 1);
            if (squeezeW && squeezeD && squeezeC) return new BufferShape(1, src.h, 1, 1, 1);
            if (squeezeW && squeezeH && squeezeC) return new BufferShape(1, src.d, 1, 1, 1);
            if (squeezeW && squeezeH) return new BufferShape(2, src.d, src.c, 1, 1);
            if (squeezeW && squeezeD) return new BufferShape(2, src.h, src.c, 1, 1);
            if (squeezeH && squeezeD) return new BufferShape(2, src.w, src.c, 1, 1);
            if (squeezeH && squeezeC) return new BufferShape(2, src.w, src.d, 1, 1);
            if (squeezeW && squeezeC) return new BufferShape(2, src.h, src.d, 1, 1);
            if (squeezeD && squeezeC) return new BufferShape(2, src.w, src.h, 1, 1);
            if (squeezeW) return new BufferShape(3, src.h, src.d, 1, src.c);
            if (squeezeH) return new BufferShape(3, src.w, src.d, 1, src.c);
            if (squeezeD) return new BufferShape(3, src.w, src.h, 1, src.c);
            if (squeezeC) return new BufferShape(3, src.w, src.h, 1, src.d);
            return src;
        }

        internal static NcnnTensorBuffer ResolveSqueezeView(NcnnTensorBuffer src, NcnnParamModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            var shape = ResolveSqueezeShape(GetShapeOf(src), layer);
            return src.Reshape(shape.dims, shape.w, shape.h, shape.d, shape.c);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, NcnnParamModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            var startsExpr = layer.GetString(19, null);
            var endsExpr = layer.GetString(20, null);
            var axesExpr = layer.GetString(21, null);
            var starts = layer.GetInts(-23309, null);
            var ends = layer.GetInts(-23310, null);
            var axes = layer.GetInts(-23311, null);

            var hasExprSlice = !string.IsNullOrWhiteSpace(startsExpr) && !string.IsNullOrWhiteSpace(endsExpr);
            if (hasExprSlice)
            {
                starts = EvaluateExpressionListOrNull(startsExpr, bottomShapes, layer);
                ends = EvaluateExpressionListOrNull(endsExpr, bottomShapes, layer);
                axes = EvaluateExpressionListOrNull(axesExpr, bottomShapes, layer);
            }

            if (starts != null && ends != null && starts.Length > 0 && ends.Length > 0)
                return ResolveCropRoiFromSlice(srcShape, starts, ends, axes, layer.name);

            var dims = srcShape.dims;
            var w = srcShape.w;
            var h = srcShape.h;
            var d = srcShape.d;
            var c = srcShape.c;

            var woffset = layer.GetInt(0, 0);
            var hoffset = layer.GetInt(1, 0);
            var doffset = layer.GetInt(13, 0);
            var coffset = layer.GetInt(2, 0);
            var outw = w;
            var outh = h;
            var outd = d;
            var outc = c;
            var woffset2 = layer.GetInt(6, 0);
            var hoffset2 = layer.GetInt(7, 0);
            var doffset2 = layer.GetInt(15, 0);
            var coffset2 = layer.GetInt(8, 0);

            if (dims == 1)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
            }
            else if (dims == 2)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
            }
            else if (dims == 3)
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
                outc = c - coffset - coffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(5))
                    outc = Math.Min(layer.GetInt(5, outc), outc);
            }
            else
            {
                outw = w - woffset - woffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(3))
                    outw = Math.Min(layer.GetInt(3, outw), outw);
                outh = h - hoffset - hoffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(4))
                    outh = Math.Min(layer.GetInt(4, outh), outh);
                outd = d - doffset - doffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(14))
                    outd = Math.Min(layer.GetInt(14, outd), outd);
                outc = c - coffset - coffset2;
                if (layer.intParams != null && layer.intParams.ContainsKey(5))
                    outc = Math.Min(layer.GetInt(5, outc), outc);
            }

            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, outw, outh, outd, outc), srcShape, layer.name);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, BufferShape referenceShape, NcnnParamModel.Layer layer)
        {
            if (layer == null)
                throw new ArgumentNullException(nameof(layer));

            var woffset = layer.GetInt(0, 0);
            var hoffset = layer.GetInt(1, 0);
            var doffset = layer.GetInt(13, 0);
            var coffset = layer.GetInt(2, 0);

            if (srcShape.dims == 1)
                return ValidateCropRoi(new CropRoi(woffset, 0, 0, 0, referenceShape.w, 1, 1, 1), srcShape, layer.name);
            if (srcShape.dims == 2)
                return ValidateCropRoi(new CropRoi(woffset, hoffset, 0, 0, referenceShape.w, referenceShape.h, 1, 1), srcShape, layer.name);
            if (srcShape.dims == 3)
                return ValidateCropRoi(new CropRoi(woffset, hoffset, 0, coffset, referenceShape.w, referenceShape.h, 1, referenceShape.dims == 3 ? referenceShape.c : srcShape.c), srcShape, layer.name);
            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, referenceShape.w, referenceShape.h, referenceShape.d, referenceShape.dims == 4 ? referenceShape.c : srcShape.c), srcShape, layer.name);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, int[] paramData, NcnnParamModel.Layer layer)
        {
            if (paramData == null)
                throw new ArgumentNullException(nameof(paramData));

            CropRoi roi;
            if (srcShape.dims == 1)
                roi = new CropRoi(paramData[0], 0, 0, 0, paramData[3], 1, 1, 1);
            else if (srcShape.dims == 2)
                roi = new CropRoi(paramData[0], paramData[1], 0, 0, paramData[3], paramData[4], 1, 1);
            else if (srcShape.dims == 3)
                roi = new CropRoi(paramData[0], paramData[1], 0, paramData[2], paramData[3], paramData[4], 1, paramData[5]);
            else
                roi = new CropRoi(paramData[0], paramData[1], paramData[2], paramData[3], paramData[4], paramData[5], paramData[6], paramData[7]);

            return ValidateCropRoi(roi, srcShape, layer?.name);
        }

        internal static CropRoi ResolveCropRoiFromSlice(BufferShape srcShape, int[] starts, int[] ends, int[] axes, string layerName)
        {
            if (starts == null || ends == null || starts.Length == 0 || ends.Length == 0)
                throw new InvalidOperationException("Crop slice arrays are empty: " + layerName);

            var dims = srcShape.dims;
            var woffset = 0;
            var hoffset = 0;
            var doffset = 0;
            var coffset = 0;
            var outw = srcShape.w;
            var outh = srcShape.h;
            var outd = srcShape.d;
            var outc = srcShape.c;

            int[] actualAxes;
            if (axes == null || axes.Length == 0)
            {
                actualAxes = new int[Math.Max(starts.Length, dims)];
                for (var i = 0; i < actualAxes.Length; i++)
                    actualAxes[i] = i;
            }
            else
            {
                actualAxes = axes;
            }

            var numAxis = axes == null || axes.Length == 0 ? Math.Min(dims, starts.Length) : Math.Min(starts.Length, axes.Length);
            for (var i = 0; i < numAxis; i++)
            {
                var axis = actualAxes[i];
                if (axis < 0)
                    axis += dims;

                var start = starts[i];
                var end = ends[Math.Min(i, ends.Length - 1)];

                if (dims == 1)
                {
                    ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (dims == 2)
                {
                    if (axis == 0) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                    else if (axis == 1) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (dims == 3)
                {
                    if (axis == 0) ApplySliceBounds(srcShape.c, ref start, ref end, out coffset, out outc);
                    else if (axis == 1) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                    else if (axis == 2) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
                    continue;
                }

                if (axis == 0) ApplySliceBounds(srcShape.c, ref start, ref end, out coffset, out outc);
                else if (axis == 1) ApplySliceBounds(srcShape.d, ref start, ref end, out doffset, out outd);
                else if (axis == 2) ApplySliceBounds(srcShape.h, ref start, ref end, out hoffset, out outh);
                else if (axis == 3) ApplySliceBounds(srcShape.w, ref start, ref end, out woffset, out outw);
            }

            return ValidateCropRoi(new CropRoi(woffset, hoffset, doffset, coffset, outw, outh, outd, outc), srcShape, layerName);
        }

        private static void ApplySliceBounds(int axisSize, ref int start, ref int end, out int offset, out int outSize)
        {
            if (start == -233) start = 0;
            if (end == -233) end = axisSize;
            if (start < 0) start = axisSize + start;
            if (end <= 0) end = axisSize + end;
            if (end == int.MaxValue) end = axisSize;
            offset = Mathf.Clamp(start, 0, axisSize);
            var clampedEnd = Mathf.Clamp(end, offset, axisSize);
            outSize = Mathf.Max(0, clampedEnd - offset);
        }

        internal static CropRoi ValidateCropRoi(CropRoi roi, BufferShape srcShape, string layerName)
        {
            if (roi.outw <= 0 || roi.outh <= 0 || roi.outd <= 0 || roi.outc <= 0)
                throw new InvalidOperationException("Crop produced empty output: " + layerName);

            if (roi.woffset < 0 || roi.hoffset < 0 || roi.doffset < 0 || roi.coffset < 0)
                throw new InvalidOperationException("Crop negative offset: " + layerName);

            if (roi.woffset + roi.outw > srcShape.w
                || roi.hoffset + roi.outh > srcShape.h
                || roi.doffset + roi.outd > srcShape.d
                || roi.coffset + roi.outc > srcShape.c)
            {
                throw new InvalidOperationException("Crop roi out of range: " + layerName);
            }

            return roi;
        }

        internal NcnnTensorBuffer ApplyCrop(
            ComputeBuffer srcBuf,
            NcnnTensorBuffer srcView,
            CropRoi roi,
            List<IDisposable> tempOwned)
        {
            if (srcBuf == null)
                throw new ArgumentNullException(nameof(srcBuf));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));

            var needsCropW = roi.woffset != 0 || roi.outw != srcView.w;
            var needsCropH = roi.hoffset != 0 || roi.outh != srcView.h;
            var needsCropD = roi.doffset != 0 || roi.outd != srcView.d;
            var needsCropC = roi.coffset != 0 || roi.outc != srcView.c;
            if (!needsCropW && !needsCropH && !needsCropD && !needsCropC)
                return srcView;

            var currentBuf = srcBuf;
            var currentView = srcView;

            void ApplyAxisSlice(int tensorAxis, int begin, int outW, int outH, int outD, int outC)
            {
                var outCount = outW * outH * outD * outC;
                var outBuf = RentTempBuffer(outCount, sizeof(float));
                _ops.Slice(currentBuf, currentView.dims, currentView.w, currentView.h, currentView.d, currentView.c, tensorAxis, begin, outW, outH, outD, outC, outBuf);
                var outView = new NcnnTensorBuffer(outBuf, currentView.dims, outW, outH, outD, outC, false);
                tempOwned?.Add(outBuf);
                currentBuf = outBuf;
                currentView = outView;
            }

            if (srcView.dims == 1)
            {
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, 1, 1, 1);
                return currentView;
            }

            if (srcView.dims == 2)
            {
                if (needsCropH)
                    ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, 1, 1);
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, 1, 1);
                return currentView;
            }

            if (srcView.dims == 3)
            {
                if (needsCropC)
                    ApplyAxisSlice(2, roi.coffset, currentView.w, currentView.h, 1, roi.outc);
                if (needsCropH)
                    ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, 1, currentView.c);
                if (needsCropW)
                    ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, 1, currentView.c);
                return currentView;
            }

            if (needsCropC)
                ApplyAxisSlice(3, roi.coffset, currentView.w, currentView.h, currentView.d, roi.outc);
            if (needsCropD)
                ApplyAxisSlice(2, roi.doffset, currentView.w, currentView.h, roi.outd, currentView.c);
            if (needsCropH)
                ApplyAxisSlice(1, roi.hoffset, currentView.w, roi.outh, currentView.d, currentView.c);
            if (needsCropW)
                ApplyAxisSlice(0, roi.woffset, roi.outw, currentView.h, currentView.d, currentView.c);

            return currentView;
        }

        internal NcnnTensorBuffer ShuffleChannelCpu(ComputeBuffer srcBuffer, NcnnTensorBuffer srcView, int group, bool reverse)
        {
            if (srcBuffer == null)
                throw new ArgumentNullException(nameof(srcBuffer));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));
            if (srcView.dims < 3)
                throw new InvalidOperationException("ShuffleChannel expects dims>=3");

            var channels = srcView.c;
            if (channels <= 0)
                throw new InvalidOperationException("ShuffleChannel invalid channels");
            if (channels % Mathf.Max(1, group) != 0)
                throw new InvalidOperationException("ShuffleChannel invalid group: " + group + " for c=" + channels);

            var actualGroup = reverse ? channels / Mathf.Max(1, group) : Mathf.Max(1, group);
            var channelsPerGroup = channels / actualGroup;
            var featureSize = srcView.w * srcView.h * srcView.d;

            var srcData = new float[srcBuffer.count];
            srcBuffer.GetData(srcData);
            var dstData = new float[srcData.Length];

            for (var i = 0; i < actualGroup; i++)
            {
                for (var j = 0; j < channelsPerGroup; j++)
                {
                    var srcChannel = channelsPerGroup * i + j;
                    var dstChannel = actualGroup * j + i;
                    Array.Copy(srcData, srcChannel * featureSize, dstData, dstChannel * featureSize, featureSize);
                }
            }

            var outBuffer = RentTempBuffer(dstData.Length, sizeof(float));
            outBuffer.SetData(dstData);
            return new NcnnTensorBuffer(outBuffer, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
        }

        internal static (int mode, int size, int total, NcnnTensorBuffer outputView) ResolveBinaryBroadcast(
            NcnnTensorBuffer aView,
            NcnnTensorBuffer bView,
            int aCount,
            int bCount,
            string layerName)
        {
            if (aCount == bCount)
            {
                return (0, 0, aCount, aView ?? bView);
            }

            if (aView != null && bView != null && aView.dims == 2 && bView.dims == 2)
            {
                // row-wise broadcast: [h,w] op [1,w]
                if (aView.w == bView.w && bView.h == 1 && aView.h > 1)
                    return (2, bCount, aCount, aView);
                if (aView.w == bView.w && aView.h == 1 && bView.h > 1)
                    return (1, aCount, bCount, bView);

                // column-wise broadcast: [h,w] op [h,1]
                if (aView.h == bView.h && bView.w == 1 && aView.w > 1)
                    return (2, bCount, aCount, aView);
                if (aView.h == bView.h && aView.w == 1 && bView.w > 1)
                    return (1, aCount, bCount, bView);
            }

            if (aView != null && bView != null)
            {
                if (IsChannelVectorBroadcastSource(aView, bView))
                    return (3, bView.w * bView.h * bView.d, bCount, bView);
                if (IsChannelVectorBroadcastSource(bView, aView))
                    return (4, aView.w * aView.h * aView.d, aCount, aView);
            }

            if (aCount < bCount && bCount % aCount == 0)
                return (1, aCount, bCount, bView);
            if (bCount < aCount && aCount % bCount == 0)
                return (2, bCount, aCount, aView);

            throw new InvalidOperationException("BinaryOp broadcast not supported: " + layerName + " | " + aCount + " vs " + bCount);
        }

        private static bool IsChannelVectorBroadcastSource(NcnnTensorBuffer vectorView, NcnnTensorBuffer tensorView)
        {
            if (vectorView == null || tensorView == null)
                return false;
            if (tensorView.dims < 3 || tensorView.c <= 0)
                return false;
            if (vectorView.elementCount != tensorView.c)
                return false;

            return vectorView.dims switch
            {
                1 => true,
                2 => vectorView.w == 1 || vectorView.h == 1,
                3 => vectorView.w == 1 && vectorView.h == 1,
                4 => vectorView.w == 1 && vectorView.h == 1 && vectorView.d == 1,
                _ => false
            };
        }

        internal bool TryExpand2DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 2 || targetView.dims != 2)
                return false;
            if (sourceView.w == targetView.w && sourceView.h == targetView.h)
                return false;

            bool isRowVector = sourceView.h == 1 && sourceView.w == targetView.w && targetView.h > 1;
            bool isColumnVector = sourceView.w == 1 && sourceView.h == targetView.h && targetView.w > 1;
            if (!isRowVector && !isColumnVector)
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);

            var expandedData = new float[targetView.w * targetView.h];
            if (isRowVector)
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var rowBase = y * targetView.w;
                    Array.Copy(srcData, 0, expandedData, rowBase, targetView.w);
                }
            }
            else
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var value = srcData[y];
                    var rowBase = y * targetView.w;
                    for (var x = 0; x < targetView.w; x++)
                        expandedData[rowBase + x] = value;
                }
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand1DTo2DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 1 || targetView.dims != 2)
                return false;
            if (sourceView.w != targetView.w && sourceView.w != targetView.h)
                return false;

            // Match ncnn binaryop.cpp behavior:
            // if vec length == other.h -> reshape(1, len) => column broadcast
            // else reshape(len, 1) => row broadcast
            bool columnVector = sourceView.w == targetView.h;
            bool rowVector = !columnVector && sourceView.w == targetView.w;
            if (!columnVector && !rowVector)
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);
            var expandedData = new float[targetView.w * targetView.h];

            if (columnVector)
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var value = srcData[y];
                    var rowBase = y * targetView.w;
                    for (var x = 0; x < targetView.w; x++)
                        expandedData[rowBase + x] = value;
                }
            }
            else
            {
                for (var y = 0; y < targetView.h; y++)
                {
                    var rowBase = y * targetView.w;
                    Array.Copy(srcData, 0, expandedData, rowBase, targetView.w);
                }
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand3DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            NcnnTensorBuffer sourceView,
            NcnnTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out NcnnTensorBuffer expandedView)
        {
            expandedBuffer = null;
            expandedView = null;

            if (sourceBuffer == null || sourceView == null || targetView == null)
                return false;
            if (sourceView.dims != 3 || targetView.dims != 3)
                return false;
            if (sourceView.w == targetView.w && sourceView.h == targetView.h && sourceView.c == targetView.c)
                return false;

            // Common CLIP image path broadcast: [1,1,C] op [W,H,C].
            if (!(sourceView.w == 1 && sourceView.h == 1 && sourceView.c == targetView.c))
                return false;

            var srcData = new float[sourceBuffer.count];
            sourceBuffer.GetData(srcData);
            var expandedData = new float[targetView.w * targetView.h * targetView.c];
            var plane = targetView.w * targetView.h;
            for (var c = 0; c < targetView.c; c++)
            {
                var value = srcData[c];
                var dstBase = c * plane;
                for (var i = 0; i < plane; i++)
                    expandedData[dstBase + i] = value;
            }

            expandedBuffer = RentTempBuffer(expandedData.Length, sizeof(float));
            expandedBuffer.SetData(expandedData);
            expandedView = new NcnnTensorBuffer(expandedBuffer, 3, targetView.w, targetView.h, 1, targetView.c, false);
            return true;
        }

        public static Dictionary<string, int> BuildBlobUseCount(NcnnParamModel model)
        {
            var use = new Dictionary<string, int>(StringComparer.Ordinal);
            for (var i = 0; i < model.layers.Count; i++)
            {
                var layer = model.layers[i];
                if (layer.bottomNames == null)
                    continue;
                for (var b = 0; b < layer.bottomNames.Length; b++)
                {
                    var name = layer.bottomNames[b];
                    if (string.IsNullOrEmpty(name))
                        continue;
                    use.TryGetValue(name, out var count);
                    use[name] = count + 1;
                }
            }
            return use;
        }

        public static float ParseLeakySlope(NcnnParamModel.Layer layer)
        {
            if (layer.intParams == null || !layer.intParams.TryGetValue(-23310, out var s) || string.IsNullOrWhiteSpace(s))
                return 0.2f;
            var parts = s.Split(',');
            if (parts.Length >= 2 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                return v;
            return 0.2f;
        }

        public static float[] ParseActivationParams(NcnnParamModel.Layer layer)
        {
            if (layer == null)
                return Array.Empty<float>();
            return layer.GetFloats(-23310, Array.Empty<float>());
        }

        public static float ApplyActivationScalarCpu(float v, int activationType, float param0 = 0f, float param1 = 0f)
        {
            if (activationType == 1)
                return Mathf.Max(0f, v);
            if (activationType == 2)
                return v < 0f ? v * param0 : v;
            if (activationType == 3)
                return Mathf.Clamp(v, param0, param1);
            if (activationType == 4)
                return 1f / (1f + Mathf.Exp(-v));
            if (activationType == 5)
                return v * (float)Math.Tanh(Mathf.Log(Mathf.Exp(v) + 1f));
            if (activationType == 6)
                return v * Mathf.Clamp(v * param0 + param1, 0f, 1f);
            return v;
        }

        public static (float coeffA, float coeffB) ParseEltwiseCoeff(NcnnParamModel.Layer layer)
        {
            if (layer.intParams == null || !layer.intParams.TryGetValue(-23301, out var s) || string.IsNullOrWhiteSpace(s))
                return (1f, 1f);
            var parts = s.Split(',');
            if (parts.Length >= 3
                && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var a)
                && float.TryParse(parts[2], NumberStyles.Float, CultureInfo.InvariantCulture, out var b))
                return (a, b);
            return (1f, 1f);
        }

        public static Vector4[] PackBiasToO4(float[] b, int outC, int outPacks)
        {
            var packed = new Vector4[outPacks];
            for (var op = 0; op < outPacks; op++)
            {
                var x = op * 4 + 0 < outC ? b[op * 4 + 0] : 0f;
                var y = op * 4 + 1 < outC ? b[op * 4 + 1] : 0f;
                var z = op * 4 + 2 < outC ? b[op * 4 + 2] : 0f;
                var w = op * 4 + 3 < outC ? b[op * 4 + 3] : 0f;
                packed[op] = new Vector4(x, y, z, w);
            }
            return packed;
        }

        public static Vector4[] PackWeightsToO4I4K(float[] w, int outC, int inC, int k, int outPacks, int inPacks)
        {
            var packed = new Vector4[outPacks * inPacks * k * k * 4];
            if (w != null
                && outC > 0
                && inC > 0
                && k > 0
                && (outC & 3) == 0
                && (inC & 3) == 0
                && outPacks * 4 == outC
                && inPacks * 4 == inC)
            {
                PackWeightsToO4I4KAligned(w, packed, inC, k, outPacks, inPacks);
                return packed;
            }

            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var ky = 0; ky < k; ky++)
                    {
                        for (var kx = 0; kx < k; kx++)
                        {
                            var kIndex = ky * k + kx;
                            for (var lane = 0; lane < 4; lane++)
                            {
                                var oc = op * 4 + lane;
                                var baseIndex = ((((op * inPacks + ip) * k + ky) * k + kx) * 4) + lane;
                                var v = Vector4.zero;
                                for (var icLane = 0; icLane < 4; icLane++)
                                {
                                    var ic = ip * 4 + icLane;
                                    if (oc < outC && ic < inC)
                                    {
                                        var srcIndex = (((oc * inC + ic) * k + ky) * k + kx);
                                        v[icLane] = w[srcIndex];
                                    }
                                }
                                packed[baseIndex] = v;
                            }
                        }
                    }
                }
            }
            return packed;
        }

        public static Vector4[] PackWeightsToO4I4K3D(float[] w, int outC, int inC, int kernelW, int kernelH, int kernelD, int outPacks, int inPacks)
        {
            var kernelPlane = Mathf.Max(1, kernelW * kernelH);
            var kernelVolume = kernelPlane * Mathf.Max(1, kernelD);
            var packed = new Vector4[outPacks * inPacks * kernelVolume * 4];
            if (w == null || w.Length == 0 || outC <= 0 || inC <= 0 || kernelW <= 0 || kernelH <= 0 || kernelD <= 0)
                return packed;

            for (var op = 0; op < outPacks; op++)
            {
                for (var ip = 0; ip < inPacks; ip++)
                {
                    for (var kz = 0; kz < kernelD; kz++)
                    {
                        for (var ky = 0; ky < kernelH; ky++)
                        {
                            for (var kx = 0; kx < kernelW; kx++)
                            {
                                var kernelIndex = (kz * kernelPlane) + (ky * kernelW) + kx;
                                var dstBase = ((((op * inPacks + ip) * kernelVolume) + kernelIndex) * 4);
                                for (var ocLane = 0; ocLane < 4; ocLane++)
                                {
                                    var oc = op * 4 + ocLane;
                                    var v = Vector4.zero;
                                    for (var icLane = 0; icLane < 4; icLane++)
                                    {
                                        var ic = ip * 4 + icLane;
                                        if (oc < outC && ic < inC)
                                        {
                                            var srcIndex = ((((oc * inC + ic) * kernelD + kz) * kernelH + ky) * kernelW + kx);
                                            v[icLane] = w[srcIndex];
                                        }
                                    }
                                    packed[dstBase + ocLane] = v;
                                }
                            }
                        }
                    }
                }
            }

            return packed;
        }

        private static void PackWeightsToO4I4KAligned(float[] src, Vector4[] dst, int inC, int k, int outPacks, int inPacks)
        {
            var kArea = k * k;
            var minParallelWeights = 1 << 20;
            if (src.Length >= minParallelWeights && outPacks >= 4)
            {
                Parallel.For(0, outPacks, op => PackWeightsToO4I4KAlignedOp(src, dst, inC, kArea, inPacks, op));
                return;
            }

            for (var op = 0; op < outPacks; op++)
                PackWeightsToO4I4KAlignedOp(src, dst, inC, kArea, inPacks, op);
        }

        private static void PackWeightsToO4I4KAlignedOp(float[] src, Vector4[] dst, int inC, int kArea, int inPacks, int op)
        {
            var ocBase = op * 4;
            var ocStride = inC * kArea;
            var opDstBase = op * inPacks * kArea * 4;

            for (var ip = 0; ip < inPacks; ip++)
            {
                var icBase = ip * 4;
                var ipDstBase = opDstBase + ip * kArea * 4;
                var srcBase0 = (ocBase * inC + icBase) * kArea;

                for (var ki = 0; ki < kArea; ki++)
                {
                    var dstBase = ipDstBase + ki * 4;
                    var srcBase = srcBase0 + ki;
                    dst[dstBase + 0] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 1] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 2] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                    srcBase += ocStride;
                    dst[dstBase + 3] = new Vector4(src[srcBase], src[srcBase + kArea], src[srcBase + kArea * 2], src[srcBase + kArea * 3]);
                }
            }
        }

        public static Vector4[] PackDepthWiseWeightsToP4KhKw(float[] w, int channels, int kernelW, int kernelH, int packs)
        {
            var packed = new Vector4[packs * kernelW * kernelH];
            for (var p = 0; p < packs; p++)
            {
                for (var ky = 0; ky < kernelH; ky++)
                {
                    for (var kx = 0; kx < kernelW; kx++)
                    {
                        var baseIndex = (p * kernelH + ky) * kernelW + kx;
                        var v = Vector4.zero;
                        for (var lane = 0; lane < 4; lane++)
                        {
                            var c = p * 4 + lane;
                            if (c < channels)
                            {
                                var srcIndex = ((c * kernelH + ky) * kernelW + kx);
                                v[lane] = w[srcIndex];
                            }
                        }
                        packed[baseIndex] = v;
                    }
                }
            }
            return packed;
        }

        internal bool ShouldCompareTextureLayer(string layerName)
        {
            if (!string.IsNullOrWhiteSpace(layerName))
            {
                if (DebugCompareTextureLayers != null
                    && (DebugCompareTextureLayers.Contains(layerName) || DebugCompareTextureLayers.Contains("*")))
                    return true;
            }

            return false;
        }

        internal bool ShouldCompareTextureConvLayer(string layerName)
        {
            if (ShouldCompareTextureLayer(layerName))
                return true;

            if (!string.IsNullOrWhiteSpace(layerName))
            {
                if (DebugCompareTextureConvLayers != null
                    && (DebugCompareTextureConvLayers.Contains(layerName) || DebugCompareTextureConvLayers.Contains("*")))
                {
                    return true;
                }
            }

            return false;
        }

        internal bool ShouldKeepRawConvWeightsForTexturePath(string layerName, ConvPack conv, bool needGeneralTexturePack, bool needDepthWiseTexturePack)
        {
            if (KeepRawConvWeightsForTexturePath)
                return true;
            if (conv == null)
                return true;
            if (MatchesForceBufferToken(ForceBufferLayerNames, layerName))
                return true;
            if (ForceBufferConvolutionAll || ForceBufferConvolution)
                return true;
            if (ShouldCompareTextureConvLayer(layerName))
                return true;
            return !needGeneralTexturePack && !needDepthWiseTexturePack;
        }

        internal static void UploadRawConvWeights(ConvPack pack, float[] weights, float[] bias)
        {
            if (pack == null)
                throw new ArgumentNullException(nameof(pack));
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (bias == null)
                throw new ArgumentNullException(nameof(bias));
            pack.rawWeight = new ComputeBuffer(weights.Length, sizeof(float), ComputeBufferType.Structured);
            NcnnGpuResourceTracker.RegisterBuffer(pack.rawWeight, weights.Length, sizeof(float), "NcnnRepro.ConvRawWeight");
            pack.rawBias = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
            NcnnGpuResourceTracker.RegisterBuffer(pack.rawBias, bias.Length, sizeof(float), "NcnnRepro.ConvRawBias");
            pack.rawWeight.SetData(weights);
            pack.rawBias.SetData(bias);
        }

        internal void CompareTextureConvPath(
            string layerName,
            string bottomName,
            ConvPack conv,
            int outW,
            int outH,
            RenderTexture textureOutput,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            try
            {
                var srcBuf = GetOrConvertToBuffer(bottomName, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                var srcTensor = TryGetBufferView(bottomName, bufferBlobs, bufferViews);
                if (srcBuf == null || srcTensor == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: source buffer unavailable");
                    return;
                }

                LogBufferStats(layerName, "src", srcBuf, srcTensor.w * srcTensor.h * srcTensor.d * srcTensor.c);
                if (conv.rawWeight == null || conv.rawBias == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: raw convolution weights were not kept");
                    return;
                }
                LogBufferStats(layerName, "weight", conv.rawWeight, conv.weightSize);
                LogBufferStats(layerName, "bias", conv.rawBias, conv.outC);

                using var refTensor = RentTempTensorBuffer(3, outW, outH, 1, conv.outC);
                if (conv.isDepthWise || conv.group > 1 || conv.kernelW != 3 || conv.kernelH != 3 || conv.strideW != conv.strideH || conv.padLeft != conv.padTop)
                {
                    _ops.ConvDepthWise(
                        srcTensor,
                        conv.rawWeight,
                        conv.rawBias,
                        conv.outC,
                        conv.group,
                        conv.kernelW,
                        conv.kernelH,
                        conv.strideW,
                        conv.strideH,
                        conv.padLeft,
                        conv.padTop,
                        conv.dilationW,
                        conv.dilationH,
                        conv.activationType,
                        conv.activationSlope,
                        refTensor);
                }
                else
                {
                    _ops.Conv3x3(srcTensor, conv.rawWeight, conv.rawBias, conv.outC, conv.strideW, conv.padLeft, conv.activationType, conv.activationSlope, refTensor);
                }

                var logicalCount = outW * outH * conv.outC;
                var physicalCount = outW * outH * conv.outPacks * 4;
                var texturePhysical = RentTempBuffer(physicalCount, sizeof(float));
                try
                {
                    _ops.Pack4ToBufferCHW(textureOutput, outW, outH, conv.outPacks * 4, texturePhysical);

                    var refData = new float[logicalCount];
                    refTensor.buffer.GetData(refData);

                    var texPhysicalData = new float[physicalCount];
                    texturePhysical.GetData(texPhysicalData);

                    double sumAbs = 0d;
                    float maxAbs = 0f;
                    var validCount = 0;
                    var refNanCount = 0;
                    var texNanCount = 0;
                    var preview = new List<string>(8);
                    var compareCount = Mathf.Min(logicalCount, texPhysicalData.Length);
                    for (var i = 0; i < compareCount; i++)
                    {
                        var rv = refData[i];
                        var tv = texPhysicalData[i];
                        var refFinite = !float.IsNaN(rv) && !float.IsInfinity(rv);
                        var texFinite = !float.IsNaN(tv) && !float.IsInfinity(tv);
                        if (!refFinite) refNanCount++;
                        if (!texFinite) texNanCount++;
                        if (!refFinite || !texFinite)
                        {
                            if (preview.Count < 8)
                                preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture) + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                            continue;
                        }

                        var diff = Mathf.Abs(rv - tv);
                        sumAbs += diff;
                        if (diff > maxAbs)
                            maxAbs = diff;
                        validCount++;
                        if (preview.Count < 8)
                            preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture) + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                    }

                    var meanAbs = validCount > 0 ? (float)(sumAbs / validCount) : float.NaN;
                    DebugLog?.Invoke(layerName
                        + " | texture_vs_buffer mean_abs=" + meanAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | max_abs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | count=" + compareCount
                        + " | valid=" + validCount
                        + " | ref_nan=" + refNanCount
                        + " | tex_nan=" + texNanCount);
                    for (var i = 0; i < preview.Count; i++)
                        DebugLog?.Invoke(layerName + " | sample[" + i + "] " + preview[i]);
                }
                finally
                {
                    ReturnTempBuffer(texturePhysical);
                }
            }
            catch (Exception e)
            {
                DebugLog?.Invoke(layerName + " | compare failed: " + e.Message);
            }
        }

        internal void CompareTextureInnerProductPath(
            string layerName,
            string bottomName,
            InnerProductPack ip,
            RenderTexture textureOutput,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, NcnnTensorBuffer> bufferViews,
            List<IDisposable> tempOwned)
        {
            try
            {
                using var srcTensor = GetReadableTensorInput(bottomName, textureBlobs, bufferBlobs, textureShapes, bufferViews, tempOwned);
                if (srcTensor == null || srcTensor.buffer == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: source buffer unavailable");
                    return;
                }
                if (ip == null || ip.w == null || ip.b == null)
                {
                    DebugLog?.Invoke(layerName + " | compare skipped: weights unavailable");
                    return;
                }

                LogBufferStats(layerName, "src", srcTensor.buffer, srcTensor.elementCount);
                LogBufferStats(layerName, "weight", ip.w, ip.weightSize);
                LogBufferStats(layerName, "bias", ip.b, ip.outFeatures);

                var rows = srcTensor.dims == 2 && srcTensor.w == ip.inFeatures ? srcTensor.h : 1;
                using var refTensor = rows > 1
                    ? RentTempTensorBuffer(2, ip.outFeatures, rows)
                    : RentTempTensorBuffer(1, ip.outFeatures);
                if (rows > 1)
                    Ops.InnerProduct2D(srcTensor.buffer, rows, ip.inFeatures, ip.w, ip.b, ip.outFeatures, refTensor.buffer);
                else
                    Ops.InnerProduct(srcTensor.buffer, ip.inFeatures, ip.w, ip.b, ip.outFeatures, refTensor.buffer);

                var logicalCount = refTensor.elementCount;
                var texturePhysical = RentTempBuffer(textureOutput.width * textureOutput.height * 4, sizeof(float));
                try
                {
                    Ops.Pack4ToBufferCHW(textureOutput, textureOutput.width, textureOutput.height, 4, texturePhysical);

                    var refData = new float[logicalCount];
                    refTensor.buffer.GetData(refData);

                    var texPhysicalData = new float[texturePhysical.count];
                    texturePhysical.GetData(texPhysicalData);

                    double sumAbs = 0d;
                    float maxAbs = 0f;
                    var validCount = 0;
                    var refNanCount = 0;
                    var texNanCount = 0;
                    var preview = new List<string>(8);
                    var compareCount = Mathf.Min(logicalCount, texPhysicalData.Length);
                    for (var i = 0; i < compareCount; i++)
                    {
                        var rv = refData[i];
                        var tv = texPhysicalData[i];
                        var refFinite = !float.IsNaN(rv) && !float.IsInfinity(rv);
                        var texFinite = !float.IsNaN(tv) && !float.IsInfinity(tv);
                        if (!refFinite) refNanCount++;
                        if (!texFinite) texNanCount++;
                        if (!refFinite || !texFinite)
                        {
                            if (preview.Count < 8)
                            {
                                preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture)
                                    + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                            }
                            continue;
                        }

                        var diff = Mathf.Abs(rv - tv);
                        sumAbs += diff;
                        if (diff > maxAbs)
                            maxAbs = diff;
                        validCount++;
                        if (preview.Count < 8)
                        {
                            preview.Add(i + ": ref=" + rv.ToString("G9", CultureInfo.InvariantCulture)
                                + " tex=" + tv.ToString("G9", CultureInfo.InvariantCulture));
                        }
                    }

                    var meanAbs = validCount > 0 ? (float)(sumAbs / validCount) : float.NaN;
                    DebugLog?.Invoke(layerName
                        + " | texture_vs_buffer mean_abs=" + meanAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | max_abs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                        + " | count=" + compareCount
                        + " | valid=" + validCount
                        + " | ref_nan=" + refNanCount
                        + " | tex_nan=" + texNanCount
                        + " | rows=" + rows.ToString(CultureInfo.InvariantCulture)
                        + " | out_features=" + ip.outFeatures.ToString(CultureInfo.InvariantCulture));
                    for (var i = 0; i < preview.Count; i++)
                        DebugLog?.Invoke(layerName + " | sample[" + i + "] " + preview[i]);
                }
                finally
                {
                    ReturnTempBuffer(texturePhysical);
                }
            }
            catch (Exception e)
            {
                DebugLog?.Invoke(layerName + " | compare failed: " + e.Message);
            }
        }

        internal void SynchronizeGpuBufferUse(ComputeBuffer buffer)
        {
            if (buffer == null || buffer.count <= 0)
                return;

            buffer.GetData(_gpuSyncScratch, 0, 0, 1);
        }

        internal void CompareMaxPoolingIndPath(string layerName, TensorRef src, BufferShape srcShape, RenderTexture textureOutput, RenderTexture textureIndices, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int outW, int outH)
        {
            try
            {
                var srcCount = srcShape.w * srcShape.h * srcShape.c;
                var srcBuffer = RentTempBuffer(srcCount, sizeof(float));
                try
                {
                    _ops.Pack4ToBufferCHW(src.texture, srcShape.w, srcShape.h, srcShape.c, srcBuffer);
                    var srcData = new float[srcCount];
                    srcBuffer.GetData(srcData);

                    var refData = new float[outW * outH * srcShape.c];
                    var refIndexData = new float[outW * outH * srcShape.c];
                    var srcPlane = srcShape.w * srcShape.h;
                    var outPlane = outW * outH;
                    for (var c = 0; c < srcShape.c; c++)
                    {
                        var srcBase = c * srcPlane;
                        var dstBase = c * outPlane;
                        for (var oy = 0; oy < outH; oy++)
                        {
                            var sy0 = oy * strideH - padTop;
                            for (var ox = 0; ox < outW; ox++)
                            {
                                var sx0 = ox * strideW - padLeft;
                                var best = float.NegativeInfinity;
                                var bestIndex = 0;
                                for (var ky = 0; ky < kernelH; ky++)
                                {
                                    var sy = sy0 + ky;
                                    if (sy < 0 || sy >= srcShape.h)
                                        continue;
                                    for (var kx = 0; kx < kernelW; kx++)
                                    {
                                        var sx = sx0 + kx;
                                        if (sx < 0 || sx >= srcShape.w)
                                            continue;
                                        var linear = sy * srcShape.w + sx;
                                        var v = srcData[srcBase + linear];
                                        if (v > best)
                                        {
                                            best = v;
                                            bestIndex = linear;
                                        }
                                    }
                                }

                                var dstIndex = dstBase + oy * outW + ox;
                                refData[dstIndex] = best;
                                refIndexData[dstIndex] = bestIndex;
                            }
                        }
                    }

                    var logicalCount = outW * outH * srcShape.c;
                    var texBuffer = RentTempBuffer(logicalCount, sizeof(float));
                    try
                    {
                        _ops.Pack4ToBufferCHW(textureOutput, outW, outH, srcShape.c, texBuffer);
                        var texData = new float[logicalCount];
                        texBuffer.GetData(texData);

                        float[] texIndexData = null;
                        if (textureIndices != null)
                        {
                            var idxBuffer = RentTempBuffer(logicalCount, sizeof(float));
                            try
                            {
                                _ops.Pack4ToBufferCHW(textureIndices, outW, outH, srcShape.c, idxBuffer);
                                texIndexData = new float[logicalCount];
                                idxBuffer.GetData(texIndexData);
                            }
                            finally
                            {
                                ReturnTempBuffer(idxBuffer);
                            }
                        }

                        double sumAbs = 0d;
                        float maxAbs = 0f;
                        var valid = 0;
                        var refNonFinite = 0;
                        var texNonFinite = 0;
                        double sumIdxAbs = 0d;
                        float maxIdxAbs = 0f;
                        var idxValid = 0;
                        for (var i = 0; i < logicalCount; i++)
                        {
                            var rv = refData[i];
                            var tv = texData[i];
                            var rFinite = !float.IsNaN(rv) && !float.IsInfinity(rv);
                            var tFinite = !float.IsNaN(tv) && !float.IsInfinity(tv);
                            if (!rFinite) refNonFinite++;
                            if (!tFinite) texNonFinite++;
                            if (!rFinite || !tFinite)
                                continue;

                            var diff = Mathf.Abs(rv - tv);
                            sumAbs += diff;
                            if (diff > maxAbs)
                                maxAbs = diff;
                            valid++;

                            if (texIndexData != null)
                            {
                                var idxDiff = Mathf.Abs(refIndexData[i] - texIndexData[i]);
                                sumIdxAbs += idxDiff;
                                if (idxDiff > maxIdxAbs)
                                    maxIdxAbs = idxDiff;
                                idxValid++;
                            }
                        }

                        var meanAbs = valid > 0 ? (float)(sumAbs / valid) : float.NaN;
                        var meanIdxAbs = idxValid > 0 ? (float)(sumIdxAbs / idxValid) : float.NaN;
                        DebugLog?.Invoke(layerName
                            + " | maxpool_compare mean_abs=" + meanAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | max_abs=" + maxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | idx_mean_abs=" + meanIdxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | idx_max_abs=" + maxIdxAbs.ToString("G9", CultureInfo.InvariantCulture)
                            + " | valid=" + valid.ToString(CultureInfo.InvariantCulture)
                            + " | ref_nonfinite=" + refNonFinite.ToString(CultureInfo.InvariantCulture)
                            + " | tex_nonfinite=" + texNonFinite.ToString(CultureInfo.InvariantCulture)
                            + " | shape=" + srcShape.w + "x" + srcShape.h + "x" + srcShape.c
                            + " -> " + outW + "x" + outH + "x" + srcShape.c);
                    }
                    finally
                    {
                        ReturnTempBuffer(texBuffer);
                    }
                }
                finally
                {
                    ReturnTempBuffer(srcBuffer);
                }
            }
            catch (Exception e)
            {
                DebugLog?.Invoke(layerName + " | maxpool_compare failed: " + e.Message);
            }
        }

        internal void ApplyMaxPoolingIndCpu(TensorRef src, BufferShape srcShape, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int outW, int outH, RenderTexture outRt, RenderTexture idxRt)
        {
            var srcCount = srcShape.w * srcShape.h * srcShape.c;
            var srcBuffer = RentTempBuffer(srcCount, sizeof(float));
            try
            {
                _ops.Pack4ToBufferCHW(src.texture, srcShape.w, srcShape.h, srcShape.c, srcBuffer);
                var srcData = new float[srcCount];
                srcBuffer.GetData(srcData);

                var outCount = outW * outH * srcShape.c;
                var valueData = new float[outCount];
                var indexData = new float[outCount];
                var srcPlane = srcShape.w * srcShape.h;
                var outPlane = outW * outH;

                for (var c = 0; c < srcShape.c; c++)
                {
                    var srcBase = c * srcPlane;
                    var dstBase = c * outPlane;
                    for (var oy = 0; oy < outH; oy++)
                    {
                        var sy0 = oy * strideH - padTop;
                        for (var ox = 0; ox < outW; ox++)
                        {
                            var sx0 = ox * strideW - padLeft;
                            var best = float.NegativeInfinity;
                            var bestIndex = 0;
                            for (var ky = 0; ky < kernelH; ky++)
                            {
                                var sy = sy0 + ky;
                                if (sy < 0 || sy >= srcShape.h)
                                    continue;
                                for (var kx = 0; kx < kernelW; kx++)
                                {
                                    var sx = sx0 + kx;
                                    if (sx < 0 || sx >= srcShape.w)
                                        continue;
                                    var linear = sy * srcShape.w + sx;
                                    var v = srcData[srcBase + linear];
                                    if (v > best)
                                    {
                                        best = v;
                                        bestIndex = linear;
                                    }
                                }
                            }

                            var dstIndex = dstBase + oy * outW + ox;
                            valueData[dstIndex] = best;
                            indexData[dstIndex] = bestIndex;
                        }
                    }
                }

                var valueBuffer = RentTempBuffer(outCount, sizeof(float));
                var indexBuffer = RentTempBuffer(outCount, sizeof(float));
                try
                {
                    valueBuffer.SetData(valueData);
                    indexBuffer.SetData(indexData);
                    _ops.FillPack4FromBufferCHW(valueBuffer, outW, outH, srcShape.c, outRt);
                    _ops.FillPack4FromBufferCHW(indexBuffer, outW, outH, srcShape.c, idxRt);
                    SynchronizeGpuBufferUse(indexBuffer);
                }
                finally
                {
                    ReturnTempBuffer(valueBuffer);
                    ReturnTempBuffer(indexBuffer);
                }
            }
            finally
            {
                ReturnTempBuffer(srcBuffer);
            }
        }

        internal void ApplyMaxPoolingIndCpu(
            ComputeBuffer srcBuffer,
            NcnnTensorBuffer srcView,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padTop,
            NcnnTensorBuffer outValue,
            NcnnTensorBuffer outIndex)
        {
            if (srcBuffer == null)
                throw new ArgumentNullException(nameof(srcBuffer));
            if (srcView == null)
                throw new ArgumentNullException(nameof(srcView));
            if (outValue == null)
                throw new ArgumentNullException(nameof(outValue));
            if (outIndex == null)
                throw new ArgumentNullException(nameof(outIndex));

            var srcCount = srcView.elementCount;
            var srcData = new float[srcCount];
            srcBuffer.GetData(srcData);

            var outCount = outValue.elementCount;
            var valueData = new float[outCount];
            var indexData = new float[outCount];
            var srcPlane = srcView.w * srcView.h;
            var outPlane = outValue.w * outValue.h;

            for (var c = 0; c < srcView.c; c++)
            {
                var srcBase = c * srcPlane;
                var dstBase = c * outPlane;
                for (var oy = 0; oy < outValue.h; oy++)
                {
                    var sy0 = oy * strideH - padTop;
                    for (var ox = 0; ox < outValue.w; ox++)
                    {
                        var sx0 = ox * strideW - padLeft;
                        var best = float.NegativeInfinity;
                        var bestIndex = 0;
                        for (var ky = 0; ky < kernelH; ky++)
                        {
                            var sy = sy0 + ky;
                            if (sy < 0 || sy >= srcView.h)
                                continue;
                            for (var kx = 0; kx < kernelW; kx++)
                            {
                                var sx = sx0 + kx;
                                if (sx < 0 || sx >= srcView.w)
                                    continue;
                                var linear = sy * srcView.w + sx;
                                var v = srcData[srcBase + linear];
                                if (v > best)
                                {
                                    best = v;
                                    bestIndex = linear;
                                }
                            }
                        }

                        var dstIndex = dstBase + oy * outValue.w + ox;
                        valueData[dstIndex] = best;
                        indexData[dstIndex] = bestIndex;
                    }
                }
            }

            outValue.buffer.SetData(valueData);
            outIndex.buffer.SetData(indexData);
        }

        internal void ApplyMaxUnPoolingCpu(TensorRef src, BufferShape srcShape, IndexRef idx, int outW, int outH, RenderTexture outRt)
        {
            var pooledCount = src.width * src.height * srcShape.c;
            var pooledBuffer = RentTempBuffer(pooledCount, sizeof(float));
            var indexBuffer = RentTempBuffer(src.width * src.height * srcShape.c, sizeof(float));
            try
            {
                _ops.Pack4ToBufferCHW(src.texture, src.width, src.height, srcShape.c, pooledBuffer);
                var pooledData = new float[pooledCount];
                pooledBuffer.GetData(pooledData);

                _ops.Pack4ToBufferCHW(idx.texture, idx.width, idx.height, srcShape.c, indexBuffer);
                var indexData = new float[indexBuffer.count];
                indexBuffer.GetData(indexData);

                var outPlane = outW * outH;
                var pooledPlane = src.width * src.height;
                var outData = new float[outPlane * srcShape.c];

                for (var c = 0; c < srcShape.c; c++)
                {
                    var pooledBase = c * pooledPlane;
                    var outBase = c * outPlane;
                    for (var i = 0; i < pooledPlane; i++)
                    {
                        var dstIndex = Mathf.RoundToInt(indexData[pooledBase + i]);
                        if (dstIndex < 0 || dstIndex >= outPlane)
                            continue;
                        outData[outBase + dstIndex] = pooledData[pooledBase + i];
                    }
                }

                var outBuffer = RentTempBuffer(outData.Length, sizeof(float));
                try
                {
                    outBuffer.SetData(outData);
                    _ops.FillPack4FromBufferCHW(outBuffer, outW, outH, srcShape.c, outRt);
                    SynchronizeGpuBufferUse(outBuffer);
                }
                finally
                {
                    ReturnTempBuffer(outBuffer);
                }
            }
            finally
            {
                ReturnTempBuffer(pooledBuffer);
                ReturnTempBuffer(indexBuffer);
            }
        }

        internal void ApplyMaxUnPoolingCpu(
            ComputeBuffer pooledBuffer,
            NcnnTensorBuffer pooledView,
            ComputeBuffer indexBuffer,
            NcnnTensorBuffer indexView,
            int outW,
            int outH,
            NcnnTensorBuffer outTensor)
        {
            if (pooledBuffer == null)
                throw new ArgumentNullException(nameof(pooledBuffer));
            if (pooledView == null)
                throw new ArgumentNullException(nameof(pooledView));
            if (indexBuffer == null)
                throw new ArgumentNullException(nameof(indexBuffer));
            if (indexView == null)
                throw new ArgumentNullException(nameof(indexView));
            if (outTensor == null)
                throw new ArgumentNullException(nameof(outTensor));

            var pooledData = new float[pooledView.elementCount];
            pooledBuffer.GetData(pooledData);
            var indexData = new float[indexView.elementCount];
            indexBuffer.GetData(indexData);

            var outPlane = outW * outH;
            var pooledPlane = pooledView.w * pooledView.h;
            var outData = new float[outPlane * pooledView.c];

            for (var c = 0; c < pooledView.c; c++)
            {
                var pooledBase = c * pooledPlane;
                var outBase = c * outPlane;
                for (var i = 0; i < pooledPlane; i++)
                {
                    var dstIndex = Mathf.RoundToInt(indexData[pooledBase + i]);
                    if (dstIndex < 0 || dstIndex >= outPlane)
                        continue;
                    outData[outBase + dstIndex] = pooledData[pooledBase + i];
                }
            }

            outTensor.buffer.SetData(outData);
        }

        internal void LogBufferStats(string layerName, string kind, ComputeBuffer buffer, int logicalCount)
        {
            if (DebugLog == null || buffer == null)
                return;

            var count = logicalCount > 0 ? Mathf.Min(logicalCount, buffer.count) : buffer.count;
            if (count <= 0)
            {
                DebugLog(layerName + " | " + kind + " count=0");
                return;
            }

            var data = new float[count];
            buffer.GetData(data, 0, 0, count);

            var finite = 0;
            var nan = 0;
            var inf = 0;
            var min = float.PositiveInfinity;
            var max = float.NegativeInfinity;
            var preview = new List<string>(4);
            for (var i = 0; i < data.Length; i++)
            {
                var v = data[i];
                if (float.IsNaN(v))
                {
                    nan++;
                }
                else if (float.IsInfinity(v))
                {
                    inf++;
                }
                else
                {
                    finite++;
                    if (v < min) min = v;
                    if (v > max) max = v;
                }

                if (preview.Count < 4)
                    preview.Add(v.ToString("G9", CultureInfo.InvariantCulture));
            }

            DebugLog(layerName
                + " | " + kind
                + " finite=" + finite
                + " nan=" + nan
                + " inf=" + inf
                + " min=" + (finite > 0 ? min.ToString("G9", CultureInfo.InvariantCulture) : "NaN")
                + " max=" + (finite > 0 ? max.ToString("G9", CultureInfo.InvariantCulture) : "NaN")
                + " sample=" + string.Join(",", preview));
        }

        internal bool BufferHasAnyNonFinite(ComputeBuffer buffer, int logicalCount, out int finiteCount, out int nanCount, out int infCount)
        {
            finiteCount = 0;
            nanCount = 0;
            infCount = 0;

            if (buffer == null)
                return false;

            var count = logicalCount > 0 ? Mathf.Min(logicalCount, buffer.count) : buffer.count;
            if (count <= 0)
                return false;

            var data = new float[count];
            buffer.GetData(data, 0, 0, count);
            for (var i = 0; i < data.Length; i++)
            {
                var v = data[i];
                if (float.IsNaN(v))
                {
                    nanCount++;
                }
                else if (float.IsInfinity(v))
                {
                    infCount++;
                }
                else
                {
                    finiteCount++;
                }
            }

            return nanCount > 0 || infCount > 0;
        }
    }
}

