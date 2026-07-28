using System;
using System.Collections.Generic;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Runtime.CompilerServices;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using System.Collections;
using Aexis;
using Aexis.Async;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Execution
{
    public enum AexisInferenceExecutionMode
    {
        ProductionTextureOnly = 0,
        DebugOracle = 1
    }

    public partial class AexisGraphSession : IInferenceSession
    {
        [StructLayout(LayoutKind.Sequential)]
        private struct PackedFp16x4
        {
            public uint xy;
            public uint zw;
        }

        internal sealed class Int8WeightOnlyUpload
        {
            public ComputeBuffer packedWeights;
            public ComputeBuffer scales;
        }

        internal sealed class Int4WeightOnlyUpload
        {
            public ComputeBuffer packedWeights;
            public ComputeBuffer scales;
            public int scaleBlockSize;
        }

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

        public readonly struct AexisTextureTensorContract
        {
            public AexisTextureTensorContract(IInferenceTensor tensor)
            {
                Descriptor = tensor != null
                    ? tensor.Descriptor
                    : throw new ArgumentNullException(nameof(tensor));
                if (Descriptor == null)
                    throw new InvalidOperationException("tensor descriptor has not been published");
            }

            public TensorDescriptor Descriptor { get; }
            public AexisTextureTensor Tensor => Descriptor != null ? Descriptor.NativeTensor : null;
            public BufferShape LogicalShape => Descriptor != null ? Descriptor.LogicalShape : default;
            public BufferShape StorageShape => Descriptor != null ? Descriptor.StorageShape : default;
            public AexisTextureTensorLayoutKind LayoutKind => Descriptor != null ? Descriptor.Layout : default;
            public int Width => Tensor != null ? Tensor.Width : 0;
            public int Height => Tensor != null ? Tensor.Height : 0;
            public int Depth => Tensor != null ? Tensor.Depth : 0;
            public int Packs => Tensor != null ? Tensor.Packs : 0;
            public bool IsLinearMat => LayoutKind == AexisTextureTensorLayoutKind.LinearMat;
            public bool IsPack4Image => LayoutKind == AexisTextureTensorLayoutKind.Pack4Image;
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
            public AexisTextureTensor repoTensor;
            public AexisTextureTensorLayoutKind layoutKind;
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
            public AexisTensorBuffer view;
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
            public AexisTextureTensor repoTensor;
            public AexisTextureTensorLayoutKind layoutKind;
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

        [Serializable]
        public sealed class CommandBufferRtArenaMetrics
        {
            public bool enabled;
            public int plannedTemporaryResources;
            public int plannedPersistentResources;
            public int plannedSlots;
            public int boundResources;
            public int allocatedSlots;
            public int releasedResources;
            // Strict mode rejects this before recording an allocation. It remains
            // observable so debug-oracle execution cannot be misreported as a
            // fully compiled graph arena.
            public int unplannedTextureAllocations;
            public bool allPlannedResourcesBound;
            public bool allBoundResourcesReleased;
            // Filled only when the strict lifetime proof fails. Keeping the
            // planned blob and interval makes an arena mismatch actionable
            // without inspecting CommandBuffer internals or reading back data.
            public string[] activeResourceDiagnostics = Array.Empty<string>();
            public string[] unboundResourceDiagnostics = Array.Empty<string>();
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
            public ComputeBuffer packedWeight4Fp16;
            public ComputeBuffer packedBias4;
            public ComputeBuffer packedWeightTm23;
            public ComputeBuffer packedDepthWiseWeight4;
            public ComputeBuffer packedDepthWiseWeight4Fp16;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawWeightInt8Packed;
            public ComputeBuffer rawWeightInt8Scales;
            public ComputeBuffer rawWeightInt4Packed;
            public ComputeBuffer rawWeightInt4Scales;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(packedWeight4, "AexisGraphSession.ConvPack.Dispose"); packedWeight4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedWeight4Fp16, "AexisGraphSession.ConvPack.Dispose"); packedWeight4Fp16?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedBias4, "AexisGraphSession.ConvPack.Dispose"); packedBias4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedWeightTm23, "AexisGraphSession.ConvPack.Dispose"); packedWeightTm23?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedDepthWiseWeight4, "AexisGraphSession.ConvPack.Dispose"); packedDepthWiseWeight4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedDepthWiseWeight4Fp16, "AexisGraphSession.ConvPack.Dispose"); packedDepthWiseWeight4Fp16?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeight, "AexisGraphSession.ConvPack.Dispose"); rawWeight?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeightInt8Packed, "AexisGraphSession.ConvPack.Dispose"); rawWeightInt8Packed?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeightInt8Scales, "AexisGraphSession.ConvPack.Dispose"); rawWeightInt8Scales?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeightInt4Packed, "AexisGraphSession.ConvPack.Dispose"); rawWeightInt4Packed?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeightInt4Scales, "AexisGraphSession.ConvPack.Dispose"); rawWeightInt4Scales?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawBias, "AexisGraphSession.ConvPack.Dispose"); rawBias?.Dispose(); } catch { }
            }
        }

        public sealed class InnerProductPack : IDisposable
        {
            public int inFeatures;
            public int outFeatures;
            public int biasTerm;
            public int weightSize;
            public ComputeBuffer w;
            public ComputeBuffer wFp16;
            public ComputeBuffer wInt8Packed;
            public ComputeBuffer wInt8Scales;
            public ComputeBuffer wInt4Packed;
            public ComputeBuffer wInt4Scales;
            public int int4ScaleBlockSize;
            public ComputeBuffer b;
            public ComputeBuffer TextureWeightBinding => w ?? wInt8Packed ?? wInt4Packed;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(w, "AexisGraphSession.InnerProductPack.Dispose"); w?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(wFp16, "AexisGraphSession.InnerProductPack.Dispose"); wFp16?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(wInt8Packed, "AexisGraphSession.InnerProductPack.Dispose"); wInt8Packed?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(wInt8Scales, "AexisGraphSession.InnerProductPack.Dispose"); wInt8Scales?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(wInt4Packed, "AexisGraphSession.InnerProductPack.Dispose"); wInt4Packed?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(wInt4Scales, "AexisGraphSession.InnerProductPack.Dispose"); wInt4Scales?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(b, "AexisGraphSession.InnerProductPack.Dispose"); b?.Dispose(); } catch { }
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
            public ComputeBuffer packedWeight4Fp16;
            public ComputeBuffer packedBias4;
            public ComputeBuffer rawWeight;
            public ComputeBuffer rawBias;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(packedWeight4, "AexisGraphSession.DeconvPack.Dispose"); packedWeight4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedWeight4Fp16, "AexisGraphSession.DeconvPack.Dispose"); packedWeight4Fp16?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(packedBias4, "AexisGraphSession.DeconvPack.Dispose"); packedBias4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawWeight, "AexisGraphSession.DeconvPack.Dispose"); rawWeight?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(rawBias, "AexisGraphSession.DeconvPack.Dispose"); rawBias?.Dispose(); } catch { }
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
            public ComputeBuffer bDataFp16;
            public ComputeBuffer bDataInt8Packed;
            public ComputeBuffer bDataInt8Scales;
            public ComputeBuffer bDataInt4Packed;
            public ComputeBuffer bDataInt4Scales;
            public int int4ScaleBlockSize;
            public ComputeBuffer cData;
            public bool ownsBData = true;
            public bool ownsBDataInt8 = true;
            public bool ownsBDataInt4 = true;
            public float[] bDataCpu;
            public float[] cDataCpu;
            public ComputeBuffer TextureWeightBinding => bData ?? bDataInt8Packed ?? bDataInt4Packed;

            public void Dispose()
            {
                try { if (ownsBData) { AexisGpuResourceTracker.ReleaseBuffer(bData, "AexisGraphSession.GemmPack.Dispose"); bData?.Dispose(); } } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(bDataFp16, "AexisGraphSession.GemmPack.Dispose"); bDataFp16?.Dispose(); } catch { }
                try { if (ownsBDataInt8) { AexisGpuResourceTracker.ReleaseBuffer(bDataInt8Packed, "AexisGraphSession.GemmPack.Dispose"); bDataInt8Packed?.Dispose(); } } catch { }
                try { if (ownsBDataInt8) { AexisGpuResourceTracker.ReleaseBuffer(bDataInt8Scales, "AexisGraphSession.GemmPack.Dispose"); bDataInt8Scales?.Dispose(); } } catch { }
                try { if (ownsBDataInt4) { AexisGpuResourceTracker.ReleaseBuffer(bDataInt4Packed, "AexisGraphSession.GemmPack.Dispose"); bDataInt4Packed?.Dispose(); } } catch { }
                try { if (ownsBDataInt4) { AexisGpuResourceTracker.ReleaseBuffer(bDataInt4Scales, "AexisGraphSession.GemmPack.Dispose"); bDataInt4Scales?.Dispose(); } } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(cData, "AexisGraphSession.GemmPack.Dispose"); cData?.Dispose(); } catch { }
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
            public RenderTexture linearMatRt;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(data, "AexisGraphSession.MemoryDataPack.Dispose"); data?.Dispose(); } catch { }
                try { if (channelVectorTexture != null) UnityEngine.Object.DestroyImmediate(channelVectorTexture); } catch { }
                try
                {
                    if (pack4Rt != null)
                    {
                        AexisGpuResourceTracker.ReleaseTexture(pack4Rt, "AexisGraphSession.MemoryDataPack.Dispose");
                        pack4Rt.Release();
                        UnityEngine.Object.DestroyImmediate(pack4Rt);
                    }
                }
                catch { }
                try
                {
                    if (linearMatRt != null)
                    {
                        AexisGpuResourceTracker.ReleaseTexture(linearMatRt, "AexisGraphSession.MemoryDataPack.Dispose");
                        linearMatRt.Release();
                        UnityEngine.Object.DestroyImmediate(linearMatRt);
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
            public ComputeBuffer wInt8Packed;
            public ComputeBuffer wInt8Scales;
            public ComputeBuffer wInt4Packed;
            public ComputeBuffer wInt4Scales;
            public int int4ScaleBlockSize;
            public ComputeBuffer b;
            public bool ownsW = true;
            public bool ownsWInt8 = true;
            public bool ownsWInt4 = true;
            public ComputeBuffer WeightBinding => w ?? wInt8Packed ?? wInt4Packed;

            public void Dispose()
            {
                try { if (ownsW) { AexisGpuResourceTracker.ReleaseBuffer(w, "AexisGraphSession.EmbedPack.Dispose"); w?.Dispose(); } } catch { }
                try { if (ownsWInt8) { AexisGpuResourceTracker.ReleaseBuffer(wInt8Packed, "AexisGraphSession.EmbedPack.Dispose.Int8Packed"); wInt8Packed?.Dispose(); } } catch { }
                try { if (ownsWInt8) { AexisGpuResourceTracker.ReleaseBuffer(wInt8Scales, "AexisGraphSession.EmbedPack.Dispose.Int8Scales"); wInt8Scales?.Dispose(); } } catch { }
                try { if (ownsWInt4) { AexisGpuResourceTracker.ReleaseBuffer(wInt4Packed, "AexisGraphSession.EmbedPack.Dispose.Int4Packed"); wInt4Packed?.Dispose(); } } catch { }
                try { if (ownsWInt4) { AexisGpuResourceTracker.ReleaseBuffer(wInt4Scales, "AexisGraphSession.EmbedPack.Dispose.Int4Scales"); wInt4Scales?.Dispose(); } } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(b, "AexisGraphSession.EmbedPack.Dispose"); b?.Dispose(); } catch { }
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
                try { AexisGpuResourceTracker.ReleaseBuffer(gamma, "AexisGraphSession.LayerNormPack.Dispose"); gamma?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(beta, "AexisGraphSession.LayerNormPack.Dispose"); beta?.Dispose(); } catch { }
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
                try { AexisGpuResourceTracker.ReleaseBuffer(gamma, "AexisGraphSession.GroupNormPack.Dispose"); gamma?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(beta, "AexisGraphSession.GroupNormPack.Dispose"); beta?.Dispose(); } catch { }
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
                try { AexisGpuResourceTracker.ReleaseBuffer(biasA, "AexisGraphSession.BatchNormPack.Dispose"); biasA?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(scaleB, "AexisGraphSession.BatchNormPack.Dispose"); scaleB?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(biasA4, "AexisGraphSession.BatchNormPack.Dispose"); biasA4?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(scaleB4, "AexisGraphSession.BatchNormPack.Dispose"); scaleB4?.Dispose(); } catch { }
            }
        }

        // Immutable model weights, packed once for the Pack4 bias dispatch.
        public sealed class BiasPack : IDisposable
        {
            public int channels;
            public ComputeBuffer bias4;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(bias4, "AexisGraphSession.BiasPack.Dispose"); bias4?.Dispose(); } catch { }
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
                try { AexisGpuResourceTracker.ReleaseBuffer(qW, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); qW?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(qB, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); qB?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(kW, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); kW?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(kB, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); kB?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(vW, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); vW?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(vB, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); vB?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(oW, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); oW?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(oB, "AexisGraphSession.MultiHeadAttentionPack.Dispose"); oB?.Dispose(); } catch { }
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
            public bool causal;
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
                try { AexisGpuResourceTracker.ReleaseBuffer(minSizeBuffer, "AexisGraphSession.PriorBoxPack.Dispose"); minSizeBuffer?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(maxSizeBuffer, "AexisGraphSession.PriorBoxPack.Dispose"); maxSizeBuffer?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(aspectRatioBuffer, "AexisGraphSession.PriorBoxPack.Dispose"); aspectRatioBuffer?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(varianceBuffer, "AexisGraphSession.PriorBoxPack.Dispose"); varianceBuffer?.Dispose(); } catch { }
            }
        }

        public sealed class InferResult : IDisposable
        {
            private readonly Dictionary<string, TensorRef> _textureBlobs;
            private readonly Dictionary<string, BufferShape> _textureShapes;
            private readonly Dictionary<string, ComputeBuffer> _bufferBlobs;
            private readonly Dictionary<string, BufferRef> _bufferRefs;
            private readonly Dictionary<string, AexisTensorBuffer> _bufferViews;
            private readonly List<IDisposable> _tempOwned;
            private readonly AexisGraphSession _owner;
            private readonly bool _disallowTextureToBufferFallback;
            private readonly HashSet<RenderTexture> _visitedTextures = new HashSet<RenderTexture>();
            private readonly HashSet<ComputeBuffer> _visitedBuffers = new HashSet<ComputeBuffer>();

            internal InferResult(
                Dictionary<string, TensorRef> textureBlobs,
                Dictionary<string, BufferShape> textureShapes,
                Dictionary<string, ComputeBuffer> bufferBlobs,
                Dictionary<string, BufferRef> bufferRefs,
                Dictionary<string, AexisTensorBuffer> bufferViews,
                List<IDisposable> tempOwned,
                AexisGraphSession owner,
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
                    EnsureNcnnTextureTensor(tr, existingLogicalShape, GetTextureStorageShape(tr, existingLogicalShape));
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

            public bool TryGetExistingTextureDescriptor(
                string name,
                out BufferShape logicalShape,
                out BufferShape storageShape)
            {
                logicalShape = default;
                storageShape = default;
                if (!TryGetExistingTextureContract(
                        _textureBlobs,
                        _textureShapes,
                        name,
                        out _,
                        out var contract))
                {
                    return false;
                }

                logicalShape = contract.LogicalShape;
                storageShape = contract.StorageShape;
                return true;
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

            internal static float[] ReadExistingTextureData(
                RenderTexture texture,
                BufferShape logicalShape,
                BufferShape storageShape,
                AexisTextureTensorLayoutKind layoutKind)
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

                    if (layoutKind == AexisTextureTensorLayoutKind.LinearMat || texture.dimension == TextureDimension.Tex2D)
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
                    var logicalW = Mathf.Max(1, logicalShape.w);
                    var logicalH = logicalShape.dims >= 2 ? Mathf.Max(1, logicalShape.h) : 1;
                    var logicalD = logicalShape.dims == 4 ? Mathf.Max(1, logicalShape.d) : 1;
                    var logicalC = logicalShape.dims >= 3 ? Mathf.Max(1, logicalShape.c) : 1;
                    if (logicalW > physicalW || logicalH > physicalH || logicalD > physicalD || logicalC > physicalC)
                        throw new InvalidOperationException("texture logical crop exceeds physical storage");
                    for (var c = 0; c < logicalC; c++)
                    for (var z = 0; z < logicalD; z++)
                    for (var y = 0; y < logicalH; y++)
                    for (var x = 0; x < logicalW; x++)
                    {
                        var sourceIndex = ((c * physicalD + z) * physicalH + y) * physicalW + x;
                        var destinationIndex = ((c * logicalD + z) * logicalH + y) * logicalW + x;
                        logicalValues[destinationIndex] = physical[sourceIndex];
                    }
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
                var logicalHeight = logicalShape.dims == 1 ? 1 : Mathf.Max(1, logicalShape.h);
                var logicalPackWidth = Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f);
                var tileRows = storageShape.w > 0 ? Mathf.CeilToInt(logicalPackWidth / (float)storageShape.w) : 0;
                if ((logicalShape.dims != 1 && logicalShape.dims != 2)
                    || storageShape.dims != 3
                    || storageShape.w <= 0
                    || storageShape.h != logicalHeight * tileRows
                    || storageShape.d != 1
                    || storageShape.c != 4
                    || texture.dimension != TextureDimension.Tex2DArray
                    || texture.width != storageShape.w
                    || texture.height != storageShape.h
                    || texture.volumeDepth < 1)
                {
                    return false;
                }

                values = new float[logicalShape.w * logicalHeight];
                ReadRenderTextureSlice(texture, readback, 0);
                var raw = readback.GetRawTextureData<float>();
                for (var row = 0; row < logicalHeight; row++)
                {
                    for (var packIndex = 0; packIndex < logicalPackWidth; packIndex++)
                    {
                        var tileY = packIndex / storageShape.w;
                        var tileX = packIndex - tileY * storageShape.w;
                        var physicalY = row * tileRows + tileY;
                        var srcBase = (physicalY * storageShape.w + tileX) * 4;
                        var dstBase = row * logicalShape.w + packIndex * 4;
                        for (var lane = 0; lane < 4 && dstBase + lane < values.Length && packIndex * 4 + lane < logicalShape.w; lane++)
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
            public AexisTensorBuffer GetBufferView(string name)
            {
                if (_bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                    return view;

                var buf = GetOrMaterializeBuffer(name);
                if (_bufferViews.TryGetValue(name, out view) && view != null && view.buffer != null)
                    return view;

                if (_textureShapes.TryGetValue(name, out var shape))
                {
                    view = new AexisTensorBuffer(buf, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                    _bufferViews[name] = view;
                    return view;
                }

                view = new AexisTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
                _bufferViews[name] = view;
                return view;
            }

            public bool TryGetExistingBufferView(string name, out AexisTensorBuffer view)
            {
                if (_bufferViews.TryGetValue(name, out view) && view != null && view.buffer != null)
                    return true;

                view = null;
                return false;
            }
#endif

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

#if UNITY_EDITOR || AIIMAGE_INFERENCE_DEBUG_ORACLE
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
        public InferenceBackend Backend => InferenceBackend.GpuCompute;
        public InferenceSessionState State => _isDisposed
            ? InferenceSessionState.Disposed
            : Model == null
                ? InferenceSessionState.Created
                : InferenceSessionState.Ready;

        public AexisGraphModel Model { get; private set; }
        public IReadOnlyList<AexisBaseLayer> LayerRepros { get; private set; }
        public ModelLoadProfile LastLoadProfile { get; private set; }
        public bool ForceBufferBinaryOpAll { get; set; }
        public ComputeBuffer SharedTokenEmbeddingWeights { get; set; }
        public ComputeBuffer SharedTokenEmbeddingWeightsInt8Packed { get; set; }
        public ComputeBuffer SharedTokenEmbeddingWeightsInt8Scales { get; set; }
        public ComputeBuffer SharedTokenEmbeddingWeightsInt4Packed { get; set; }
        public ComputeBuffer SharedTokenEmbeddingWeightsInt4Scales { get; set; }
        public int SharedTokenEmbeddingWeightsInt4ScaleBlockSize { get; set; }
        public int SharedTokenEmbeddingElementCount { get; set; }
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
        // Model-scoped activation island. The named layer and all following layers retain FP32
        // activation storage while an otherwise FP16 manifest remains active.
        public string Fp32ActivationStartLayerName
        {
            get => _fp32ActivationStartLayerName;
            set
            {
                if (string.Equals(_fp32ActivationStartLayerName, value, StringComparison.Ordinal))
                    return;
                _fp32ActivationStartLayerName = value;
                _fp32ActivationIslandLayers = null;
            }
        }

        internal readonly Dictionary<string, ConvPack> _conv = new Dictionary<string, ConvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, DeconvPack> _deconv = new Dictionary<string, DeconvPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, InnerProductPack> _innerProduct = new Dictionary<string, InnerProductPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GemmPack> _gemm = new Dictionary<string, GemmPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, MemoryDataPack> _memoryData = new Dictionary<string, MemoryDataPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, EmbedPack> _embed = new Dictionary<string, EmbedPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, LayerNormPack> _layerNorm = new Dictionary<string, LayerNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, GroupNormPack> _groupNorm = new Dictionary<string, GroupNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, BatchNormPack> _batchNorm = new Dictionary<string, BatchNormPack>(StringComparer.Ordinal);
        internal readonly Dictionary<string, BiasPack> _bias = new Dictionary<string, BiasPack>(StringComparer.Ordinal);
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
        private string _fp32ActivationStartLayerName;
        private HashSet<string> _fp32ActivationIslandLayers;
        private HashSet<string> _fp32IndexSelectionProducerLayers;
        private HashSet<string> _fp32SensitiveInputProducerLayers;

        private readonly AexisOps _ops;

        public const bool EnableWinograd23 = false;
        public bool PreferTexturePathForFaceDetector { get; set; }
        public bool ForceBufferConvolutionAll { get; set; }
        public bool KeepRawConvWeightsForTexturePath { get; set; } = true;
        public bool EnableMhaParallelSoftmax { get; set; }
        public bool EnableMhaQkvFusion { get; set; }
        public bool EnableAttentionMatMulPack4Specializations { get; set; }
        public bool EnableVistaTailPack4Specializations { get; set; }
        // Preserves a model's verified pre-migration FP32 tensor contract while newer
        // Pack4-linear specializations remain available to FP16 and other runners.
        public bool PreserveLegacyFp32Execution { get; set; }
        // Keeps attention and linear projections in the model's established Pack4 layout
        // without changing its activation or weight precision.
        public bool UseLegacyPack4AttentionLayout { get; set; }
        private RenderTextureFormat _tensorTextureFormat = RenderTextureFormat.ARGBHalf;
        public RenderTextureFormat TensorTextureFormat
        {
            get
            {
                if (ModelManifest != null)
                    return UsesFp16ActivationStorage ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGBFloat;
                return AppliedPrecisionMode == AexisPrecisionMode.FP16
                    ? RenderTextureFormat.ARGBHalf
                    : AppliedPrecisionMode == AexisPrecisionMode.FP32
                        ? RenderTextureFormat.ARGBFloat
                        : _tensorTextureFormat;
            }
            set => _tensorTextureFormat = value;
        }
        public ModelManifest ModelManifest { get; private set; }
        public AexisPrecisionMode AppliedPrecisionMode { get; private set; } = AexisPrecisionMode.Auto;
        public bool UsesFp16WeightStorage => ModelManifest?.precision?.weightDataType == TensorDataType.Float16;
        // Unity exposes no portable BF16 RenderTexture format. BF16 tensors are kept
        // in ARGBFloat storage and rounded by the Pack4 Cast path, so capacity and
        // accumulation remain FP32 while the declared logical precision is BF16.
        public bool UsesBf16ActivationStorage => ModelManifest?.precision?.activationDataType == TensorDataType.BFloat16;
        public bool UsesInt8WeightOnly => ModelManifest?.IsInt8WeightOnly == true;
        public bool UsesInt4WeightOnly => ModelManifest?.IsInt4WeightOnly == true;
        public bool UsesQuantizedWeightOnly => UsesInt8WeightOnly || UsesInt4WeightOnly;
        public bool UsesFp16ActivationStorage => ModelManifest?.precision?.activationDataType == TensorDataType.Float16;
        internal bool UsesFp16WeightsForCurrentLayer => UsesFp16WeightStorage;

        // Model-level precision is the fallback; a mixed plan is authoritative for
        // its declared layer. BF16 deliberately resolves to ARGBFloat because Unity
        // has no portable BF16 RenderTexture, while FP16 uses ARGBHalf.
        internal bool TryGetMixedPrecisionNodePlan(string layerName, string operatorName, out MixedPrecisionNodePlan plan)
        {
            foreach (var candidate in ModelManifest?.mixedPrecision?.nodePlans ?? Array.Empty<MixedPrecisionNodePlan>())
            {
                if (candidate == null || !string.Equals(candidate.layerName, layerName, StringComparison.Ordinal))
                    continue;
                if (!string.IsNullOrWhiteSpace(candidate.operatorName)
                    && !string.Equals(candidate.operatorName, operatorName, StringComparison.Ordinal))
                    break;
                plan = candidate;
                return true;
            }
            plan = null;
            return false;
        }

        internal bool UsesFp16WeightsForLayer(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            return TryGetMixedPrecisionNodePlan(layer?.name, operatorName, out var plan)
                ? plan.weightDataType == TensorDataType.Float16
                : UsesFp16WeightStorage;
        }

        internal RenderTextureFormat ResolveActivationTextureFormat(AexisGraphModel.Layer layer, int dims)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (!TryGetMixedPrecisionNodePlan(layer?.name, operatorName, out var plan))
            {
                // A manifest owns the default precision. Manifest-less sessions may
                // deliberately select a texture format (the face detector uses
                // ARGBFloat), so do not discard that explicit runtime contract.
                return ModelManifest != null
                    ? ResolveActivationTextureFormat(dims)
                    : TensorTextureFormat;
            }
            var activationType = plan.activationDataType;
            return activationType == TensorDataType.Float16 ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGBFloat;
        }
        public bool UsesInt8WeightOnlyForLayer(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (ModelManifest?.quantization?.nodePlans != null && ModelManifest.quantization.nodePlans.Length > 0)
            {
                return ModelManifest.TryGetQuantizedNodePlan(layer?.name, operatorName, out var plan)
                    && (plan.mode == QuantizedNodeMode.Int8WeightOnly || plan.mode == QuantizedNodeMode.Int8W8A8);
            }
            return ModelManifest?.UsesInt8WeightOnlyForOperator(operatorName) == true;
        }

        public sealed class CommandBufferInferResult : IDisposable
        {
            private readonly AexisGraphSession _owner;
            private readonly CommandBuffer _commandBuffer;
            private readonly Dictionary<string, ComputeTexture> _textures;
            private readonly Dictionary<string, BufferShape> _logicalShapes;
            private bool _disposed;

            internal CommandBufferInferResult(
                AexisGraphSession owner,
                CommandBuffer commandBuffer,
                Dictionary<string, ComputeTexture> textures,
                Dictionary<string, BufferShape> logicalShapes)
            {
                _owner = owner ?? throw new ArgumentNullException(nameof(owner));
                _commandBuffer = commandBuffer ?? throw new ArgumentNullException(nameof(commandBuffer));
                _textures = textures ?? new Dictionary<string, ComputeTexture>(StringComparer.Ordinal);
                _logicalShapes = logicalShapes ?? new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            }

            public IReadOnlyCollection<string> OutputNames => _textures.Keys;

            public ComputeTexture GetTexture(string name)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(CommandBufferInferResult));
                if (!_textures.TryGetValue(name, out var texture) || texture == null)
                    throw new InvalidOperationException("CommandBuffer output texture not found: " + name);
                return texture;
            }

            public BufferShape GetLogicalShape(string name)
            {
                if (_disposed)
                    throw new ObjectDisposedException(nameof(CommandBufferInferResult));
                if (!_logicalShapes.TryGetValue(name, out var shape))
                    throw new InvalidOperationException("CommandBuffer output logical shape not found: " + name);
                return shape;
            }

            public void Dispose()
            {
                if (_disposed)
                    return;
                _disposed = true;
                var released = new HashSet<ComputeTexture>();
                foreach (var texture in _textures.Values)
                {
                    if (texture != null && released.Add(texture))
                        _owner.ReturnTempArray(_commandBuffer, texture);
                }
                _textures.Clear();
                _logicalShapes.Clear();
            }
        }

        // Recurrent operators deliberately carry no mutable CPU state.
        // Weights are model tensors and caches are published as graph textures.
        public sealed class ShortConvPack : IDisposable
        {
            public int groups;
            public int kernelSize;
            public void Dispose() { }
        }

        public sealed class GatedDeltaRulePack : IDisposable
        {
            public int heads;
            public int keyDim;
            public int valueDim;
            public float epsilon = 1e-6f;
            public void Dispose() { }
        }

        // The P2 recurrent profile keeps only immutable gate weights in GPU
        // buffers. Sequence activations and the externally-visible state remain
        // Pack4 Texture2DArrays throughout command-buffer execution.
        public sealed class RecurrentPack : IDisposable
        {
            public int kind;
            public int inputSize;
            public int hiddenSize;
            public ComputeBuffer inputWeights;
            public ComputeBuffer recurrentWeights;
            public ComputeBuffer bias;

            public void Dispose()
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(inputWeights, "AexisGraphSession.RecurrentPack.Dispose"); inputWeights?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(recurrentWeights, "AexisGraphSession.RecurrentPack.Dispose"); recurrentWeights?.Dispose(); } catch { }
                try { AexisGpuResourceTracker.ReleaseBuffer(bias, "AexisGraphSession.RecurrentPack.Dispose"); bias?.Dispose(); } catch { }
            }
        }
        internal bool UsesInt8WeightsForLayer(AexisGraphModel.Layer layer) => UsesInt8WeightOnlyForLayer(layer);

        public bool UsesInt4WeightOnlyForLayer(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (ModelManifest?.quantization?.nodePlans != null && ModelManifest.quantization.nodePlans.Length > 0)
            {
                return ModelManifest.TryGetQuantizedNodePlan(layer?.name, operatorName, out var plan)
                    && plan.mode == QuantizedNodeMode.Int4WeightOnly;
            }
            return ModelManifest?.UsesInt4WeightOnlyForOperator(operatorName) == true;
        }

        internal bool UsesInt4WeightsForLayer(AexisGraphModel.Layer layer) => UsesInt4WeightOnlyForLayer(layer);
        internal bool UsesQuantizedWeightsForLayer(AexisGraphModel.Layer layer) => UsesInt8WeightsForLayer(layer) || UsesInt4WeightsForLayer(layer);

        internal void ConfigureInt8ActivationQuantization(AexisGraphModel.Layer layer)
        {
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (TryGetActivationQuantizationPlan(layer?.name, operatorName, out var activationPlan))
            {
                _ops.SetInt8ActivationQuantization(activationPlan);
                return;
            }
            QuantizedNodePlan plan = null;
            ModelManifest?.TryGetQuantizedNodePlan(layer?.name, operatorName, out plan);
            _ops.SetInt8ActivationQuantization(plan);
        }

        internal bool TryGetActivationQuantizationPlan(string layerName, string operatorName, out QuantizedActivationPlan plan)
        {
            foreach (var candidate in ModelManifest?.mixedPrecision?.activationPlans ?? Array.Empty<QuantizedActivationPlan>())
            {
                if (candidate == null || !string.Equals(candidate.layerName, layerName, StringComparison.Ordinal))
                    continue;
                if (!string.IsNullOrWhiteSpace(candidate.operatorName)
                    && !string.Equals(candidate.operatorName, operatorName, StringComparison.Ordinal))
                    break;
                plan = candidate;
                return true;
            }
            plan = null;
            return false;
        }

        internal void ResetInt8ActivationQuantization()
        {
            _ops.SetInt8ActivationQuantization((QuantizedNodePlan)null);
        }
        public long TemporaryTextureBudgetBytes { get; set; }
        public int AttentionKvCacheTextureCapacity { get; set; }
        // Opt-in only: some graphs require functional rather than mutable cache tensors.
        public bool EnableInPlaceAttentionKvCache { get; set; }
        public AexisInferenceExecutionMode ExecutionMode { get; set; } = AexisInferenceExecutionMode.ProductionTextureOnly;
        // Production CommandBuffer execution is always planned strictly. DebugOracle is the
        // only explicit relaxation path and remains unavailable in non-debug builds.
        public bool StrictTextureInference => !IsExplicitDebugOracleExecution;
        public string StrictTextureTargetDtype { get; set; } = "FP16";
        public string StrictTextureTargetLayout { get; set; } = AexisTexturePlanLayout.Packed4;
        // External texture inputs are float textures even when they carry exact Int32
        // index values. Callers must explicitly declare those logical dtypes here.
        public IDictionary<string, string> StrictTextureInputLogicalDtypes { get; } = new Dictionary<string, string>(StringComparer.Ordinal);
        public AexisTextureExecutionPlan LastTextureExecutionPlan { get; private set; }
        public CommandBufferRtArenaMetrics LastCommandBufferRtArena { get; private set; }
        public bool DisallowBufferAccess { get; set; }
        public bool DisallowBufferOutputs { get; set; }
        public bool DisallowBufferToTextureMaterialization { get; set; }
        public bool ForceBufferOutputsForDims4 { get; set; }
        public ISet<string> ForceBufferLayerTypes { get; set; }
        public ISet<string> ForceBufferLayerNames { get; set; }
        public ISet<string> DebugCompareTextureLayers { get; set; }
        public ISet<string> DebugCompareTextureConvLayers { get; set; }
        public ISet<string> DebugCompareMaxPoolingLayers { get; set; }
        public ISet<string> DebugLayerReadbackBlobs { get; set; }
        public Action<string, string, float[]> DebugLayerTextureReadback { get; set; }
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
        public AexisOps Ops => _ops;
        public bool gpuLayerProfileEnabled = false;
        public bool useExperimentalIteratePath = false;
        public bool LayerRuntimeProfileEnabled { get; set; }
        public bool LayerRuntimeProfileSyncGpu { get; set; }
        public string LayerRuntimeProfilePathKindOverride { get; set; }
        public LayerRuntimeProfile LastRuntimeProfile { get; private set; }
        public long ManagedLoadGarbageCollectionIntervalBytes { get; set; }
        public string TimingSplitSyncAfterTopName { get; set; }
        public Action<string, double> OnTimingSplitSyncPoint { get; set; }
        public event Action<string, string, int, int, int, int, double> OnConvComplete;
        private const int FallbackMaxTextureArraySlices = 2048;
        private const int FallbackMaxTextureSize = 16384;
        private string _currentExecutingLayerName;
        private string _currentExecutingLayerTypeName;
        private int _currentExecutingLayerIndex = -1;
        private AexisLayerBufferContext _currentBufferContext;
        private CommandBufferRtArena _activeCommandBufferRtArena;

        public void ApplyModelManifest(ModelManifest manifest)
        {
            if (manifest == null)
                throw new ArgumentNullException(nameof(manifest));
            manifest.Validate();
            if (Model != null)
            {
                throw new InvalidOperationException(
                    "Model manifest precision must be applied before model weights are loaded"
                    + " | model=" + manifest.modelId
                    + " | rejected_fallback=FP32-weight-reuse");
            }

            ModelManifest = manifest;
            TensorTextureFormat = manifest.precision.activationDataType == TensorDataType.Float16
                ? RenderTextureFormat.ARGBHalf
                : RenderTextureFormat.ARGBFloat;
            // targetDtype describes activation textures.  The plan separately records
            // whether immutable weights require the D2 INT8 capability.
            StrictTextureTargetDtype = manifest.precision.activationDataType == TensorDataType.Float16
                ? "FP16"
                : manifest.precision.activationDataType == TensorDataType.BFloat16 ? "BF16" : "FP32";
            StrictTextureTargetLayout = AexisTexturePlanLayout.Packed4;
        }

        public void SetAppliedPrecisionMode(AexisPrecisionMode precisionMode)
        {
            AppliedPrecisionMode = precisionMode;
            if (ModelManifest == null)
            {
                _tensorTextureFormat = precisionMode == AexisPrecisionMode.FP16
                    ? RenderTextureFormat.ARGBHalf
                    : precisionMode == AexisPrecisionMode.FP32
                        ? RenderTextureFormat.ARGBFloat
                        : _tensorTextureFormat;
                StrictTextureTargetDtype = precisionMode == AexisPrecisionMode.FP32 ? "FP32"
                    : precisionMode == AexisPrecisionMode.BF16 ? "BF16" : "FP16";
            }
        }

        public RenderTextureFormat ResolveActivationTextureFormat(int dims)
        {
            if (ModelManifest == null)
                return AppliedPrecisionMode == AexisPrecisionMode.Auto
                    ? ResolveTensorTextureFormat(dims)
                    : TensorTextureFormat;
            return UsesFp16ActivationStorage ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGBFloat;
        }

        internal RenderTextureFormat ResolveSensitiveOutputTextureFormat()
        {
            if (ModelManifest == null)
                return TensorTextureFormat;
            return ModelManifest.precision.sensitiveOutputDataType == TensorDataType.Float16
                ? RenderTextureFormat.ARGBHalf
                : RenderTextureFormat.ARGBFloat;
        }

        internal bool RequiresFp32SensitiveOutputStorage(AexisGraphModel.Layer layer)
        {
            if (layer == null
                || ModelManifest?.precision?.sensitiveOutputDataType != TensorDataType.Float32
                || layer.topNames == null)
                return false;

            var outputBlobName = ResolveDefaultOutputBlobName();
            for (var i = 0; i < layer.topNames.Length; i++)
            {
                if (string.Equals(layer.topNames[i], outputBlobName, StringComparison.Ordinal))
                    return true;
            }
            return false;
        }

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

        // Buffer execution is diagnostic-only. A debug flag must not silently relax
        // production inference; callers have to select DebugOracle explicitly.
        public bool IsDebugOracleExecution => IsExplicitDebugOracleExecution;

        private bool IsExplicitDebugOracleExecution => IsDebugOracleBuild
            && ExecutionMode == AexisInferenceExecutionMode.DebugOracle;

        private void EnsureCommandBufferTextureExecutionPlan(
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            ISet<string> fixedBufferInputBlobs = null,
            string stopAfterTopName = null)
        {
            if (Model == null)
                throw new InvalidOperationException("model not loaded");
            EnsureProductionDebugOverridesRejected();

            var inputs = new List<AexisTexturePlanTensorDescriptor>();
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
                var storageShape = ResolveExternalTextureInputStorageShape(
                    logicalShape,
                    texture.width,
                    texture.height,
                    texture.dimension,
                    depth,
                    texture.format);
                var physicalDtype = ResolveTexturePlanDtype(texture.format);
                var fixedBufferUpload = fixedBufferInputBlobs != null && fixedBufferInputBlobs.Contains(kv.Key);
                inputs.Add(new AexisTexturePlanTensorDescriptor
                {
                    blob = kv.Key,
                    logicalShape = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c },
                    storageShape = new[] { storageShape.dims, storageShape.w, storageShape.h, storageShape.d, storageShape.c },
                    layout = StrictTextureTargetLayout,
                    dtype = physicalDtype,
                    logicalDtype = fixedBufferUpload && IsFixedIntegerInput(kv.Key)
                        ? "Int32"
                        : ResolveStrictInputLogicalDtype(kv.Key, physicalDtype),
                    fixedInputUpload = fixedBufferUpload,
                    aliasGroup = "input:" + kv.Key,
                    textureBacked = true
                });
            }

            CompleteTextureExecutionPlan(inputs, IsExplicitDebugOracleExecution, stopAfterTopName);
        }

        // Immediate execution uses the same CommandBuffer Pack4 admission gate. This
        // keeps an RT-only caller from running a graph that could not be scheduled in
        // the public asynchronous CommandBuffer API.
        internal void EnsureImmediateTextureExecutionPlan(
            Dictionary<string, TensorRef> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            ISet<string> fixedBufferInputBlobs = null,
            string stopAfterTopName = null)
        {
            if (Model == null)
                throw new InvalidOperationException("model not loaded");
            EnsureProductionDebugOverridesRejected();

            var inputs = new List<AexisTexturePlanTensorDescriptor>();
            foreach (var kv in textureInputs ?? new Dictionary<string, TensorRef>(StringComparer.Ordinal))
            {
                var tensor = kv.Value;
                if (tensor?.texture == null)
                    throw new ArgumentNullException("textureInputs[\"" + kv.Key + "\"]");

                var logicalShape = textureInputShapes != null && textureInputShapes.TryGetValue(kv.Key, out var suppliedShape)
                    ? suppliedShape
                    : GetTextureShape(textureInputShapes, tensor, kv.Key);
                var storageShape = GetTextureStorageShape(tensor, logicalShape);
                var physicalDtype = ResolveTexturePlanDtype(tensor.texture.format);
                var fixedBufferUpload = fixedBufferInputBlobs != null && fixedBufferInputBlobs.Contains(kv.Key);
                inputs.Add(new AexisTexturePlanTensorDescriptor
                {
                    blob = kv.Key,
                    logicalShape = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c },
                    storageShape = new[] { storageShape.dims, storageShape.w, storageShape.h, storageShape.d, storageShape.c },
                    layout = StrictTextureTargetLayout,
                    dtype = physicalDtype,
                    logicalDtype = fixedBufferUpload && IsFixedIntegerInput(kv.Key)
                        ? "Int32"
                        : ResolveStrictInputLogicalDtype(kv.Key, physicalDtype),
                    fixedInputUpload = fixedBufferUpload,
                    aliasGroup = "input:" + kv.Key,
                    textureBacked = true
                });
            }

            CompleteTextureExecutionPlan(inputs, IsExplicitDebugOracleExecution, stopAfterTopName);
        }

        private string ResolveStrictInputLogicalDtype(string blobName, string physicalDtype)
        {
            if (!string.IsNullOrWhiteSpace(blobName)
                && StrictTextureInputLogicalDtypes.TryGetValue(blobName, out var declared)
                && !string.IsNullOrWhiteSpace(declared))
                return declared.Trim();
            return ResolveLogicalDtype(physicalDtype);
        }

        private void CompleteTextureExecutionPlan(
            List<AexisTexturePlanTensorDescriptor> inputs,
            bool debugOracleRelaxed,
            string stopAfterTopName = null)
        {
            var explicitInt8LayerNames = GetExplicitInt8WeightOnlyLayerNamesForPlan();
            var explicitInt4LayerNames = GetExplicitInt4WeightOnlyLayerNamesForPlan();
            LastTextureExecutionPlan = AexisTextureExecutionPlanner.Analyze(Model, new AexisTextureExecutionPlanRequest
            {
                modelName = LastLoadProfile?.modelMagic ?? string.Empty,
                targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                targetDtype = StrictTextureTargetDtype,
                targetLayout = StrictTextureTargetLayout,
                strict = StrictTextureInference,
                debugOracleRelaxed = debugOracleRelaxed,
                int8WeightOnly = UsesInt8WeightOnly,
                int8WeightOnlyOperators = ModelManifest?.quantization?.quantizedOperators ?? Array.Empty<string>(),
                int8WeightOnlyLayerNames = explicitInt8LayerNames,
                int8WeightOnlyLayerSelectionExplicit = explicitInt8LayerNames != null,
                int4WeightOnly = UsesInt4WeightOnly,
                int4WeightOnlyOperators = ModelManifest?.quantization?.quantizedOperators ?? Array.Empty<string>(),
                int4WeightOnlyLayerNames = explicitInt4LayerNames,
                int4WeightOnlyLayerSelectionExplicit = explicitInt4LayerNames != null,
                fp32ActivationLayerNames = GetFp32ActivationLayerNamesForPlan(),
                stopAfterTopName = stopAfterTopName,
                inputs = inputs.ToArray(),
                nodeVerifier = VerifyStrictCommandBufferPack4Node
            });
            AexisTextureExecutionPlanner.ThrowIfDispatchRejected(LastTextureExecutionPlan);
        }

        private string[] GetFp32ActivationLayerNamesForPlan()
        {
            if (!UsesFp16ActivationStorage || Model?.layers == null)
            {
                return Array.Empty<string>();
            }

            var names = new List<string>();
            var previousName = _currentExecutingLayerName;
            var previousTypeName = _currentExecutingLayerTypeName;
            var previousIndex = _currentExecutingLayerIndex;
            try
            {
                for (var index = 0; index < Model.layers.Count; index++)
                {
                    var layer = Model.layers[index];
                    if (string.IsNullOrWhiteSpace(layer?.name))
                        continue;

                    _currentExecutingLayerName = layer.name;
                    _currentExecutingLayerTypeName = layer.typeName;
                    _currentExecutingLayerIndex = index;
                    if (RequiresFp32IndexSelectionInputStorage()
                        || RequiresFp32SensitiveInputStorage()
                        || UsesFp32ActivationIsland()
                        || (RequiresFp32AccumulatorOutput(layer.typeName)
                            && ModelManifest?.precision?.sensitiveOutputDataType == TensorDataType.Float32))
                    {
                        names.Add(layer.name);
                    }
                }
            }
            finally
            {
                _currentExecutingLayerName = previousName;
                _currentExecutingLayerTypeName = previousTypeName;
                _currentExecutingLayerIndex = previousIndex;
            }
            return names.Distinct(StringComparer.Ordinal).ToArray();
        }

        internal void BeginCommandBufferRtArena(CommandBuffer commandBuffer)
        {
            if (_activeCommandBufferRtArena != null)
                throw new InvalidOperationException("A CommandBuffer RT arena is already active for this session.");
            if (LastTextureExecutionPlan == null || !LastTextureExecutionPlan.dispatchAllowed)
                throw new InvalidOperationException("A verified texture execution plan is required before binding the CommandBuffer RT arena.");

            _activeCommandBufferRtArena = new CommandBufferRtArena(this, commandBuffer, LastTextureExecutionPlan);
            LastCommandBufferRtArena = _activeCommandBufferRtArena.Metrics;
        }

        internal void CompleteCommandBufferRtArena()
        {
            _activeCommandBufferRtArena?.Complete();
        }

        internal void EndCommandBufferRtArena()
        {
            var arena = _activeCommandBufferRtArena;
            _activeCommandBufferRtArena = null;
            arena?.Dispose();
        }

        public AexisModelPreflightReport AnalyzeLoadedModelPreflight(AexisModelPreflightRequest request)
        {
            if (Model == null)
                throw new InvalidOperationException("model not loaded");
            request ??= new AexisModelPreflightRequest();
            request.nodeVerifier = VerifyStrictCommandBufferPack4Node;
            return AexisModelPreflight.Analyze(Model, request);
        }

        private string[] GetExplicitInt8WeightOnlyLayerNamesForPlan()
        {
            if (!UsesInt8WeightOnly)
                return null;
            var quantization = ModelManifest?.quantization;
            if (quantization == null)
                return null;
            if (quantization.quantizedOperators != null && quantization.quantizedOperators.Length > 0)
                return null;
            if (quantization.nodePlans == null || quantization.nodePlans.Length == 0)
                return null;
            return quantization.nodePlans
                .Where(plan => plan != null
                    && plan.mode != QuantizedNodeMode.Float
                    && !string.IsNullOrWhiteSpace(plan.layerName))
                .Select(plan => plan.layerName)
                .Distinct(StringComparer.Ordinal)
                .ToArray();
        }

        private string[] GetExplicitInt4WeightOnlyLayerNamesForPlan()
        {
            if (!UsesInt4WeightOnly)
                return null;
            var quantization = ModelManifest?.quantization;
            if (quantization == null)
                return null;
            if (quantization.quantizedOperators != null && quantization.quantizedOperators.Length > 0)
                return null;
            if (quantization.nodePlans == null || quantization.nodePlans.Length == 0)
                return null;
            return quantization.nodePlans
                .Where(plan => plan != null
                    && plan.mode != QuantizedNodeMode.Float
                    && !string.IsNullOrWhiteSpace(plan.layerName))
                .Select(plan => plan.layerName)
                .Distinct(StringComparer.Ordinal)
                .ToArray();
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPack4Node(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!string.Equals(request?.targetBackend, AexisOperatorCapabilityBackend.CommandBuffer, StringComparison.Ordinal)
                || (!string.Equals(request?.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(request?.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase)
                    && !string.Equals(request?.targetDtype, "BF16", StringComparison.OrdinalIgnoreCase))
                || !string.Equals(request?.targetLayout, AexisTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase))
            {
                return RejectStrictCommandBufferPack4Node("The loaded runtime profile supports only FP16, BF16-emulated, or FP32 Packed4 CommandBuffer branches.");
            }
            if ((string.Equals(request.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(request.targetDtype, "BF16", StringComparison.OrdinalIgnoreCase))
                && TensorTextureFormat != RenderTextureFormat.ARGBFloat)
                return RejectStrictCommandBufferPack4Node("FP32/BF16 Pack4 requires TensorTextureFormat=ARGBFloat for texture-native intermediate/output storage.");

            if (ModelManifest != null
                && string.Equals(request.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                && UsesFp16WeightStorage
                && !TryVerifyFp16WeightStorage(layer, out var fp16WeightReason))
            {
                return RejectStrictCommandBufferPack4Node(fp16WeightReason);
            }

            if (UsesQuantizedWeightsForLayer(layer) && !TryVerifyQuantizedWeightOnlyStorage(layer, out var quantizedWeightReason))
                return RejectStrictCommandBufferPack4Node(quantizedWeightReason);

            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            switch (operatorName)
            {
                case "Convolution":
                    return VerifyStrictCommandBufferConvolution(layer, inputs, request);
                case "Convolution1D":
                    return VerifyStrictCommandBufferConvolution1D(layer, inputs, request);
                case "ConvolutionDepthWise":
                    return VerifyStrictCommandBufferDepthWiseConvolution(layer, inputs, request);
                case "ConvolutionDepthWise1D":
                    return VerifyStrictCommandBufferConvolutionDepthWise1D(layer, inputs, request);
                case "Convolution3D":
                    return VerifyStrictCommandBufferConvolution3D(layer, inputs, request);
                case "ConvolutionDepthWise3D":
                case "ConvDw3D":
                    return VerifyStrictCommandBufferConvolutionDepthWise3D(layer, inputs, request);
                case "Deconvolution":
                    return VerifyStrictCommandBufferDeconvolution(layer, inputs, request, depthWiseLayer: false);
                case "Deconvolution1D":
                    return VerifyStrictCommandBufferDeconvolution1D(layer, inputs, request);
                case "DeconvolutionDepthWise1D":
                case "DeconvDw1D":
                    return VerifyStrictCommandBufferDeconvolutionDepthWise1D(layer, inputs, request);
                case "DeconvolutionDepthWise":
                    return VerifyStrictCommandBufferDeconvolution(layer, inputs, request, depthWiseLayer: true);
                case "Deconvolution3D":
                    return VerifyStrictCommandBufferDeconvolution3D(layer, inputs, request);
                case "DeconvolutionDepthWise3D":
                case "DeconvDw3D":
                    return VerifyStrictCommandBufferDeconvolutionDepthWise3D(layer, inputs, request);
                case "StatisticsPooling":
                case "StatsPooling":
                    return VerifyStrictCommandBufferStatisticsPooling(layer, inputs, request);
                case "Spectrogram":
                    return VerifyStrictCommandBufferSpectrogram(layer, inputs, request, inverse: false);
                case "InverseSpectrogram":
                case "InvSpectrogram":
                    return VerifyStrictCommandBufferSpectrogram(layer, inputs, request, inverse: true);
                case "RNN":
                    return VerifyStrictCommandBufferRecurrent(layer, inputs, request, AexisRecurrentKind.Rnn);
                case "GRU":
                    return VerifyStrictCommandBufferRecurrent(layer, inputs, request, AexisRecurrentKind.Gru);
                case "LSTM":
                    return VerifyStrictCommandBufferRecurrent(layer, inputs, request, AexisRecurrentKind.Lstm);
                case "Embed":
                    return VerifyStrictCommandBufferEmbed(layer, inputs, request);
                case "Eltwise":
                    return VerifyStrictCommandBufferEltwise(layer, inputs, request);
                case "Concat":
                    return VerifyStrictCommandBufferConcat(layer, inputs, request);
                case "BinaryOp":
                    return VerifyStrictCommandBufferBinaryOp(layer, inputs, request);
                case "ReLU":
                    return VerifyStrictCommandBufferRelu(layer, inputs, request);
                case "Sigmoid":
                    return VerifyStrictCommandBufferSigmoid(layer, inputs, request);
                case "Interp":
                    return VerifyStrictCommandBufferInterp3DOr2D(layer, inputs, request);
                case "PixelShuffle":
                    return VerifyStrictCommandBufferPixelShuffle(layer, inputs, request);
                case "UnaryOp":
                    return VerifyStrictCommandBufferUnaryOp(layer, inputs, request);
                case "AbsVal":
                    return VerifyStrictCommandBufferUnaryAlias(layer, inputs, request, 0, "abs");
                case "TanH":
                    return VerifyStrictCommandBufferUnaryAlias(layer, inputs, request, 16, "tanh");
                case "BNLL":
                    return VerifyStrictCommandBufferPointwise(layer, inputs, request, operatorName);
                case "Exp":
                case "Log":
                case "Power":
                case "Threshold":
                case "ThresholdedRelu":
                case "ELU":
                case "Erf":
                case "HardSigmoid":
                case "HardSwish":
                case "Mish":
                case "SELU":
                case "Shrink":
                case "Softplus":
                case "Softsign":
                case "IsInf":
                case "IsNaN":
                case "CELU":
                case "Swish":
                case "Clip":
                    return VerifyStrictCommandBufferPointwise(layer, inputs, request, operatorName);
                case "Trilu":
                    return VerifyStrictCommandBufferTrilu(layer, inputs, request);
                case "CumSum":
                case "CumulativeSum":
                    return VerifyStrictCommandBufferCumSum(layer, inputs, request);
                case "GELU":
                    return VerifyStrictCommandBufferGelu(layer, inputs, request);
                case "pnnx.Expression":
                    return VerifyStrictCommandBufferPnnxExpression(layer, request);
                case "MemoryData":
                    return VerifyStrictCommandBufferMemoryData(layer, request);
                case "InnerProduct":
                    return VerifyStrictCommandBufferInnerProduct(layer, inputs, request);
                case "Pooling":
                    return VerifyStrictCommandBufferPooling(layer, inputs, request);
                case "Pooling1D":
                    return VerifyStrictCommandBufferPooling1D(layer, inputs, request);
                case "MaxPoolingInd":
                    return VerifyStrictCommandBufferMaxPoolingInd(layer, inputs, request);
                case "MaxUnPooling":
                    return VerifyStrictCommandBufferMaxUnPooling(layer, inputs, request);
                case "Pooling3D":
                    return VerifyStrictCommandBufferPooling3D(layer, inputs, request);
                case "Reduction":
                    return VerifyStrictCommandBufferReduction(layer, inputs, request);
                case "BatchNorm":
                    return VerifyStrictCommandBufferBatchNorm(layer, inputs, request);
                case "PReLU":
                    return VerifyStrictCommandBufferPRelu(layer, inputs, request);
                case "LRN":
                    return VerifyStrictCommandBufferLrn(layer, inputs, request);
                case "InstanceNorm":
                    return VerifyStrictCommandBufferInstanceNorm(layer, inputs, request);
                case "MVN":
                    return VerifyStrictCommandBufferMvn(layer, inputs, request);
                case "Bias":
                    return VerifyStrictCommandBufferBias(layer, inputs, request);
                case "CopyTo":
                    return VerifyStrictCommandBufferCopyTo(layer, inputs, request);
                case "aten::to":
                    return VerifyStrictCommandBufferAtenTo(layer, inputs, request);
                case "Crop":
                    return VerifyStrictCommandBufferCrop(layer, inputs, request);
                case "GroupNorm":
                    return VerifyStrictCommandBufferGroupNorm(layer, inputs, request);
                case "Padding":
                    return VerifyStrictCommandBufferPadding(layer, inputs, request);
                case "Quantize":
                case "Dequantize":
                case "Requantize":
                    return VerifyStrictCommandBufferQuantization(layer, inputs, request, operatorName);
                case "Reorg":
                    return VerifyStrictCommandBufferReorg(layer, inputs, request);
                case "Scale":
                    return VerifyStrictCommandBufferScale(layer, inputs, request);
                case "Unfold":
                    return VerifyStrictCommandBufferUnfold(layer, inputs, request);
                case "ExtractPatches":
                    return VerifyStrictCommandBufferExtractPatches(layer, inputs, request);
                case "Reshape":
                    return VerifyStrictCommandBufferReshape(layer, inputs, request);
                case "Flatten":
                    return VerifyStrictCommandBufferFlatten(layer, inputs, request);
                case "Squeeze":
                    return VerifyStrictCommandBufferSqueeze(layer, inputs, request);
                case "ExpandDims":
                    return VerifyStrictCommandBufferExpandDims(layer, inputs, request);
                case "Permute":
                    return VerifyStrictCommandBufferPermute(layer, inputs, request);
                case "ShuffleChannel":
                    return VerifyStrictCommandBufferShuffleChannel(layer, inputs, request);
                case "Gemm":
                    return VerifyStrictCommandBufferGemm(layer, inputs, request);
                case "LayerNorm":
                    return VerifyStrictCommandBufferLayerNorm(layer, inputs, request);
                case "RMSNorm":
                    return VerifyStrictCommandBufferRmsNorm(layer, inputs, request);
                case "Slice":
                    return VerifyStrictCommandBufferSlice(layer, inputs, request);
                case "Tile":
                    return VerifyStrictCommandBufferTile(layer, inputs, request);
                case "Packing":
                    return VerifyStrictCommandBufferPacking(layer, inputs, request);
                case "Cast":
                    return VerifyStrictCommandBufferCast(layer, inputs, request);
                case "MatMul":
                    return VerifyStrictCommandBufferMatMul(layer, inputs, request);
                case "GridSample":
                case "DeformableConv2D":
                case "Fold":
                case "Flip":
                case "GLU":
                case "Einsum":
                case "Diag":
                case "SPP":
                case "ROIAlign":
                case "ROIPooling":
                case "PSROIPooling":
                case "Proposal":
                case "DetectionOutput":
                case "YoloDetectionOutput":
                case "Yolov3DetectionOutput":
                case "YoloDetectOut":
                case "Yolov3DetectOut":
                    return VerifyStrictCommandBufferP1Vision(layer, inputs, request);
                case "RandomUniformLike":
                case "RandomNormalLike":
                case "RandomUniform":
                case "RandomNormal":
                case "Bernoulli":
                case "RandomLike":
                    return VerifyStrictCommandBufferDeterministicRandom(layer, inputs, request);
                case "Multinomial":
                    return VerifyStrictCommandBufferMultinomial(layer, inputs, request);
                case "Softmax":
                    return VerifyStrictCommandBufferSoftmax(layer, inputs, request);
                case "SDPA":
                    return VerifyStrictCommandBufferSdpa(layer, inputs, request);
                case "MultiHeadAttention":
                    return VerifyStrictCommandBufferMultiHeadAttention(layer, inputs, request);
                case "RotaryEmbed":
                    return VerifyStrictCommandBufferRotaryEmbed(layer, inputs, request);
                case "ShortConv":
                    return VerifyStrictCommandBufferShortConv(layer, inputs, request);
                case "GatedDeltaRule":
                    return VerifyStrictCommandBufferGatedDeltaRule(layer, inputs, request);
                case "DeepFillV2ContextualAttention":
                    return VerifyStrictCommandBufferDeepFillV2ContextualAttention(layer, inputs, request);
                case "Nms":
                    return VerifyStrictCommandBufferNonMaxSuppression(layer, inputs, request);
                case "NonZero":
                case "Compress":
                case "GatherND":
                case "Scatter":
                case "ScatterElements":
                case "ScatterND":
                    return VerifyStrictCommandBufferDataIndex(layer, inputs, request, operatorName);
                case "Shape":
                case "Size":
                case "Range":
                case "ConstantOfShape":
                case "Expand":
                case "Where":
                case "Gather":
                case "GatherElements":
                case "ArgMax":
                case "ArgMin":
                case "TopK":
                case "OneHot":
                    return VerifyStrictCommandBufferSentisTextureNode(layer, inputs, request, operatorName);
                default:
                    return RejectStrictCommandBufferPack4Node("No loaded-runtime Pack4 proof exists for operator " + (operatorName ?? string.Empty) + ".");
            }
        }

        private bool TryVerifyFp16WeightStorage(AexisGraphModel.Layer layer, out string reason)
        {
            reason = null;
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            if (string.Equals(operatorName, "Convolution", StringComparison.Ordinal))
            {
                if (!_conv.TryGetValue(layer.name, out var conv)
                    || conv.packedWeight4Fp16 == null
                    || conv.isDepthWise
                    || conv.group != 1
                    || conv.kernelW != conv.kernelH)
                {
                    reason = "FP16 Conv requires the verified Pack4General half4-weight profile (group=1 and square kernel); packed tail lanes are cleared before storage, and FP32 weight or Buffer fallbacks are prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "Gemm", StringComparison.Ordinal))
            {
                if (!_gemm.TryGetValue(layer.name, out var gemm)
                    || (gemm.constantB && gemm.bDataFp16 == null))
                {
                    reason = "FP16 Gemm requires a loaded packed-half constant-B upload; FP32 constant weights are not a valid FP16 plan substitute.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "ConvolutionDepthWise", StringComparison.Ordinal))
            {
                if (!_conv.TryGetValue(layer.name, out var depthWise)
                    || depthWise.packedDepthWiseWeight4Fp16 == null
                    || !depthWise.isDepthWise
                    || depthWise.group != depthWise.inC
                    || depthWise.outC % depthWise.group != 0
                    || (depthWise.outC & 3) != 0)
                {
                    reason = "FP16 ConvolutionDepthWise requires the verified Pack4 half4-weight profile (group=inC, integral depthwise multiplier, and output channels divisible by four); FP32 weight and Buffer fallbacks are prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "Deconvolution", StringComparison.Ordinal))
            {
                if (!_deconv.TryGetValue(layer.name, out var deconv)
                    || deconv.packedWeight4Fp16 == null
                    || deconv.group != 1
                    || deconv.kernelW != deconv.kernelH)
                {
                    reason = "FP16 Deconvolution requires the verified Pack4General half4-weight profile (group=1 and square kernel); FP32 weight or Buffer fallbacks are prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal))
            {
                if (!_innerProduct.TryGetValue(layer.name, out var innerProduct)
                    || innerProduct.wFp16 == null)
                {
                    reason = "FP16 InnerProduct requires a loaded packed-half immutable weight upload for its verified texture-native Gemm lowering; FP32 weight and Buffer fallbacks are prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "Convolution3D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "DeconvolutionDepthWise", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution3D", StringComparison.Ordinal)
                )
            {
                reason = "This operator has no verified FP16 weight-storage CommandBuffer Pack4 implementation; strict FP16 planning refuses an FP32-weight or Buffer fallback.";
                return false;
            }

            return true;
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferEmbed(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var indices, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (indices.dims < 1 || indices.dims > 2
                || !HasStrictAnyRankLinearMatStorage(inputs[0], indices)
                || !string.Equals(inputs[0].dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(inputs[0].logicalDtype, "Int32", StringComparison.Ordinal))
            {
                return RejectStrictCommandBufferPack4Node(
                    "Embed requires an exact FP32 RFloat LinearMat Int32 index texture; Buffer activations and lossy FP16 token ids are prohibited.");
            }
            if (!SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                return RejectStrictCommandBufferPack4Node("Embed requires RFloat RenderTexture support for exact texture-native token ids.");
            if (!_embed.TryGetValue(layer.name, out var embed) || embed == null || embed.WeightBinding == null
                || embed.numOutput <= 0 || embed.inputDim <= 0)
            {
                return RejectStrictCommandBufferPack4Node("Embed immutable GPU weights were not loaded for this layer.");
            }

            var words = GetStrictPlanElementCount(indices);
            if (words <= 0 || words > int.MaxValue)
                return RejectStrictCommandBufferPack4Node("Embed index element count is outside the supported LinearMat profile.");
            var output = new BufferShape(2, embed.numOutput, (int)words, 1, 1);
            if (!TryResolveStrictLinearStorage(output, out var storage, out var storageReason))
                return RejectStrictCommandBufferPack4Node("Embed output LinearMat profile rejected: " + storageReason);

            // Embed's shader reads the uploaded texture indices and writes a real
            // RFloat LinearMat activation.  This is a verified GPU texture path,
            // not a ComputeBuffer materialization exception.
            return AcceptStrictDataIndexNode(
                layer,
                request,
                "command-buffer-linearmat:embed-index-texture",
                new[] { output },
                new[] { storage },
                new[] { "Float32" });
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolution3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolutionDepthWise3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictCdhwPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_conv.TryGetValue(layer.name, out var conv) || conv == null)
                return RejectStrictCommandBufferPack4Node("Packed 3D depthwise convolution weights were not loaded for this layer.");
            if (input.c != conv.inC)
                return RejectStrictCommandBufferPack4Node("Input channels do not match the loaded 3D depthwise convolution profile.");
            if (!TryValidateCommandBuffer3dDepthWiseConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("CommandBuffer 3D depthwise convolution profile rejected: " + profileReason);

            var output = new BufferShape(
                4,
                ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight),
                ComputeConvOut(input.h, conv.kernelH, conv.dilationH, conv.strideH, conv.padTop, conv.padBottom),
                ComputeConvOut(input.d, conv.kernelD, conv.dilationD, conv.strideD, conv.padFront, conv.padBehind),
                conv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                return RejectStrictCommandBufferPack4Node("The 3D depthwise convolution profile resolves a non-positive CDHW output extent.");
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:depthwise-convolution3d-cdhw");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolution3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolutionDepthWise3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictCdhwPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_deconv.TryGetValue(layer.name, out var deconv) || deconv == null || deconv.packedWeight4 == null || deconv.packedBias4 == null)
                return RejectStrictCommandBufferPack4Node("Packed 3D depthwise deconvolution weights were not loaded for this layer.");
            if (input.c != deconv.inC || deconv.group != deconv.inC || deconv.inC != deconv.outC)
                return RejectStrictCommandBufferPack4Node("Only one-to-one group=inC=outC CDHW depthwise deconvolution is implemented.");
            if (deconv.kernelW <= 0 || deconv.kernelH <= 0 || deconv.kernelD <= 0
                || deconv.strideW <= 0 || deconv.strideH <= 0 || deconv.strideD <= 0
                || deconv.dilationW <= 0 || deconv.dilationH <= 0 || deconv.dilationD <= 0
                || deconv.padLeft < 0 || deconv.padRight < 0 || deconv.padTop < 0 || deconv.padBottom < 0 || deconv.padFront < 0 || deconv.padBehind < 0
                || deconv.outputPadRight < 0 || deconv.outputPadBottom < 0 || deconv.outputPadBehind < 0
                || deconv.weightSize != deconv.outC * deconv.kernelW * deconv.kernelH * deconv.kernelD)
                return RejectStrictCommandBufferPack4Node("Depthwise 3D deconvolution parameters do not satisfy the immutable CDHW profile.");
            if (!IsCommandBufferConvActivationSupported(deconv.activationType))
                return RejectStrictCommandBufferPack4Node("Depthwise 3D deconvolution activation supports only none, ReLU, LeakyReLU, or Sigmoid.");

            var output = new BufferShape(4,
                ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight),
                ComputeDeconvOut(input.h, deconv.kernelH, deconv.dilationH, deconv.strideH, deconv.padTop, deconv.padBottom, deconv.outputPadBottom),
                ComputeDeconvOut(input.d, deconv.kernelD, deconv.dilationD, deconv.strideD, deconv.padFront, deconv.padBehind, deconv.outputPadBehind),
                deconv.outC);
            if (output.w <= 0 || output.h <= 0 || output.d <= 0)
                return RejectStrictCommandBufferPack4Node("The depthwise 3D deconvolution profile resolves a non-positive CDHW output extent.");
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:depthwise-deconvolution3d-cdhw");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPooling3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private static bool TryValidateCommandBuffer3dDepthWiseConvProfile(ConvPack conv, out string reason)
        {
            reason = null;
            if (conv == null || conv.packedDepthWiseWeight4 == null || conv.packedBias4 == null)
                reason = "immutable Pack4 depthwise weights/bias are unavailable";
            else if (conv.group != conv.inC || conv.inC != conv.outC)
                reason = "only one-to-one group=inC=outC depthwise CDHW convolution is implemented";
            else if (conv.inC <= 0 || conv.kernelW <= 0 || conv.kernelH <= 0 || conv.kernelD <= 0
                || conv.strideW <= 0 || conv.strideH <= 0 || conv.strideD <= 0
                || conv.dilationW <= 0 || conv.dilationH <= 0 || conv.dilationD <= 0)
                reason = "channels, kernel, stride, and dilation must be positive on W/H/D";
            else if (conv.padLeft < 0 || conv.padRight < 0 || conv.padTop < 0 || conv.padBottom < 0 || conv.padFront < 0 || conv.padBehind < 0)
                reason = "negative or auto padding is unsupported";
            else if (!IsCommandBufferConvActivationSupported(conv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (conv.weightSize != conv.outC * conv.kernelW * conv.kernelH * conv.kernelD)
                reason = "weight_data_size does not match the one-to-one depthwise CDHW profile";
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolution(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer convolution requires a 2D Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("CommandBuffer convolution requires exact rank-3 Pack4 descriptor storage.");
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDepthWiseConvolution(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer depthwise convolution requires a 2D Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("CommandBuffer depthwise convolution requires exact rank-3 Pack4 descriptor storage.");
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolution1D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.d != 1 || input.c != 1)
                return RejectStrictCommandBufferPack4Node("Convolution1D requires logical dims=2 [width,channels] texture input.");
            var padValue = layer.GetFloat(18, 0f);
            if (layer.GetInt(19, 0) != 0 || !IsStrictFinite(padValue) || Math.Abs(padValue) > 0f)
                return RejectStrictCommandBufferPack4Node("Convolution1D requires immutable weights and zero pad_value.");
            if (!HasStrictLinearMatStorage(inputs[0], input) && !HasStrictScalar2DPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Convolution1D requires exact LinearMat or scalar rank-2 Pack4 storage.");
            if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not ConvPack conv)
                return RejectStrictCommandBufferPack4Node("Packed Convolution1D weights were not loaded for this layer.");
            if (conv.packedWeight4 == null || conv.packedBias4 == null || conv.group != 1)
                return RejectStrictCommandBufferPack4Node("Convolution1D requires loaded immutable O4I4K weights/bias and group=1.");
            if (input.h != conv.inC)
                return RejectStrictCommandBufferPack4Node("Convolution1D input channels do not match the loaded immutable weight profile.");
            if (!TryValidateCommandBuffer2dConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("Convolution1D profile rejected: " + profileReason);
            var outputWidth = ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            if (outputWidth <= 0)
                return RejectStrictCommandBufferPack4Node("Convolution1D produces a non-positive output width.");
            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(2, outputWidth, conv.outC, 1, 1),
                request,
                "command-buffer-pack4:convolution1d");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConvolutionDepthWise1D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.d != 1 || input.c != 1)
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D requires native logical dims=2 [width,channels].");
            var padValue = layer.GetFloat(18, 0f);
            if (layer.GetInt(19, 0) != 0 || !IsStrictFinite(padValue) || Math.Abs(padValue) > 0f)
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D requires immutable weights and zero pad_value.");
            if (!HasStrictLinearMatStorage(inputs[0], input) && !HasStrictScalar2DPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D requires exact LinearMat or scalar rank-2 texture storage.");
            if (!_conv.TryGetValue(layer.name, out var conv) || conv == null || !conv.isDepthWise
                || conv.packedDepthWiseWeight4 == null || conv.packedBias4 == null)
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D immutable depthwise Pack4 weights are unavailable.");
            if (input.h != conv.inC)
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D profile rejected: input channel mismatch.");
            if (!TryValidateCommandBuffer2dConvProfile(conv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D profile rejected: " + profileReason);
            var outputWidth = ComputeConvOut(input.w, conv.kernelW, conv.dilationW, conv.strideW, conv.padLeft, conv.padRight);
            if (outputWidth <= 0)
                return RejectStrictCommandBufferPack4Node("ConvolutionDepthWise1D produces a non-positive output width.");
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(2, outputWidth, conv.outC, 1, 1), request, "command-buffer-pack4:convolution-depthwise1d");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolution1D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.d != 1 || input.c != 1)
                return RejectStrictCommandBufferPack4Node("Deconvolution1D requires native logical dims=2 [width,channels].");
            if (layer.GetInt(28, 0) != 0 || layer.GetInt(20, 0) != 0)
                return RejectStrictCommandBufferPack4Node("Deconvolution1D requires immutable weights and output_w=0; output padding remains supported.");
            if (!HasStrictLinearMatStorage(inputs[0], input) && !HasStrictScalar2DPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Deconvolution1D requires exact LinearMat or scalar rank-2 texture storage.");
            if (!_deconv.TryGetValue(layer.name, out var deconv) || deconv == null || deconv.group != 1
                || deconv.packedWeight4 == null || deconv.packedBias4 == null)
                return RejectStrictCommandBufferPack4Node("Deconvolution1D immutable Pack4 weights are unavailable.");
            if (input.h != deconv.inC)
                return RejectStrictCommandBufferPack4Node("Deconvolution1D profile rejected: input channel mismatch.");
            if (!TryValidateCommandBuffer1dDeconvProfile(deconv, out var profileReason))
                return RejectStrictCommandBufferPack4Node("Deconvolution1D profile rejected: " + profileReason);
            var outputWidth = ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            if (outputWidth <= 0)
                return RejectStrictCommandBufferPack4Node("Deconvolution1D produces a non-positive output width.");
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(2, outputWidth, deconv.outC, 1, 1), request, "command-buffer-pack4:deconvolution1d");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolutionDepthWise1D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.d != 1 || input.c != 1 || input.w <= 0)
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D requires native logical dims=2 [width,channels].");
            if (layer.GetInt(28, 0) != 0 || layer.GetInt(20, 0) != 0)
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D requires immutable weights and output_w=0.");
            if (!HasStrictLinearMatStorage(inputs[0], input) && !HasStrictScalar2DPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D requires exact LinearMat or scalar rank-2 texture storage.");
            if (!_deconv.TryGetValue(layer.name, out var deconv) || deconv == null
                || deconv.rawWeight == null || deconv.rawBias == null
                || deconv.inC <= 0 || deconv.inC != deconv.outC || deconv.group != deconv.inC
                || input.h != deconv.inC || deconv.kernelW <= 0 || deconv.kernelH != 1
                || deconv.strideW <= 0 || deconv.dilationW <= 0 || deconv.padLeft < 0 || deconv.padRight < 0
                || deconv.weightSize != deconv.outC * deconv.kernelW)
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D parameters do not satisfy the immutable one-to-one Pack4 profile.");
            if (!IsCommandBufferConvActivationSupported(deconv.activationType))
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D activation supports only none, ReLU, LeakyReLU, or Sigmoid.");
            var outputWidth = ComputeDeconvOut(input.w, deconv.kernelW, deconv.dilationW, deconv.strideW, deconv.padLeft, deconv.padRight, deconv.outputPadRight);
            if (outputWidth <= 0)
                return RejectStrictCommandBufferPack4Node("DeconvolutionDepthWise1D produces a non-positive output width.");
            var verification = AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(2, outputWidth, deconv.outC, 1, 1),
                request,
                "command-buffer-pack4:depthwise-deconvolution1d");
            if (!verification.accepted)
                return verification;

            // The rank-2 LinearMat boundary is converted into an exact Pack4 array,
            // processed by the texture kernel, then converted back. Both conversion
            // intermediates are bounded by the loaded immutable profile and must be
            // in the compiled RT arena rather than allocated opportunistically.
            var scratchDtype = ResolvePhysicalTextureDtype(request.targetDtype);
            verification.scratch = new[]
            {
                CreateStrictPack4ScratchDescriptor(layer, "packed-input", new BufferShape(3, input.w, 1, 1, deconv.inC), request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "packed-output", new BufferShape(3, outputWidth, 1, 1, deconv.outC), request, scratchDtype)
            };
            return verification;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferStatisticsPooling(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.w <= 0 || input.h <= 0 || input.d != 1 || input.c != 1)
                return RejectStrictCommandBufferPack4Node("StatisticsPooling requires a static rank-2 [frames,channels] input.");
            if (!HasStrictLinearMatStorage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("StatisticsPooling requires exact FP32 LinearMat texture storage.");

            var includeStd = layer.GetInt(0, 1);
            var epsilon = layer.GetFloat(1, 0f);
            if ((includeStd != 0 && includeStd != 1) || !IsStrictFinite(epsilon) || epsilon < 0f)
                return RejectStrictCommandBufferPack4Node("StatisticsPooling requires include_std=0|1 and finite non-negative epsilon.");

            var outputRows = includeStd != 0 ? checked(input.h * 2) : input.h;
            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(2, 1, outputRows, 1, 1),
                request,
                "command-buffer-pack4:statistics-pooling-linearmat");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSpectrogram(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            bool inverse)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 2 || input.w <= 0 || input.h <= 0 || input.d != 1 || input.c != 1)
                return RejectStrictCommandBufferPack4Node((inverse ? "InverseSpectrogram" : "Spectrogram") + " requires a static rank-2 LinearMat input.");
            if (!HasStrictLinearMatStorage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node((inverse ? "InverseSpectrogram" : "Spectrogram") + " requires exact FP32 LinearMat texture storage.");

            var nfft = layer.GetInt(0, 0);
            var hop = layer.GetInt(1, 0);
            var channels = layer.GetInt(2, 0);
            if (nfft < 2 || nfft > 256 || (nfft & 1) != 0 || hop <= 0 || hop > nfft || channels <= 0)
                return RejectStrictCommandBufferPack4Node("Spectrogram requires an even n_fft in [2,256], hop in [1,n_fft], and explicit positive channels.");

            BufferShape output;
            if (!inverse)
            {
                if (input.h != channels || input.w < nfft || (input.w - nfft) % hop != 0)
                    return RejectStrictCommandBufferPack4Node("Spectrogram input must be exact [samples,channels] with complete static frames.");
                var frames = 1 + (input.w - nfft) / hop;
                if (frames > int.MaxValue / channels)
                    return RejectStrictCommandBufferPack4Node("Spectrogram output rows exceed the static 32-bit texture descriptor range.");
                output = new BufferShape(2, 2 * (nfft / 2 + 1), frames * channels, 1, 1);
            }
            else
            {
                var complexBins = 2 * (nfft / 2 + 1);
                if (input.w != complexBins || input.h % channels != 0)
                    return RejectStrictCommandBufferPack4Node("InverseSpectrogram input must be exact one-sided complex [2*(n_fft/2+1),frames*channels] storage.");
                var frames = input.h / channels;
                if (frames <= 0 || frames > 1 + (int.MaxValue - nfft) / hop)
                    return RejectStrictCommandBufferPack4Node("InverseSpectrogram frame count exceeds the static output descriptor range.");
                output = new BufferShape(2, nfft + hop * (frames - 1), channels, 1, 1);
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                request,
                inverse ? "command-buffer-pack4:inverse-spectrogram-linearmat" : "command-buffer-pack4:spectrogram-linearmat");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferRecurrent(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            AexisRecurrentKind kind)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            try
            {
                AexisRecurrentLayer.ValidateLayerContract(layer, kind);
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node(exception.Message);
            }
            if (input.dims != 3 || input.w <= 0 || input.w > 256 || input.h != 1 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("The bounded recurrent profile requires static [sequence<=256,1,input_channels] Pack4 input storage.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("The bounded recurrent profile requires exact Pack4 Texture2DArray descriptor storage.");
            if (!string.Equals(inputs[0].dtype, "FP32", StringComparison.Ordinal)
                || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal))
            {
                return RejectStrictCommandBufferPack4Node("The bounded recurrent profile requires FP32 activation and logical dtype contracts.");
            }
            if (!_extraPacks.TryGetValue(layer.name, out var loaded) || loaded is not RecurrentPack pack
                || pack.kind != (int)kind || pack.inputSize != input.c || pack.hiddenSize <= 0 || pack.hiddenSize > 256
                || pack.inputWeights == null || pack.recurrentWeights == null || pack.bias == null)
            {
                return RejectStrictCommandBufferPack4Node("The bounded recurrent profile requires matching loaded immutable gate weights and FP32 output state.");
            }
            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, input.w, 1, 1, pack.hiddenSize),
                request,
                "command-buffer-pack4:bounded-" + kind.ToString().ToLowerInvariant() + "-fp32");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeconvolution(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            bool depthWiseLayer)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer deconvolution requires a 2D Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("CommandBuffer deconvolution requires exact rank-3 Pack4 descriptor storage.");
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
            if (conv == null || (conv.rawWeight == null && conv.rawWeightInt8Packed == null && conv.rawWeightInt4Packed == null) || conv.rawBias == null)
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

        private static bool TryValidateCommandBuffer1dDeconvProfile(DeconvPack deconv, out string reason)
        {
            reason = null;
            if (deconv == null || deconv.packedWeight4 == null || deconv.packedBias4 == null)
                reason = "immutable packed O4I4K weights/bias are unavailable";
            else if (deconv.inC <= 0 || deconv.outC <= 0 || deconv.group != 1)
                reason = "positive input/output channels and group=1 are required";
            else if (deconv.kernelW <= 0 || deconv.kernelH != 1 || deconv.strideW <= 0 || deconv.strideH != 1
                || deconv.dilationW <= 0 || deconv.dilationH != 1)
                reason = "1D kernel/stride/dilation geometry is invalid";
            else if (deconv.padLeft < 0 || deconv.padRight < 0 || deconv.padTop != 0 || deconv.padBottom != 0
                || deconv.outputPadRight < 0 || deconv.outputPadBottom != 0)
                reason = "1D padding or output padding is invalid";
            else if (!IsCommandBufferConvActivationSupported(deconv.activationType))
                reason = "activation supports only none, ReLU, LeakyReLU, or Sigmoid";
            else if (deconv.weightSize != deconv.outC * deconv.inC * deconv.kernelW)
                reason = "weight_data_size does not match 1D OIK";
            return reason == null;
        }

        private static bool IsCommandBufferConvActivationSupported(int activationType)
        {
            return activationType == 0 || activationType == 1 || activationType == 2 || activationType == 4;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPooling(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Pooling profile requires a 2D Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Pooling requires exact rank-3 Pack4 descriptor storage.");
            if (layer.GetInt(7, 0) != 0)
                return RejectStrictCommandBufferPack4Node("Adaptive pooling does not have a verified CommandBuffer Pack4 path.");
            var poolingType = layer.GetInt(0, 0);
            if (poolingType != 0 && poolingType != 1)
                return RejectStrictCommandBufferPack4Node("Pooling type must be max (0) or average (1).");
            var includePad = layer.GetInt(6, 0);
            if (includePad != 0 && includePad != 1)
                return RejectStrictCommandBufferPack4Node("Pooling avgpool_count_include_pad must be 0 or 1.");
            var globalPooling = layer.GetInt(4, 0) != 0;
            if (globalPooling)
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    new BufferShape(3, 1, 1, 1, input.c),
                    request,
                    "command-buffer-pack4:pooling-global");
            }

            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            var padMode = layer.GetInt(5, 0);
            if (!AexisPoolingLayer.TryResolvePack4Geometry(
                input.w, input.h, kernelW, kernelH, strideW, strideH,
                padLeft, padRight, padTop, padBottom, padMode,
                out _, out _, out _, out _, out var outW, out var outH, out var geometryReason))
            {
                return RejectStrictCommandBufferPack4Node("Pooling geometry is invalid: " + geometryReason + ".");
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, outW, outH, 1, input.c),
                request,
                "command-buffer-pack4:pooling");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMaxPoolingInd(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("MaxPoolingInd CommandBuffer Pack4 requires a rank-3 2D activation.");
            if (layer?.topNames == null || layer.topNames.Length != 2)
                return RejectStrictCommandBufferPack4Node("MaxPoolingInd CommandBuffer Pack4 requires value and index output blobs.");

            var kernelW = layer.GetInt(1, 0);
            var kernelH = layer.GetInt(11, kernelW);
            var strideW = layer.GetInt(2, 1);
            var strideH = layer.GetInt(12, strideW);
            var padLeft = layer.GetInt(3, 0);
            var padRight = layer.GetInt(14, padLeft);
            var padTop = layer.GetInt(13, padLeft);
            var padBottom = layer.GetInt(15, padTop);
            if (kernelW <= 0 || kernelH <= 0 || strideW <= 0 || strideH <= 0
                || padLeft < 0 || padRight < 0 || padTop < 0 || padBottom < 0)
            {
                return RejectStrictCommandBufferPack4Node("MaxPoolingInd requires positive kernel/stride and explicit non-negative padding.");
            }

            var outW = (input.w + padLeft + padRight - kernelW) / strideW + 1;
            var outH = (input.h + padTop + padBottom - kernelH) / strideH + 1;
            if (outW <= 0 || outH <= 0)
                return RejectStrictCommandBufferPack4Node("MaxPoolingInd produces a non-positive output shape.");

            var outputShape = new[] { 3, outW, outH, 1, input.c };
            var sourceShape = new[] { input.dims, input.w, input.h, input.d, input.c };
            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = "command-buffer-pack4:max-pooling-indices",
                outputs = new[]
                {
                    new AexisTexturePlanTensorDescriptor
                    {
                        blob = layer.topNames[0],
                        logicalShape = outputShape,
                        storageShape = (int[])outputShape.Clone(),
                        layout = request.targetLayout,
                        dtype = ResolvePhysicalTextureDtype(request.targetDtype),
                        aliasGroup = "computed:" + layer.name + ":value",
                        textureBacked = true
                    },
                    new AexisTexturePlanTensorDescriptor
                    {
                        blob = layer.topNames[1],
                        logicalShape = (int[])outputShape.Clone(),
                        storageShape = (int[])outputShape.Clone(),
                        sourceLogicalShape = sourceShape,
                        layout = request.targetLayout,
                        dtype = "FP32",
                        aliasGroup = "computed:" + layer.name + ":indices",
                        textureBacked = true
                    }
                }
            };
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMaxUnPooling(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (inputs == null || inputs.Count != 2)
                return RejectStrictCommandBufferPack4Node("MaxUnPooling requires value and index texture descriptors.");
            if (!TryGetStrictPlanShape(inputs[0], out var values, out var reason)
                || !TryGetStrictPlanShape(inputs[1], out var indices, out reason))
                return RejectStrictCommandBufferPack4Node(reason);
            // The index tensor is made by MaxPoolingInd before a decoder can
            // reduce its feature channels.  The Pack4 shader consumes the first
            // value pack-count of its index texture, so a wider index tensor is a
            // valid texture-native profile; it is neither a reinterpretation nor
            // a buffer fallback.
            if (values.dims != 3 || values.d != 1
                || indices.dims != 3 || indices.d != 1
                || values.w != indices.w || values.h != indices.h
                || indices.c < values.c)
            {
                return RejectStrictCommandBufferPack4Node(
                    "MaxUnPooling requires rank-3 Pack4 values and indices with matching pooled spatial dimensions and index channels covering every value channel.");
            }
            if (!TryGetSourceShape(inputs[1], out var source))
                return RejectStrictCommandBufferPack4Node("MaxUnPooling index descriptor lacks its originating pre-pool activation shape.");
            if (source.dims != 3 || source.d != 1 || source.c < values.c)
                return RejectStrictCommandBufferPack4Node("MaxUnPooling index source contract does not cover the pooled activation channels.");

            // Runtime dispatch allocates the output by the value tensor's packs;
            // retain its logical channel count even when the saved index source
            // has more channels.
            return AcceptStrictCommandBufferPack4Node(
                layer,
                new BufferShape(3, source.w, source.h, 1, values.c),
                request,
                "command-buffer-pack4:max-unpooling-indices");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferEltwise(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferConcat(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length < 2)
                return RejectStrictCommandBufferPack4Node("Concat CommandBuffer Pack4 requires two or more inputs.");

            if (shapes[0].dims == 1 || shapes[0].dims == 2)
                return VerifyStrictCommandBufferLowDimConcat(layer, inputs, shapes, request);
            if (shapes[0].dims != 3 && shapes[0].dims != 4)
                return RejectStrictCommandBufferPack4Node("Concat CommandBuffer Pack4 requires rank-one through rank-four inputs.");

            var axis = layer.GetInt(0, 0);
            if (axis < 0)
                axis += shapes[0].dims;
            if (axis < 0 || axis >= shapes[0].dims)
                return RejectStrictCommandBufferPack4Node("Concat axis is outside the input rank.");
            var tensorAxis = MapNcnnAxisToTensorAxis(shapes[0].dims, axis);
            var channelAxis = shapes[0].dims == 4 ? 3 : 2;
            if (shapes[0].dims == 3 && tensorAxis == 0)
            {
                var outputWidth = 0;
                for (var index = 0; index < shapes.Length; index++)
                {
                    var shape = shapes[index];
                    if (shape.dims != 3
                        || shape.h != shapes[0].h
                        || shape.d != shapes[0].d
                        || shape.c != shapes[0].c
                        || !HasStrictExactPack4Storage(inputs[index], shape))
                    {
                        return RejectStrictCommandBufferPack4Node(
                            "Rank-three width Concat requires exact Pack4 inputs with matching height, depth, and channel descriptors.");
                    }
                    outputWidth += shape.w;
                }

                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    new BufferShape(3, outputWidth, shapes[0].h, 1, shapes[0].c),
                    request,
                    "command-buffer-pack4:concat-width");
            }
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferLowDimConcat(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            IReadOnlyList<BufferShape> shapes,
            AexisTextureExecutionPlanRequest request)
        {
            var first = shapes[0];
            var axis = layer.GetInt(0, 0);
            if (axis < 0)
                axis += first.dims;
            if (axis < 0 || axis >= first.dims)
                return RejectStrictCommandBufferPack4Node("Concat axis is outside the input rank.");

            var tensorAxis = MapNcnnAxisToTensorAxis(first.dims, axis);
            if ((first.dims == 1 && tensorAxis != 0)
                || (first.dims == 2 && tensorAxis != 0 && tensorAxis != 1))
            {
                return RejectStrictCommandBufferPack4Node("Low-dimensional Concat supports only its width or height axis.");
            }

            var outW = first.w;
            var outH = first.h;
            for (var index = 0; index < shapes.Count; index++)
            {
                var shape = shapes[index];
                if (shape.dims != first.dims
                    || !HasStrictScalarLikePlanStorage(inputs[index], shape)
                    || (tensorAxis != 0 && shape.w != first.w)
                    || (tensorAxis != 1 && shape.h != first.h)
                    || shape.d != first.d
                    || shape.c != first.c)
                {
                    return RejectStrictCommandBufferPack4Node("Low-dimensional Concat requires matching descriptor-backed scalar storage on every non-concatenated axis.");
                }

                if (index > 0)
                {
                    if (tensorAxis == 0)
                        outW += shape.w;
                    else
                        outH += shape.h;
                }
            }

            var output = new BufferShape(first.dims, outW, outH, 1, 1);
            var storage = new BufferShape(3, outW, outH, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                storage,
                request,
                "command-buffer-pack4:concat-low-dim");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferBinaryOp(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            var operation = layer.GetInt(0, 0);
            if (operation < 0 || operation > 27)
                return RejectStrictCommandBufferPack4Node("The BinaryOp code is outside the verified CommandBuffer Pack4 kernel range.");

            if (layer.GetInt(1, 0) != 0)
            {
                if (!TryGetSingleStrictPlanShape(inputs, out var scalarInput, out var scalarReason))
                    return RejectStrictCommandBufferPack4Node(scalarReason);
                if (scalarInput.dims == 2 && HasStrictLinearMatStorage(inputs[0], scalarInput))
                {
                    return AcceptStrictCommandBufferPack4Node(
                        layer,
                        scalarInput,
                        CopyStrictStorage(inputs[0]),
                        request,
                        "command-buffer-linearmat:binary-scalar");
                }
                // Rank-two projection activations can be physically stored as a
                // Pack4-linear Texture2DArray.  The CommandBuffer layer dispatches
                // AexisBinaryOpPack4 with a single physical slice for this exact
                // layout, retaining the logical [w,h] matrix and its storage
                // descriptor; no texture-to-buffer conversion is involved.
                if (scalarInput.dims == 2 && HasStrictPack4LinearMatStorage(inputs[0], scalarInput))
                {
                    return AcceptStrictCommandBufferPack4Node(
                        layer,
                        scalarInput,
                        CopyStrictStorage(inputs[0]),
                        request,
                        "command-buffer-pack4:binary-pack4-linear-scalar");
                }
                if (HasStrictScalarLikePlanStorage(inputs[0], scalarInput))
                {
                    return AcceptStrictCommandBufferPack4Node(
                        layer,
                        scalarInput,
                        CopyStrictStorage(inputs[0]),
                        request,
                        "command-buffer-pack4:binary-scalar");
                }
                if (scalarInput.dims < 3 || scalarInput.dims > 4)
                    return RejectStrictCommandBufferPack4Node("The verified scalar BinaryOp profile requires scalar rank-1/rank-2 texture storage or a rank-3/rank-4 Pack4 texture.");
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

            if (TryResolveStrictCommandBufferScalarSingleBinary(
                    inputs[0],
                    shapes[0],
                    inputs[1],
                    shapes[1],
                    out var scalarSingleOutput,
                    out var scalarSingleStorage))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    scalarSingleOutput,
                    scalarSingleStorage,
                    request,
                    "command-buffer-pack4:binary-scalar-single-broadcast");
            }

            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && HasStrictLinearMatStorage(inputs[0], shapes[0])
                && HasStrictLinearMatStorage(inputs[1], shapes[1]))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(inputs[0]),
                    request,
                    "command-buffer-linearmat:binary-exact");
            }

            // Legacy MHA exposes its FP32 output as a scalar one-slice
            // Texture2DArray. Residuals with a LinearMat use the dedicated
            // texture kernel and keep the LinearMat physical descriptor.
            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && ((HasStrictLinearMatStorage(inputs[0], shapes[0])
                        && HasStrictScalar2DPack4Storage(inputs[1], shapes[1]))
                    || (HasStrictScalar2DPack4Storage(inputs[0], shapes[0])
                        && HasStrictLinearMatStorage(inputs[1], shapes[1]))))
            {
                var linearInput = HasStrictLinearMatStorage(inputs[0], shapes[0]) ? inputs[0] : inputs[1];
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(linearInput),
                    request,
                    "command-buffer-linearmat:binary-scalar-array");
            }

            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && HasStrictPack4LinearMatStorage(inputs[0], shapes[0])
                && HasStrictPack4LinearMatStorage(inputs[1], shapes[1]))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(inputs[0]),
                    request,
                    "command-buffer-pack4:binary-pack4-linear-exact");
            }

            // The scalar rank-two path stores one logical scalar per Pack4 texel
            // and dispatches AexisBinaryOpPack4 directly. CodeFormer uses this
            // exact profile for its normalization multiply chain.
            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && HasStrictScalar2DPack4Storage(inputs[0], shapes[0])
                && HasStrictScalar2DPack4Storage(inputs[1], shapes[1]))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(inputs[0]),
                    request,
                    "command-buffer-pack4:binary-scalar2d-exact");
            }

            // The mixed matrix kernel reads a Pack4-Linear texture and a scalar
            // LinearMat texture with the same logical matrix shape, then publishes
            // Pack4-Linear output. This is used by SD residual adds after Gemm.
            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && ((HasStrictPack4LinearMatStorage(inputs[0], shapes[0])
                        && HasStrictLinearMatStorage(inputs[1], shapes[1]))
                    || (HasStrictLinearMatStorage(inputs[0], shapes[0])
                        && HasStrictPack4LinearMatStorage(inputs[1], shapes[1]))))
            {
                var pack4Input = HasStrictPack4LinearMatStorage(inputs[0], shapes[0]) ? inputs[0] : inputs[1];
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(pack4Input),
                    request,
                    "command-buffer-pack4:binary-pack4-linear-mixed");
            }

            // Legacy MHA publishes its FP32 projection result as a scalar
            // Texture2DArray rather than an RFloat Texture2D. The dedicated
            // MixedArray kernel consumes that scalar-array layout directly and
            // returns the Pack4-linear residual without a buffer conversion.
            if (shapes[0].dims == 2
                && StrictPlanShapesEqual(shapes[0], shapes[1])
                && ((HasStrictPack4LinearMatStorage(inputs[0], shapes[0])
                        && HasStrictScalar2DPack4Storage(inputs[1], shapes[1]))
                    || (HasStrictScalar2DPack4Storage(inputs[0], shapes[0])
                        && HasStrictPack4LinearMatStorage(inputs[1], shapes[1]))))
            {
                var pack4Input = HasStrictPack4LinearMatStorage(inputs[0], shapes[0]) ? inputs[0] : inputs[1];
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    shapes[0],
                    CopyStrictStorage(pack4Input),
                    request,
                    "command-buffer-pack4:binary-pack4-linear-scalar-array");
            }

            if ((shapes[0].dims == 3 || shapes[0].dims == 4)
                && StrictPlanShapesEqual(shapes[0], shapes[1]))
            {
                return AcceptStrictCommandBufferPack4Node(layer, shapes[0], request, "command-buffer-pack4:binary-exact");
            }

            if (TryResolveStrictCommandBufferChannelVector(inputs[0], shapes[0], inputs[1], shapes[1], out var channelVectorOutput))
                return AcceptStrictCommandBufferPack4Node(layer, channelVectorOutput, request, "command-buffer-pack4:binary-channel-vector");

            if (TryResolveStrictCommandBufferSpatialBroadcast(
                    inputs[0],
                    shapes[0],
                    inputs[1],
                    shapes[1],
                    out var spatialBroadcastOutput))
                return AcceptStrictCommandBufferPack4Node(layer, spatialBroadcastOutput, request, "command-buffer-pack4:binary-spatial-broadcast");

            return RejectStrictCommandBufferPack4Node("BinaryOp does not match an exact, channel-vector, or verified spatial CommandBuffer Pack4 descriptor profile.");
        }

        private static bool TryResolveStrictCommandBufferScalarSingleBinary(
            AexisTexturePlanTensorDescriptor firstDescriptor,
            BufferShape first,
            AexisTexturePlanTensorDescriptor secondDescriptor,
            BufferShape second,
            out BufferShape output,
            out BufferShape storage)
        {
            output = default;
            storage = default;
            if (!HasStrictScalarLikePlanStorage(firstDescriptor, first)
                || !HasStrictScalarLikePlanStorage(secondDescriptor, second)
                || !AexisBinaryOpLayer.TryResolveScalarSingleBroadcastShapes(first, second, out _, out output, out storage))
            {
                return false;
            }
            return true;
        }

        private bool TryVerifyQuantizedWeightOnlyStorage(AexisGraphModel.Layer layer, out string reason)
        {
            reason = null;
            var operatorName = string.IsNullOrWhiteSpace(layer?.typeName) ? layer?.type.ToString() : layer.typeName;
            var usesInt4 = UsesInt4WeightOnlyForLayer(layer);
            if (string.Equals(operatorName, "Convolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "ConvolutionDepthWise", StringComparison.Ordinal))
            {
                if (!_conv.TryGetValue(layer.name, out var conv)
                    || (usesInt4 ? conv.rawWeightInt4Packed == null || conv.rawWeightInt4Scales == null : conv.rawWeightInt8Packed == null || conv.rawWeightInt8Scales == null)
                    || conv.rawBias == null)
                {
                    reason = (usesInt4 ? "INT4" : "INT8") + " Conv requires immutable packed " + (usesInt4 ? "INT4" : "INT8") + " OIHW weights, per-output-channel scales, and FP32 bias; FP32 weight or Buffer fallback is prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "Gemm", StringComparison.Ordinal))
            {
                if (!_gemm.TryGetValue(layer.name, out var gemm)
                    || !gemm.constantB
                    || (usesInt4 ? gemm.bDataInt4Packed == null || gemm.bDataInt4Scales == null : gemm.bDataInt8Packed == null || gemm.bDataInt8Scales == null))
                {
                    reason = (usesInt4 ? "INT4" : "INT8") + " Gemm requires immutable packed " + (usesInt4 ? "INT4" : "INT8") + " constant-B weights and per-output-channel scales; FP32 weight or Buffer fallback is prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "InnerProduct", StringComparison.Ordinal))
            {
                if (!_innerProduct.TryGetValue(layer.name, out var innerProduct)
                    || (usesInt4 ? innerProduct.wInt4Packed == null || innerProduct.wInt4Scales == null : innerProduct.wInt8Packed == null || innerProduct.wInt8Scales == null)
                    || innerProduct.b == null)
                {
                    reason = (usesInt4 ? "INT4" : "INT8") + " InnerProduct requires immutable packed " + (usesInt4 ? "INT4" : "INT8") + " weights, per-output-channel scales, and FP32 bias; FP32 weight or Buffer fallback is prohibited.";
                    return false;
                }
                return true;
            }

            if (string.Equals(operatorName, "Convolution1D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Convolution3D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution", StringComparison.Ordinal)
                || string.Equals(operatorName, "DeconvolutionDepthWise", StringComparison.Ordinal)
                || string.Equals(operatorName, "Deconvolution3D", StringComparison.Ordinal)
                || string.Equals(operatorName, "Embed", StringComparison.Ordinal)
                || string.Equals(operatorName, "MultiHeadAttention", StringComparison.Ordinal)
                || string.Equals(operatorName, "BatchNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "GroupNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "LayerNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "Scale", StringComparison.Ordinal)
                || string.Equals(operatorName, "PReLU", StringComparison.Ordinal)
                || string.Equals(operatorName, "Normalize", StringComparison.Ordinal))
            {
                reason = "Selective quantization has no verified packed-weight CommandBuffer kernel for " + operatorName + "; strict quant planning rejects an FP32 parameter or Buffer fallback.";
                return false;
            }

            return true;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferNonMaxSuppression(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!string.Equals(request?.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase))
                return RejectStrictCommandBufferPack4Node("NonMaxSuppression uses exact FP32 LinearMat textures for boxes, scores, padded indices, and count.");
            if (inputs == null || inputs.Count != 2 || layer?.topNames == null || layer.topNames.Length != 2)
                return RejectStrictCommandBufferPack4Node("NonMaxSuppression requires exactly boxes/scores inputs and padded-index/count texture outputs.");
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!HasStrictAnyRankLinearMatStorage(inputs[0], shapes[0])
                || !HasStrictAnyRankLinearMatStorage(inputs[1], shapes[1])
                || !string.Equals(inputs[0].dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(inputs[1].dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal)
                || !string.Equals(inputs[1].logicalDtype, "Float32", StringComparison.Ordinal))
            {
                return RejectStrictCommandBufferPack4Node("NonMaxSuppression requires exact FP32 LinearMat Float32 boxes and scores; Pack4 activation reinterpretation and buffer materialization are not permitted.");
            }

            var boxes = shapes[0];
            var scores = shapes[1];
            var capacity = ReadStrictLayerInt(layer, "capacity", 0, 0);
            var maxOutputPerClass = ReadStrictLayerInt(layer, "max_output_boxes_per_class", 1, -1);
            var centerPointBox = ReadStrictLayerInt(layer, "center_point_box", 2, 0);
            var iouThreshold = layer.GetFloat(3, 0f);
            if (boxes.dims != 2 || boxes.w != 4 || boxes.h <= 0
                || scores.dims != 2 || scores.w != boxes.h || scores.h <= 0
                || capacity <= 0 || maxOutputPerClass < 0 || maxOutputPerClass > capacity
                || centerPointBox < 0 || centerPointBox > 1 || iouThreshold < 0f || iouThreshold > 1f)
            {
                return RejectStrictCommandBufferPack4Node("NonMaxSuppression requires static batch=1 boxes[num_boxes,4], scores[num_classes,num_boxes], capacity>0, max_output_boxes_per_class<=capacity, center_point_box=0|1, and iou_threshold in [0,1].");
            }

            var output = new BufferShape(2, 3, capacity, 1, 1);
            var count = new BufferShape(1, 1, 1, 1, 1);
            return AcceptStrictLinearTextureNode(
                layer,
                request,
                "command-buffer-linearmat:bounded-nms",
                new[] { output, count },
                new[] { "Int32", "Int32" });
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDataIndex(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            string operatorName)
        {
            if (inputs == null || inputs.Count == 0 || inputs[0]?.logicalShape == null || inputs[0].logicalShape.Length != 5)
                return RejectStrictCommandBufferPack4Node("Data-index operator requires a physical Pack4 descriptor for every input.");
            for (var i = 0; i < inputs.Count; i++)
            {
                if (inputs[i] == null || !inputs[i].textureBacked || inputs[i].layout != AexisTexturePlanLayout.Packed4
                    || inputs[i].logicalShape == null || inputs[i].logicalShape.Length != 5
                    || inputs[i].storageShape == null || inputs[i].storageShape.Length != 5)
                    return RejectStrictCommandBufferPack4Node("Data-index inputs must all be texture-backed Packed4/LinearMat descriptors with logical and storage shapes.");
            }
            if (!string.Equals(request.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase))
                return RejectStrictCommandBufferPack4Node("Data-index LinearMat kernels use RFloat texture storage and require an FP32 strict target.");
            if (!SystemInfo.SupportsRenderTextureFormat(RenderTextureFormat.RFloat))
                return RejectStrictCommandBufferPack4Node("Data-index LinearMat kernels require RFloat render-texture support on the active graphics device.");
            for (var i = 0; i < inputs.Count; i++)
            {
                var inputShape = StrictShape(inputs[i]);
                if (!TryResolveStrictLinearStorage(inputShape, out _, out var storageReason))
                    return RejectStrictCommandBufferPack4Node("Data-index input " + i + " is not representable as LinearMat storage: " + storageReason);
                if (!HasStrictAnyRankLinearMatStorage(inputs[i], inputShape))
                    return RejectStrictCommandBufferPack4Node("Data-index inputs must use exact RFloat LinearMat storage; buffer materialization and implicit storage reinterpretation are prohibited.");
                if (!string.Equals(inputs[i].dtype, "FP32", StringComparison.OrdinalIgnoreCase))
                    return RejectStrictCommandBufferPack4Node("Data-index inputs must use physical FP32 RFloat texture storage.");
            }
            if ((operatorName == "NonZero" || operatorName == "Compress") && (layer.topNames == null || layer.topNames.Length < 2))
                return RejectStrictCommandBufferPack4Node("NonZero/Compress require a second GPU-resident count output; no CPU count readback is permitted.");
            if (operatorName == "NonZero" || operatorName == "Compress")
            {
                var capacity = ReadStrictLayerInt(layer, "capacity", 30, 0);
                if (capacity <= 0)
                    return RejectStrictCommandBufferPack4Node("NonZero/Compress require a positive fixed texture capacity parameter 30.");
                var source = StrictShape(inputs[0]);
                if (source.dims != 1 || capacity < GetStrictPlanElementCount(source))
                    return RejectStrictCommandBufferPack4Node("NonZero/Compress capacity must cover the full static rank-1 input so results cannot be truncated.");
                if (!string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal))
                    return RejectStrictCommandBufferPack4Node("NonZero/Compress P0 data input must use logical Float32 values in RFloat storage.");
                if (operatorName == "Compress")
                {
                    var axis = ReadStrictLayerInt(layer, "axis", 0, 0);
                    if (inputs.Count != 2 || StrictShape(inputs[1]).dims != 1
                        || GetStrictPlanElementCount(StrictShape(inputs[1])) != GetStrictPlanElementCount(source)
                        || (axis != 0 && axis != -1)
                        || (!string.Equals(inputs[1].logicalDtype, "Bool", StringComparison.Ordinal)
                            && !string.Equals(inputs[1].logicalDtype, "Int32", StringComparison.Ordinal)))
                        return RejectStrictCommandBufferPack4Node("Compress condition must be a rank-1 logical Bool/Int32 texture with the same element count as Float32 data, and axis must be 0 or -1.");
                }
                else if (inputs.Count != 1)
                {
                    return RejectStrictCommandBufferPack4Node("NonZero requires exactly one input.");
                }
                var output = operatorName == "NonZero"
                    ? new BufferShape(2, capacity, 1, 1, 1)
                    : new BufferShape(1, capacity, 1, 1, 1);
                var count = new BufferShape(1, 1, 1, 1, 1);
                return AcceptStrictLinearTextureNode(
                    layer,
                    request,
                    "command-buffer-linearmat:bounded-compaction",
                    new[] { output, count },
                    new[] { operatorName == "NonZero" ? "Int32" : inputs[0].logicalDtype, "Int32" });
            }
            if (operatorName == "GatherND")
            {
                var data = StrictShape(inputs[0]);
                var indices = inputs.Count > 1 ? StrictShape(inputs[1]) : default;
                if (inputs.Count != 2 || data.dims != 1 || indices.dims != 2 || indices.w != 1
                    || ReadStrictLayerInt(layer, "batch_dims", 0, 0) != 0 || ReadStrictLayerInt(layer, "index_depth", 1, 1) != 1
                    || !HasStrictLayerProof(layer, "indices_in_range")
                    || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal)
                    || !string.Equals(layer.GetString("index_dtype", null), "Int32", StringComparison.Ordinal)
                    || !string.Equals(inputs[1].logicalDtype, "Int32", StringComparison.Ordinal))
                    return RejectStrictCommandBufferPack4Node("GatherND requires rank-1 Float32 data, batch_dims=0, rank-2 [N,1] Int32 indices, index_depth=1, and an indices_in_range proof.");
                var output = new BufferShape(1, indices.h, 1, 1, 1);
                return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:gathernd-linear", new[] { output }, new[] { inputs[0].logicalDtype });
            }
            var scatterNd = operatorName == "ScatterND";
            var scatterData = StrictShape(inputs[0]);
            var scatterIndices = inputs.Count > 1 ? StrictShape(inputs[1]) : default;
            var scatterUpdates = inputs.Count > 2 ? StrictShape(inputs[2]) : default;
            var scatterAxis = ReadStrictLayerInt(layer, "axis", 0, 0);
            var validScatterShapes = inputs.Count == 3 && scatterData.dims == 1 && scatterUpdates.dims == 1
                && (scatterNd
                    ? scatterIndices.dims == 2 && scatterIndices.w == 1 && scatterIndices.h == scatterUpdates.w
                    : scatterIndices.dims == 1 && scatterIndices.w == scatterUpdates.w && scatterAxis == 0);
            if (!validScatterShapes
                || !HasStrictLayerProof(layer, "unique_indices")
                || !HasStrictLayerProof(layer, "indices_in_range")
                || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal)
                || !string.Equals(inputs[2].logicalDtype, inputs[0].logicalDtype, StringComparison.Ordinal)
                || !string.Equals(layer.GetString("index_dtype", null), "Int32", StringComparison.Ordinal)
                || !string.Equals(inputs[1].logicalDtype, "Int32", StringComparison.Ordinal)
                || (scatterNd && ReadStrictLayerInt(layer, "index_depth", 1, 1) != 1)
                || !string.Equals(layer.GetString("reduction", null), "none", StringComparison.Ordinal))
                return RejectStrictCommandBufferPack4Node(scatterNd
                    ? "ScatterND requires matching Float32 rank-1 data/updates, index_depth=1, reduction=none, and in-range unique rank-2 [N,1] Int32 indices."
                    : "Scatter/ScatterElements require matching Float32 rank-1 data/updates, axis=0, reduction=none, and in-range unique rank-1 Int32 indices.");
            var dataShape = inputs[0].logicalShape;
            var scatterOutput = new BufferShape(dataShape[0], dataShape[1], dataShape[2], dataShape[3], dataShape[4]);
            return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:scatter-unique", new[] { scatterOutput }, new[] { inputs[0].logicalDtype });
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSentisTextureNode(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            string operatorName)
        {
            if (!string.Equals(request?.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase))
                return RejectStrictCommandBufferPack4Node(operatorName + " uses exact RFloat LinearMat storage and requires an FP32 strict target.");

            try
            {
                if (operatorName == "Range" || operatorName == "ConstantOfShape")
                {
                    if (inputs != null && inputs.Count != 0)
                        return RejectStrictCommandBufferPack4Node(operatorName + " requires all runtime shape/value inputs to be statically folded.");

                    if (operatorName == "Range")
                    {
                        if ((!AexisShapeIndexLayerUtil.TryGetFloat(layer, "start", out var start) && !AexisShapeIndexLayerUtil.TryGetFloat(layer, 0, out start))
                            || (!AexisShapeIndexLayerUtil.TryGetFloat(layer, "limit", out var limit) && !AexisShapeIndexLayerUtil.TryGetFloat(layer, 1, out limit)))
                            return RejectStrictCommandBufferPack4Node("Range requires static start and limit parameters.");
                        var delta = AexisShapeIndexLayerUtil.GetFloat(layer, 2, "delta", 1f);
                        if (!IsStrictFinite(start) || !IsStrictFinite(limit) || !IsStrictFinite(delta) || Math.Abs(delta) < 1e-12f)
                            return RejectStrictCommandBufferPack4Node("Range requires finite start/limit and a finite non-zero delta.");
                        var span = (limit - start) / delta;
                        if (!IsStrictFinite(span) || span <= 0f || Math.Ceiling(span) > int.MaxValue)
                            return RejectStrictCommandBufferPack4Node("Range requires a positive statically bounded output length.");
                        var count = (int)Math.Ceiling(span);
                        var logicalDtype = layer.GetString("logical_dtype", "Float32");
                        if (!IsStrictRFloatLogicalDtype(logicalDtype)
                            || (string.Equals(logicalDtype, "Int32", StringComparison.Ordinal)
                                && (!IsStrictExactRFloatInteger(start)
                                    || !IsStrictExactRFloatInteger(delta)
                                    || !IsStrictExactRFloatInteger(start + (count - 1d) * delta))))
                        {
                            return RejectStrictCommandBufferPack4Node("Range logical dtype must be Float32 or FP32-exact Int32 for every generated endpoint.");
                        }
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:range-static", new[] { new BufferShape(1, count, 1, 1, 1) }, new[] { logicalDtype });
                    }

                    if (!AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var constantShape))
                        return RejectStrictCommandBufferPack4Node("ConstantOfShape requires a static shape parameter.");
                    var constantOutput = AexisShapeIndexLayerUtil.FromAxisSizes(constantShape);
                    var fill = AexisShapeIndexLayerUtil.GetFloat(layer, 1, "value", AexisShapeIndexLayerUtil.GetFloat(layer, 0, "fill", 0f));
                    var constantDtype = layer.GetString("logical_dtype", "Float32");
                    if (!IsStrictFinite(fill) || !IsStrictRFloatLogicalDtype(constantDtype)
                        || (string.Equals(constantDtype, "Int32", StringComparison.Ordinal) && !IsStrictExactRFloatInteger(fill)))
                        return RejectStrictCommandBufferPack4Node("ConstantOfShape fill must be finite Float32 or an FP32-exact Int32 value.");
                    return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:constant-of-shape-static", new[] { constantOutput }, new[] { constantDtype });
                }

                if (!TryGetStrictSentisTextureInputs(inputs, out var shapes, out var inputReason))
                    return RejectStrictCommandBufferPack4Node(operatorName + " input contract failed: " + inputReason);

                switch (operatorName)
                {
                    case "Shape":
                    {
                        if (shapes.Length != 1)
                            return RejectStrictCommandBufferPack4Node("Shape requires exactly one input.");
                        var rank = shapes[0].dims;
                        var start = AexisShapeIndexLayerUtil.GetInt(layer, 0, "start", 0);
                        var end = AexisShapeIndexLayerUtil.GetInt(layer, 1, "end", rank);
                        if (start < 0) start += rank;
                        if (end < 0) end += rank;
                        start = Mathf.Clamp(start, 0, rank);
                        end = Mathf.Clamp(end, 0, rank);
                        if (end <= start)
                            return RejectStrictCommandBufferPack4Node("Shape requires a non-empty static start/end slice.");
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:shape-descriptor", new[] { new BufferShape(1, end - start, 1, 1, 1) }, new[] { "Int32" });
                    }
                    case "Size":
                        if (shapes.Length != 1)
                            return RejectStrictCommandBufferPack4Node("Size requires exactly one input.");
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:size-descriptor", new[] { new BufferShape(1, 1, 1, 1, 1) }, new[] { "Int32" });
                    case "Expand":
                    {
                        if (shapes.Length != 1 || !AexisShapeIndexLayerUtil.TryGetShapeParam(layer, out var requested))
                            return RejectStrictCommandBufferPack4Node("Expand requires one input and a static shape parameter.");
                        var output = AexisShapeIndexLayerUtil.ResolveExpandShape(shapes[0], requested);
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:expand-static", new[] { output }, new[] { inputs[0].logicalDtype });
                    }
                    case "Where":
                    {
                        if (shapes.Length != 3)
                            return RejectStrictCommandBufferPack4Node("Where requires condition, true-value, and false-value inputs.");
                        var conditionDtype = inputs[0].logicalDtype;
                        if (!string.Equals(conditionDtype, "Bool", StringComparison.Ordinal)
                            && !string.Equals(conditionDtype, "Int32", StringComparison.Ordinal))
                            return RejectStrictCommandBufferPack4Node("Where condition must have logical Bool or Int32 semantics.");
                        if (!string.Equals(inputs[1].logicalDtype, inputs[2].logicalDtype, StringComparison.Ordinal)
                            || !IsStrictRFloatLogicalDtype(inputs[1].logicalDtype))
                            return RejectStrictCommandBufferPack4Node("Where true/false values must have the same Float32 or Int32 logical dtype.");
                        var output = AexisShapeIndexLayerUtil.BroadcastShapes(shapes);
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:where-broadcast", new[] { output }, new[] { inputs[1].logicalDtype });
                    }
                    case "Gather":
                    case "GatherElements":
                    {
                        if (shapes.Length != 2 || !string.Equals(inputs[1].logicalDtype, "Int32", StringComparison.Ordinal)
                            || !HasStrictLayerProof(layer, "indices_in_range"))
                            return RejectStrictCommandBufferPack4Node(operatorName + " requires two inputs, logical Int32 indices, and an explicit indices_in_range proof.");
                        var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                        BufferShape output;
                        if (operatorName == "Gather")
                            output = AexisShapeIndexLayerUtil.ResolveGatherShape(shapes[0], shapes[1], axis);
                        else
                        {
                            AexisShapeIndexLayerUtil.ValidateGatherElementsShape(layer, shapes[0], shapes[1], axis);
                            output = shapes[1];
                        }
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:" + operatorName.ToLowerInvariant() + "-in-range", new[] { output }, new[] { inputs[0].logicalDtype });
                    }
                    case "ArgMax":
                    case "ArgMin":
                    {
                        if (shapes.Length != 1 || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal))
                            return RejectStrictCommandBufferPack4Node(operatorName + " requires one logical Float32 input.");
                        var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", 0);
                        var keepDims = AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepdims", AexisShapeIndexLayerUtil.GetInt(layer, 1, "keepDims", 1));
                        var selectLast = AexisShapeIndexLayerUtil.GetInt(layer, 2, "selectLastIndex", 0);
                        if ((keepDims != 0 && keepDims != 1) || (selectLast != 0 && selectLast != 1))
                            return RejectStrictCommandBufferPack4Node(operatorName + " keepdims/select_last_index must be 0 or 1.");
                        var output = AexisShapeIndexLayerUtil.ResolveArgReduceShape(shapes[0], axis, keepDims != 0);
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:" + operatorName.ToLowerInvariant(), new[] { output }, new[] { "Int32" });
                    }
                    case "TopK":
                    {
                        if (shapes.Length != 1 || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal))
                            return RejectStrictCommandBufferPack4Node("TopK requires one logical Float32 input.");
                        if (!AexisShapeIndexLayerUtil.TryGetInt(layer, "k", out var k) && !AexisShapeIndexLayerUtil.TryGetInt(layer, 1, out k))
                            return RejectStrictCommandBufferPack4Node("TopK requires a static k parameter.");
                        var axis = AexisShapeIndexLayerUtil.GetInt(layer, 0, "axis", -1);
                        var largest = AexisShapeIndexLayerUtil.GetInt(layer, 2, "largest", 1);
                        var sorted = AexisShapeIndexLayerUtil.GetInt(layer, 3, "sorted", 1);
                        if ((largest != 0 && largest != 1) || (sorted != 0 && sorted != 1))
                            return RejectStrictCommandBufferPack4Node("TopK largest/sorted must be 0 or 1.");
                        var output = AexisShapeIndexLayerUtil.ResolveTopKShape(shapes[0], axis, k);
                        var outputCount = layer?.topNames?.Length ?? 0;
                        if (outputCount < 1 || outputCount > 2)
                            return RejectStrictCommandBufferPack4Node("TopK requires one values output and an optional logical Int32 indices output.");
                        return outputCount == 1
                            ? AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:topk-static", new[] { output }, new[] { "Float32" })
                            : AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:topk-static", new[] { output, output }, new[] { "Float32", "Int32" });
                    }
                    case "OneHot":
                    {
                        if (shapes.Length != 1 || !string.Equals(inputs[0].logicalDtype, "Int32", StringComparison.Ordinal))
                            return RejectStrictCommandBufferPack4Node("OneHot requires one logical Int32 indices input.");
                        if (!AexisShapeIndexLayerUtil.TryGetInt(layer, "depth", out var depth)
                            && !AexisShapeIndexLayerUtil.TryGetInt(layer, 1, out depth)
                            && !AexisShapeIndexLayerUtil.TryGetInt(layer, 0, out depth))
                            return RejectStrictCommandBufferPack4Node("OneHot requires a static depth parameter.");
                        var axis = AexisShapeIndexLayerUtil.GetInt(layer, 2, "axis", -1);
                        var output = AexisShapeIndexLayerUtil.ResolveOneHotShape(shapes[0], axis, depth, out _);
                        var outputDtype = layer.GetString("logical_dtype", "Float32");
                        var onValue = AexisShapeIndexLayerUtil.GetFloat(layer, 3, "on_value", 1f);
                        var offValue = AexisShapeIndexLayerUtil.GetFloat(layer, 4, "off_value", 0f);
                        if (!IsStrictRFloatLogicalDtype(outputDtype) || !IsStrictFinite(onValue) || !IsStrictFinite(offValue)
                            || (string.Equals(outputDtype, "Int32", StringComparison.Ordinal)
                                && (!IsStrictExactRFloatInteger(onValue) || !IsStrictExactRFloatInteger(offValue))))
                            return RejectStrictCommandBufferPack4Node("OneHot values must be finite Float32 or FP32-exact Int32 constants.");
                        return AcceptStrictLinearTextureNode(layer, request, "command-buffer-linearmat:onehot-static", new[] { output }, new[] { outputDtype });
                    }
                    default:
                        return RejectStrictCommandBufferPack4Node("No static Sentis texture verifier exists for " + operatorName + ".");
                }
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node(operatorName + " static texture contract failed: " + exception.Message);
            }
        }

        private static bool TryGetStrictSentisTextureInputs(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            out BufferShape[] shapes,
            out string reason)
        {
            if (!TryGetStrictPlanShapes(inputs, out shapes, out reason))
                return false;
            for (var index = 0; index < shapes.Length; index++)
            {
                var descriptor = inputs[index];
                var linear = HasStrictAnyRankLinearMatStorage(descriptor, shapes[index]);
                var pack4 = HasStrictExactPack4Storage(descriptor, shapes[index]) || HasStrictScalarPack4Storage(descriptor, shapes[index]);
                if (!linear && !pack4)
                {
                    reason = "input " + index.ToString(CultureInfo.InvariantCulture) + " is not exact LinearMat or direct Pack4 texture storage";
                    return false;
                }
                if (linear && !TryResolveStrictLinearStorage(shapes[index], out _, out reason))
                {
                    reason = "input " + index.ToString(CultureInfo.InvariantCulture) + " " + reason;
                    return false;
                }
            }
            return true;
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictLinearTextureNode(
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanRequest request,
            string executionPath,
            BufferShape[] logicalShapes,
            string[] logicalDtypes)
        {
            if (logicalShapes == null || logicalDtypes == null || logicalShapes.Length != logicalDtypes.Length)
                return RejectStrictCommandBufferPack4Node("LinearMat verifier output contract is invalid.");
            var storage = new BufferShape[logicalShapes.Length];
            for (var index = 0; index < logicalShapes.Length; index++)
            {
                if (!TryResolveStrictLinearStorage(logicalShapes[index], out storage[index], out var reason))
                    return RejectStrictCommandBufferPack4Node("LinearMat output " + index.ToString(CultureInfo.InvariantCulture) + " is not representable: " + reason);
                if (!IsStrictRFloatLogicalDtype(logicalDtypes[index]))
                    return RejectStrictCommandBufferPack4Node("LinearMat output logical dtype must be Float32 or Int32.");
            }
            return AcceptStrictDataIndexNode(layer, request, executionPath, logicalShapes, storage, logicalDtypes);
        }

        private static bool TryResolveStrictLinearStorage(BufferShape logicalShape, out BufferShape storage, out string reason)
        {
            storage = ResolveLinearMatStorageShape(logicalShape);
            var capacity = (long)Mathf.Max(1, storage.w) * Mathf.Max(1, storage.h);
            var required = GetStrictPlanElementCount(logicalShape);
            if (storage.w > SystemInfo.maxTextureSize || storage.h > SystemInfo.maxTextureSize || capacity < required)
            {
                reason = "required elements=" + required.ToString(CultureInfo.InvariantCulture)
                    + " exceed exact 2D texture capacity=" + capacity.ToString(CultureInfo.InvariantCulture) + ".";
                return false;
            }
            reason = null;
            return true;
        }

        private static bool IsStrictRFloatLogicalDtype(string logicalDtype)
        {
            return string.Equals(logicalDtype, "Float32", StringComparison.Ordinal)
                || string.Equals(logicalDtype, "Int32", StringComparison.Ordinal);
        }

        private static bool IsStrictFinite(double value)
        {
            return !double.IsNaN(value) && !double.IsInfinity(value);
        }

        private static bool IsStrictExactRFloatInteger(double value)
        {
            const double MaxExactRFloatInteger = 16777216d;
            return IsStrictFinite(value) && Math.Truncate(value) == value && Math.Abs(value) <= MaxExactRFloatInteger;
        }

        private static BufferShape StrictShape(AexisTexturePlanTensorDescriptor descriptor)
        {
            var shape = descriptor.logicalShape;
            return new BufferShape(shape[0], shape[1], shape[2], shape[3], shape[4]);
        }

        private static int ReadStrictLayerInt(AexisGraphModel.Layer layer, string namedKey, int parameterKey, int defaultValue)
        {
            if (layer?.stringParams != null && layer.stringParams.TryGetValue(namedKey, out var named)
                && int.TryParse(named, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedNamed))
                return parsedNamed;
            if (layer?.intParams != null && layer.intParams.TryGetValue(parameterKey, out var keyed)
                && int.TryParse(keyed, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsedKeyed))
                return parsedKeyed;
            return defaultValue;
        }

        private static bool HasStrictLayerProof(AexisGraphModel.Layer layer, string namedKey)
        {
            return layer?.stringParams != null
                && layer.stringParams.TryGetValue(namedKey, out var value)
                && int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out var parsed)
                && parsed == 1;
        }

        private static bool TryReadStrictLayerInt(AexisGraphModel.Layer layer, string namedKey, int parameterKey, out int value)
        {
            value = 0;
            if (layer?.stringParams != null && layer.stringParams.TryGetValue(namedKey, out var named)
                && int.TryParse(named, NumberStyles.Integer, CultureInfo.InvariantCulture, out value))
                return true;
            return layer?.intParams != null
                && layer.intParams.TryGetValue(parameterKey, out var keyed)
                && int.TryParse(keyed, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool HasStrictScalarLikePlanStorage(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            if (storage == null || storage.Length != 5
                || logicalShape.dims < 1 || logicalShape.dims > 2
                || logicalShape.w <= 0 || logicalShape.h <= 0
                || storage[1] != logicalShape.w
                || storage[2] != (logicalShape.dims == 1 ? 1 : logicalShape.h)
                || storage[3] != 1
                || (storage[4] != 1
                    // An ncnn scalar can be carried in the first lane of a
                    // single Pack4 texel (notably the UNet timestep input).
                    // BinaryOpScalarSingleBroadcast reads that texel directly;
                    // permit only the one-element case, never a general
                    // vector-lane reinterpretation or buffer materialization.
                    && !(storage[4] == 4 && GetStrictPlanElementCount(logicalShape) == 1)))
            {
                return false;
            }

            return storage[0] == 1 || storage[0] == 2 || storage[0] == 3;
        }

        private static bool TryResolveStrictCommandBufferSpatialBroadcast(
            AexisTexturePlanTensorDescriptor firstDescriptor,
            BufferShape first,
            AexisTexturePlanTensorDescriptor secondDescriptor,
            BufferShape second,
            out BufferShape output)
        {
            output = default;
            if ((first.dims != 3 && first.dims != 4)
                || first.dims != second.dims
                || !HasStrictExactPack4Storage(firstDescriptor, first)
                || !HasStrictExactPack4Storage(secondDescriptor, second))
            {
                return false;
            }

            if (IsStrictCommandBufferWidthVector(first) && IsStrictCommandBufferWidthVectorTarget(second, first.w))
            {
                output = second;
                return true;
            }

            if (IsStrictCommandBufferWidthVector(second) && IsStrictCommandBufferWidthVectorTarget(first, second.w))
            {
                output = first;
                return true;
            }

            // AexisBinaryOpPack4Broadcast modes 3/4 replicate a single packed
            // channel lane across every output channel at the same spatial point.
            // DeepFillV2 uses this exact [W,H,D,3] x [W,H,D,1] profile to apply
            // its mask without materialising a multi-channel texture.
            var matchingSpatialExtent = first.w == second.w
                && first.h == second.h
                && first.d == second.d;
            if (matchingSpatialExtent && first.c == 1 && second.c > 1)
            {
                output = second;
                return true;
            }

            if (matchingSpatialExtent && second.c == 1 && first.c > 1)
            {
                output = first;
                return true;
            }

            if (first.c != second.c)
                return false;

            // Modes 5/6 use the first texel in a Pack4 row for a width-1
            // tensor. This keeps row-wise broadcast entirely texture-native.
            var matchingRowsAndChannels = first.h == second.h
                && first.d == second.d
                && first.c == second.c;
            if (matchingRowsAndChannels && first.w == 1 && second.w > 1)
            {
                output = second;
                return true;
            }

            if (matchingRowsAndChannels && second.w == 1 && first.w > 1)
            {
                output = first;
                return true;
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

        private static bool IsStrictCommandBufferWidthVector(BufferShape shape)
        {
            return (shape.dims == 3 || shape.dims == 4)
                && shape.w > 0
                && shape.h == 1
                && shape.d == 1
                && shape.c == 1;
        }

        private static bool IsStrictCommandBufferWidthVectorTarget(BufferShape shape, int width)
        {
            return (shape.dims == 3 || shape.dims == 4)
                && shape.w == width
                && shape.h > 0
                && shape.d > 0
                && shape.c > 0;
        }

        private static bool TryResolveStrictCommandBufferChannelVector(
            AexisTexturePlanTensorDescriptor firstDescriptor,
            BufferShape first,
            AexisTexturePlanTensorDescriptor secondDescriptor,
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
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape,
            int expectedChannels)
        {
            var storage = descriptor?.storageShape;
            if (descriptor == null
                || (logicalShape.dims != 1 && (logicalShape.dims != 2 || logicalShape.h != 1))
                || logicalShape.w != expectedChannels)
            {
                return false;
            }

            // AexisBinaryOpLayer accepts either an already-packed texture vector
            // or an RFloat LinearMat vector. The latter is transformed by the
            // explicit CommandBuffer ReshapeLinearMatToPack4 kernel before the
            // real channel-vector BinaryOp dispatch.
            return HasStrictLinearMatStorage(descriptor, logicalShape)
                || (storage != null
                    && storage.Length == 5
                    && storage[0] == 3
                    && storage[1] == expectedChannels
                    && storage[2] == 1
                    && storage[3] == 1
                    && storage[4] == 1);
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp3DOr2D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            var reason = (string)null;
            var input = default(BufferShape);
            if (inputs == null || inputs.Count == 0
                || !TryGetStrictPlanShape(inputs[0], out input, out reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims == 4)
            {
                if (inputs.Count != 1)
                    return RejectStrictCommandBufferPack4Node("The CDHW Interp profile supports exactly one source descriptor; descriptor-only size expressions are a 2D-only path.");
                return VerifyStrictCommandBufferInterp3D(layer, inputs, input, request);
            }

            return VerifyStrictCommandBufferInterp(layer, inputs, request);
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp3D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            BufferShape input,
            AexisTextureExecutionPlanRequest request)
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
            AexisGraphModel.Layer layer,
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInterp(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            var reason = (string)null;
            var input = default(BufferShape);
            if (inputs == null || inputs.Count == 0
                || !TryGetStrictPlanShape(inputs[0], out input, out reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1)
                return RejectStrictCommandBufferPack4Node("CommandBuffer Interp requires a 2D Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("CommandBuffer Interp requires exact rank-3 Pack4 source descriptor storage.");

            var sizeExpr = layer.GetString(9, null);
            if (!string.IsNullOrWhiteSpace(sizeExpr))
            {
                if (layer?.bottomNames == null || layer.bottomNames.Length != inputs.Count)
                    return RejectStrictCommandBufferPack4Node("Interp size_expr requires one texture descriptor for every declared bottom.");

                var bottomShapes = new BufferShape[inputs.Count];
                for (var index = 0; index < inputs.Count; index++)
                {
                    if (!TryGetStrictPlanShape(inputs[index], out bottomShapes[index], out reason))
                        return RejectStrictCommandBufferPack4Node(reason);
                }

                try
                {
                    var sizes = EvaluateExpressionList(sizeExpr, bottomShapes, layer);
                    if (sizes == null || sizes.Count < 1 || sizes.Count > 2)
                        return RejectStrictCommandBufferPack4Node("2D Interp size_expr must resolve one or two descriptor-only extents.");

                    var expressionOutputWidth = sizes[0];
                    var expressionOutputHeight = sizes.Count == 1 ? input.h : sizes[1];
                    if (expressionOutputWidth <= 0 || expressionOutputHeight <= 0)
                        return RejectStrictCommandBufferPack4Node("Interp size_expr resolved a non-positive output extent.");
                    if (expressionOutputWidth == input.w && expressionOutputHeight == input.h)
                        return AcceptStrictCommandBufferPack4NoopAlias(layer, inputs[0], request);

                    return AcceptStrictCommandBufferPack4Node(
                        layer,
                        new BufferShape(3, expressionOutputWidth, expressionOutputHeight, 1, input.c),
                        request,
                        "command-buffer-pack4:interp-descriptor-size-expression");
                }
                catch (Exception ex)
                {
                    return RejectStrictCommandBufferPack4Node(
                        "Interp size_expr is not a valid descriptor-only shape expression: " + ex.Message);
                }
            }

            if (inputs.Count != 1)
                return RejectStrictCommandBufferPack4Node("Static 2D Interp accepts exactly one source descriptor.");

            var resizeType = layer.GetInt(0, 0);
            if (resizeType != 0 && resizeType != 1 && resizeType != 2 && resizeType != 3)
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPixelShuffle(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferUnaryOp(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            var operation = layer.GetInt(0, 0);
            if (operation < 0 || operation > 28)
                return RejectStrictCommandBufferPack4Node("The UnaryOp code is outside the verified CommandBuffer Pack4 kernel range.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:unary");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferRelu(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("ReLU requires an exact rank-3/rank-4 Pack4 texture descriptor.");
            var slope = layer.GetFloat(0, 0f);
            if (float.IsNaN(slope) || float.IsInfinity(slope))
                return RejectStrictCommandBufferPack4Node("ReLU slope must be finite.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:relu");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSigmoid(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            var exactStorage = HasStrictExactPack4Storage(inputs[0], input)
                || (input.dims <= 2 && HasStrictAnyRankLinearMatStorage(inputs[0], input))
                || HasStrictPack4LinearMatStorage(inputs[0], input);
            if (!exactStorage)
                return RejectStrictCommandBufferPack4Node("Sigmoid requires exact LinearMat or rank-3/rank-4 Pack4 texture storage.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:sigmoid");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferUnaryAlias(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            int operation,
            string operationName)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (operation < 0 || operation > 17)
                return RejectStrictCommandBufferPack4Node("The " + operationName + " unary alias has no verified CommandBuffer Pack4 kernel.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:unary-" + operationName);
        }

        internal static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPointwise(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            string operatorName)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 1 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node(operatorName + " supports only rank-1 through rank-4 Pack4 tensors.");
            if (!HasStrictExactPack4Storage(inputs[0], input)
                && !HasStrictScalarPack4Storage(inputs[0], input)
                && !HasStrictPack4LinearMatStorage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node(operatorName + " requires exact scalar-Pack4, Pack4-Linear, or rank-3/rank-4 Pack4 texture storage.");
            if (string.Equals(operatorName, "Clip", StringComparison.Ordinal)
                && layer.GetFloat(0, -1e30f) > layer.GetFloat(1, 1e30f))
                return RejectStrictCommandBufferPack4Node("Clip minimum must not exceed maximum.");
            if (string.Equals(operatorName, "IsInf", StringComparison.Ordinal)
                && ((layer.GetInt(0, 1) != 0 && layer.GetInt(0, 1) != 1)
                    || (layer.GetInt(1, 1) != 0 && layer.GetInt(1, 1) != 1)))
                return RejectStrictCommandBufferPack4Node("IsInf detect_negative/detect_positive must be 0 or 1.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:pointwise-" + operatorName.ToLowerInvariant());
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGelu(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 1 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("GELU has no verified CommandBuffer Pack4 path for this rank.");
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:gelu");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferTrilu(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 2 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("Trilu requires a descriptor-backed logical rank from 2 through 4.");

            var upper = layer.GetInt(0, 1);
            if (upper != 0 && upper != 1)
                return RejectStrictCommandBufferPack4Node("Trilu upper must be 0 or 1.");

            var descriptor = inputs[0];
            var storageValues = descriptor.storageShape;
            var storage = new BufferShape(storageValues[0], storageValues[1], storageValues[2], storageValues[3], storageValues[4]);
            var exactPack4 = HasStrictExactPack4Storage(descriptor, input);
            var linearRankTwo = input.dims == 2 && HasStrictLinearMatStorage(descriptor, input);
            var scalarPack4RankTwo = input.dims == 2 && HasStrictScalar2DPack4Storage(descriptor, input);
            if (!exactPack4 && !linearRankTwo && !scalarPack4RankTwo)
            {
                return RejectStrictCommandBufferPack4Node(
                    "Trilu requires exact final-axis X/Y texture storage; packed-lane or rematerialized matrix layouts are not accepted.");
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                storage,
                request,
                linearRankTwo ? "command-buffer-linearmat:trilu" : "command-buffer-pack4:trilu");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferCumSum(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!TryReadStrictLayerInt(layer, "axis", 0, out var axis))
            {
                if (string.Equals(layer.typeName, "CumulativeSum", StringComparison.Ordinal))
                    axis = 0;
                else
                    return RejectStrictCommandBufferPack4Node("ONNX CumSum requires a statically lowered axis parameter.");
            }
            if (axis < -input.dims || axis >= input.dims)
                return RejectStrictCommandBufferPack4Node("CumSum axis is outside the logical input rank.");

            var exclusive = ReadStrictLayerInt(layer, "exclusive", 1, 0);
            var reverse = ReadStrictLayerInt(layer, "reverse", 2, 0);
            if ((exclusive != 0 && exclusive != 1) || (reverse != 0 && reverse != 1))
                return RejectStrictCommandBufferPack4Node("CumSum exclusive and reverse flags must be 0 or 1.");

            var descriptor = inputs[0];
            var directPack4 = HasStrictExactPack4Storage(descriptor, input)
                || HasStrictScalar2DPack4Storage(descriptor, input)
                || (input.dims == 1 && HasStrictScalarLikePlanStorage(descriptor, input));
            if (!directPack4 && !HasStrictAnyRankLinearMatStorage(descriptor, input))
                return RejectStrictCommandBufferPack4Node("CumSum requires exact LinearMat or direct Pack4 texture storage; buffer materialization is prohibited.");

            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                ResolveLinearMatStorageShape(input),
                request,
                "command-buffer-linearmat:cumsum");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPnnxExpression(
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanRequest request)
        {
            if (!AexisPnnxExpressionLayer.TryResolveConstantValueCount(layer, out var valueCount, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            // This mirrors ExecuteCommandBuffer exactly: a rank-one logical list is materialized
            // into a one-pack scalar Texture2DArray by AexisOps.FillScalarTexture.
            var logicalShape = new BufferShape(1, valueCount, 1, 1, 1);
            var storageShape = new BufferShape(3, valueCount, 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                logicalShape,
                storageShape,
                request,
                "command-buffer-pack4:pnnx-expression-constant");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMemoryData(
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanRequest request)
        {
            if (!_memoryData.TryGetValue(layer.name, out var memory) || memory == null)
                return RejectStrictCommandBufferPack4Node("MemoryData was not loaded for this node.");
            if (memory.linearMatRt != null
                && memory.linearMatRt.IsCreated()
                && (memory.dims == 1 || memory.dims == 2)
                && memory.w > 0
                && memory.h > 0)
            {
                var linearLogicalShape = new BufferShape(memory.dims, memory.w, memory.h, 1, 1);
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    linearLogicalShape,
                    ResolveLinearMatStorageShape(linearLogicalShape),
                    request,
                    "command-buffer-pack4:memory-data-linear-mat",
                    layer.GetString("logical_dtype", null));
            }
            if (memory.pack4Rt != null && memory.pack4Rt.IsCreated()
                && (memory.dims == 3 || memory.dims == 4)
                && memory.w > 0 && memory.h > 0 && memory.d > 0 && memory.c > 0
                && memory.pack4RtDepth == memory.d * ((memory.c + 3) / 4))
            {
                var packedLogicalShape = new BufferShape(memory.dims, memory.w, memory.h, memory.d, memory.c);
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    packedLogicalShape,
                    packedLogicalShape,
                    request,
                    "command-buffer-pack4:memory-data-pack4",
                    layer.GetString("logical_dtype", null));
            }
            if (memory.channelVectorTexture == null
                || memory.dims != 3
                || memory.w != 1
                || memory.h != 1
                || memory.d != 1
                || memory.c <= 0)
            {
                return RejectStrictCommandBufferPack4Node("Only loaded 1x1 channel-vector MemoryData has a texture-native CommandBuffer path.");
            }

            var logicalShape = new BufferShape(3, 1, 1, 1, memory.c);
            var storageShape = logicalShape;
            return AcceptStrictCommandBufferPack4Node(
                layer,
                logicalShape,
                storageShape,
                request,
                "command-buffer-pack4:memory-data-channel-vector-pack4");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInnerProduct(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 1 && input.dims != 2)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer InnerProduct profile requires a vector or matrix input.");
            if (!_innerProduct.TryGetValue(layer.name, out var innerProduct)
                || innerProduct == null
                || innerProduct.TextureWeightBinding == null
                || innerProduct.b == null
                || innerProduct.inFeatures <= 0
                || innerProduct.outFeatures <= 0)
            {
                return RejectStrictCommandBufferPack4Node("The loaded InnerProduct weights or bias are unavailable.");
            }
            // FP16 activations may multiply immutable FP32 weights through the
            // native Gemm texture kernel.  The runtime selects wFp16 only when
            // the model explicitly owns a half-weight upload; requiring it here
            // would reject that valid all-GPU texture path and tempt a Buffer
            // fallback.  The binding itself remains mandatory and immutable.
            if (input.w != innerProduct.inFeatures)
                return RejectStrictCommandBufferPack4Node("InnerProduct input width does not match the loaded weight profile.");

            var linearMatInput = HasStrictLinearMatStorage(inputs[0], input);
            var pack4LinearMatInput = input.dims == 2 && HasStrictPack4LinearMatStorage(inputs[0], input);
            var scalarPack4Input = HasStrictScalarPack4Storage(inputs[0], input);
            if (!linearMatInput && !pack4LinearMatInput && !scalarPack4Input)
            {
                return RejectStrictCommandBufferPack4Node("InnerProduct input storage does not prove a LinearMat, Pack4-Linear, or scalar Pack4 CommandBuffer texture layout.");
            }
            if (pack4LinearMatInput && innerProduct.outFeatures % 4 != 0)
                return RejectStrictCommandBufferPack4Node("Pack4-Linear InnerProduct requires an output feature count divisible by four.");

            var output = input.dims == 2
                ? new BufferShape(2, innerProduct.outFeatures, input.h, 1, 1)
                : new BufferShape(1, innerProduct.outFeatures, 1, 1, 1);
            // Mirror TryExecuteCommandBufferTexturePath. A legacy/F32-sensitive
            // projection stays in exact RFloat LinearMat storage; only the
            // explicitly admitted FP16 path publishes Pack4-linear output.
            var usePack4LinearMat = UsesFp16ActivationStorage
                && output.dims == 2
                && innerProduct.outFeatures % 4 == 0
                && (linearMatInput || pack4LinearMatInput);
            usePack4LinearMat = usePack4LinearMat
                && !PreserveLegacyFp32Execution
                && !UseLegacyPack4AttentionLayout
                && !HasDirectSoftmaxConsumerForStrictPlan(layer)
                && !RequiresFp32SensitiveOutputStorage(layer);
            if (pack4LinearMatInput && !usePack4LinearMat)
            {
                return RejectStrictCommandBufferPack4Node(
                    "The loaded InnerProduct Pack4-linear input requires the FP16 Pack4-linear projection profile; the active legacy or sensitive-output contract selects LinearMat instead.");
            }
            var outputStorage = usePack4LinearMat
                ? ResolvePack4LinearMatStorageShape(output)
                : linearMatInput
                ? ResolveLinearMatStorageShape(output)
                : new BufferShape(3, output.w, output.dims == 2 ? output.h : 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                outputStorage,
                request,
                usePack4LinearMat
                    ? "command-buffer-pack4:inner-product-pack4-linear-mat"
                    : linearMatInput
                    ? "command-buffer-pack4:inner-product-linear-mat"
                    : "command-buffer-pack4:inner-product-scalar-pack4");
        }

        private bool HasDirectSoftmaxConsumerForStrictPlan(AexisGraphModel.Layer layer)
        {
            if (Model?.layers == null || layer?.topNames == null)
                return false;

            foreach (var topName in layer.topNames)
            {
                if (string.IsNullOrWhiteSpace(topName))
                    continue;
                foreach (var consumer in Model.layers)
                {
                    if (consumer?.type != AexisLayerTypes.Softmax || consumer.bottomNames == null)
                        continue;
                    if (consumer.bottomNames.Any(bottom => string.Equals(bottom, topName, StringComparison.Ordinal)))
                        return true;
                }
            }

            return false;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferShuffleChannel(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.d != 1 || !HasStrictExactPack4Storage(inputs[0], input))
            {
                return RejectStrictCommandBufferPack4Node(
                    "ShuffleChannel requires an exact rank-3 Pack4 Texture2DArray activation.");
            }

            var group = layer.GetInt(0, 1);
            var reverse = layer.GetInt(1, 0);
            if (group <= 0 || input.c % group != 0)
                return RejectStrictCommandBufferPack4Node("ShuffleChannel group must be positive and divide the logical channel count.");
            if (reverse != 0 && reverse != 1)
                return RejectStrictCommandBufferPack4Node("ShuffleChannel reverse must be 0 or 1.");

            // AexisShuffleChannelLayer dispatches ShuffleChannelPack4 over the
            // existing texture-array packs.  It preserves both logical and
            // physical shape without readback or activation materialization.
            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                "command-buffer-pack4:shuffle-channel");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferReduction(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            if (input.dims == 2
                && (HasStrictLinearMatStorage(inputs[0], input)
                    || HasStrictScalar2DPack4Storage(inputs[0], input)))
            {
                if (!TryResolveStrictLinearMat2DReduction(layer, input, out var linearOutput, out reason))
                    return RejectStrictCommandBufferPack4Node(reason);
                var scalarPack4Input = HasStrictScalar2DPack4Storage(inputs[0], input);
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    linearOutput,
                    scalarPack4Input
                        ? new BufferShape(3, Mathf.Max(1, linearOutput.w), Mathf.Max(1, linearOutput.h), 1, 1)
                        : ResolveLinearMatStorageShape(linearOutput),
                    request,
                    scalarPack4Input
                        ? "command-buffer-pack4:reduction-scalar2d"
                        : "command-buffer-linearmat:reduction-2d");
            }

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

        private static bool TryResolveStrictLinearMat2DReduction(
            AexisGraphModel.Layer layer,
            BufferShape input,
            out BufferShape output,
            out string reason)
        {
            output = default;
            reason = null;
            if (input.dims != 2 || input.w <= 0 || input.h <= 0)
            {
                reason = "The LinearMat Reduction profile requires a non-empty rank-2 input.";
                return false;
            }

            var operation = layer.GetInt(0, 0);
            if (operation < 0 || operation > 10)
            {
                reason = "The LinearMat Reduction operation is outside the verified CommandBuffer subset.";
                return false;
            }

            var reduceAll = layer.GetInt(1, 1) != 0;
            var keepDims = layer.GetInt(4, 0) != 0;
            if (reduceAll)
            {
                output = keepDims
                    ? new BufferShape(2, 1, 1, 1, 1)
                    : new BufferShape(1, 1, 1, 1, 1);
                return true;
            }

            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length != 1)
            {
                reason = "The LinearMat Reduction profile requires one static rank-2 axis, or reduce-all.";
                return false;
            }

            var axis = axes[0] < 0 ? axes[0] + input.dims : axes[0];
            if (axis == 1)
            {
                output = keepDims
                    ? new BufferShape(2, 1, input.h, 1, 1)
                    : new BufferShape(1, input.h, 1, 1, 1);
                return true;
            }
            if (axis == 0)
            {
                output = keepDims
                    ? new BufferShape(2, input.w, 1, 1, 1)
                    : new BufferShape(1, input.w, 1, 1, 1);
                return true;
            }

            reason = "The LinearMat Reduction axis is outside rank-2 input dimensions.";
            return false;
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMvn(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if ((input.dims != 3 && input.dims != 4) || input.w <= 0 || input.h <= 0 || input.d <= 0 || input.c <= 0)
                return RejectStrictCommandBufferPack4Node("MVN requires a static dims=3/4 Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("MVN requires exact rank-3/rank-4 Pack4 descriptor storage.");
            if (layer.GetFloat(2, 0.0001f) < 0f)
                return RejectStrictCommandBufferPack4Node("MVN epsilon must be non-negative.");
            if ((layer.GetInt(0, 0) != 0 && layer.GetInt(0, 0) != 1)
                || (layer.GetInt(1, 0) != 0 && layer.GetInt(1, 0) != 1))
                return RejectStrictCommandBufferPack4Node("MVN normalize_variance and across_channels must be 0 or 1.");
            if (!_extraPacks.ContainsKey(layer.name))
                return RejectStrictCommandBufferPack4Node("MVN dispatch constants are not loaded.");
            var verification = AcceptStrictCommandBufferPack4Node(layer, input, request,
                layer.GetInt(1, 0) != 0 ? "command-buffer-pack4:mvn-across-channels" : "command-buffer-pack4:mvn-per-channel");
            if (!verification.accepted)
                return verification;

            var groups = layer.GetInt(1, 0) != 0 ? 1 : input.c;
            verification.scratch = new[]
            {
                CreateStrictPack4ScratchDescriptor(layer, "mvn-stats-a", new BufferShape(3, groups, 1, 1, 4), request),
                CreateStrictPack4ScratchDescriptor(layer, "mvn-stats-b", new BufferShape(3, groups, 1, 1, 4), request)
            };
            return verification;
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPRelu(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 1 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("PReLU requires a static logical rank from 1 through 4.");
            if (!HasStrictExactPack4Storage(inputs[0], input)
                && !HasStrictScalarPack4Storage(inputs[0], input)
                && !HasStrictAnyRankLinearMatStorage(inputs[0], input))
            {
                return RejectStrictCommandBufferPack4Node("PReLU requires exact Pack4 or LinearMat texture storage.");
            }
            if (!_extraPacks.TryGetValue(layer.name, out var packObject)
                || packObject is not PReluPack pack
                || pack.slope == null || pack.slopeCpu == null
                || pack.numSlope <= 0 || pack.slopeCpu.Length != pack.numSlope)
            {
                return RejectStrictCommandBufferPack4Node("PReLU requires loaded immutable FP32 slope constants.");
            }
            if (pack.numSlope != 1
                && ((input.dims != 3 && input.dims != 4) || pack.numSlope != input.c))
            {
                return RejectStrictCommandBufferPack4Node("PReLU non-scalar slope count must equal the logical channel count for rank-3/rank-4 input.");
            }
            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                pack.numSlope == 1 ? "command-buffer-pack4:prelu-scalar" : "command-buffer-pack4:prelu-channel");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferLrn(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || input.w <= 0 || input.h <= 0 || input.c <= 0)
                return RejectStrictCommandBufferPack4Node("LRN requires a static rank-3 CHW activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input)
                && !HasStrictAnyRankLinearMatStorage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("LRN requires exact Pack4 or LinearMat texture storage.");
            if (!_extraPacks.TryGetValue(layer.name, out var packObject)
                || packObject is not LrnPack pack
                || (pack.regionType != 0 && pack.regionType != 1)
                || pack.localSize <= 0 || float.IsNaN(pack.alpha) || float.IsInfinity(pack.alpha)
                || float.IsNaN(pack.beta) || float.IsInfinity(pack.beta)
                || float.IsNaN(pack.bias) || float.IsInfinity(pack.bias))
            {
                return RejectStrictCommandBufferPack4Node("LRN requires loaded finite parameters, region_type=0|1, and local_size>0.");
            }
            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                pack.regionType == 0 ? "command-buffer-pack4:lrn-across-channels" : "command-buffer-pack4:lrn-within-channel");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPooling1D(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!AexisPooling1DLayer.TryResolveSpec(layer, input, out var spec, out reason))
                return RejectStrictCommandBufferPack4Node("Pooling1D contract failed: " + reason + ".");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Pooling1D requires exact rank-3 Pack4 descriptor storage.");
            var profile = spec.global ? "global" : spec.adaptive ? "adaptive" : "windowed";
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(3, spec.outputWidth, 1, 1, input.c), request, "command-buffer-pack4:pooling-1d-" + profile);
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferBatchNorm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferInstanceNorm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 && input.dims != 4)
                return RejectStrictCommandBufferPack4Node("InstanceNorm requires a rank-3/rank-4 Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("InstanceNorm requires exact rank-3/rank-4 Pack4 descriptor storage.");
            if (!UseNcnnStyleGroupNorm)
                return RejectStrictCommandBufferPack4Node("InstanceNorm requires the texture-native ncnn-style group normalization path.");
            if (!_extraPacks.TryGetValue(layer.name, out var packObject)
                || packObject is not GroupNormPack pack
                || !pack.affine
                || pack.gamma == null
                || pack.beta == null
                || pack.channels != input.c
                || pack.group != input.c
                || pack.eps < 0f)
            {
                return RejectStrictCommandBufferPack4Node("Loaded immutable InstanceNorm scale/bias or channel/epsilon metadata does not match the input descriptor.");
            }
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:instance-norm");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferBias(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 && input.dims != 4)
                return RejectStrictCommandBufferPack4Node("The verified Bias profile requires a rank-3/rank-4 Pack4 activation.");
            if (!HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Bias requires exact rank-3/rank-4 Pack4 descriptor storage.");
            if (!_bias.TryGetValue(layer.name, out var bias)
                || bias == null
                || bias.channels != input.c
                || bias.bias4 == null)
            {
                return RejectStrictCommandBufferPack4Node("The loaded immutable Bias Pack4 constants do not match the input descriptor.");
            }
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:bias");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferCopyTo(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (inputs == null || inputs.Count != 2 || inputs[0] == null || inputs[1] == null)
                return RejectStrictCommandBufferPack4Node("CopyTo requires exactly two texture-backed inputs.");
            if (!string.Equals(inputs[0].layout, inputs[1].layout, StringComparison.OrdinalIgnoreCase)
                || !string.Equals(inputs[0].dtype, inputs[1].dtype, StringComparison.OrdinalIgnoreCase))
            {
                return RejectStrictCommandBufferPack4Node("CopyTo destination and source layout/dtype contracts must match.");
            }
            if (inputs[0].logicalShape == null || inputs[0].logicalShape.Length != 5
                || inputs[1].logicalShape == null || inputs[1].logicalShape.Length != 5
                || inputs[0].storageShape == null || inputs[0].storageShape.Length != 5
                || inputs[1].storageShape == null || inputs[1].storageShape.Length != 5)
            {
                return RejectStrictCommandBufferPack4Node("CopyTo requires complete logical and storage descriptors for both inputs.");
            }

            var self = StrictShape(inputs[0]);
            var src = StrictShape(inputs[1]);
            if (!HasStrictExactPack4Storage(inputs[0], self) || !HasStrictExactPack4Storage(inputs[1], src))
                return RejectStrictCommandBufferPack4Node("CopyTo requires exact rank-3/rank-4 Pack4 storage for both inputs.");
            if (!AexisCopyToLayer.TryResolveOffsets(layer, self, src, out _, out var reason))
                return RejectStrictCommandBufferPack4Node("CopyTo ROI contract failed: " + reason + ".");
            return AcceptStrictCommandBufferPack4Node(layer, self, request, "command-buffer-pack4:copyto-roi");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferAtenTo(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out _, out var reason))
                return RejectStrictCommandBufferPack4Node("aten::to dtype-preserving alias requires one descriptor-backed data input: " + reason);
            return AcceptStrictCommandBufferPack4NoopAlias(layer, inputs[0], request);
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferCrop(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason) || shapes.Length < 1 || shapes.Length > 2)
                return RejectStrictCommandBufferPack4Node("Crop requires one data input and at most one descriptor-backed reference input: " + reason);
            var source = shapes[0];
            if (source.dims != 3 || !HasStrictExactPack4Storage(inputs[0], source))
                return RejectStrictCommandBufferPack4Node("Crop requires an exact rank-3 Pack4 source descriptor.");
            if (shapes.Length > 1 && layer.GetInt(0, 0) == -233)
                return RejectStrictCommandBufferPack4Node("Crop param_data requires runtime value readback and is not a strict texture-native profile.");

            CropRoi roi;
            try
            {
                roi = ResolveCropRoi(source, layer, shapes);
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node("Crop ROI resolution failed: " + exception.Message);
            }
            if (roi.outw <= 0 || roi.outh <= 0 || roi.outd != 1 || roi.outc <= 0)
                return RejectStrictCommandBufferPack4Node("Crop requires a non-empty rank-3 ROI with no depth slicing.");
            var output = new BufferShape(3, roi.outw, roi.outh, 1, roi.outc);
            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:crop-roi");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGroupNorm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!UseNcnnStyleGroupNorm)
                return RejectStrictCommandBufferPack4Node("GroupNorm requires the ncnn-style texture-native normalization path.");
            if (!_groupNorm.TryGetValue(layer.name, out var pack) || pack == null
                || !pack.affine || pack.gamma == null || pack.beta == null
                || pack.channels <= 0 || pack.group <= 0 || pack.channels % pack.group != 0
                || pack.eps < 0f)
            {
                return RejectStrictCommandBufferPack4Node("Loaded GroupNorm affine constants, group count, channels, or epsilon are invalid.");
            }

            var pack4 = (input.dims == 3 || input.dims == 4)
                && input.c == pack.channels
                && HasStrictExactPack4Storage(inputs[0], input);
            var linear = input.dims == 2
                && input.h == pack.channels
                && HasStrictAnyRankLinearMatStorage(inputs[0], input);
            if (!pack4 && !linear)
                return RejectStrictCommandBufferPack4Node("GroupNorm requires exact rank-3/rank-4 Pack4 storage or rank-2 LinearMat with height=channels.");

            var verification = AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                linear ? "command-buffer-linearmat:groupnorm" : "command-buffer-pack4:groupnorm",
                inputs[0].logicalDtype);
            if (!verification.accepted)
                return verification;

            // GroupNorm's texture kernels write mean and variance to two FP32
            // Pack4 textures. LinearMat additionally materializes its
            // immutable, statically-shaped Pack4 view and its result before
            // returning to LinearMat. Declare all of them to the static RT arena
            // so CommandBuffer execution never falls back to opportunistic RTs.
            var scratch = new List<AexisTexturePlanTensorDescriptor>
            {
                CreateStrictPack4ScratchDescriptor(layer, "groupnorm-stats-a", new BufferShape(3, pack.group, 1, 1, 4), request),
                CreateStrictPack4ScratchDescriptor(layer, "groupnorm-stats-b", new BufferShape(3, pack.group, 1, 1, 4), request)
            };
            if (linear)
            {
                var packed = new BufferShape(3, input.w, 1, 1, pack.channels);
                scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "groupnorm-linear-packed-input", packed, request));
                scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "groupnorm-linear-packed-output", packed, request));
            }
            verification.scratch = scratch.ToArray();
            return verification;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPadding(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || !HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Padding requires an exact rank-3 Pack4 input.");

            var top = layer.GetInt(0, 0);
            var bottom = layer.GetInt(1, 0);
            var left = layer.GetInt(2, 0);
            var right = layer.GetInt(3, 0);
            var type = layer.GetInt(4, 0);
            var value = layer.GetFloat(5, 0f);
            var perChannel = layer.GetInt(6, 0);
            var front = layer.GetInt(7, 0);
            var behind = layer.GetInt(8, 0);
            if (front != 0 || behind != 0 || perChannel != 0)
                return RejectStrictCommandBufferPack4Node("Padding texture profile does not support channel/depth padding or per-channel pad values.");
            if (type < 0 || type > 2 || !IsStrictFinite(value))
                return RejectStrictCommandBufferPack4Node("Padding type must be constant(0), replicate(1), or reflect-101(2), with a finite value.");
            if (top < 0 || bottom < 0 || left < 0 || right < 0)
                return RejectStrictCommandBufferPack4Node("Padding strict texture profile requires non-negative spatial padding.");
            if (type == 2
                && ((left > 0 || right > 0) && (input.w <= 1 || left >= input.w || right >= input.w)
                    || (top > 0 || bottom > 0) && (input.h <= 1 || top >= input.h || bottom >= input.h)))
            {
                return RejectStrictCommandBufferPack4Node("Padding reflect-101 requires each padded input axis to have length > 1 and every pad to be smaller than that axis.");
            }
            var outputW = input.w + left + right;
            var outputH = input.h + top + bottom;
            if (outputW <= 0 || outputH <= 0)
                return RejectStrictCommandBufferPack4Node("Padding produces an empty spatial output.");
            if (top == 0 && bottom == 0 && left == 0 && right == 0)
                return AcceptStrictCommandBufferPack4NoopAlias(layer, inputs[0], request);
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(3, outputW, outputH, 1, input.c), request, "command-buffer-pack4:padding-spatial");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferQuantization(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request,
            string operatorName)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            var supportedStorage = HasStrictExactPack4Storage(inputs[0], input)
                || HasStrictAnyRankLinearMatStorage(inputs[0], input);
            if (!supportedStorage)
                return RejectStrictCommandBufferPack4Node(operatorName + " requires exact Pack4 or LinearMat texture storage.");

            var axisSize = input.dims <= 1 ? input.w : input.dims == 2 ? input.h : input.c;
            if (operatorName == "Quantize")
            {
                if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not QuantizePack pack
                    || !IsValidStrictQuantArray(pack.scaleDataSize, axisSize, pack.scaleCpu, pack.scale))
                    return RejectStrictCommandBufferPack4Node("Quantize requires loaded finite scalar or full-axis scale constants.");
                if (!string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.Ordinal)
                    && !string.Equals(inputs[0].logicalDtype, "Float16", StringComparison.Ordinal))
                    return RejectStrictCommandBufferPack4Node("Quantize input must have logical Float32 or Float16 semantics.");
                return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:quantize", "Int8");
            }

            if (!string.Equals(inputs[0].logicalDtype, "Int8", StringComparison.Ordinal))
                return RejectStrictCommandBufferPack4Node(operatorName + " input must have logical Int8 semantics.");
            if (operatorName == "Dequantize")
            {
                if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not DequantizePack pack
                    || !IsValidStrictQuantArray(pack.scaleDataSize, axisSize, pack.scaleCpu, pack.scale)
                    || !IsValidOptionalStrictQuantArray(pack.biasDataSize, axisSize, pack.biasCpu, pack.bias))
                    return RejectStrictCommandBufferPack4Node("Dequantize requires loaded finite scalar or full-axis scale/bias constants.");
                return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:dequantize", ResolveLogicalDtype(request.targetDtype));
            }

            if (!_extraPacks.TryGetValue(layer.name, out var requantObject) || requantObject is not RequantizePack requant
                || !IsValidStrictQuantArray(requant.scaleInDataSize, axisSize, requant.scaleInCpu, requant.scaleIn)
                || !IsValidStrictQuantArray(requant.scaleOutDataSize, axisSize, requant.scaleOutCpu, requant.scaleOut)
                || !IsValidOptionalStrictQuantArray(requant.biasDataSize, axisSize, requant.biasCpu, requant.bias)
                || requant.activationType < 0 || requant.activationType > 6
                || !IsStrictFinite(requant.activationParam0) || !IsStrictFinite(requant.activationParam1))
            {
                return RejectStrictCommandBufferPack4Node("Requantize requires loaded finite scalar/full-axis constants and activation type 0..6.");
            }
            return AcceptStrictCommandBufferPack4Node(layer, input, CopyStrictStorage(inputs[0]), request, "command-buffer-pack4:requantize", "Int8");
        }

        private static bool IsValidStrictQuantArray(int size, int axisSize, float[] values, ComputeBuffer buffer)
        {
            return (size == 1 || size == axisSize)
                && values != null && values.Length == size
                && buffer != null
                && values.All(value => IsStrictFinite(value));
        }

        private static bool IsValidOptionalStrictQuantArray(int size, int axisSize, float[] values, ComputeBuffer buffer)
        {
            return size == 0 && values == null && buffer == null
                || IsValidStrictQuantArray(size, axisSize, values, buffer);
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferReorg(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || !HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Reorg requires an exact rank-3 Pack4 input.");
            var stride = layer.GetInt(0, 1);
            var mode = layer.GetInt(1, 0);
            if (stride != 2 || mode != 0 || input.w % 2 != 0 || input.h % 2 != 0)
                return RejectStrictCommandBufferPack4Node("Reorg texture kernel requires stride=2, mode=0, and even spatial dimensions.");
            if (input.c > int.MaxValue / 4)
                return RejectStrictCommandBufferPack4Node("Reorg output channel count exceeds the descriptor range.");
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(3, input.w / 2, input.h / 2, 1, input.c * 4), request, "command-buffer-pack4:reorg-stride2");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferScale(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims != 3 || !HasStrictExactPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Scale scalar texture kernel requires an exact rank-3 Pack4 input.");
            if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not ScalePack pack
                || pack.dynamic || pack.scaleDataSize != 1 || pack.biasTerm
                || pack.scale == null || pack.scaleCpu == null || pack.scaleCpu.Length != 1 || !IsStrictFinite(pack.scaleCpu[0]))
            {
                return RejectStrictCommandBufferPack4Node("Scale texture kernel requires one loaded finite static scalar and no bias.");
            }
            return AcceptStrictCommandBufferPack4Node(layer, input, request, "command-buffer-pack4:scale-scalar");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferUnfold(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not UnfoldPack pack)
                return RejectStrictCommandBufferPack4Node("Loaded Unfold geometry is missing.");
            var supportedInput = input.dims == 3 && HasStrictExactPack4Storage(inputs[0], input)
                || input.dims == 2 && HasStrictAnyRankLinearMatStorage(inputs[0], input);
            if (!supportedInput)
                return RejectStrictCommandBufferPack4Node("Unfold requires exact rank-3 Pack4 or rank-2 LinearMat input storage.");
            if (pack.kernelW <= 0 || pack.kernelH <= 0 || pack.strideW <= 0 || pack.strideH <= 0
                || pack.dilationW <= 0 || pack.dilationH <= 0 || !IsStrictFinite(pack.padValue))
                return RejectStrictCommandBufferPack4Node("Unfold kernel, stride, dilation, and pad value parameters are invalid.");

            var extentW = (long)pack.dilationW * (pack.kernelW - 1) + 1;
            var extentH = (long)pack.dilationH * (pack.kernelH - 1) + 1;
            if (!TryResolveStrictUnfoldPadding(input.w, input.h, extentW, extentH, pack, out var left, out var right, out var top, out var bottom))
                return RejectStrictCommandBufferPack4Node("Unfold padding parameters are unsupported.");
            var paddedW = (long)input.w + left + right;
            var paddedH = (long)input.h + top + bottom;
            var outW = (paddedW - extentW) / pack.strideW + 1;
            var outH = (paddedH - extentH) / pack.strideH + 1;
            var channels = input.dims == 3 ? input.c : 1;
            var columns = outW * outH;
            var rows = (long)pack.kernelW * pack.kernelH * channels;
            if (outW <= 0 || outH <= 0 || columns > int.MaxValue || rows > int.MaxValue)
                return RejectStrictCommandBufferPack4Node("Unfold produces an empty or oversized output texture.");
            return AcceptStrictCommandBufferPack4Node(layer, new BufferShape(2, (int)columns, (int)rows, 1, 1), request, "command-buffer-pack4:unfold-static");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferExtractPatches(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_extraPacks.TryGetValue(layer.name, out var packObject) || packObject is not UnfoldPack pack)
                return RejectStrictCommandBufferPack4Node("Loaded ExtractPatches geometry is missing.");
            if (pack.kernelW <= 0 || pack.kernelH <= 0 || pack.strideW <= 0 || pack.strideH <= 0
                || pack.dilationW <= 0 || pack.dilationH <= 0 || pack.padLeft < 0 || pack.padRight < 0
                || pack.padTop < 0 || pack.padBottom < 0 || !IsStrictFinite(pack.padValue))
            {
                return RejectStrictCommandBufferPack4Node("ExtractPatches requires positive kernel/stride/dilation, non-negative explicit padding, and a finite pad value.");
            }

            var exactRank3 = input.dims == 3 && HasStrictExactPack4Storage(inputs[0], input);
            var storage = inputs[0]?.storageShape;
            var foldedRank4 = input.dims == 4 && storage != null && storage.Length == 5
                && storage[0] == 4 && storage[1] == input.w && storage[2] == (long)input.h * input.d
                && storage[3] == 1 && storage[4] == input.c;
            if (!exactRank3 && !foldedRank4)
                return RejectStrictCommandBufferPack4Node("ExtractPatches requires exact rank-3 Pack4 or descriptor-proven rank-4 Fold-D texture storage.");

            try
            {
                var extentW = checked((long)pack.dilationW * (pack.kernelW - 1L) + 1L);
                var extentH = checked((long)pack.dilationH * (pack.kernelH - 1L) + 1L);
                var outW = FloorDivStrict((long)input.w + pack.padLeft + pack.padRight - extentW, pack.strideW) + 1L;
                var outH = FloorDivStrict((long)input.h + pack.padTop + pack.padBottom - extentH, pack.strideH) + 1L;
                var area = checked((long)pack.kernelW * pack.kernelH);
                var outD = input.dims == 4 ? checked((long)input.d * area) : 1L;
                var outC = input.dims == 4 ? input.c : checked((long)input.c * area);
                if (outW <= 0 || outH <= 0 || outD <= 0 || outC <= 0
                    || outW > int.MaxValue || outH > int.MaxValue || outD > int.MaxValue || outC > int.MaxValue)
                    return RejectStrictCommandBufferPack4Node("ExtractPatches produces an empty or oversized logical output.");
                var logicalOutput = new BufferShape(input.dims, (int)outW, (int)outH, (int)outD, (int)outC);
                if (input.dims == 3)
                    return AcceptStrictCommandBufferPack4Node(layer, logicalOutput, request, "command-buffer-pack4:extract-patches");
                var foldedHeight = checked(outH * outD);
                if (foldedHeight > int.MaxValue)
                    return RejectStrictCommandBufferPack4Node("ExtractPatches Fold-D output height exceeds the descriptor range.");
                var storageOutput = new BufferShape(4, (int)outW, (int)foldedHeight, 1, (int)outC);
                return AcceptStrictCommandBufferPack4Node(layer, logicalOutput, storageOutput, request, "command-buffer-pack4:extract-patches-fold-d");
            }
            catch (OverflowException)
            {
                return RejectStrictCommandBufferPack4Node("ExtractPatches shape arithmetic overflowed the descriptor range.");
            }
        }

        private static long FloorDivStrict(long numerator, long denominator)
        {
            if (denominator <= 0) throw new ArgumentOutOfRangeException(nameof(denominator));
            var quotient = numerator / denominator;
            var remainder = numerator % denominator;
            return remainder < 0 ? quotient - 1 : quotient;
        }

        private static bool TryResolveStrictUnfoldPadding(
            int width,
            int height,
            long extentW,
            long extentH,
            UnfoldPack pack,
            out int left,
            out int right,
            out int top,
            out int bottom)
        {
            left = pack.padLeft;
            right = pack.padRight;
            top = pack.padTop;
            bottom = pack.padBottom;
            if (left > 0 || right > 0 || top > 0 || bottom > 0)
                return left >= 0 && right >= 0 && top >= 0 && bottom >= 0;
            if (left == -233 && right == -233 && top == -233 && bottom == -233)
            {
                var widthPad = extentW + (width - 1L) / pack.strideW * pack.strideW - width;
                var heightPad = extentH + (height - 1L) / pack.strideH * pack.strideH - height;
                right = (int)Math.Max(0L, widthPad / 2);
                left = (int)Math.Max(0L, widthPad - right);
                bottom = (int)Math.Max(0L, heightPad / 2);
                top = (int)Math.Max(0L, heightPad - bottom);
                return true;
            }
            if (left == 0 && right == 0 && top == 0 && bottom == 0)
                return true;
            return false;
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferReshape(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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

            if (input.dims <= 2 && output.dims <= 2
                && HasStrictLinearMatStorage(inputs[0], input))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    ResolveLinearMatStorageShape(output),
                    request,
                    "command-buffer-linearmat:reshape-2d");
            }

            if (input.dims <= 2 && output.dims <= 2
                && HasStrictPack4LinearMatStorage(inputs[0], input))
            {
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    ResolvePack4LinearMatStorageShape(output),
                    request,
                    "command-buffer-pack4:reshape-pack4-linear-2d");
            }

            if (input.dims >= 3 && input.dims <= 4 && output.dims >= 1 && output.dims <= 2
                && (CanUseStrictWidthPreservingPack4ToLinearMatReshape(input, output)
                    || CanUseStrictPack4ToLinearMatReshape(layer)
                    || CanUseStrictCodeFormerStylePack4ToLinearMatReshape(layer, input, output)))
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

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferFlatten(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            var total = GetStrictPlanElementCount(input);
            if (total > int.MaxValue)
                return RejectStrictCommandBufferPack4Node("Flatten output exceeds the supported Pack4 texture extent.");

            var output = new BufferShape(1, (int)total, 1, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                new BufferShape(3, output.w, 1, 1, 1),
                request,
                "command-buffer-pack4:flatten-pack4");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSqueeze(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            try
            {
                var output = ResolveSqueezeShape(input, layer);
                return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], output, request);
            }
            catch (Exception exception)
            {
                return RejectStrictCommandBufferPack4Node("Squeeze shape resolution failed: " + exception.Message);
            }
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferExpandDims(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!TryResolveStrictExpandDimsShape(layer, input, out var output, out reason))
                return RejectStrictCommandBufferPack4Node(reason);
            return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], output, request);
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferTile(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!TryResolveStrictTileShape(layer, input, out var output, out var identity, out reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (identity)
                return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], output, request);

            var storage = input.dims <= 2 && HasStrictLinearMatStorage(inputs[0], input)
                ? ResolveLinearMatStorageShape(output)
                : output.dims <= 2
                    ? new BufferShape(3, output.w, output.dims == 2 ? output.h : 1, 1, 1)
                    : output;
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                storage,
                request,
                input.dims <= 2 && HasStrictLinearMatStorage(inputs[0], input)
                    ? "command-buffer-pack4:tile-linear-mat"
                    : "command-buffer-pack4:tile-pack4");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPacking(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            var outElemPack = layer.GetInt(0, 1);
            var castFrom = layer.GetInt(2, 0);
            var castTo = layer.GetInt(3, 0);
            if (outElemPack != 1 && outElemPack != 4)
                return RejectStrictCommandBufferPack4Node("Packing supports only out_elempack=1 or 4 on the CommandBuffer Pack4 backend.");
            if (outElemPack == 4 && (castTo == 0 || castFrom == castTo))
                return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], input, request);

            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                castTo != 0 && castFrom != castTo
                    ? "command-buffer-pack4:packing-cast"
                    : "command-buffer-pack4:packing-repack");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferCast(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            var typeFrom = layer.GetInt(0, 0);
            var typeTo = layer.GetInt(1, 0);
            if (typeFrom < 1 || typeFrom > 7 || typeTo < 1 || typeTo > 7)
                return RejectStrictCommandBufferPack4Node("Cast supports FP32(1), FP16(2), BF16(4), Int8(3), Int32(5), UInt8(6), and logical Bool(7).");
            if (typeFrom == typeTo)
                return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], input, request);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(inputs[0]),
                request,
                "command-buffer-pack4:cast-pack4");
        }

        private static BufferShape CopyStrictStorage(AexisTexturePlanTensorDescriptor descriptor)
        {
            var storage = descriptor?.storageShape;
            return new BufferShape(storage[0], storage[1], storage[2], storage[3], storage[4]);
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Alias(
            AexisGraphModel.Layer layer,
            AexisTexturePlanTensorDescriptor source,
            BufferShape logicalShape,
            AexisTextureExecutionPlanRequest request)
        {
            if (source == null || source.storageShape == null || source.storageShape.Length != 5)
                return RejectStrictCommandBufferPack4Node("Descriptor alias requires a source logical/storage contract.");

            var outputNames = layer?.topNames ?? Array.Empty<string>();
            var storage = (int[])source.storageShape.Clone();
            var logical = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c };
            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = outputNames.Length > 0,
                usesDescriptorAlias = outputNames.Length > 0,
                executionPath = "descriptor-alias",
                reason = outputNames.Length > 0 ? null : "The node has no output blobs.",
                outputs = outputNames.Select(name => new AexisTexturePlanTensorDescriptor
                {
                    blob = name,
                    logicalShape = (int[])logical.Clone(),
                    storageShape = (int[])storage.Clone(),
                    layout = request.targetLayout,
                    dtype = ResolvePhysicalTextureDtype(request.targetDtype),
                    aliasGroup = source.aliasGroup,
                    textureBacked = source.textureBacked
                }).ToArray()
            };
        }

        private static bool TryResolveStrictExpandDimsShape(
            AexisGraphModel.Layer layer,
            BufferShape input,
            out BufferShape output,
            out string reason)
        {
            output = default;
            reason = null;
            var axes = layer.GetInts(-23303, null);
            if (axes == null || axes.Length == 0)
                axes = layer.GetInts(3, Array.Empty<int>());
            if (axes == null || axes.Length == 0)
            {
                reason = "ExpandDims requires static axes metadata.";
                return false;
            }

            try
            {
                var dims = input.dims;
                var values = new[] { input.w, input.h, input.dims == 4 ? input.d : input.c, input.dims == 4 ? input.c : 1 };
                for (var index = 0; index < axes.Length; index++)
                {
                    var outDims = dims + 1;
                    if (outDims > 4)
                        throw new InvalidOperationException("ExpandDims would exceed rank four.");
                    var ncnnAxis = axes[index] < 0 ? axes[index] + outDims : axes[index];
                    if (ncnnAxis < 0 || ncnnAxis >= outDims)
                        throw new InvalidOperationException("ExpandDims axis is outside the target rank.");
                    var axis = MapNcnnAxisToTensorAxis(outDims, ncnnAxis);
                    var next = new[] { 1, 1, 1, 1 };
                    for (var i = 0; i < outDims; i++)
                        next[i] = i == axis ? 1 : values[i < axis ? i : i - 1];
                    values = next;
                    dims = outDims;
                }

                output = dims == 1
                    ? new BufferShape(1, values[0], 1, 1, 1)
                    : dims == 2
                        ? new BufferShape(2, values[0], values[1], 1, 1)
                        : dims == 3
                            ? new BufferShape(3, values[0], values[1], 1, values[2])
                            : new BufferShape(4, values[0], values[1], values[2], values[3]);
                return true;
            }
            catch (Exception exception)
            {
                reason = "ExpandDims shape resolution failed: " + exception.Message;
                return false;
            }
        }

        private static bool TryResolveStrictTileShape(
            AexisGraphModel.Layer layer,
            BufferShape input,
            out BufferShape output,
            out bool identity,
            out string reason)
        {
            output = default;
            identity = false;
            reason = null;
            var repeats = layer.GetInts(-23302, null) ?? layer.GetInts(2, null) ?? layer.GetInts(-23330, null) ?? layer.GetInts(30, null);
            var repeatW = 1;
            var repeatH = 1;
            var repeatD = 1;
            var repeatC = 1;
            var repeatCount = repeats?.Length ?? 0;
            if (repeatCount == 0)
            {
                var axis = layer.GetInt(0, 0);
                var tiles = layer.GetInt(1, 1);
                if (tiles <= 0)
                {
                    reason = "Tile count must be static and positive.";
                    return false;
                }
                if (axis < 0)
                    axis += input.dims;
                if (axis < 0 || axis >= input.dims)
                {
                    reason = "Tile axis is outside the descriptor-backed input rank.";
                    return false;
                }
                if (input.dims == 1) repeatW = tiles;
                else if (input.dims == 2) { if (axis == 0) repeatH = tiles; else repeatW = tiles; }
                else if (input.dims == 3) { if (axis == 0) repeatC = tiles; else if (axis == 1) repeatH = tiles; else repeatW = tiles; }
                else { if (axis == 0) repeatC = tiles; else if (axis == 1) repeatD = tiles; else if (axis == 2) repeatH = tiles; else repeatW = tiles; }
            }
            else if (repeatCount <= 4 && repeats.All(value => value > 0))
            {
                if (repeatCount == 1) repeatW = repeats[0];
                else if (repeatCount == 2) { repeatH = repeats[0]; repeatW = repeats[1]; }
                else if (repeatCount == 3 && input.dims == 4) { repeatD = repeats[0]; repeatH = repeats[1]; repeatW = repeats[2]; }
                else if (repeatCount == 3) { repeatC = repeats[0]; repeatH = repeats[1]; repeatW = repeats[2]; }
                else { repeatC = repeats[0]; repeatD = repeats[1]; repeatH = repeats[2]; repeatW = repeats[3]; }
            }
            else
            {
                reason = "Tile repeats must be static positive metadata with rank one through four.";
                return false;
            }

            var dims = Mathf.Max(input.dims, repeatCount);
            output = dims == 1
                ? new BufferShape(1, input.w * repeatW, 1, 1, 1)
                : dims == 2
                    ? new BufferShape(2, input.w * repeatW, input.h * repeatH, 1, 1)
                    : dims == 3
                        ? new BufferShape(3, input.w * repeatW, input.h * repeatH, 1, input.c * repeatC)
                        : new BufferShape(4, input.w * repeatW, input.h * repeatH, input.d * repeatD, input.c * repeatC);
            identity = repeatW == 1 && repeatH == 1 && repeatD == 1 && repeatC == 1 && StrictPlanShapesEqual(input, output);
            return true;
        }

        private static bool HasStrictPack4LinearMatStorage(
            AexisTexturePlanTensorDescriptor descriptor,
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
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            if (descriptor?.storageShape == null || descriptor.storageShape.Length != 5
                || logicalShape.dims < 1 || logicalShape.dims > 2)
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
            AexisTexturePlanTensorDescriptor descriptor,
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

        private bool CanUseStrictPack4ToLinearMatReshape(AexisGraphModel.Layer layer)
        {
            if (Model?.layers == null || layer?.topNames == null || layer.topNames.Length == 0)
                return false;

            var consumer = FindStrictEffectiveSingleConsumer(layer.topNames[0]);
            return consumer != null
                && (consumer.type == AexisLayerTypes.Permute
                    || consumer.type == AexisLayerTypes.Gemm
                    || consumer.type == AexisLayerTypes.InnerProduct
                    || consumer.type == AexisLayerTypes.Sigmoid
                    || consumer.type == AexisLayerTypes.BinaryOp);
        }

        // The CodeFormer generator reshapes a [W,H,D,C] Pack4 activation to
        // [C,W*H*D], then immediately feeds the result to a Split/Reduction
        // normalization chain. AexisReshapeLayer dispatches this through
        // ReshapePack4ToLinearMat; retain the same narrowly-scoped proof here
        // so strict planning describes the actual CommandBuffer transform.
        private bool CanUseStrictCodeFormerStylePack4ToLinearMatReshape(
            AexisGraphModel.Layer layer,
            BufferShape input,
            BufferShape output)
        {
            if (input.dims != 3 && input.dims != 4)
                return false;
            if (output.dims != 2
                || input.w <= 0 || input.h <= 0 || input.d <= 0 || input.c <= 0
                || output.w != input.c || output.h <= 0)
            {
                return false;
            }

            var expectedRows = (long)input.w * input.h * input.d;
            if (expectedRows != output.h)
                return false;

            var consumer = FindStrictEffectiveSingleConsumer(layer?.topNames != null && layer.topNames.Length > 0
                ? layer.topNames[0]
                : null);
            return consumer != null
                && (consumer.type == AexisLayerTypes.Split
                    || consumer.type == AexisLayerTypes.Reduction);
        }

        // Mirrors the execution-side reshape contract. aten::to and Noop are
        // descriptor aliases in strict CommandBuffer execution, so they do not change
        // the storage mapping established by the preceding reshape.
        private AexisGraphModel.Layer FindStrictEffectiveSingleConsumer(string blobName)
        {
            if (Model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return null;

            var currentBlob = blobName;
            for (var hop = 0; hop < 8; hop++)
            {
                AexisGraphModel.Layer consumer = null;
                foreach (var candidate in Model.layers)
                {
                    if (candidate?.bottomNames == null || !candidate.bottomNames.Contains(currentBlob))
                        continue;
                    if (consumer != null)
                        return null;
                    consumer = candidate;
                }

                if (consumer == null
                    || (consumer.type != AexisLayerTypes.AtenTo && consumer.type != AexisLayerTypes.Noop)
                    || consumer.topNames == null
                    || consumer.topNames.Length != 1
                    || string.IsNullOrWhiteSpace(consumer.topNames[0]))
                    return consumer;

                currentBlob = consumer.topNames[0];
            }

            return null;
        }

        // The Pack4-to-LinearMat kernel has a direct width-preserving mapping: it keeps
        // the NCNN width axis and flattens every remaining logical axis into rows.  This
        // does not depend on the immediate consumer (a Split alias may sit before the
        // LayerNorm/Gemm attention preparation chain), so requiring a direct consumer
        // would reject a CommandBuffer path which the runtime actually executes.
        private static bool CanUseStrictWidthPreservingPack4ToLinearMatReshape(BufferShape input, BufferShape output)
        {
            if (input.dims != 3 && input.dims != 4)
                return false;
            if (output.dims != 2 || input.w <= 0 || input.h <= 0 || input.d <= 0 || input.c <= 0)
                return false;
            if (output.w != input.w || output.h <= 0)
                return false;

            var rows = (long)input.h * input.d * input.c;
            return rows == output.h;
        }

        private static long GetStrictPlanElementCount(BufferShape shape)
        {
            var total = 1L;
            var extents = new[] { shape.w, shape.h, shape.d, shape.c };
            for (var index = 0; index < extents.Length; index++)
            {
                var extent = Math.Max(1, extents[index]);
                if (total > long.MaxValue / extent)
                    return long.MaxValue;
                total *= extent;
            }
            return total;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferPermute(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
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
                    if (HasStrictPack4LinearMatStorage(inputs[0], input))
                    {
                        return AcceptStrictCommandBufferPack4Node(
                            layer,
                            output,
                            ResolvePack4LinearMatStorageShape(output),
                            request,
                            "command-buffer-pack4:permute-pack4-linear-2d");
                    }
                    if (HasStrictScalar2DPack4Storage(inputs[0], input))
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGemm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!_gemm.TryGetValue(layer.name, out var gemm) || gemm == null)
                return RejectStrictCommandBufferPack4Node("The loaded Gemm runtime profile is unavailable.");

            if (!gemm.constantB)
                return VerifyStrictCommandBufferTextureMatMulGemm(layer, gemm, inputs, request);

            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (gemm.TextureWeightBinding == null)
                return RejectStrictCommandBufferPack4Node("The loaded Gemm constant-B weights are unavailable.");
            if (input.dims != 2 || gemm.transA || gemm.constantK <= 0 || gemm.constantN <= 0)
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
            var pack4LinearOutput = outputColumns % 4 == 0;
            // Keep the planner descriptor in lockstep with the verified tiled
            // CommandBuffer Gemm path used for wide vocabulary projections.
            var tiledPack4LinearOutput = pack4LinearOutput
                && Mathf.CeilToInt(outputColumns / 4f) > GetMaxTextureSize();
            var storage = tiledPack4LinearOutput
                ? ResolvePack4TiledLinearMatStorageShape(output)
                : pack4LinearOutput
                    ? ResolvePack4LinearMatStorageShape(output)
                    : ResolveLinearMatStorageShape(output);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                storage,
                request,
                tiledPack4LinearOutput
                    ? "command-buffer-pack4:gemm-linear-to-pack4-tiled-linear"
                    : pack4LinearOutput
                    ? "command-buffer-pack4:gemm-linear-to-pack4-linear"
                    : "command-buffer-pack4:gemm-linear-mat");
        }

        // Dynamic two-input Gemm is used by the VAE attention blocks. The execution
        // path materializes only LinearMat inputs into temporary Pack4 RTs, dispatches
        // MatMulPack4Cdhw, and never creates a ComputeBuffer activation. Keep the
        // planner admission exactly to the two 2D texture mappings accepted there.
        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferTextureMatMulGemm(
            AexisGraphModel.Layer layer,
            GemmPack gemm,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason) || shapes.Length != 2)
                return RejectStrictCommandBufferPack4Node("The dynamic Gemm texture-matmul profile requires exactly two descriptor-backed inputs: " + reason);
            if (gemm.transA || gemm.constantC || gemm.broadcastTypeC != 0)
                return RejectStrictCommandBufferPack4Node("The dynamic Gemm texture-matmul profile requires non-transposed A with no constant C/broadcast.");

            var a = shapes[0];
            var b = shapes[1];
            var aSupported = a.dims == 2 && (HasStrictLinearMatStorage(inputs[0], a) || HasStrictScalar2DPack4Storage(inputs[0], a));
            var bSupported = b.dims == 2 && (HasStrictLinearMatStorage(inputs[1], b) || HasStrictScalar2DPack4Storage(inputs[1], b));
            if (!aSupported || !bSupported)
                return RejectStrictCommandBufferPack4Node("The dynamic Gemm texture-matmul profile requires rank-2 LinearMat or scalar-Pack4 texture inputs.");

            var k = a.w;
            var bRows = b.h;
            var bColumns = b.w;
            var kFromB = gemm.transB ? bColumns : bRows;
            var outputColumns = gemm.transB ? bRows : bColumns;
            if ((gemm.constantK > 0 && k != gemm.constantK) || k != kFromB || a.h <= 0 || outputColumns <= 0)
                return RejectStrictCommandBufferPack4Node("The dynamic Gemm texture-matmul K/N dimensions do not match its two descriptor-backed inputs.");
            if (float.IsNaN(gemm.alpha) || float.IsInfinity(gemm.alpha))
                return RejectStrictCommandBufferPack4Node("The dynamic Gemm texture-matmul alpha must be finite.");

            var output = new BufferShape(2, outputColumns, a.h, 1, 1);
            var useLinearOutput = HasStrictLinearMatStorage(inputs[0], a) || HasStrictLinearMatStorage(inputs[1], b);
            var storage = useLinearOutput
                ? ResolveLinearMatStorageShape(output)
                : new BufferShape(3, output.w, output.h, 1, 1);
            return AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                storage,
                request,
                useLinearOutput
                    ? "command-buffer-pack4:gemm-texture-matmul-to-linear-mat"
                    : "command-buffer-pack4:gemm-texture-matmul-scalar-pack4");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferLayerNorm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_layerNorm.TryGetValue(layer.name, out var pack)
                || pack == null
                || !pack.affine
                || pack.affineSize != input.w
                || pack.gamma == null
                || pack.beta == null)
            {
                return RejectStrictCommandBufferPack4Node("LayerNorm requires loaded affine FP32 parameters over the logical width axis.");
            }
            if (input.dims < 2 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("LayerNorm has no verified CommandBuffer Pack4 path for this rank.");

            var descriptor = inputs[0];
            var verifiedStorage = input.dims == 2
                ? HasStrictLinearMatStorage(descriptor, input) || HasStrictPack4LinearMatStorage(descriptor, input)
                : IsStrictAttentionMatMulInput(descriptor, input);
            if (!verifiedStorage)
                return RejectStrictCommandBufferPack4Node("LayerNorm input storage does not prove a LinearMat or Pack4 texture mapping.");

            var storage = descriptor.storageShape;
            var verification = AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                new BufferShape(storage[0], storage[1], storage[2], storage[3], storage[4]),
                request,
                "command-buffer-pack4:layernorm-width-fp32-accumulate");
            if (!verification.accepted || !HasStrictLinearMatStorage(descriptor, input))
                return verification;

            // LinearMat inputs are explicitly materialized to one-slice Pack4
            // arrays for the width-normalized kernel, then normalized into a
            // second Pack4 scratch before being reshaped back to the LinearMat
            // graph output. Both allocations are real CommandBuffer RTs.
            var scratchDtype = IsStrictFp32ActivationIslandLayer(request, layer)
                ? "FP32"
                : ResolvePhysicalTextureDtype(request.targetDtype);
            var pack4Scratch = new BufferShape(3, input.w, input.h, 1, 1);
            verification.scratch = new[]
            {
                CreateStrictPack4ScratchDescriptor(layer, "layernorm-packed-input", pack4Scratch, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "layernorm-packed-output", pack4Scratch, request, scratchDtype)
            };
            return verification;
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferRmsNorm(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_extraPacks.TryGetValue(layer.name, out var extra) || extra is not RmsNormPack pack
                || !pack.affine
                || pack.affineSize != input.w
                || pack.gamma == null
                || !IsStrictFinite(pack.eps)
                || pack.eps <= 0f)
            {
                return RejectStrictCommandBufferPack4Node(
                    "RMSNorm requires loaded immutable affine FP32 parameters over the logical width axis and a finite positive epsilon.");
            }

            // The dedicated LinearMat kernel is the Qwen profile.  Exact Pack4
            // rank-three/four storage uses the same texture-native RMSNorm kernel;
            // neither branch creates an activation ComputeBuffer.
            var descriptor = inputs[0];
            var usesLinearMat = input.dims == 2
                && (HasStrictLinearMatStorage(descriptor, input) || HasStrictPack4LinearMatStorage(descriptor, input));
            var usesPack4 = (input.dims == 3 || input.dims == 4)
                && IsStrictAttentionMatMulInput(descriptor, input);
            if (!usesLinearMat && !usesPack4)
            {
                return RejectStrictCommandBufferPack4Node(
                    "RMSNorm requires verified rank-two LinearMat/Pack4-Linear storage or exact rank-three/rank-four Pack4 texture storage.");
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                CopyStrictStorage(descriptor),
                request,
                usesLinearMat
                    ? "command-buffer-pack4:rmsnorm-linear-mat-fp32-accumulate"
                    : "command-buffer-pack4:rmsnorm-pack4-fp32-accumulate");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSlice(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (layer?.topNames == null || layer.topNames.Length == 0)
                return RejectStrictCommandBufferPack4Node("Slice has no output blobs.");
            if (input.dims < 1 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("Slice supports only descriptor-backed rank-one through rank-four tensors.");
            var linearMatInput = input.dims <= 2 && HasStrictLinearMatStorage(inputs[0], input);
            var pack4LinearInput = input.dims == 2 && HasStrictPack4LinearMatStorage(inputs[0], input);
            var scalarPack4Input = input.dims <= 2 && HasStrictScalarLikePlanStorage(inputs[0], input);
            if (input.dims <= 2 && !linearMatInput && !pack4LinearInput && !scalarPack4Input)
                return RejectStrictCommandBufferPack4Node("Rank-one/rank-two Slice requires verified LinearMat, Pack4-Linear, or scalar Pack4 descriptor storage.");
            if (input.dims >= 3 && input.dims != 4 && !IsStrictAttentionMatMulInput(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Rank-three Slice requires exact Pack4 logical/storage descriptor mapping.");
            if (input.dims == 4 && !HasStrictCdhwPack4Storage(inputs[0], input))
                return RejectStrictCommandBufferPack4Node("Rank-four Slice requires exact CDHW Pack4 descriptor storage.");

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

            var outputs = new AexisTexturePlanTensorDescriptor[layer.topNames.Length];
            var identity = layer.topNames.Length == 1;
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
                identity &= begin == 0 && StrictPlanShapesEqual(input, output);
                var storage = input.dims <= 2
                    ? linearMatInput
                        ? ResolveLinearMatStorageShape(output)
                        : pack4LinearInput
                            ? ResolvePack4LinearMatStorageShape(output)
                            : new BufferShape(3, output.w, output.dims == 2 ? output.h : 1, 1, 1)
                    : output;

                outputs[index] = new AexisTexturePlanTensorDescriptor
                {
                    blob = layer.topNames[index],
                    logicalShape = new[] { output.dims, output.w, output.h, output.d, output.c },
                    storageShape = new[] { storage.dims, storage.w, storage.h, storage.d, storage.c },
                    layout = request.targetLayout,
                    dtype = ResolvePhysicalTextureDtype(request.targetDtype),
                    aliasGroup = "computed:" + (layer.name ?? layer.typeName ?? "slice") + ":" + index,
                    textureBacked = true
                };
                begin += size;
            }

            if (begin != axisSize)
                return RejectStrictCommandBufferPack4Node("Slice sizes do not cover the descriptor-backed input axis.");

            if (identity)
                return AcceptStrictCommandBufferPack4Alias(layer, inputs[0], input, request);

            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = input.dims <= 2
                    ? linearMatInput
                        ? "command-buffer-pack4:slice-linear-mat"
                        : pack4LinearInput
                            ? "command-buffer-pack4:slice-pack4-linear"
                            : "command-buffer-pack4:slice-scalar-pack4"
                    : input.dims == 4
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

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeterministicRandom(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (AexisDeterministicRandomLayer.IsStaticRandom(layer))
            {
                try
                {
                    AexisDeterministicRandomLayer.ValidateLayer(layer);
                    var output = AexisDeterministicRandomLayer.ResolveStaticOutputShape(layer);
                    return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:deterministic-static-rng");
                }
                catch (Exception exception) when (exception is ArgumentException
                    || exception is InvalidOperationException
                    || exception is NotSupportedException
                    || exception is OverflowException)
                {
                    return RejectStrictCommandBufferPack4Node("Static deterministic RNG profile rejected: " + exception.Message);
                }
            }
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length != 1 || (shapes[0].dims != 3 && shapes[0].dims != 4)
                || !HasStrictExactPack4Storage(inputs[0], shapes[0]))
            {
                return RejectStrictCommandBufferPack4Node(
                    "Deterministic RNG requires one rank-3/rank-4 exact Pack4 Texture2DArray input; LinearMat, buffer, and descriptor-only storage are rejected.");
            }
            try
            {
                AexisDeterministicRandomLayer.ValidateLayer(layer);
                return AcceptStrictCommandBufferPack4Node(layer, shapes[0], request, "command-buffer-pack4:deterministic-rng");
            }
            catch (Exception exception) when (exception is ArgumentException
                || exception is InvalidOperationException
                || exception is NotSupportedException
                || exception is OverflowException)
            {
                return RejectStrictCommandBufferPack4Node("Deterministic RNG profile rejected: " + exception.Message);
            }
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMultinomial(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!string.Equals(request?.targetDtype, "FP32", StringComparison.OrdinalIgnoreCase))
            {
                return RejectStrictCommandBufferPack4Node(
                    "Bounded Multinomial emits exact FP32 Pack4 lanes for logical Int32 indices; FP16/BF16 cannot preserve the categorical index contract.");
            }
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            try
            {
                AexisMultinomialLayer.ValidateLayer(layer);
                var profile = AexisMultinomialLayer.ValidateShape(input, layer.name);
                profile.samples = layer.GetInt(0);
                if (!HasStrictExactPack4Storage(inputs[0], input)
                    || !string.Equals(inputs[0].dtype, "FP32", StringComparison.OrdinalIgnoreCase)
                    || !string.Equals(inputs[0].logicalDtype, "Float32", StringComparison.OrdinalIgnoreCase))
                {
                    return RejectStrictCommandBufferPack4Node(
                        "Bounded Multinomial requires an exact FP32 Pack4 logits descriptor with logical Float32 values.");
                }
                var output = AexisMultinomialLayer.OutputShape(profile);
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    output,
                    output,
                    request,
                    "command-buffer-pack4:bounded-multinomial",
                    "Int32");
            }
            catch (Exception exception) when (exception is ArgumentException
                || exception is InvalidOperationException
                || exception is NotSupportedException
                || exception is OverflowException)
            {
                return RejectStrictCommandBufferPack4Node("Bounded Multinomial profile rejected: " + exception.Message);
            }
        }

        // P1 vision layers have a native Pack4 implementation. Admission is based on
        // the same descriptor/parameter proof used by the dispatch, rather than the
        // broad capability catalog: this is what keeps strict inference texture-only.
        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferP1Vision(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            try
            {
                AexisP1VisionSchema.Validate(layer);
                for (var index = 0; index < shapes.Length; index++)
                {
                    var exactPack4 = HasStrictExactPack4Storage(inputs[index], shapes[index])
                        || HasStrictScalarPack4Storage(inputs[index], shapes[index]);
                    var linearMat = HasStrictLinearMatStorage(inputs[index], shapes[index])
                        || HasStrictPack4LinearMatStorage(inputs[index], shapes[index]);
                    if (!exactPack4 && !linearMat)
                        return RejectStrictCommandBufferPack4Node(
                            "P1 vision input " + index.ToString(CultureInfo.InvariantCulture)
                            + " is neither exact Pack4 nor descriptor-proven LinearMat storage.");
                }

                if (string.Equals(layer.typeName, "Flip", StringComparison.Ordinal))
                {
                    if (shapes.Length != 1 || !HasStrictExactPack4Storage(inputs[0], shapes[0]))
                        return RejectStrictCommandBufferPack4Node("Flip requires a rank-3/rank-4 exact Pack4 Texture2DArray input.");
                    AexisFlipLayer.ValidatePack4Profile(layer, shapes[0]);
                    return AcceptStrictCommandBufferPack4Node(layer, shapes[0], request, "command-buffer-pack4:p1-flip");
                }

                var input0 = shapes.Length > 0 ? shapes[0] : new BufferShape(3, 1, 1, 1, 0);
                var input1 = shapes.Length > 1 ? shapes[1] : new BufferShape(3, 1, 1, 1, 0);
                var input2 = shapes.Length > 2 ? shapes[2] : new BufferShape(3, 1, 1, 1, 0);
                var dispatch = AexisNativeP1VisionLayer.DescribeDispatch(this, layer, input0, input1, input2);
                if (dispatch.output.w <= 0 || dispatch.output.h <= 0 || dispatch.output.d <= 0 || dispatch.output.c <= 0)
                    return RejectStrictCommandBufferPack4Node("P1 vision dispatch resolved a non-positive output extent.");
                return AcceptStrictCommandBufferPack4Node(
                    layer,
                    dispatch.output,
                    request,
                    "command-buffer-pack4:p1-" + (layer.typeName ?? "vision").ToLowerInvariant());
            }
            catch (Exception exception) when (exception is ArgumentException
                || exception is InvalidOperationException
                || exception is NotSupportedException
                || exception is OverflowException)
            {
                return RejectStrictCommandBufferPack4Node("P1 vision profile rejected: " + exception.Message);
            }
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMatMul(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
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
            AexisTexturePlanTensorDescriptor descriptor,
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

        // KV caches retain the logical sequence length while their Texture2DArray can
        // reserve additional rows for future decoding steps. W/D/C remain an exact
        // Pack4 attention mapping; only the physical sequence height may be larger.
        private static bool IsStrictAttentionKvCacheInput(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape shape)
        {
            var storage = descriptor?.storageShape;
            return shape.dims == 3
                && shape.w > 0
                && shape.h > 0
                && shape.d == 1
                && shape.c > 0
                && storage != null
                && storage.Length == 5
                && storage[0] == 3
                && storage[1] == shape.w
                && storage[2] >= shape.h
                && storage[3] == 1
                && storage[4] == shape.c;
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSoftmax(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetSingleStrictPlanShape(inputs, out var input, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (input.dims < 2 || input.dims > 4)
                return RejectStrictCommandBufferPack4Node("The verified CommandBuffer Softmax profile requires a rank-2 through rank-4 texture.");
            var pack4LinearMat = input.dims == 2 && HasStrictPack4LinearMatStorage(inputs[0], input);
            var inputStorage = input.dims == 2
                ? HasStrictLinearMatStorage(inputs[0], input)
                    || HasStrictScalar2DPack4Storage(inputs[0], input)
                    || pack4LinearMat
                : IsStrictAttentionMatMulInput(inputs[0], input);
            if (!inputStorage)
                return RejectStrictCommandBufferPack4Node("Softmax input storage does not prove the required LinearMat or Pack4 physical mapping.");

            var ncnnAxis = layer.GetInt(0, 0);
            if (ncnnAxis < 0)
                ncnnAxis += input.dims;
            if (ncnnAxis < 0 || ncnnAxis >= input.dims)
                return RejectStrictCommandBufferPack4Node("The Softmax axis is outside the descriptor-backed input rank.");

            var tensorAxis = MapNcnnAxisToTensorAxis(input.dims, ncnnAxis);
            if (pack4LinearMat && (tensorAxis < 0 || tensorAxis > 1))
                return RejectStrictCommandBufferPack4Node("Pack4 LinearMat Softmax supports only logical width or height axes.");

            var mode = layer.GetInt(10, 0);
            if (mode < 0 || mode > 2)
                return RejectStrictCommandBufferPack4Node("The Softmax mode is outside softmax/log-softmax/hardmax.");

            var modeName = mode == 1 ? "log-softmax" : mode == 2 ? "hardmax" : "softmax";
            return AcceptStrictCommandBufferPack4Node(
                layer,
                input,
                request,
                "command-buffer-pack4:"
                    + (pack4LinearMat ? "pack4-linear-mat-" : string.Empty)
                    + modeName
                    + "-axis-fp32-accumulate");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferSdpa(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_extraPacks.TryGetValue(layer.name, out var extra) || extra is not SdpaPack pack)
                return RejectStrictCommandBufferPack4Node("SDPA parameters were not loaded for this node.");
            if (pack.int8ScaleTerm)
                return RejectStrictCommandBufferPack4Node("SDPA int8 scale mode is outside the verified Pack4 profile.");

            var firstPastCacheInput = pack.attnMask ? 4 : 3;
            var expectedInputCount = firstPastCacheInput;
            var hasPastCache = false;
            if (pack.kvCache)
            {
                if (layer.topNames == null || layer.topNames.Length != 3)
                {
                    return RejectStrictCommandBufferPack4Node(
                        "SDPA kv-cache requires exactly attention, key-cache, and value-cache output blobs.");
                }

                if (shapes.Length == firstPastCacheInput + 2)
                {
                    hasPastCache = true;
                    expectedInputCount += 2;
                }
                else if (shapes.Length != firstPastCacheInput)
                {
                    return RejectStrictCommandBufferPack4Node(
                        "SDPA kv-cache accepts Q/K/V (and an optional mask) on the first step, or both texture-backed past key/value caches on later steps.");
                }
            }

            if (shapes.Length != expectedInputCount)
                return RejectStrictCommandBufferPack4Node("SDPA input count or int8 scale mode is outside the verified Pack4 profile.");
            if (!IsStrictAttentionMatMulInput(inputs[0], shapes[0])
                || !IsStrictAttentionMatMulInput(inputs[1], shapes[1])
                || !IsStrictAttentionMatMulInput(inputs[2], shapes[2]))
            {
                return RejectStrictCommandBufferPack4Node("SDPA Q/K/V require exact rank-3 Pack4 texture descriptors.");
            }
            if (shapes[0].dims != 3 || shapes[1].dims != 3 || shapes[2].dims != 3
                || shapes[0].w != shapes[1].w || shapes[1].h != shapes[2].h
                || shapes[0].c != shapes[2].c || shapes[0].c % shapes[1].c != 0 || shapes[1].h > 4096)
            {
                return RejectStrictCommandBufferPack4Node("SDPA Q/current-K/current-V dimensions are outside the verified Pack4 broadcast profile.");
            }
            var attentionKeyLength = shapes[1].h;
            var attentionValueLength = shapes[2].h;
            if (hasPastCache)
            {
                var pastKeyIndex = firstPastCacheInput;
                var pastValueIndex = pastKeyIndex + 1;
                var pastKey = shapes[pastKeyIndex];
                var pastValue = shapes[pastValueIndex];
                if (!IsStrictAttentionKvCacheInput(inputs[pastKeyIndex], pastKey)
                    || !IsStrictAttentionKvCacheInput(inputs[pastValueIndex], pastValue))
                {
                    return RejectStrictCommandBufferPack4Node(
                        "SDPA past key/value require rank-3 Pack4 Texture2DArray descriptors whose physical height is at least their logical sequence length.");
                }
                if (pastKey.w != shapes[1].w || pastKey.d != shapes[1].d || pastKey.c != shapes[1].c
                    || pastValue.w != shapes[2].w || pastValue.d != shapes[2].d || pastValue.c != shapes[2].c
                    || pastKey.h != pastValue.h)
                {
                    return RejectStrictCommandBufferPack4Node(
                        "SDPA past key/value logical shapes do not match the current key/value cache profile.");
                }

                attentionKeyLength += pastKey.h;
                attentionValueLength += pastValue.h;
                if (attentionKeyLength != attentionValueLength || attentionKeyLength > 4096)
                {
                    return RejectStrictCommandBufferPack4Node(
                        "SDPA concatenated key/value sequence length must match and remain within the verified 4096-token Pack4 profile.");
                }
            }
            if (pack.attnMask
                && (shapes[3].dims != 2 || shapes[3].w != attentionKeyLength || shapes[3].h != shapes[0].h
                    || !HasStrictScalar2DPack4Storage(inputs[3], shapes[3])))
            {
                return RejectStrictCommandBufferPack4Node("SDPA mask requires an exact Pack4 scalar [keyLength,queryLength] texture.");
            }

            var output = new BufferShape(3, shapes[2].w, shapes[0].h, 1, shapes[0].c);
            if (pack.kvCache)
            {
                var keyCache = new BufferShape(3, shapes[1].w, attentionKeyLength, 1, shapes[1].c);
                var valueCache = new BufferShape(3, shapes[2].w, attentionValueLength, 1, shapes[2].c);
                var cacheStorageHeight = hasPastCache
                    ? Math.Max(attentionKeyLength, AttentionKvCacheTextureCapacity)
                    : attentionKeyLength;
                var keyCacheStorage = new BufferShape(3, keyCache.w, cacheStorageHeight, 1, keyCache.c);
                var valueCacheStorage = new BufferShape(3, valueCache.w, cacheStorageHeight, 1, valueCache.c);
                return AcceptStrictCommandBufferPack4Outputs(
                    layer,
                    request,
                    "command-buffer-pack4:sdpa-mask-causal-kv-cache",
                    new[] { output, keyCache, valueCache },
                    new[] { output, keyCacheStorage, valueCacheStorage });
            }

            return AcceptStrictCommandBufferPack4Node(layer, output, request, "command-buffer-pack4:sdpa-mask-causal-no-kv-cache");
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferMultiHeadAttention(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (!_multiHeadAttention.TryGetValue(layer.name, out var pack) || pack == null)
                return RejectStrictCommandBufferPack4Node("MultiHeadAttention parameters were not loaded for this node.");
            if (pack.kvCache)
                return RejectStrictCommandBufferPack4Node("MultiHeadAttention kv-cache is not implemented for CommandBuffer Pack4 execution.");
            if (layer?.topNames == null || layer.topNames.Length != 1)
                return RejectStrictCommandBufferPack4Node("MultiHeadAttention CommandBuffer Pack4 execution requires exactly one output blob.");
            if (shapes.Length < 1 || shapes.Length > 4)
            {
                return RejectStrictCommandBufferPack4Node(
                    "MultiHeadAttention requires one self-attention input, two/three cross-attention inputs, or an optional attention-mask input.");
            }

            var queryIndex = -1;
            var keyIndex = -1;
            var valueIndex = -1;
            var maskIndex = -1;
            try
            {
                AexisMultiHeadAttentionLayer.ResolveBottomBlobIndices(
                    layer.bottomNames?.Length ?? 0,
                    pack.attnMask,
                    pack.kvCache,
                    out queryIndex,
                    out keyIndex,
                    out valueIndex,
                    out maskIndex);
                if (!HasStrictMultiHeadAttentionLinearInput(inputs, shapes, queryIndex)
                    || !HasStrictMultiHeadAttentionLinearInput(inputs, shapes, keyIndex)
                    || !HasStrictMultiHeadAttentionLinearInput(inputs, shapes, valueIndex))
                {
                    return RejectStrictCommandBufferPack4Node(
                        "MultiHeadAttention query/key/value require descriptor-backed LinearMat or Pack4-Linear rank-2 textures.");
                }

                var query = shapes[queryIndex];
                var key = shapes[keyIndex];
                var value = shapes[valueIndex];
                if (query.w != pack.qdim || key.w != pack.kdim || value.w != pack.vdim
                    || query.h <= 0 || key.h != query.h || value.h != query.h
                    || pack.embedDim <= 0 || pack.numHeads <= 0 || pack.embedDim % pack.numHeads != 0)
                {
                    return RejectStrictCommandBufferPack4Node("MultiHeadAttention query/key/value or head dimensions are outside the verified profile.");
                }

                if (maskIndex >= 0)
                {
                    if (!HasStrictMultiHeadAttentionLinearInput(inputs, shapes, maskIndex))
                        return RejectStrictCommandBufferPack4Node("MultiHeadAttention mask requires a descriptor-backed LinearMat or Pack4-Linear rank-2 texture.");
                    var mask = shapes[maskIndex];
                    if (mask.w != key.h || mask.h != query.h)
                        return RejectStrictCommandBufferPack4Node("MultiHeadAttention mask dimensions must be [key_sequence, query_sequence].");
                }
            }
            catch (InvalidOperationException exception)
            {
                return RejectStrictCommandBufferPack4Node("MultiHeadAttention input profile rejected: " + exception.Message);
            }

            var rows = shapes[queryIndex].h;
            var output = new BufferShape(2, pack.qdim, rows, 1, 1);
            var outputScalarStorage = new BufferShape(3, output.w, output.h, 1, 1);
            var packedOutputStorage = ResolvePack4LinearMatStorageShape(output);
            var useLegacyAttentionLayout = UseLegacyPack4AttentionLayout || PreserveLegacyFp32Execution;
            var verification = AcceptStrictCommandBufferPack4Node(
                layer,
                output,
                useLegacyAttentionLayout ? outputScalarStorage : packedOutputStorage,
                request,
                "command-buffer-pack4:mha-mask-no-kv-cache");
            if (!verification.accepted)
                return verification;

            // Keep the static arena in lockstep with TryExecuteCommandBufferTexturePath.
            // MHA uses scalar Pack4 intermediates for the projections, then CDHW
            // Pack4 arrays for attention. Inputs are materialized only when their
            // descriptor says LinearMat or Pack4-Linear; equivalent descriptor
            // aliases share that materialization exactly as the runtime does.
            var scratchDtype = IsStrictFp32ActivationIslandLayer(request, layer)
                ? "FP32"
                : ResolvePhysicalTextureDtype(request.targetDtype);
            var headDim = pack.embedDim / pack.numHeads;
            var scalar = new BufferShape(3, pack.embedDim, rows, 1, 1);
            var heads = new BufferShape(4, headDim, rows, 1, pack.numHeads);
            var keyTransposed = new BufferShape(4, rows, headDim, 1, pack.numHeads);
            var scores = new BufferShape(4, rows, rows, 1, pack.numHeads);
            var contextPermuted = new BufferShape(4, pack.numHeads, rows, 1, headDim);
            var scratch = new List<AexisTexturePlanTensorDescriptor>
            {
                CreateStrictPack4ScratchDescriptor(layer, "mha-q-projection", scalar, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-k-projection", scalar, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-v-projection", scalar, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-q-scaled", scalar, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-q-heads", heads, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-k-heads", heads, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-v-heads", heads, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-k-heads-transposed", keyTransposed, request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "mha-scores", scores, request, scratchDtype)
            };

            if (maskIndex >= 0)
            {
                scratch.Add(CreateStrictPack4ScratchDescriptor(
                    layer,
                    "mha-attention-mask-tiled",
                    scores,
                    request,
                    scratchDtype));
                scratch.Add(CreateStrictPack4ScratchDescriptor(
                    layer,
                    "mha-scores-biased",
                    scores,
                    request,
                    scratchDtype));
            }

            scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "mha-weights", scores, request, scratchDtype));
            scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "mha-context-heads", heads, request, scratchDtype));
            scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "mha-context-permuted", contextPermuted, request, scratchDtype));
            scratch.Add(CreateStrictPack4ScratchDescriptor(layer, "mha-context-flat", scalar, request, scratchDtype));
            scratch.Add(CreateStrictPack4ScratchDescriptor(
                layer,
                useLegacyAttentionLayout ? "mha-output-packed" : "mha-output-scalar",
                useLegacyAttentionLayout ? packedOutputStorage : outputScalarStorage,
                request,
                scratchDtype));

            // Query always materializes from the admitted LinearMat/Pack4-Linear
            // input. K and V use that same texture when the plan proves an alias.
            scratch.Add(CreateStrictPack4ScratchDescriptor(
                layer,
                "mha-q-scalar-input",
                new BufferShape(3, shapes[queryIndex].w, shapes[queryIndex].h, 1, 1),
                request,
                scratchDtype));
            if (!StrictMhaInputsShareStorage(inputs, queryIndex, keyIndex))
            {
                scratch.Add(CreateStrictPack4ScratchDescriptor(
                    layer,
                    "mha-k-scalar-input",
                    new BufferShape(3, shapes[keyIndex].w, shapes[keyIndex].h, 1, 1),
                    request,
                    scratchDtype));
            }
            if (!StrictMhaInputsShareStorage(inputs, queryIndex, valueIndex)
                && !StrictMhaInputsShareStorage(inputs, keyIndex, valueIndex))
            {
                scratch.Add(CreateStrictPack4ScratchDescriptor(
                    layer,
                    "mha-v-scalar-input",
                    new BufferShape(3, shapes[valueIndex].w, shapes[valueIndex].h, 1, 1),
                    request,
                    scratchDtype));
            }
            if (maskIndex >= 0)
            {
                scratch.Add(CreateStrictPack4ScratchDescriptor(
                    layer,
                    "mha-attention-mask-scalar-input",
                    new BufferShape(3, shapes[maskIndex].w, shapes[maskIndex].h, 1, 1),
                    request,
                    scratchDtype));
            }

            verification.scratch = scratch.ToArray();
            return verification;
        }

        private static bool StrictMhaInputsShareStorage(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            int firstIndex,
            int secondIndex)
        {
            if (firstIndex == secondIndex)
                return true;
            if (firstIndex < 0 || secondIndex < 0 || firstIndex >= inputs.Count || secondIndex >= inputs.Count)
                return false;
            var first = inputs[firstIndex];
            var second = inputs[secondIndex];
            return first != null
                && second != null
                && !string.IsNullOrWhiteSpace(first.aliasGroup)
                && string.Equals(first.aliasGroup, second.aliasGroup, StringComparison.Ordinal);
        }

        private static bool HasStrictMultiHeadAttentionLinearInput(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            IReadOnlyList<BufferShape> shapes,
            int index)
        {
            return index >= 0
                && index < inputs.Count
                && index < shapes.Count
                && shapes[index].dims == 2
                && (HasStrictLinearMatStorage(inputs[index], shapes[index])
                    || HasStrictPack4LinearMatStorage(inputs[index], shapes[index]));
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferRotaryEmbed(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);
            if (shapes.Length != 3)
                return RejectStrictCommandBufferPack4Node("RotaryEmbed requires source, cosine cache, and sine cache texture descriptors.");
            if (!_extraPacks.TryGetValue(layer.name, out var extra) || extra is not RotaryEmbedPack)
                return RejectStrictCommandBufferPack4Node("RotaryEmbed parameters were not loaded for this node.");

            var source = shapes[0];
            if (source.dims != 3 || source.d != 1 || source.w < 2 || (source.w & 1) != 0)
                return RejectStrictCommandBufferPack4Node("RotaryEmbed requires an even-width rank-3 [embed,sequence,head] Pack4 source texture.");
            if (!IsStrictAttentionMatMulInput(inputs[0], source)
                || !IsStrictAttentionMatMulInput(inputs[1], shapes[1])
                || !IsStrictAttentionMatMulInput(inputs[2], shapes[2]))
            {
                return RejectStrictCommandBufferPack4Node("RotaryEmbed source/cosine/sine inputs require exact rank-3 Pack4 Texture2DArray descriptors.");
            }

            var requiredCacheElements = (long)source.h * (source.w / 2);
            if (GetStrictPlanElementCount(shapes[1]) < requiredCacheElements
                || GetStrictPlanElementCount(shapes[2]) < requiredCacheElements)
            {
                return RejectStrictCommandBufferPack4Node("RotaryEmbed cosine/sine cache textures are smaller than sequence_length * embed_dim / 2.");
            }

            return AcceptStrictCommandBufferPack4Node(
                layer,
                source,
                request,
                "command-buffer-pack4:rotary-embed");
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferShortConv(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason) || shapes.Length != 3)
                return RejectStrictCommandBufferPack4Node("ShortConv requires exact weight, mixed activation, and cache texture descriptors: " + reason);
            if (layer?.topNames == null || layer.topNames.Length != 2)
                return RejectStrictCommandBufferPack4Node("ShortConv requires exactly activation and cache output blobs.");

            var weight = shapes[0];
            var mixed = shapes[1];
            var cache = shapes[2];
            if (weight.dims != 3 || weight.w <= 0 || weight.h != 1 || weight.d != 1 || weight.c != mixed.w
                || mixed.dims != 2 || mixed.w <= 0 || mixed.h <= 0
                || cache.dims != 2 || cache.w <= 0 || cache.h <= 0
                || cache.w * 4 != mixed.w || weight.w != cache.h)
            {
                return RejectStrictCommandBufferPack4Node(
                    "ShortConv requires Qwen's texture-native depthwise profile: weight=[kernel,1,groups], mixed=[groups,sequence], and cache=[groups/4,kernel].");
            }
            if (!HasStrictExactPack4Storage(inputs[0], weight)
                || !(HasStrictLinearMatStorage(inputs[1], mixed) || HasStrictPack4LinearMatStorage(inputs[1], mixed))
                || !HasStrictShortConvCacheStorage(inputs[2], cache))
            {
                return RejectStrictCommandBufferPack4Node(
                    "ShortConv requires immutable Pack4 weight storage, LinearMat or Pack4-Linear mixed storage, and the exact texture-native cache mapping.");
            }

            var mixedStorage = ResolvePack4LinearMatStorageShape(mixed);
            var cacheStorage = new BufferShape(3, cache.w, cache.h, 1, 4);
            return AcceptStrictCommandBufferPack4Outputs(
                layer,
                request,
                "command-buffer-pack4:shortconv-causal-texture",
                new[] { mixed, cache },
                new[] { mixedStorage, cacheStorage });
        }

        private static bool HasStrictShortConvCacheStorage(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape cache)
        {
            var storage = descriptor?.storageShape;
            if (storage == null || storage.Length != 5)
                return false;

            // The caller-owned initial cache is a Texture2DArray that exposes its
            // four lanes as logical scalar width.  After ShortConv, the same native
            // mapping is explicitly published as Packed4 storage.
            return (storage[0] == 2
                    && storage[1] == cache.w
                    && storage[2] == cache.h
                    && storage[3] == 1
                    && storage[4] == 1)
                || (storage[0] == 3
                    && storage[1] == cache.w
                    && storage[2] == cache.h
                    && storage[3] == 1
                    && (storage[4] == 1 || storage[4] == 4));
        }

        private AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferGatedDeltaRule(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason) || shapes.Length != 8)
                return RejectStrictCommandBufferPack4Node("GatedDeltaRule requires four scalar and four Pack4 state descriptors: " + reason);
            if (layer?.topNames == null || layer.topNames.Length != 2)
                return RejectStrictCommandBufferPack4Node("GatedDeltaRule requires exactly value and recurrent-state output blobs.");
            if (!_extraPacks.TryGetValue(layer.name, out var extra) || extra is not GatedDeltaRulePack pack
                || !IsStrictFinite(pack.epsilon) || pack.epsilon <= 0f)
            {
                return RejectStrictCommandBufferPack4Node("GatedDeltaRule requires a loaded finite positive recurrent epsilon.");
            }

            for (var index = 0; index < 4; index++)
            {
                var scalar = shapes[index];
                if ((scalar.dims != 1 && scalar.dims != 2)
                    || !(HasStrictLinearMatStorage(inputs[index], scalar) || HasStrictPack4LinearMatStorage(inputs[index], scalar)))
                {
                    return RejectStrictCommandBufferPack4Node(
                        "GatedDeltaRule scalar input " + index + " requires verified LinearMat or Pack4-Linear texture storage.");
                }
            }

            var query = shapes[4];
            var key = shapes[5];
            var value = shapes[6];
            var state = shapes[7];
            if (!IsStrictAttentionMatMulInput(inputs[4], query)
                || !IsStrictAttentionMatMulInput(inputs[5], key)
                || !IsStrictAttentionMatMulInput(inputs[6], value)
                || !IsStrictAttentionMatMulInput(inputs[7], state)
                || query.dims != 3 || key.dims != 3 || value.dims != 3 || state.dims != 3
                || query.w != key.w || query.h != key.h || query.c != key.c
                || value.h != query.h || value.c != query.c
                // The recurrent state stores four value elements in each float4
                // texel. Its width is consequently the Pack4 value width, while
                // Q/K/V retain the sequence-in-lane representation used by the
                // GatedDeltaRule kernel.
                || state.w * 4 != value.w || state.h != query.w || state.c != query.h)
            {
                return RejectStrictCommandBufferPack4Node(
                    "GatedDeltaRule requires exact Qwen Pack4 Q/K/V=[dimension,heads,sequence] and state=[ceil(valueDimension/4),keyDimension,heads] textures.");
            }

            return AcceptStrictCommandBufferPack4Outputs(
                layer,
                request,
                "command-buffer-pack4:gated-delta-rule-recurrent",
                new[] { value, state },
                new[] { CopyStrictStorage(inputs[6]), CopyStrictStorage(inputs[7]) });
        }

        private static AexisTextureExecutionPlanNodeVerification VerifyStrictCommandBufferDeepFillV2ContextualAttention(
            AexisGraphModel.Layer layer,
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
            AexisTextureExecutionPlanRequest request)
        {
            if (inputs == null || inputs.Count != 2)
                return RejectStrictCommandBufferPack4Node("DeepFillV2 contextual attention requires exactly feature and mask Pack4 texture inputs.");
            if (!TryGetStrictPlanShapes(inputs, out var shapes, out var reason))
                return RejectStrictCommandBufferPack4Node(reason);

            var feature = shapes[0];
            var mask = shapes[1];
            if (!HasStrictExactPack4Storage(inputs[0], feature) || !HasStrictExactPack4Storage(inputs[1], mask))
            {
                return RejectStrictCommandBufferPack4Node(
                    "DeepFillV2 contextual attention requires exact Texture2DArray Pack4 feature and mask descriptors; buffer or LinearMat activation fallback is prohibited.");
            }
            if (feature.dims != 3 || feature.w != 100 || feature.h != 128 || feature.d != 1 || feature.c != 96)
                return RejectStrictCommandBufferPack4Node("DeepFillV2 feature must be the verified d3:100x128x96 Pack4 tensor.");
            if (mask.dims != 3 || mask.w != 400 || mask.h != 512 || mask.d != 1 || mask.c != 1)
                return RejectStrictCommandBufferPack4Node("DeepFillV2 mask must be the verified d3:400x512x1 Pack4 tensor.");

            var ksize = layer.GetInt(0, 3);
            var rate = layer.GetInt(1, 2);
            var stride = layer.GetInt(2, 1);
            var softmaxScale = layer.GetFloat(3, 10f);
            var patchEpsilon = layer.GetFloat(4, 1e-4f);
            var maskDownsample = layer.GetInt(5, 8);
            if (ksize != 3 || rate != 2 || stride != 1 || maskDownsample != 8)
                return RejectStrictCommandBufferPack4Node("DeepFillV2 contextual attention supports only ksize=3, rate=2, stride=1, mask_downsample=8.");
            if (!IsStrictFinite(softmaxScale) || softmaxScale <= 0f || !IsStrictFinite(patchEpsilon) || patchEpsilon <= 0f)
                return RejectStrictCommandBufferPack4Node("DeepFillV2 contextual attention requires finite positive softmax_scale and patch_epsilon.");

            var verification = AcceptStrictCommandBufferPack4Node(
                layer,
                feature,
                CopyStrictStorage(inputs[0]),
                request,
                "command-buffer-pack4:deepfillv2-contextual-attention");
            if (!verification.accepted)
                return verification;

            var scratchDtype = ResolveStrictOutputTextureDtype(feature, CopyStrictStorage(inputs[0]), request);
            verification.scratch = new[]
            {
                CreateStrictPack4ScratchDescriptor(layer, "deepfill-patch-stats", new BufferShape(3, 50, 64, 1, 4), request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "deepfill-scores", new BufferShape(3, 50, 64, 1, 3200), request, scratchDtype),
                CreateStrictPack4ScratchDescriptor(layer, "deepfill-weights", new BufferShape(3, 50, 64, 1, 3200), request, scratchDtype)
            };
            return verification;
        }

        private static bool TryGetSingleStrictCdhwPlanShape(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
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
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, AexisTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase)
                && logicalShape.dims == 4
                && storage != null
                && storage.Length == 5
                && storage[0] == 4
                && storage[1] == logicalShape.w
                && storage[2] == logicalShape.h
                && storage[3] == logicalShape.d
                && storage[4] == logicalShape.c;
        }

        private static bool HasStrictExactPack4Storage(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, AexisTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase)
                && (logicalShape.dims == 3 || logicalShape.dims == 4)
                && storage != null
                && storage.Length == 5
                && storage[0] == logicalShape.dims
                && storage[1] == logicalShape.w
                && storage[2] == logicalShape.h
                && storage[3] == logicalShape.d
                && storage[4] == logicalShape.c;
        }

        private static bool HasStrictScalarPack4Storage(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            return descriptor != null
                && descriptor.textureBacked
                && string.Equals(descriptor.layout, AexisTexturePlanLayout.Packed4, StringComparison.OrdinalIgnoreCase)
                && (logicalShape.dims == 1 || logicalShape.dims == 2)
                && storage != null
                && storage.Length == 5
                && storage[0] == 3
                && storage[1] == logicalShape.w
                && storage[2] == (logicalShape.dims == 2 ? logicalShape.h : 1)
                && storage[3] == 1
                && storage[4] == 1;
        }

        private static bool HasStrictAnyRankLinearMatStorage(
            AexisTexturePlanTensorDescriptor descriptor,
            BufferShape logicalShape)
        {
            var storage = descriptor?.storageShape;
            var expected = ResolveLinearMatStorageShape(logicalShape);
            return descriptor != null
                && descriptor.textureBacked
                && storage != null
                && storage.Length == 5
                && storage[0] == expected.dims
                && storage[1] == expected.w
                && storage[2] == expected.h
                && storage[3] == expected.d
                && storage[4] == expected.c;
        }

        private static bool TryGetSingleStrictPlanShape(
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
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
            IReadOnlyList<AexisTexturePlanTensorDescriptor> inputs,
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
            AexisTexturePlanTensorDescriptor descriptor,
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

        private static bool TryGetSourceShape(AexisTexturePlanTensorDescriptor descriptor, out BufferShape shape)
        {
            shape = default;
            var source = descriptor?.sourceLogicalShape;
            if (source == null || source.Length != 5
                || source[0] < 1 || source[0] > 4
                || source[1] <= 0 || source[2] <= 0 || source[3] <= 0 || source[4] <= 0)
            {
                return false;
            }
            shape = new BufferShape(source[0], source[1], source[2], source[3], source[4]);
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

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Node(
            AexisGraphModel.Layer layer,
            BufferShape outputShape,
            AexisTextureExecutionPlanRequest request,
            string executionPath)
        {
            return AcceptStrictCommandBufferPack4Node(layer, outputShape, outputShape, request, executionPath);
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Node(
            AexisGraphModel.Layer layer,
            BufferShape logicalShape,
            BufferShape storageShape,
            AexisTextureExecutionPlanRequest request,
            string executionPath,
            string logicalDtype = null)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            if (outputNames.Length == 0)
                return RejectStrictCommandBufferPack4Node("The node has no output blobs.");
            if (!TryValidateStrictTextureOutput(logicalShape, storageShape, out var capacityReason))
                return RejectStrictCommandBufferPack4Node(capacityReason);
            var logical = new[] { logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c };
            var storage = new[] { storageShape.dims, storageShape.w, storageShape.h, storageShape.d, storageShape.c };
            var physicalDtype = ResolveStrictOutputTextureDtype(logicalShape, storageShape, request, layer);
            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = executionPath,
                reason = null,
                outputs = outputNames.Select((name, index) => new AexisTexturePlanTensorDescriptor
                {
                    blob = name,
                    logicalShape = (int[])logical.Clone(),
                    storageShape = (int[])storage.Clone(),
                    layout = request.targetLayout,
                    dtype = physicalDtype,
                    logicalDtype = string.IsNullOrWhiteSpace(logicalDtype) ? ResolveLogicalDtype(request.targetDtype) : logicalDtype,
                    aliasGroup = "computed:" + (layer?.name ?? layer?.typeName ?? "layer") + ":" + index,
                    textureBacked = true
                }).ToArray()
            };
        }

        private static AexisTexturePlanTensorDescriptor CreateStrictPack4ScratchDescriptor(
            AexisGraphModel.Layer layer,
            string suffix,
            BufferShape shape,
            AexisTextureExecutionPlanRequest request,
            string physicalDtype = "FP32")
        {
            var identity = GetStrictPack4ScratchIdentity(layer, suffix);
            return new AexisTexturePlanTensorDescriptor
            {
                blob = identity,
                logicalShape = new[] { shape.dims, shape.w, shape.h, shape.d, shape.c },
                storageShape = new[] { shape.dims, shape.w, shape.h, shape.d, shape.c },
                layout = request.targetLayout,
                // MVN reduction statistics use FP32 accumulation even when the
                // activation profile is FP16; this is still a Pack4 RT, not a
                // buffer-side fallback or readback.
                dtype = physicalDtype,
                logicalDtype = "Float32",
                aliasGroup = identity,
                textureBacked = true
            };
        }

        internal static string GetStrictPack4ScratchIdentity(AexisGraphModel.Layer layer, string suffix)
        {
            var layerName = string.IsNullOrWhiteSpace(layer?.name) ? (layer?.typeName ?? "layer") : layer.name;
            return "scratch:" + layerName + ":" + (suffix ?? string.Empty);
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4Outputs(
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanRequest request,
            string executionPath,
            BufferShape[] logicalShapes,
            BufferShape[] storageShapes)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            if (logicalShapes == null || storageShapes == null
                || outputNames.Length == 0 || outputNames.Length != logicalShapes.Length
                || logicalShapes.Length != storageShapes.Length)
            {
                return RejectStrictCommandBufferPack4Node(
                    "CommandBuffer Pack4 verifier output contract does not match the graph outputs.");
            }

            var outputs = new AexisTexturePlanTensorDescriptor[outputNames.Length];
            for (var index = 0; index < outputNames.Length; index++)
            {
                var logical = logicalShapes[index];
                var storage = storageShapes[index];
                if (!TryValidateStrictTextureOutput(logical, storage, out var capacityReason))
                {
                    return RejectStrictCommandBufferPack4Node(
                        "CommandBuffer Pack4 output " + index.ToString(CultureInfo.InvariantCulture) + " is invalid: " + capacityReason);
                }

                outputs[index] = new AexisTexturePlanTensorDescriptor
                {
                    blob = outputNames[index],
                    logicalShape = new[] { logical.dims, logical.w, logical.h, logical.d, logical.c },
                    storageShape = new[] { storage.dims, storage.w, storage.h, storage.d, storage.c },
                    layout = request.targetLayout,
                    dtype = ResolveStrictOutputTextureDtype(logicalShapes[index], storageShapes[index], request, layer),
                    logicalDtype = ResolveLogicalDtype(request.targetDtype),
                    aliasGroup = "computed:" + (layer?.name ?? layer?.typeName ?? "layer") + ":" + index,
                    textureBacked = true
                };
            }

            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = true,
                executionPath = executionPath,
                outputs = outputs
            };
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictDataIndexNode(
            AexisGraphModel.Layer layer,
            AexisTextureExecutionPlanRequest request,
            string executionPath,
            BufferShape[] logicalShapes,
            BufferShape[] storageShapes,
            string[] logicalDtypes)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            if (logicalShapes == null || storageShapes == null || logicalDtypes == null
                || outputNames.Length == 0 || outputNames.Length != logicalShapes.Length
                || logicalShapes.Length != storageShapes.Length || logicalShapes.Length != logicalDtypes.Length)
                return RejectStrictCommandBufferPack4Node("Data-index verifier output contract does not match the graph outputs.");

            var outputs = new AexisTexturePlanTensorDescriptor[outputNames.Length];
            for (var index = 0; index < outputNames.Length; index++)
            {
                var logical = logicalShapes[index];
                var storage = storageShapes[index];
                if (!TryValidateStrictTextureOutput(logical, storage, out var capacityReason))
                    return RejectStrictCommandBufferPack4Node("Data-index output " + index.ToString(CultureInfo.InvariantCulture) + " is invalid: " + capacityReason);
                outputs[index] = new AexisTexturePlanTensorDescriptor
                {
                    blob = outputNames[index],
                    logicalShape = new[] { logical.dims, logical.w, logical.h, logical.d, logical.c },
                    storageShape = new[] { storage.dims, storage.w, storage.h, storage.d, storage.c },
                    layout = request.targetLayout,
                    dtype = "FP32",
                    logicalDtype = string.IsNullOrWhiteSpace(logicalDtypes[index]) ? "Float32" : logicalDtypes[index],
                    aliasGroup = "computed:" + (layer?.name ?? layer?.typeName ?? "data-index") + ":" + index,
                    textureBacked = true
                };
            }
            return new AexisTextureExecutionPlanNodeVerification { accepted = true, executionPath = executionPath, outputs = outputs };
        }

        private static string ResolveLogicalDtype(string physicalDtype)
        {
            return string.Equals(physicalDtype, "FP16", StringComparison.OrdinalIgnoreCase) ? "Float16"
                : string.Equals(physicalDtype, "BF16", StringComparison.OrdinalIgnoreCase) ? "BFloat16"
                : string.Equals(physicalDtype, "FP32", StringComparison.OrdinalIgnoreCase) ? "Float32"
                : physicalDtype ?? string.Empty;
        }

        private static string ResolvePhysicalTextureDtype(string requestedDtype)
        {
            return string.Equals(requestedDtype, "BF16", StringComparison.OrdinalIgnoreCase) ? "FP32" : requestedDtype;
        }

        private static string ResolveStrictOutputTextureDtype(
            BufferShape logicalShape,
            BufferShape storageShape,
            AexisTextureExecutionPlanRequest request,
            AexisGraphModel.Layer layer = null)
        {
            if (IsStrictFp32ActivationIslandLayer(request, layer))
                return "FP32";
            var expectedLinearStorage = ResolveLinearMatStorageShape(logicalShape);
            if (StrictPlanShapesEqual(storageShape, expectedLinearStorage))
                return "FP32";
            return ResolvePhysicalTextureDtype(request.targetDtype);
        }

        private static bool IsStrictFp32ActivationIslandLayer(
            AexisTextureExecutionPlanRequest request,
            AexisGraphModel.Layer layer)
        {
            if (!string.Equals(request?.targetDtype, "FP16", StringComparison.OrdinalIgnoreCase)
                || string.IsNullOrWhiteSpace(layer?.name))
            {
                return false;
            }

            return (request.fp32ActivationLayerNames ?? Array.Empty<string>())
                .Any(name => string.Equals(name, layer.name, StringComparison.Ordinal));
        }

        private static AexisTextureExecutionPlanNodeVerification RejectStrictCommandBufferPack4Node(string reason)
        {
            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = false,
                reason = reason ?? "The loaded runtime profile rejected this CommandBuffer Pack4 node."
            };
        }

        private static bool TryValidateStrictTextureOutput(
            BufferShape logicalShape,
            BufferShape storageShape,
            out string reason)
        {
            if (logicalShape.dims < 1 || logicalShape.dims > 4
                || storageShape.dims < 1 || storageShape.dims > 4
                || logicalShape.w <= 0 || logicalShape.h <= 0 || logicalShape.d <= 0 || logicalShape.c <= 0
                || storageShape.w <= 0 || storageShape.h <= 0 || storageShape.d <= 0 || storageShape.c <= 0)
            {
                reason = "Strict texture output requires positive rank-1 through rank-4 logical and storage extents.";
                return false;
            }

            var logicalElements = GetStrictPlanElementCount(logicalShape);
            if (logicalElements > int.MaxValue)
            {
                reason = "Strict texture output logical element count exceeds the supported 32-bit shader descriptor range.";
                return false;
            }

            var maxTextureSize = GetMaxTextureSize();
            if (storageShape.w > maxTextureSize || storageShape.h > maxTextureSize)
            {
                reason = "Strict texture output storage extent exceeds SystemInfo.maxTextureSize="
                    + maxTextureSize.ToString(CultureInfo.InvariantCulture) + ".";
                return false;
            }

            if (storageShape.dims >= 3)
            {
                var packs = Math.Max(1L, (storageShape.c + 3L) / 4L);
                var slices = storageShape.dims == 4 ? packs * storageShape.d : packs;
                var maxSlices = GetMaxTextureArraySlices();
                if (slices > maxSlices)
                {
                    reason = "Strict texture output requires " + slices.ToString(CultureInfo.InvariantCulture)
                        + " Texture2DArray slices, exceeding SystemInfo.maxTextureArraySlices="
                        + maxSlices.ToString(CultureInfo.InvariantCulture) + ".";
                    return false;
                }
            }

            reason = null;
            return true;
        }

        private static AexisTextureExecutionPlanNodeVerification AcceptStrictCommandBufferPack4NoopAlias(
            AexisGraphModel.Layer layer,
            AexisTexturePlanTensorDescriptor source,
            AexisTextureExecutionPlanRequest request)
        {
            var outputNames = layer?.topNames ?? Array.Empty<string>();
            return new AexisTextureExecutionPlanNodeVerification
            {
                accepted = outputNames.Length > 0,
                usesDescriptorAlias = outputNames.Length > 0,
                executionPath = "descriptor-alias",
                reason = outputNames.Length > 0 ? null : "The node has no output blobs.",
                outputs = outputNames.Select(name => new AexisTexturePlanTensorDescriptor
                {
                    blob = name,
                    logicalShape = source.logicalShape == null ? Array.Empty<int>() : (int[])source.logicalShape.Clone(),
                    storageShape = source.storageShape == null ? Array.Empty<int>() : (int[])source.storageShape.Clone(),
                    layout = request.targetLayout,
                    dtype = ResolvePhysicalTextureDtype(request.targetDtype),
                    logicalDtype = source.logicalDtype,
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

        private void EnsureProductionDebugOverridesRejected()
        {
            if (!IsDebugOracleExecution && HasDebugOracleIntent())
            {
                throw new InvalidOperationException(
                    "ProductionTextureOnly rejects debug-oracle Buffer/CPU overrides. "
                    + "Select ExecutionMode=DebugOracle in a debug-oracle build for diagnostics; "
                    + "production inference has no Buffer fallback.");
            }
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

        internal void SetCurrentBufferExecutionContext(AexisLayerBufferContext context)
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

        // Binds compiler-assigned temporary graph activations to stable
        // CommandBuffer RT IDs. This deliberately does not invent a buffer
        // arena: every slot is a Pack4 RenderTexture/Texture2DArray resource.
        private sealed class CommandBufferRtArena : IDisposable
        {
            private sealed class ResourceState
            {
                public AexisTextureExecutionPlanResource resource;
                public bool bound;
            }

            private sealed class SlotState
            {
                public int slot;
                public int nameId;
                public bool allocated;
                public bool active;
                public bool persistent;
                public string trackerLabel;
                public AexisTextureExecutionPlanResource activeResource;
            }

            private readonly AexisGraphSession _owner;
            private readonly CommandBuffer _commandBuffer;
            private readonly List<ResourceState> _resources = new List<ResourceState>();
            private readonly Dictionary<int, SlotState> _slots = new Dictionary<int, SlotState>();
            private readonly Queue<string> _recentTransitions = new Queue<string>();
            private bool _disposed;

            public CommandBufferRtArena(AexisGraphSession owner, CommandBuffer commandBuffer, AexisTextureExecutionPlan plan)
            {
                _owner = owner ?? throw new ArgumentNullException(nameof(owner));
                _commandBuffer = commandBuffer ?? throw new ArgumentNullException(nameof(commandBuffer));
                Metrics = new CommandBufferRtArenaMetrics { enabled = true };
                foreach (var resource in plan?.memory?.resources ?? Array.Empty<AexisTextureExecutionPlanResource>())
                {
                    if (resource == null || resource.allocationSlot < 0
                        || resource.externalInput
                        || (!resource.temporary && !resource.producedByGraph))
                        continue;
                    _resources.Add(new ResourceState { resource = resource });
                    if (resource.temporary) Metrics.plannedTemporaryResources++;
                    else Metrics.plannedPersistentResources++;
                    if (_slots.ContainsKey(resource.allocationSlot))
                        continue;
                    _slots.Add(resource.allocationSlot, new SlotState
                    {
                        slot = resource.allocationSlot,
                        persistent = resource.persistent,
                        nameId = Shader.PropertyToID("_AexisStaticRtArena_" + Guid.NewGuid().ToString("N")),
                        trackerLabel = "AexisGraphSession.CommandBufferRtArena(slot="
                            + resource.allocationSlot.ToString(CultureInfo.InvariantCulture) + ")"
                    });
                }
                Metrics.plannedSlots = _slots.Count;
            }

            public CommandBufferRtArenaMetrics Metrics { get; }

            public bool TryRentArray(
                int layerIndex,
                int width,
                int height,
                int depth,
                RenderTextureFormat format,
                RenderTextureDescriptor descriptor,
                AexisTemporaryRtDescriptor temporaryDescriptor,
                string plannedResourceName,
                out ComputeTexture texture)
            {
                return TryRent(layerIndex, true, width, height, depth, format, descriptor, temporaryDescriptor, plannedResourceName, out texture);
            }

            public bool TryRentMat(
                int layerIndex,
                int width,
                int height,
                RenderTextureFormat format,
                RenderTextureDescriptor descriptor,
                AexisTemporaryRtDescriptor temporaryDescriptor,
                out ComputeTexture texture)
            {
                return TryRent(layerIndex, false, width, height, 1, format, descriptor, temporaryDescriptor, null, out texture);
            }

            private bool TryRent(
                int layerIndex,
                bool array,
                int width,
                int height,
                int depth,
                RenderTextureFormat format,
                RenderTextureDescriptor descriptor,
                AexisTemporaryRtDescriptor temporaryDescriptor,
                string plannedResourceName,
                out ComputeTexture texture)
            {
                texture = null;
                if (layerIndex < 0)
                    return false;

                var candidates = _resources.Where(state => !state.bound
                    && state.resource.firstLayerIndex == layerIndex
                    && (string.IsNullOrWhiteSpace(plannedResourceName)
                        || string.Equals(state.resource.representativeBlob, plannedResourceName, StringComparison.Ordinal))
                    && Matches(state.resource.descriptor, array, width, height, depth, format))
                    .ToArray();
                if (string.IsNullOrWhiteSpace(plannedResourceName) && candidates.Length > 1)
                {
                    throw new InvalidOperationException(
                        "Compiled RT arena has ambiguous unnamed allocation"
                        + " | layer=" + layerIndex.ToString(CultureInfo.InvariantCulture)
                        + " | storage=" + width.ToString(CultureInfo.InvariantCulture) + "x"
                        + height.ToString(CultureInfo.InvariantCulture) + "x"
                        + depth.ToString(CultureInfo.InvariantCulture)
                        + " | format=" + format
                        + " | candidates=" + string.Join(",", candidates.Select(state => state.resource.representativeBlob))
                        + " | required_action=bind the graph output or kernel scratch identity when renting the CommandBuffer Pack4 RT.");
                }
                var candidate = candidates.FirstOrDefault();
                if (candidate == null)
                    return false;

                if (!_slots.TryGetValue(candidate.resource.allocationSlot, out var slot))
                    throw new InvalidOperationException("Compiled RT arena slot is missing for " + candidate.resource.representativeBlob + ".");
                if (slot.active)
                {
                    var active = slot.activeResource;
                    throw new InvalidOperationException(
                        "Compiled RT arena liveness violation: slot=" + slot.slot.ToString(CultureInfo.InvariantCulture)
                        + " is reused before its prior graph activation was released."
                        + " | request_layer=" + layerIndex.ToString(CultureInfo.InvariantCulture)
                        + " | requested_blob=" + candidate.resource.representativeBlob
                        + " | requested_range=" + candidate.resource.firstLayerIndex.ToString(CultureInfo.InvariantCulture)
                        + ".." + candidate.resource.lastLayerIndex.ToString(CultureInfo.InvariantCulture)
                        + " | active_blob=" + (active?.representativeBlob ?? string.Empty)
                        + " | active_range=" + (active?.firstLayerIndex ?? -1).ToString(CultureInfo.InvariantCulture)
                        + ".." + (active?.lastLayerIndex ?? -1).ToString(CultureInfo.InvariantCulture)
                        + " | recent_arena_transitions=" + string.Join("; ", _recentTransitions));
                }
                if (!slot.allocated)
                {
                    _commandBuffer.GetTemporaryRT(slot.nameId, descriptor);
                    AexisGpuResourceTracker.RegisterTemporaryTextureHandle(slot.nameId, temporaryDescriptor);
                    slot.allocated = true;
                    Metrics.allocatedSlots++;
                }

                slot.active = true;
                slot.activeResource = candidate.resource;
                candidate.bound = true;
                Metrics.boundResources++;
                RecordTransition("rent layer=" + layerIndex.ToString(CultureInfo.InvariantCulture)
                    + " slot=" + slot.slot.ToString(CultureInfo.InvariantCulture)
                    + " blob=" + candidate.resource.representativeBlob);
                texture = new ComputeTexture
                {
                    nameID = slot.nameId,
                    width = width,
                    height = height,
                    depth = depth,
                    dimension = descriptor.dimension,
                    format = format,
                    trackerLabel = slot.trackerLabel + ":" + candidate.resource.representativeBlob,
                    isTemporary = true,
                    arenaSlot = slot.slot,
                    temporaryDescriptor = temporaryDescriptor
                };
                return true;
            }

            public bool TryReturn(ComputeTexture texture)
            {
                if (texture == null || texture.arenaSlot < 0)
                    return false;
                if (!_slots.TryGetValue(texture.arenaSlot, out var slot))
                    throw new InvalidOperationException("Unknown compiled RT arena slot " + texture.arenaSlot.ToString(CultureInfo.InvariantCulture) + ".");
                if (!slot.active)
                    throw new InvalidOperationException("Compiled RT arena slot was returned twice: " + texture.arenaSlot.ToString(CultureInfo.InvariantCulture) + ".");

                var releasedResource = slot.activeResource;
                slot.active = false;
                slot.activeResource = null;
                texture.isReleased = true;
                Metrics.releasedResources++;
                RecordTransition("return slot=" + slot.slot.ToString(CultureInfo.InvariantCulture)
                    + " blob=" + (releasedResource?.representativeBlob ?? string.Empty));
                if (slot.persistent)
                {
                    AexisGpuResourceTracker.ReleaseTextureHandle(texture.nameID, texture.trackerLabel ?? slot.trackerLabel);
                    _commandBuffer.ReleaseTemporaryRT(texture.nameID);
                    slot.allocated = false;
                }
                return true;
            }

            private void RecordTransition(string transition)
            {
                const int capacity = 16;
                if (_recentTransitions.Count == capacity)
                    _recentTransitions.Dequeue();
                _recentTransitions.Enqueue(transition);
            }

            public void RecordUnplannedTextureAllocation()
            {
                Metrics.unplannedTextureAllocations++;
            }

            public void Complete()
            {
                Metrics.allPlannedResourcesBound = _resources.All(resource => resource.bound);
                Metrics.allBoundResourcesReleased = _slots.Values
                    .Where(slot => !slot.persistent)
                    .All(slot => !slot.active);
                Metrics.unboundResourceDiagnostics = _resources
                    .Where(resource => !resource.bound)
                    .OrderBy(resource => resource.resource.firstLayerIndex)
                    .ThenBy(resource => resource.resource.representativeBlob, StringComparer.Ordinal)
                    .Select(resource => "slot=" + resource.resource.allocationSlot.ToString(CultureInfo.InvariantCulture)
                        + " | persistent=" + resource.resource.persistent
                        + " | blob=" + resource.resource.representativeBlob
                        + " | range=" + resource.resource.firstLayerIndex.ToString(CultureInfo.InvariantCulture)
                        + ".." + resource.resource.lastLayerIndex.ToString(CultureInfo.InvariantCulture))
                    .ToArray();
                Metrics.activeResourceDiagnostics = _slots.Values
                    .Where(slot => slot.active)
                    .OrderBy(slot => slot.slot)
                    .Select(slot => "slot=" + slot.slot.ToString(CultureInfo.InvariantCulture)
                        + " | persistent=" + slot.persistent
                        + " | blob=" + (slot.activeResource?.representativeBlob ?? string.Empty)
                        + " | range=" + (slot.activeResource?.firstLayerIndex ?? -1).ToString(CultureInfo.InvariantCulture)
                        + ".." + (slot.activeResource?.lastLayerIndex ?? -1).ToString(CultureInfo.InvariantCulture))
                    .ToArray();
                if (!Metrics.allPlannedResourcesBound || !Metrics.allBoundResourcesReleased)
                {
                    throw new InvalidOperationException(
                        "Compiled CommandBuffer RT arena did not satisfy the static graph lifetime proof"
                        + " | planned=" + Metrics.plannedTemporaryResources.ToString(CultureInfo.InvariantCulture)
                        + " | planned_persistent=" + Metrics.plannedPersistentResources.ToString(CultureInfo.InvariantCulture)
                        + " | bound=" + Metrics.boundResources.ToString(CultureInfo.InvariantCulture)
                        + " | released=" + Metrics.releasedResources.ToString(CultureInfo.InvariantCulture)
                        + " | active_slots=" + _slots.Values.Count(slot => slot.active).ToString(CultureInfo.InvariantCulture)
                        + " | active_resources=" + string.Join("; ", Metrics.activeResourceDiagnostics)
                        + " | unbound_resources=" + string.Join("; ", Metrics.unboundResourceDiagnostics));
                }
            }

            public void Dispose()
            {
                if (_disposed)
                    return;
                _disposed = true;
                foreach (var slot in _slots.Values)
                {
                    if (!slot.allocated || slot.persistent)
                        continue;
                    AexisGpuResourceTracker.ReleaseTextureHandle(slot.nameId, slot.trackerLabel);
                    _commandBuffer.ReleaseTemporaryRT(slot.nameId);
                }
            }

            private static bool Matches(
                AexisTexturePlanTensorDescriptor descriptor,
                bool array,
                int width,
                int height,
                int depth,
                RenderTextureFormat format)
            {
                var shape = descriptor?.storageShape;
                if (shape == null || shape.Length != 5 || shape[1] != width || shape[2] != height)
                    return false;
                var dims = shape[0];
                var packs = Math.Max(1, (shape[4] + 3) / 4);
                if (array)
                {
                    var expectedDepth = dims == 4 ? Math.Max(1, shape[3]) * packs : packs;
                    if (dims < 3 || expectedDepth != depth)
                        return false;
                }
                else if (dims > 2 || depth != 1)
                {
                    return false;
                }

                return string.Equals(descriptor.dtype, "FP16", StringComparison.OrdinalIgnoreCase)
                    ? format == RenderTextureFormat.ARGBHalf
                    : format == RenderTextureFormat.ARGBFloat || format == RenderTextureFormat.RFloat;
            }
        }

        internal void SetCurrentExecutingLayer(AexisGraphModel.Layer layer)
        {
            _currentExecutingLayerName = layer?.name;
            _currentExecutingLayerTypeName = layer?.typeName;
            _currentExecutingLayerIndex = Model?.layers?.IndexOf(layer) ?? -1;
            if (ModelManifest != null
                && UsesFp16WeightStorage
                && !TryVerifyFp16WeightStorage(layer, out var reason))
            {
                throw new StrictTextureInferencePlanException(new AexisTextureExecutionPlan
                {
                    targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                    targetDtype = "FP16",
                    targetLayout = AexisTexturePlanLayout.Packed4,
                    diagnostics = new[]
                    {
                        new AexisTextureExecutionPlanDiagnostic
                        {
                            layer = layer?.name ?? string.Empty,
                            operatorName = layer?.typeName ?? layer?.type.ToString() ?? string.Empty,
                            capabilityStatus = AexisOperatorCapabilityStatus.Partial,
                            code = "fp16-weight-profile-rejected",
                            reason = reason,
                            targetBackend = AexisOperatorCapabilityBackend.CommandBuffer,
                            targetDtype = "FP16",
                            targetLayout = AexisTexturePlanLayout.Packed4,
                            rejectedPaths = new[] { "FP32-weight", "Buffer", "materialize-from-buffer" },
                            recommendedAction = "Use a verified half4/packed-half kernel profile or select the FP32 manifest."
                        }
                    }
                });
            }
        }

        internal void ClearCurrentExecutingLayer()
        {
            _currentExecutingLayerName = null;
            _currentExecutingLayerTypeName = null;
            _currentExecutingLayerIndex = -1;
        }

        public sealed class CmdInferResult : IDisposable
        {
            private readonly Dictionary<string, CmdTensorRef> _blobs;
            private readonly Dictionary<string, BufferShape> _shapes;
            private readonly AexisGraphSession _owner;
            private readonly RenderTexture _readbackTexture;

            internal CmdInferResult(
                Dictionary<string, CmdTensorRef> blobs,
                Dictionary<string, BufferShape> shapes,
                AexisGraphSession owner,
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
            if (!IsDebugOracleExecution)
                return false;
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
            AexisTensorBuffer view,
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

        public AexisGraphSession(AexisOps ops)
        {
            _ops = ops ?? throw new ArgumentNullException(nameof(ops));
        }

        public void LoadModel(string paramText, AexisWeightReader br, Action<LoadProgress> onProgress = null)
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

            var pnnxModel = AexisGraphModelParser.Parse(pnnxParamText);
            return AexisGraphModelParser.MergeStringParamsByLayerName(Model, pnnxModel, overwriteExisting);
        }

        public async Task LoadModelAsync(
            string paramText,
            AexisWeightReader br,
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
                    // The sample batch harness drives its own UniTask loop and cannot
                    // resume a BCL Task.Yield continuation. Batch loading therefore
                    // keeps the same layer loop but runs it synchronously.
                    if (shouldYield && !Application.isBatchMode)
                        await AexisAsync.YieldFrame();
                }
            }
            catch
            {
                try { Release(); } catch { }
                throw;
            }
        }

        private IEnumerable<LoadProgress> EnumerateModelLoad(string paramText, AexisWeightReader br)
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
            _fp32IndexSelectionProducerLayers = null;
            _fp32SensitiveInputProducerLayers = null;
            _fp32ActivationIslandLayers = null;
            Model = AexisGraphModelParser.Parse(paramText);
            LayerRepros = AexisLayerFactory.CreateModelLayers(Model?.layers);
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
            long managedBytesSinceCollection = 0;
            for (var i = 0; i < totalLayers; i++)
            {
                var layer = Model.layers[i];
                var layerSw = Stopwatch.StartNew();
                var layerRepro = LayerRepros != null && i < LayerRepros.Count ? LayerRepros[i] : null;
                var metrics = layerRepro != null ? layerRepro.LoadLayer(this, layer, br) : LoadLayer(layer, br);
                layerSw.Stop();
                totalLoadMs += layerSw.ElapsedMilliseconds;

                AccumulateLayerProfile(profile, layer?.typeName, metrics, layerSw.ElapsedMilliseconds);
                managedBytesSinceCollection = checked(managedBytesSinceCollection + Math.Max(0L, metrics.bytesRead));
                if (ManagedLoadGarbageCollectionIntervalBytes > 0
                    && managedBytesSinceCollection >= ManagedLoadGarbageCollectionIntervalBytes)
                {
                    var cleanupSw = Stopwatch.StartNew();
                    GC.Collect(GC.MaxGeneration, GCCollectionMode.Forced, true, false);
                    cleanupSw.Stop();
                    profile.managedCleanupCount++;
                    profile.managedCleanupMs += cleanupSw.ElapsedMilliseconds;
                    totalLoadMs += cleanupSw.ElapsedMilliseconds;
                    managedBytesSinceCollection = 0;
                }

                var progress01 = totalLayers > 0
                    ? 0.05f + 0.94f * ((float)(i + 1) / totalLayers)
                    : 0.99f;
                yield return new LoadProgress("layer", i + 1, totalLayers, layer?.name, layer?.typeName, progress01);
            }

            profile.totalMs = totalLoadMs;
            yield return new LoadProgress("complete", totalLayers, totalLayers, null, null, 1f);
        }

        private LayerLoadMetrics LoadLayer(AexisGraphModel.Layer layer, AexisWeightReader br)
        {
            if (layer == null)
                return default;

            var bytesStart = br.Position;
            long readMs = 0;
            long uploadMs = 0;
            long packMs = 0;
            var phaseSw = new Stopwatch();

            if (layer.type == AexisLayerTypes.Convolution
                || layer.type == AexisLayerTypes.ConvolutionDepthWise)
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
                pack.isDepthWise = layer.type == AexisLayerTypes.ConvolutionDepthWise;

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
                if (UsesInt4WeightOnlyForLayer(layer))
                    UploadInt4WeightOnlyConvWeights(pack, w, b, layer.name);
                else if (UsesInt8WeightOnlyForLayer(layer))
                    UploadInt8WeightOnlyConvWeights(pack, w, b, layer.name);
                else
                    UploadRawConvWeights(pack, w, b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                if (needGeneralTexturePack && !UsesQuantizedWeightsForLayer(layer))
                {
                    phaseSw.Restart();
                    var w4 = PackWeightsToO4I4K(w, pack.outC, pack.inC, pack.kernelW, pack.outPacks, pack.inPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(pack.packedWeight4, w4.Length, sizeof(float) * 4, "AexisGraphSession.ConvPackedWeight4:" + layer.name);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "AexisGraphSession.ConvPackedBias4:" + layer.name);
                    pack.packedWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);
                    if (UsesFp16WeightStorage)
                        pack.packedWeight4Fp16 = NewFp16Vector4Buffer(w4, "AexisGraphSession.ConvPackedWeight4Fp16:" + layer.name);

                    if (EnableWinograd23
                        && pack.kernelW == 3
                        && pack.kernelH == 3
                        && pack.strideW == 1
                        && pack.strideH == 1
                        && pack.padLeft == 1
                        && pack.padRight == 1
                        && pack.padTop == 1
                        && pack.padBottom == 1
                        && AexisWinograd23.CanUse(pack.kernelW, pack.padLeft, pack.inPacks, pack.outPacks))
                    {
                        pack.useWinograd23 = true;
                        var wTm = AexisWinograd23.PackWeightTm23(w, pack.outC, pack.inC, pack.outPacks, pack.inPacks);
                        pack.packedWeightTm23 = new ComputeBuffer(wTm.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                        AexisGpuResourceTracker.RegisterBuffer(pack.packedWeightTm23, wTm.Length, sizeof(float) * 4, "AexisGraphSession.ConvPackedWeightTm23:" + layer.name);
                        pack.packedWeightTm23.SetData(wTm);
                    }
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }
                else if (needDepthWiseTexturePack && !UsesQuantizedWeightsForLayer(layer))
                {
                    phaseSw.Restart();
                    var w4 = PackDepthWiseWeightsToP4KhKw(w, pack.outC, pack.kernelW, pack.kernelH, pack.outPacks);
                    var b4 = PackBiasToO4(b, pack.outC, pack.outPacks);
                    pack.packedDepthWiseWeight4 = new ComputeBuffer(w4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(pack.packedDepthWiseWeight4, w4.Length, sizeof(float) * 4, "AexisGraphSession.ConvPackedDepthWiseWeight4:" + layer.name);
                    pack.packedBias4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(pack.packedBias4, b4.Length, sizeof(float) * 4, "AexisGraphSession.ConvPackedBias4:" + layer.name);
                    pack.packedDepthWiseWeight4.SetData(w4);
                    pack.packedBias4.SetData(b4);
                    if (UsesFp16WeightStorage)
                        pack.packedDepthWiseWeight4Fp16 = NewFp16Vector4Buffer(w4, "AexisGraphSession.ConvPackedDepthWiseWeight4Fp16:" + layer.name);
                    phaseSw.Stop();
                    packMs += phaseSw.ElapsedMilliseconds;
                }

                _conv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.Deconvolution)
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
                AexisGpuResourceTracker.RegisterBuffer(pack.rawWeight, w.Length, sizeof(float), "AexisGraphSession.DeconvRawWeight:" + layer.name);
                pack.rawBias = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                AexisGpuResourceTracker.RegisterBuffer(pack.rawBias, b.Length, sizeof(float), "AexisGraphSession.DeconvRawBias:" + layer.name);
                pack.rawWeight.SetData(w);
                pack.rawBias.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _deconv[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.InnerProduct)
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
                if (UsesInt4WeightOnlyForLayer(layer))
                {
                    var quantized = NewInt4WeightOnlyUpload(
                        w,
                        ip.outFeatures,
                        ip.inFeatures,
                        outputChannelsAreContiguous: true,
                        "AexisGraphSession.InnerProductInt4WeightOnly:" + layer.name);
                    ip.wInt4Packed = quantized.packedWeights;
                    ip.wInt4Scales = quantized.scales;
                    ip.int4ScaleBlockSize = quantized.scaleBlockSize;
                }
                else if (UsesInt8WeightOnlyForLayer(layer))
                {
                    var quantized = NewInt8WeightOnlyUpload(
                        w,
                        ip.outFeatures,
                        ip.inFeatures,
                        outputChannelsAreContiguous: true,
                        "AexisGraphSession.InnerProductInt8WeightOnly:" + layer.name);
                    ip.wInt8Packed = quantized.packedWeights;
                    ip.wInt8Scales = quantized.scales;
                }
                else
                {
                    ip.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(ip.w, w.Length, sizeof(float), "AexisGraphSession.InnerProductWeight:" + layer.name);
                    ip.w.SetData(w);
                }
                if (UsesFp16WeightStorage)
                    ip.wFp16 = NewFp16Buffer(w, "AexisGraphSession.InnerProductWeightFp16:" + layer.name);
                ip.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                AexisGpuResourceTracker.RegisterBuffer(ip.b, b.Length, sizeof(float), "AexisGraphSession.InnerProductBias:" + layer.name);
                ip.b.SetData(b);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _innerProduct[layer.name] = ip;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.Gemm)
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
                    throw new InvalidOperationException("Gemm constantA is not supported in AexisGraphSession: " + layer.name);
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
                if (UsesInt4WeightOnlyForLayer(layer))
                {
                    var quantized = NewInt4WeightOnlyUpload(
                        gp.bDataCpu,
                        gp.constantN,
                        gp.constantK,
                        outputChannelsAreContiguous: gp.transB,
                        "AexisGraphSession.GemmInt4WeightOnly:" + layer.name);
                    gp.bDataInt4Packed = quantized.packedWeights;
                    gp.bDataInt4Scales = quantized.scales;
                    gp.int4ScaleBlockSize = quantized.scaleBlockSize;
                }
                else if (UsesInt8WeightOnlyForLayer(layer))
                {
                    var quantized = NewInt8WeightOnlyUpload(
                        gp.bDataCpu,
                        gp.constantN,
                        gp.constantK,
                        outputChannelsAreContiguous: gp.transB,
                        "AexisGraphSession.GemmInt8WeightOnly:" + layer.name);
                    gp.bDataInt8Packed = quantized.packedWeights;
                    gp.bDataInt8Scales = quantized.scales;
                }
                else
                {
                    gp.bData = NewBuffer(gp.bDataCpu);
                }
                if (UsesFp16WeightStorage)
                    gp.bDataFp16 = NewFp16Buffer(gp.bDataCpu, "AexisGraphSession.GemmWeightFp16:" + layer.name);
                if (gp.cDataCpu != null)
                    gp.cData = NewBuffer(gp.cDataCpu);
                if (UsesQuantizedWeightsForLayer(layer))
                    gp.bDataCpu = null;
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _gemm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.MemoryData)
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
                AexisGpuResourceTracker.RegisterBuffer(buf, a.Length, sizeof(float), "AexisGraphSession.MemoryData:" + layer.name);
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

            if (layer.type == AexisLayerTypes.Embed)
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
                    b = br.ReadTensorAsFloat32(ep.numOutput, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                ep.w = new ComputeBuffer(w.Length, sizeof(float), ComputeBufferType.Structured);
                AexisGpuResourceTracker.RegisterBuffer(ep.w, w.Length, sizeof(float), "AexisGraphSession.EmbedWeight:" + layer.name);
                ep.w.SetData(w);
                if (b != null)
                {
                    ep.b = new ComputeBuffer(b.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(ep.b, b.Length, sizeof(float), "AexisGraphSession.EmbedBias:" + layer.name);
                    ep.b.SetData(b);
                }
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;

                _embed[layer.name] = ep;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.LayerNorm)
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
                    gamma = br.ReadTensorAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    beta = br.ReadTensorAsFloat32(lp.affineSize, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    lp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(lp.gamma, gamma.Length, sizeof(float), "AexisGraphSession.LayerNormGamma:" + layer.name);
                    lp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(lp.beta, beta.Length, sizeof(float), "AexisGraphSession.LayerNormBeta:" + layer.name);
                    lp.gamma.SetData(gamma);
                    lp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _layerNorm[layer.name] = lp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.GroupNorm)
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
                    gamma = br.ReadTensorAsFloat32(gp.channels, 0, 0, 0, 1);
                    beta = br.ReadTensorAsFloat32(gp.channels, 0, 0, 0, 1);
                    phaseSw.Stop();
                    readMs += phaseSw.ElapsedMilliseconds;

                    phaseSw.Restart();
                    gp.gamma = new ComputeBuffer(gamma.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(gp.gamma, gamma.Length, sizeof(float), "AexisGraphSession.GroupNormGamma:" + layer.name);
                    gp.beta = new ComputeBuffer(beta.Length, sizeof(float), ComputeBufferType.Structured);
                    AexisGpuResourceTracker.RegisterBuffer(gp.beta, beta.Length, sizeof(float), "AexisGraphSession.GroupNormBeta:" + layer.name);
                    gp.gamma.SetData(gamma);
                    gp.beta.SetData(beta);
                    phaseSw.Stop();
                    uploadMs += phaseSw.ElapsedMilliseconds;
                }

                _groupNorm[layer.name] = gp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.BatchNorm)
            {
                var bp = new BatchNormPack();
                bp.channels = layer.GetInt(0, 0);

                phaseSw.Restart();
                var slope = br.ReadTensorAsFloat32(bp.channels, 0, 0, 0, 1);
                var mean = br.ReadTensorAsFloat32(bp.channels, 0, 0, 0, 1);
                var variance = br.ReadTensorAsFloat32(bp.channels, 0, 0, 0, 1);
                var bias = br.ReadTensorAsFloat32(bp.channels, 0, 0, 0, 1);
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
                AexisGpuResourceTracker.RegisterBuffer(bp.biasA4, a4.Length, sizeof(float) * 4, "AexisGraphSession.BatchNormBiasA4:" + layer.name);
                bp.scaleB4 = new ComputeBuffer(b4.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                AexisGpuResourceTracker.RegisterBuffer(bp.scaleB4, b4.Length, sizeof(float) * 4, "AexisGraphSession.BatchNormScaleB4:" + layer.name);
                bp.biasA4.SetData(a4);
                bp.scaleB4.SetData(b4);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
                packMs += phaseSw.ElapsedMilliseconds;

                _batchNorm[layer.name] = bp;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.Bias)
            {
                var pack = new BiasPack { channels = layer.GetInt(0, 0) };
                if (pack.channels <= 0)
                    throw new InvalidOperationException("Bias requires a positive bias_data_size: " + layer.name);

                phaseSw.Restart();
                var bias = br.ReadTensorAsFloat32(pack.channels, 0, 0, 0, 1);
                phaseSw.Stop();
                readMs += phaseSw.ElapsedMilliseconds;

                phaseSw.Restart();
                var packed = PackBiasToO4(bias, pack.channels, (pack.channels + 3) / 4);
                pack.bias4 = new ComputeBuffer(packed.Length, sizeof(float) * 4, ComputeBufferType.Structured);
                AexisGpuResourceTracker.RegisterBuffer(pack.bias4, packed.Length, sizeof(float) * 4, "AexisGraphSession.Bias4:" + layer.name);
                pack.bias4.SetData(packed);
                phaseSw.Stop();
                uploadMs += phaseSw.ElapsedMilliseconds;
                packMs += phaseSw.ElapsedMilliseconds;

                _bias[layer.name] = pack;
                return new LayerLoadMetrics(Math.Max(0, br.Position - bytesStart), readMs, uploadMs, packMs);
            }

            if (layer.type == AexisLayerTypes.MultiHeadAttention)
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
                var qB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var kW = ReadClipMatAsFloat32(br, mp.embedDim * mp.kdim, 0, 0, 0, 0);
                var kB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var vW = ReadClipMatAsFloat32(br, mp.embedDim * mp.vdim, 0, 0, 0, 0);
                var vB = br.ReadTensorAsFloat32(mp.embedDim, 0, 0, 0, 1);
                var oW = ReadClipMatAsFloat32(br, mp.qdim * mp.embedDim, 0, 0, 0, 0);
                var oB = br.ReadTensorAsFloat32(mp.qdim, 0, 0, 0, 1);
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

            var bufferInputs = new Dictionary<string, AexisTensorBuffer>(StringComparer.Ordinal);
            foreach (var kv in inputBuffers)
            {
                if (kv.Value == null)
                    throw new ArgumentNullException("inputBuffers[\"" + kv.Key + "\"]");
                bufferInputs[kv.Key] = new AexisTensorBuffer(kv.Value, 1, kv.Value.count, 1, 1, 1, false);
            }

            return InferWithMultiInputs(null, bufferInputs, null, null, stopAfterTopName);
        }

        public InferResult InferWithMultiInputs(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null)
        {
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            EnsureProductionDebugOverridesRejected();
            if ((textureInputs == null || textureInputs.Count == 0) && (bufferInputs == null || bufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
                return InferWithMultiInputsByLayerRepros(textureInputs, bufferInputs, pinnedNames, textureInputShapes, stopAfterTopName, startAtTopName);

            return null;
        }

        /// <summary>
        /// Executes the same strict texture-backed graph plan as <see cref="InferWithMultiInputs"/>,
        /// yielding between small groups of layers so Player UI events and cancellation can run.
        /// </summary>
        public Task<InferResult> InferWithMultiInputsAsync(
            Dictionary<string, RenderTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> bufferInputs,
            ICollection<string> pinnedNames = null,
            Dictionary<string, BufferShape> textureInputShapes = null,
            string stopAfterTopName = null,
            string startAtTopName = null,
            CancellationToken cancellationToken = default,
            int yieldEveryLayers = 12)
        {
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            EnsureProductionDebugOverridesRejected();
            if ((textureInputs == null || textureInputs.Count == 0) && (bufferInputs == null || bufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));
            if (LayerRepros != null && LayerRepros.Count == Model.layers.Count)
            {
                return InferWithMultiInputsByLayerReprosAsync(
                    textureInputs,
                    bufferInputs,
                    pinnedNames,
                    textureInputShapes,
                    stopAfterTopName,
                    startAtTopName,
                    cancellationToken,
                    yieldEveryLayers);
            }

            return Task.FromResult<InferResult>(null);
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
            EnsureCommandBufferTextureExecutionPlan(
                new Dictionary<string, ComputeTexture>(StringComparer.Ordinal) { [inputBlobName] = inputPack4 },
                new Dictionary<string, BufferShape>(StringComparer.Ordinal) { [inputBlobName] = legacyInputLogicalShape });
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
                if (l.type == AexisLayerTypes.Input)
                    continue;

                if (l.type == AexisLayerTypes.Split)
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

                if (l.type == AexisLayerTypes.Concat)
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

                if (l.type == AexisLayerTypes.Reshape)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    blobs[l.topNames[0]] = src;
                    src.refs++;
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.Padding)
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

                if (l.type == AexisLayerTypes.Pooling)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var poolingType = l.GetInt(0, 0);
                    var kernelW = l.GetInt(1, 0);
                    var kernelH = l.GetInt(11, kernelW);
                    var strideW = l.GetInt(2, 1);
                    var strideH = l.GetInt(12, strideW);
                    var padLeft = l.GetInt(3, 0);
                    var padRight = l.GetInt(14, padLeft);
                    var padTop = l.GetInt(13, padLeft);
                    var padBottom = l.GetInt(15, padTop);
                    var globalPooling = l.GetInt(4, 0);
                    var padMode = l.GetInt(5, 0);
                    var includePad = l.GetInt(6, 0) != 0;
                    var adaptivePooling = l.GetInt(7, 0);
                    if (globalPooling != 0 || adaptivePooling != 0)
                        throw new InvalidOperationException("Pooling(global/adaptive) not supported");

                    if (!AexisPoolingLayer.TryResolvePack4Geometry(
                        src.width, src.height, kernelW, kernelH, strideW, strideH,
                        padLeft, padRight, padTop, padBottom, padMode,
                        out padLeft, out _, out padTop, out _, out var outW, out var outH, out var reason))
                        throw new InvalidOperationException("Pooling geometry is invalid: " + reason);
                    var outArr = RentTempArray(cmd, outW, outH, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.PoolingPack4(cmd, src.texture, src.packs, kernelW, kernelH, strideW, strideH, padLeft, padTop, poolingType, outArr, includePad);
                    blobs[l.topNames[0]] = new CmdTensorRef { texture = outArr, width = outW, height = outH, packs = src.packs, refs = 1, owned = true };
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.Softmax)
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

                if (l.type == AexisLayerTypes.Convolution || l.type == AexisLayerTypes.ConvolutionDepthWise)
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

                if (l.type == AexisLayerTypes.Deconvolution || l.type == AexisLayerTypes.DeconvolutionDepthWise)
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

                if (l.type == AexisLayerTypes.Eltwise)
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

                if (l.type == AexisLayerTypes.BinaryOp)
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

                if (l.type == AexisLayerTypes.UnaryOp)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var srcShape = GetCmdShape(null, blobs, l.bottomNames[0]);
                    var storageShape = GetCmdStorageShape(src, srcShape);
                    var opType = l.GetInt(0, 0);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, src.texture.format);
                    _ops.UnaryOpPack4(cmd, src.texture, src.packs, opType, outArr, srcShape.dims >= 3 ? srcShape.c : 0);
                    blobs[l.topNames[0]] = CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: l.topNames[0]);
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.Swish)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var srcShape = GetCmdShape(null, blobs, l.bottomNames[0]);
                    var storageShape = GetCmdStorageShape(src, srcShape);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SwishPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: l.topNames[0]);
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.Sigmoid)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var srcShape = GetCmdShape(null, blobs, l.bottomNames[0]);
                    var storageShape = GetCmdStorageShape(src, srcShape);
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.SigmoidPack4(cmd, src.texture, src.packs, outArr);
                    blobs[l.topNames[0]] = CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: l.topNames[0]);
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.GELU)
                {
                    var src = GetCmdTensor(blobs, l.bottomNames[0]);
                    var srcShape = GetCmdShape(null, blobs, l.bottomNames[0]);
                    var storageShape = GetCmdStorageShape(src, srcShape);
                    var fast = l.GetInt(0, 0) != 0;
                    var outArr = RentTempArray(cmd, src.width, src.height, src.packs, RenderTextureFormat.ARGBHalf);
                    _ops.GeluPack4(cmd, src.texture, src.packs, fast, outArr);
                    blobs[l.topNames[0]] = CreateCmdTensorRef(outArr, srcShape, storageShape, owned: true, blobName: l.topNames[0]);
                    ConsumeCmd(cmd, blobs, remaining, l.bottomNames, pinnedNames);
                    continue;
                }

                if (l.type == AexisLayerTypes.Interp)
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
            if (textureInputs == null)
                throw new ArgumentNullException(nameof(textureInputs));
            if (textureInputs.Count == 0 && Model != null && Model.layers != null
                && Model.layers.Any(layer => layer?.bottomNames != null && layer.bottomNames.Any(name => !string.IsNullOrWhiteSpace(name))))
            {
                throw new ArgumentException(
                    "A zero-input CommandBuffer invocation is valid only for a statically closed texture plan; this model declares external activation bottoms.",
                    nameof(textureInputs));
            }
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

        /// <summary>
        /// Records strict Pack4 CommandBuffer inference with texture inputs plus
        /// optional fixed Buffer upload sources. Each fixed buffer is converted to
        /// a temporary GPU texture before the first layer; no model activation is
        /// permitted to remain in a ComputeBuffer.
        /// </summary>
        public ComputeTexture ForwardPack4WithFixedInputs(
            CommandBuffer cmd,
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, AexisTensorBuffer> fixedBufferInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            out BufferShape outputLogicalShape,
            ICollection<string> pinnedNames = null,
            string outputBlobName = null,
            string stopAfterTopName = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if ((textureInputs == null || textureInputs.Count == 0)
                && (fixedBufferInputs == null || fixedBufferInputs.Count == 0))
                throw new ArgumentNullException(nameof(textureInputs));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            EnsureProductionDebugOverridesRejected();
            if (LayerRepros == null || LayerRepros.Count != Model.layers.Count)
                throw new InvalidOperationException("ForwardPack4WithFixedInputs requires complete Pack4 CommandBuffer layer repros.");

            return ForwardPack4ByLayerRepros(
                cmd,
                textureInputs,
                textureInputShapes,
                out outputLogicalShape,
                pinnedNames,
                outputBlobName,
                stopAfterTopName,
                null,
                null,
                null,
                fixedBufferInputs);
        }

        public CommandBufferInferResult ForwardPack4Outputs(
            CommandBuffer cmd,
            Dictionary<string, ComputeTexture> textureInputs,
            Dictionary<string, BufferShape> textureInputShapes,
            ICollection<string> outputBlobNames,
            ICollection<string> pinnedNames = null)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (textureInputs == null)
                throw new ArgumentNullException(nameof(textureInputs));
            if (textureInputs.Count == 0 && Model != null && Model.layers != null
                && Model.layers.Any(layer => layer?.bottomNames != null && layer.bottomNames.Any(name => !string.IsNullOrWhiteSpace(name))))
            {
                throw new ArgumentException(
                    "A zero-input CommandBuffer invocation is valid only for a statically closed texture plan; this model declares external activation bottoms.",
                    nameof(textureInputs));
            }
            if (outputBlobNames == null)
                throw new ArgumentNullException(nameof(outputBlobNames));
            if (Model == null || _blobUseCount == null)
                throw new InvalidOperationException("model not loaded");
            if (LayerRepros == null || LayerRepros.Count != Model.layers.Count)
                throw new InvalidOperationException("Multi-output CommandBuffer execution requires the verified LayerRepro runtime path.");

            var names = outputBlobNames
                .Where(name => !string.IsNullOrWhiteSpace(name))
                .Distinct(StringComparer.Ordinal)
                .ToArray();
            if (names.Length == 0)
                throw new ArgumentException("At least one named CommandBuffer output is required.", nameof(outputBlobNames));

            var retained = new Dictionary<string, ComputeTexture>(StringComparer.Ordinal);
            var retainedShapes = new Dictionary<string, BufferShape>(StringComparer.Ordinal);
            ForwardPack4ByLayerRepros(
                cmd,
                textureInputs,
                textureInputShapes,
                out _,
                pinnedNames,
                names[0],
                null,
                names,
                retained,
                retainedShapes);
            return new CommandBufferInferResult(this, cmd, retained, retainedShapes);
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
            EnsureNcnnTextureTensor(tr);
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
                EnsureNcnnTextureTensor(tensor);
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
            return "AexisGraphSession.RentTempBuffer(" + (member ?? "?") + ":" + line.ToString(CultureInfo.InvariantCulture) + ")";
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
            AexisGpuResourceTracker.RegisterBuffer(buffer, Mathf.Max(1, count), Math.Max(1, stride), GetTempBufferLabel(callerMember, callerLine));
            return buffer;
        }

        internal void ReturnTempBuffer(ComputeBuffer buffer)
        {
            TrackTempBufferReturn(buffer);
            if (buffer == null)
                return;
            AexisGpuResourceTracker.ReleaseBuffer(buffer, "AexisGraphSession.ReturnTempBuffer");
            try { buffer.Dispose(); } catch { }
        }

        internal AexisTensorBuffer RentTempTensorBuffer(
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
            return new AexisTensorBuffer(buffer, dims, w, h, d, c, true, ReturnTempBuffer);
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
            format = ResolvePack4TextureFormat(format);
            if (WouldExceedTextureArraySliceLimit(depth))
            {
                throw new InvalidOperationException(
                    "Texture2DArray slice limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_depth=" + depth.ToString(CultureInfo.InvariantCulture)
                    + " | max_slices=" + GetMaxTextureArraySlices().ToString(CultureInfo.InvariantCulture)
                    + " | size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture));
            }

            var allocLabel = "AexisGraphSession.RentTempArray(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";

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
            AexisGpuResourceTracker.RegisterTemporaryTexture(allocated, temporaryDescriptor);
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
            format = ResolveLinearTextureFormat(format);
            if (WouldExceedTextureSizeLimit(w, h))
            {
                throw new InvalidOperationException(
                    "Texture2D size limit exceeded"
                    + " | site=" + DescribeCurrentExecutionSite()
                    + " | requested_size=" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture)
                    + " | max_size=" + GetMaxTextureSize().ToString(CultureInfo.InvariantCulture));
            }

            var allocLabel = "AexisGraphSession.RentTempMat(" + (callerMember ?? "?") + ":" + callerLine.ToString(CultureInfo.InvariantCulture) + ")";
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
            AexisGpuResourceTracker.RegisterTemporaryTexture(allocated, temporaryDescriptor);
            _ops?.FillScalarTexture(null, allocated);
            TrackTempRtRent();
            return allocated;
        }

        public void ReturnTempArray(RenderTexture rt)
        {
            if (rt == null)
                return;
            TrackTempRtReturn();
            AexisGpuResourceTracker.ReleaseTexture(rt, "AexisGraphSession.ReturnTempArray");
            RenderTexture.ReleaseTemporary(rt);
        }

        public ComputeTexture RentTempArray(CommandBuffer cmd, int w, int h, int depth, RenderTextureFormat format)
        {
            return RentTempArrayCore(cmd, w, h, depth, format, preserveRequestedFormat: false, plannedResourceName: null);
        }

        // Kernel-local plans with same-sized temporaries need a resource identity,
        // not just a texture descriptor, to preserve the compiled liveness proof.
        internal ComputeTexture RentTempArray(
            CommandBuffer cmd,
            int w,
            int h,
            int depth,
            RenderTextureFormat format,
            string plannedResourceName)
        {
            if (string.IsNullOrWhiteSpace(plannedResourceName))
                throw new ArgumentException("A planned CommandBuffer RT name is required.", nameof(plannedResourceName));
            return RentTempArrayCore(cmd, w, h, depth, format, preserveRequestedFormat: false, plannedResourceName: plannedResourceName);
        }

        // Reduction statistics are immutable FP32 accumulation storage rather
        // than activations. Keep the explicit texture format intact so the
        // compiler-declared scratch descriptor matches the CommandBuffer RT.
        internal ComputeTexture RentTempArrayExactFormat(CommandBuffer cmd, int w, int h, int depth, RenderTextureFormat format)
        {
            return RentTempArrayCore(cmd, w, h, depth, format, preserveRequestedFormat: true, plannedResourceName: null);
        }

        internal ComputeTexture RentTempArrayExactFormat(
            CommandBuffer cmd,
            int w,
            int h,
            int depth,
            RenderTextureFormat format,
            string plannedResourceName)
        {
            if (string.IsNullOrWhiteSpace(plannedResourceName))
                throw new ArgumentException("A planned CommandBuffer RT name is required.", nameof(plannedResourceName));
            return RentTempArrayCore(cmd, w, h, depth, format, preserveRequestedFormat: true, plannedResourceName: plannedResourceName);
        }

        private ComputeTexture RentTempArrayCore(
            CommandBuffer cmd,
            int w,
            int h,
            int depth,
            RenderTextureFormat format,
            bool preserveRequestedFormat,
            string plannedResourceName)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));

            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            depth = Mathf.Max(1, depth);
            format = preserveRequestedFormat ? format : ResolvePack4TextureFormat(format);
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
            var allocLabel = "AexisGraphSession.RentTempArrayCmd(" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture) + "x" + depth.ToString(CultureInfo.InvariantCulture) + ")";
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(3, w, h, 1, depth * 4),
                new BufferShape(3, w, h, 1, depth * 4),
                desc,
                allocLabel);
            if (_activeCommandBufferRtArena != null
                && _activeCommandBufferRtArena.TryRentArray(
                    _currentExecutingLayerIndex, w, h, depth, format, desc, temporaryDescriptor, plannedResourceName, out var planned))
            {
                _ops?.FillScalarTexture(cmd, null, planned);
                TrackTempRtRent();
                return planned;
            }
            if (_activeCommandBufferRtArena != null)
            {
                _activeCommandBufferRtArena.RecordUnplannedTextureAllocation();
                RejectUnplannedCommandBufferTextureAllocation("Texture2DArray", w, h, depth, format);
            }
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            cmd.GetTemporaryRT(id, desc);
            AexisGpuResourceTracker.RegisterTemporaryTextureHandle(id, temporaryDescriptor);
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
            return RentTempMatCore(cmd, w, h, format, preserveRequestedFormat: false);
        }

        // LinearMat conversions have an explicit FP32 Texture2D storage contract
        // in the strict execution plan. Preserve that format exactly instead of
        // applying the model-wide FP16 activation preference.
        internal ComputeTexture RentTempMatExactFormat(CommandBuffer cmd, int w, int h, RenderTextureFormat format)
        {
            return RentTempMatCore(cmd, w, h, format, preserveRequestedFormat: true);
        }

        private ComputeTexture RentTempMatCore(
            CommandBuffer cmd,
            int w,
            int h,
            RenderTextureFormat format,
            bool preserveRequestedFormat)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));

            w = Mathf.Max(1, w);
            h = Mathf.Max(1, h);
            format = preserveRequestedFormat ? format : ResolveLinearTextureFormat(format);
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
            var allocLabel = "AexisGraphSession.RentTempMatCmd(" + w.ToString(CultureInfo.InvariantCulture) + "x" + h.ToString(CultureInfo.InvariantCulture) + ")";
            var temporaryDescriptor = CreateTemporaryRtDescriptor(
                new BufferShape(2, w, h, 1, 1),
                new BufferShape(2, w, h, 1, 1),
                desc,
                allocLabel);
            if (_activeCommandBufferRtArena != null
                && _activeCommandBufferRtArena.TryRentMat(
                    _currentExecutingLayerIndex, w, h, format, desc, temporaryDescriptor, out var planned))
            {
                _ops?.FillScalarTexture(cmd, null, planned);
                TrackTempRtRent();
                return planned;
            }
            if (_activeCommandBufferRtArena != null)
            {
                _activeCommandBufferRtArena.RecordUnplannedTextureAllocation();
                RejectUnplannedCommandBufferTextureAllocation("Texture2D", w, h, 1, format);
            }
            EnsureTemporaryTextureBudget(temporaryDescriptor);
            cmd.GetTemporaryRT(id, desc);
            AexisGpuResourceTracker.RegisterTemporaryTextureHandle(id, temporaryDescriptor);
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
            if (_activeCommandBufferRtArena != null && _activeCommandBufferRtArena.TryReturn(t))
            {
                TrackTempRtReturn();
                return;
            }
            t.isReleased = true;
            TrackTempRtReturn();
            AexisGpuResourceTracker.ReleaseTextureHandle(t.nameID, t.trackerLabel ?? "AexisGraphSession.ReturnTempArrayCmd");
            cmd.ReleaseTemporaryRT(t.nameID);
        }

        private void RejectUnplannedCommandBufferTextureAllocation(
            string dimension,
            int width,
            int height,
            int depth,
            RenderTextureFormat format)
        {
            if (LastTextureExecutionPlan == null
                || !LastTextureExecutionPlan.strict
                || LastTextureExecutionPlan.debugOracleRelaxed)
                return;

            throw new InvalidOperationException(
                "command-buffer-pack4-unplanned-temporary-rt"
                + " | layer=" + (_currentExecutingLayerName ?? string.Empty)
                + " | operator=" + (_currentExecutingLayerTypeName ?? string.Empty)
                + " | layer_index=" + _currentExecutingLayerIndex.ToString(CultureInfo.InvariantCulture)
                + " | storage=" + dimension
                + " | size=" + width.ToString(CultureInfo.InvariantCulture) + "x" + height.ToString(CultureInfo.InvariantCulture)
                + "x" + depth.ToString(CultureInfo.InvariantCulture)
                + " | format=" + format
                + " | rejected_fallback=unplanned-texture-allocation"
                + " | action=declare-the-scratch-rt-in-the-loaded-commandbuffer-profile-before-dispatch.");
        }

        internal AexisTemporaryRtDescriptor CreateTemporaryRtDescriptor(
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

        private AexisTemporaryRtDescriptor CreateTemporaryRtDescriptor(
            BufferShape logicalShape,
            BufferShape storageShape,
            RenderTextureDescriptor renderTextureDescriptor,
            string label)
        {
            return new AexisTemporaryRtDescriptor(
                logicalShape,
                storageShape,
                renderTextureDescriptor,
                SessionId,
                DescribeCurrentExecutionSite(),
                label);
        }

        private void EnsureTemporaryTextureBudget(AexisTemporaryRtDescriptor descriptor)
        {
            AexisGpuResourceTracker.EnsureTemporaryTextureBudget(
                AexisGpuResourceTracker.EstimateTemporaryTextureBytes(descriptor),
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
            foreach (var kv in _bias) kv.Value?.Dispose();
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
            _bias.Clear();
            _multiHeadAttention.Clear();
            _extraPacks.Clear();
            Model = null;
            LayerRepros = null;
            _blobUseCount = null;
            _fp32ActivationIslandLayers = null;
            _fp32IndexSelectionProducerLayers = null;
            _fp32SensitiveInputProducerLayers = null;
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

        private RenderTextureFormat ResolvePack4TextureFormat(RenderTextureFormat requested)
        {
            if (ModelManifest == null)
                return requested == RenderTextureFormat.ARGBHalf ? TensorTextureFormat : requested;
            if (requested != RenderTextureFormat.ARGBHalf && requested != RenderTextureFormat.ARGBFloat)
                return requested;
            if (RequiresFp32IndexSelectionInputStorage()
                || RequiresFp32SensitiveInputStorage()
                || UsesFp32ActivationIsland())
                return RenderTextureFormat.ARGBFloat;
            if (RequiresFp32AccumulatorOutput(_currentExecutingLayerTypeName))
                return ResolveSensitiveOutputTextureFormat();
            return UsesFp16ActivationStorage ? RenderTextureFormat.ARGBHalf : RenderTextureFormat.ARGBFloat;
        }

        private RenderTextureFormat ResolveLinearTextureFormat(RenderTextureFormat requested)
        {
            if (ModelManifest == null)
                return requested == RenderTextureFormat.ARGBHalf ? ResolveLinearMatTextureFormat() : requested;
            if (requested != RenderTextureFormat.ARGBHalf
                && requested != RenderTextureFormat.ARGBFloat
                && requested != RenderTextureFormat.RHalf
                && requested != RenderTextureFormat.RFloat)
                return requested;
            if (RequiresFp32IndexSelectionInputStorage()
                || RequiresFp32SensitiveInputStorage()
                || UsesFp32ActivationIsland())
                return RenderTextureFormat.RFloat;
            if (RequiresFp32AccumulatorOutput(_currentExecutingLayerTypeName)
                && ModelManifest.precision.sensitiveOutputDataType == TensorDataType.Float32)
                return RenderTextureFormat.RFloat;
            return UsesFp16ActivationStorage ? RenderTextureFormat.RHalf : RenderTextureFormat.RFloat;
        }

        private bool UsesFp32ActivationIsland()
        {
            if (!UsesFp16ActivationStorage
                || string.IsNullOrWhiteSpace(Fp32ActivationStartLayerName)
                || string.IsNullOrWhiteSpace(_currentExecutingLayerName)
                || Model?.layers == null)
            {
                return false;
            }

            if (_fp32ActivationIslandLayers == null)
            {
                _fp32ActivationIslandLayers = new HashSet<string>(StringComparer.Ordinal);
                var islandStarted = false;
                for (var i = 0; i < Model.layers.Count; i++)
                {
                    var layer = Model.layers[i];
                    if (!islandStarted && string.Equals(layer?.name, Fp32ActivationStartLayerName, StringComparison.Ordinal))
                        islandStarted = true;
                    if (islandStarted && !string.IsNullOrWhiteSpace(layer?.name))
                        _fp32ActivationIslandLayers.Add(layer.name);
                }
            }

            return _fp32ActivationIslandLayers.Contains(_currentExecutingLayerName);
        }

        private bool RequiresFp32IndexSelectionInputStorage()
        {
            if (!UsesFp16ActivationStorage
                || string.IsNullOrWhiteSpace(_currentExecutingLayerName)
                || Model?.layers == null)
                return false;

            if (_fp32IndexSelectionProducerLayers == null)
            {
                var requiredBlobs = new HashSet<string>(StringComparer.Ordinal);
                for (var i = 0; i < Model.layers.Count; i++)
                {
                    var layer = Model.layers[i];
                    if (layer?.type != AexisLayerTypes.MaxPoolingInd || layer.bottomNames == null || layer.bottomNames.Length == 0)
                        continue;
                    if (!string.IsNullOrWhiteSpace(layer.bottomNames[0]))
                        requiredBlobs.Add(layer.bottomNames[0]);
                }

                _fp32IndexSelectionProducerLayers = new HashSet<string>(StringComparer.Ordinal);
                for (var i = Model.layers.Count - 1; i >= 0; i--)
                {
                    var layer = Model.layers[i];
                    if (layer?.topNames == null || layer.topNames.Length == 0)
                        continue;

                    var producesSelectionInput = false;
                    for (var topIndex = 0; topIndex < layer.topNames.Length; topIndex++)
                    {
                        if (requiredBlobs.Contains(layer.topNames[topIndex]))
                        {
                            producesSelectionInput = true;
                            break;
                        }
                    }
                    if (!producesSelectionInput)
                        continue;

                    _fp32IndexSelectionProducerLayers.Add(layer.name);
                    if (layer.bottomNames == null)
                        continue;
                    for (var bottomIndex = 0; bottomIndex < layer.bottomNames.Length; bottomIndex++)
                    {
                        if (!string.IsNullOrWhiteSpace(layer.bottomNames[bottomIndex]))
                            requiredBlobs.Add(layer.bottomNames[bottomIndex]);
                    }
                }
            }

            return _fp32IndexSelectionProducerLayers.Contains(_currentExecutingLayerName);
        }

        private bool RequiresFp32SensitiveInputStorage()
        {
            if (!UsesFp16ActivationStorage
                || string.IsNullOrWhiteSpace(_currentExecutingLayerName)
                || Model?.layers == null)
            {
                return false;
            }

            if (_fp32SensitiveInputProducerLayers == null)
            {
                var requiredBlobs = new HashSet<string>(StringComparer.Ordinal);
                for (var i = 0; i < Model.layers.Count; i++)
                {
                    var layer = Model.layers[i];
                    if (layer?.bottomNames == null
                        || (layer.type != AexisLayerTypes.LayerNorm
                            && layer.type != AexisLayerTypes.MultiHeadAttention
                            && layer.type != AexisLayerTypes.Softmax))
                    {
                        continue;
                    }

                    for (var bottomIndex = 0; bottomIndex < layer.bottomNames.Length; bottomIndex++)
                    {
                        if (!string.IsNullOrWhiteSpace(layer.bottomNames[bottomIndex]))
                            requiredBlobs.Add(layer.bottomNames[bottomIndex]);
                    }
                }

                _fp32SensitiveInputProducerLayers = new HashSet<string>(StringComparer.Ordinal);
                for (var i = Model.layers.Count - 1; i >= 0; i--)
                {
                    var layer = Model.layers[i];
                    if (layer?.topNames == null || layer.topNames.Length == 0)
                        continue;

                    var producesSensitiveInput = false;
                    for (var topIndex = 0; topIndex < layer.topNames.Length; topIndex++)
                    {
                        if (requiredBlobs.Contains(layer.topNames[topIndex]))
                        {
                            producesSensitiveInput = true;
                            break;
                        }
                    }
                    if (!producesSensitiveInput)
                        continue;

                    _fp32SensitiveInputProducerLayers.Add(layer.name);
                    if (layer.bottomNames == null
                        || (layer.type != AexisLayerTypes.Split
                            && layer.type != AexisLayerTypes.BinaryOp
                            && layer.type != AexisLayerTypes.InnerProduct
                            && layer.type != AexisLayerTypes.LayerNorm
                            && layer.type != AexisLayerTypes.GELU
                            && layer.type != AexisLayerTypes.MultiHeadAttention))
                    {
                        continue;
                    }

                    for (var bottomIndex = 0; bottomIndex < layer.bottomNames.Length; bottomIndex++)
                    {
                        if (!string.IsNullOrWhiteSpace(layer.bottomNames[bottomIndex]))
                            requiredBlobs.Add(layer.bottomNames[bottomIndex]);
                    }
                }
            }

            return _fp32SensitiveInputProducerLayers.Contains(_currentExecutingLayerName);
        }

        internal static bool RequiresFp32AccumulatorOutput(string operatorName)
        {
            return string.Equals(operatorName, "LayerNorm", StringComparison.Ordinal)
                || string.Equals(operatorName, "Softmax", StringComparison.Ordinal)
                || string.Equals(operatorName, "Reduction", StringComparison.Ordinal)
                || string.Equals(operatorName, "SDPA", StringComparison.Ordinal)
                || string.Equals(operatorName, "MultiHeadAttention", StringComparison.Ordinal)
                || string.Equals(operatorName, "Sigmoid", StringComparison.Ordinal)
                || string.Equals(operatorName, "GELU", StringComparison.Ordinal)
                || string.Equals(operatorName, "Swish", StringComparison.Ordinal);
        }

        public static RenderTextureFormat ResolveLinearMatTextureFormat()
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

        internal static BufferShape ResolvePack4TiledLinearMatStorageShape(BufferShape logicalShape)
        {
            var logicalWidth = Mathf.Max(1, logicalShape.w);
            var logicalHeight = logicalShape.dims >= 2 ? Mathf.Max(1, logicalShape.h) : 1;
            var packWidth = Mathf.CeilToInt(logicalWidth / 4f);
            var tileWidthLimit = Mathf.Max(1, Mathf.Min(8192, SystemInfo.maxTextureSize));
            var tileWidth = Mathf.Min(packWidth, tileWidthLimit);
            var tileRows = Mathf.CeilToInt(packWidth / (float)tileWidth);
            return new BufferShape(3, tileWidth, logicalHeight * tileRows, 1, 4);
        }

        internal static bool IsStrictLinearMatTexture(TensorRef tensor)
        {
            return tensor != null
                && tensor.texture != null
                && tensor.texture.dimension == TextureDimension.Tex2D
                && tensor.layoutKind == AexisTextureTensorLayoutKind.LinearMat;
        }

        internal static bool IsPack4LinearMatTexture(TensorRef tensor, BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || (logicalShape.dims != 1 && logicalShape.dims != 2))
                return false;
            if (tensor.texture.dimension != TextureDimension.Tex2DArray)
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
                && tensor.layoutKind == AexisTextureTensorLayoutKind.LinearMat;
        }

        internal static bool IsPack4LinearMatTexture(CmdTensorRef tensor, BufferShape logicalShape)
        {
            if (tensor == null || tensor.texture == null || (logicalShape.dims != 1 && logicalShape.dims != 2))
                return false;
            if (tensor.texture.dimension != TextureDimension.Tex2DArray)
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

        internal static AexisTextureTensorLayoutKind ResolveNcnnTextureLayoutKind(BufferShape logicalShape, BufferShape storageShape, int packs)
        {
            var effectivePacks = Mathf.Max(1, packs);
            if (storageShape.dims <= 2)
                return AexisTextureTensorLayoutKind.LinearMat;
            if (storageShape.dims == 3 && Mathf.Max(1, storageShape.c) <= 1 && effectivePacks <= 1)
                return AexisTextureTensorLayoutKind.LinearMat;
            if (logicalShape.dims <= 2 && effectivePacks <= 1 && Mathf.Max(1, storageShape.d) <= 1 && Mathf.Max(1, storageShape.c) <= 1)
                return AexisTextureTensorLayoutKind.LinearMat;
            return AexisTextureTensorLayoutKind.Pack4Image;
        }

        internal static AexisTextureTensor CreateNcnnTextureTensor(
            RenderTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            int packs)
        {
            if (texture == null)
                return null;

            var layoutKind = texture.dimension == TextureDimension.Tex2D
                ? AexisTextureTensorLayoutKind.LinearMat
                : ResolveNcnnTextureLayoutKind(logicalShape, storageShape, packs);

            if (layoutKind == AexisTextureTensorLayoutKind.LinearMat)
                return new AexisTextureMat(texture, logicalShape, storageShape, packs);

            var depth = Mathf.Max(1, texture.volumeDepth > 0 ? texture.volumeDepth : 1);
            return new AexisTextureImageMat(texture, logicalShape, storageShape, packs, depth);
        }

        internal static AexisTextureTensor CreateNcnnTextureTensor(
            ComputeTexture texture,
            BufferShape logicalShape,
            BufferShape storageShape,
            int packs)
        {
            if (texture == null)
                return null;

            var layoutKind = texture.dimension == TextureDimension.Tex2D
                ? AexisTextureTensorLayoutKind.LinearMat
                : ResolveNcnnTextureLayoutKind(logicalShape, storageShape, packs);
            if (layoutKind == AexisTextureTensorLayoutKind.LinearMat)
                return new AexisTextureMat(texture, logicalShape, storageShape, packs);

            var depth = Mathf.Max(1, texture.depth);
            return new AexisTextureImageMat(texture, logicalShape, storageShape, packs, depth);
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
            return new TensorProvenance("AexisGraphSession", debugName, blobName, debugName);
        }

        private static TensorPacking ResolveTensorPacking(BufferShape storageShape, AexisTextureTensorLayoutKind layout, int packCount)
        {
            var packSize = layout == AexisTextureTensorLayoutKind.Pack4Image
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
            var layout = ResolveNcnnTextureLayoutKind(logicalShape, storageShape, packs);
            var nativeTensor = CreateNcnnTextureTensor(tensor.texture, logicalShape, storageShape, packs);
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
            var layout = ResolveNcnnTextureLayoutKind(logicalShape, storageShape, packs);
            var nativeTensor = CreateNcnnTextureTensor(tensor.texture, logicalShape, storageShape, packs);
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

            EnsureNcnnTextureTensor(source);
            var sourceDescriptor = source.Descriptor;
            var targetPacks = GetTexturePackCount(storageShape, source.texture);
            var targetLayout = ResolveNcnnTextureLayoutKind(logicalShape, storageShape, targetPacks);
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

            EnsureNcnnTextureTensor(source);
            var sourceDescriptor = source.Descriptor;
            var targetPacks = GetCmdTexturePackCount(storageShape, source.texture);
            var targetLayout = ResolveNcnnTextureLayoutKind(logicalShape, storageShape, targetPacks);
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

        internal static AexisTextureTensorContract GetTextureContract(
            Dictionary<string, BufferShape> textureShapes,
            TensorRef tensor,
            string name)
        {
            if (tensor == null || tensor.texture == null)
                throw new InvalidOperationException("texture contract unavailable: " + name);

            // The per-name dictionary remains for legacy execution bookkeeping only.
            // Published tensor descriptors are the sole source of contract metadata.
            EnsureNcnnTextureTensor(tensor);
            return new AexisTextureTensorContract(tensor);
        }

        internal static AexisTextureTensorContract GetCmdTensorContract(
            CmdTensorRef tensor,
            BufferShape? fallbackLogicalShape = null,
            BufferShape? fallbackStorageShape = null)
        {
            if (tensor == null || tensor.texture == null)
                throw new ArgumentNullException(nameof(tensor));

            EnsureNcnnTextureTensor(tensor);
            return new AexisTextureTensorContract(tensor);
        }

        internal static bool TryGetExistingTextureContract(
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            string name,
            out TensorRef texture,
            out AexisTextureTensorContract contract)
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
            out AexisTextureTensorContract contract)
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

        internal static void EnsureNcnnTextureTensor(
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

        internal static void EnsureNcnnTextureTensor(
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

        private RenderTexture MaterializeTextureFromBufferViewCore(ComputeBuffer buffer, AexisTensorBuffer view, bool ignoreGuard, RenderTextureFormat? formatOverride = null)
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

        internal RenderTexture MaterializeTextureFromBufferView(ComputeBuffer buffer, AexisTensorBuffer view, RenderTextureFormat? formatOverride = null)
        {
            return MaterializeTextureFromBufferViewCore(buffer, view, ignoreGuard: false, formatOverride);
        }

        internal RenderTexture MaterializeScratchTextureFromBufferView(ComputeBuffer buffer, AexisTensorBuffer view, RenderTextureFormat? formatOverride = null)
        {
            return MaterializeTextureFromBufferViewCore(buffer, view, ignoreGuard: true, formatOverride);
        }

        // A fixed graph input may use a ComputeBuffer solely as an upload source.
        // The buffer is consumed by this dispatch before the first layer and is never
        // registered as a model activation. Integer token ids have a dedicated upload
        // kernel so their bits are converted to exact float texture values.
        internal RenderTexture UploadFixedInputTexture(string blobName, ComputeBuffer buffer, AexisTensorBuffer view)
        {
            if (string.IsNullOrWhiteSpace(blobName))
                throw new ArgumentException("A fixed input blob name is required.", nameof(blobName));
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (view == null)
                throw new ArgumentNullException(nameof(view));
            if (view.dims < 1 || view.dims > 4)
                throw new NotSupportedException("Fixed input upload requires rank 1 through 4"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims);

            if (view.dims <= 2)
            {
                var width = Mathf.Max(1, view.w);
                var height = view.dims == 1 ? 1 : Mathf.Max(1, view.h);
                var texture = RentTempMat(width, height, ResolveLinearMatTextureFormat());
                if (IsFixedIntegerInput(blobName))
                    _ops.FillLinearMatFromIntBuffer(buffer, width, height, texture);
                else
                    _ops.FillLinearMatFromBuffer(buffer, width, height, texture);
                DebugLog?.Invoke("[InferencePathAudit] fixed_input_upload=texture"
                    + " | blob=" + blobName
                    + " | logical_shape=" + DescribeShape(new BufferShape(view.dims, view.w, view.h, view.d, view.c))
                    + " | source=" + (IsFixedIntegerInput(blobName) ? "int-buffer" : "float-buffer"));
                return texture;
            }

            if (IsFixedIntegerInput(blobName))
                throw new NotSupportedException("Integer fixed input upload currently supports rank 1 or 2 only"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims
                    + " | rejected_fallback=buffer-activation");

            var upload = MaterializeTextureFromBufferViewCore(buffer, view, ignoreGuard: true);
            if (upload == null)
                throw new InvalidOperationException("Fixed input texture upload failed"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims
                    + " | rejected_fallback=buffer-activation");
            return upload;
        }

        // CommandBuffer equivalent of UploadFixedInputTexture. The fixed Buffer is
        // read once by an upload dispatch and every model layer sees only the
        // resulting texture-backed tensor.
        internal ComputeTexture UploadFixedInputCmdTexture(
            CommandBuffer cmd,
            string blobName,
            ComputeBuffer buffer,
            AexisTensorBuffer view)
        {
            if (cmd == null)
                throw new ArgumentNullException(nameof(cmd));
            if (string.IsNullOrWhiteSpace(blobName))
                throw new ArgumentException("A fixed input blob name is required.", nameof(blobName));
            if (buffer == null)
                throw new ArgumentNullException(nameof(buffer));
            if (view == null)
                throw new ArgumentNullException(nameof(view));
            if (view.dims < 1 || view.dims > 4)
                throw new NotSupportedException("Fixed CommandBuffer input upload requires rank 1 through 4"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims);

            if (view.dims <= 2)
            {
                var width = Mathf.Max(1, view.w);
                var height = view.dims == 1 ? 1 : Mathf.Max(1, view.h);
                var texture = RentTempMat(cmd, width, height, ResolveLinearMatTextureFormat());
                if (IsFixedIntegerInput(blobName))
                    _ops.FillLinearMatFromIntBuffer(cmd, buffer, width, height, texture);
                else
                    _ops.FillLinearMatFromBuffer(cmd, buffer, width, height, texture);
                return texture;
            }

            if (IsFixedIntegerInput(blobName))
                throw new NotSupportedException("Integer fixed CommandBuffer input upload currently supports rank 1 or 2 only"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims
                    + " | rejected_fallback=buffer-activation");

            var channels = Mathf.Max(1, view.c);
            var channelPacks = Mathf.Max(1, Mathf.CeilToInt(channels / 4f));
            var sliceCount = view.dims == 4
                ? Mathf.Max(1, view.d) * channelPacks
                : channelPacks;
            if (WouldExceedTextureArraySliceLimit(sliceCount))
                throw new NotSupportedException("Fixed CommandBuffer input exceeds Texture2DArray slice capacity"
                    + " | blob=" + blobName
                    + " | dims=" + view.dims
                    + " | slices=" + sliceCount
                    + " | rejected_fallback=buffer-activation");

            var upload = RentTempArray(cmd, Mathf.Max(1, view.w), Mathf.Max(1, view.h), sliceCount, ResolveTensorTextureFormat(view.dims));
            if (view.dims == 4)
                _ops.FillPack4FromBufferCDHW(cmd, buffer, view.w, view.h, view.d, channels, upload);
            else
                _ops.FillPack4FromBufferCHW(cmd, buffer, view.w, view.h, channels, upload);
            return upload;
        }

        private bool IsFixedIntegerInput(string blobName)
        {
            if (Model?.layers == null || string.IsNullOrWhiteSpace(blobName))
                return false;

            var hasConsumer = false;
            for (var layerIndex = 0; layerIndex < Model.layers.Count; layerIndex++)
            {
                var layer = Model.layers[layerIndex];
                if (layer?.bottomNames == null || Array.IndexOf(layer.bottomNames, blobName) < 0)
                    continue;
                hasConsumer = true;
                if (layer.type != AexisLayerTypes.Embed && !string.Equals(layer.typeName, "Embed", StringComparison.Ordinal))
                    return false;
            }
            return hasConsumer;
        }

        internal ComputeTexture MaterializeCmdTextureFromBufferView(CommandBuffer cmd, ComputeBuffer buffer, AexisTensorBuffer view)
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
            Dictionary<string, AexisTensorBuffer> bufferViews)
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
            Dictionary<string, AexisTensorBuffer> bufferViews)
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

        internal AexisTensorBuffer RentScratchTensorFromTexture(
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
                return new AexisTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, true, ReturnTempBuffer);
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

            return new AexisTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, true, ReturnTempBuffer);
        }

        internal AexisTensorBuffer GetReadableTensorInput(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            AexisTensorBuffer tensor,
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
            var view = new AexisTensorBuffer(buffer, logicalShape.dims, logicalShape.w, logicalShape.h, logicalShape.d, logicalShape.c, false);
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
                var componentsPerTexel = rt.format == RenderTextureFormat.RFloat ? 1L : 4L;
                var physicalCount = (long)rt.width * rt.height * sliceCount * packs * componentsPerTexel;
                var logicalCount = (long)Mathf.Max(1, logicalShape.w) * Mathf.Max(1, logicalShape.h) * Mathf.Max(1, logicalShape.d) * Mathf.Max(1, logicalShape.c);
                if (logicalCount > physicalCount)
                    throw new InvalidOperationException("texture input logical shape exceeds physical storage: " + kv.Key);

                var storageShape = ResolveExternalTextureInputStorageShape(logicalShape, rt.width, rt.height, rt.dimension, Mathf.Max(1, rt.volumeDepth), rt.format);
                textureBlobs[kv.Key] = CreateTextureRef(rt, logicalShape, storageShape, owned: false, refs: useCount, blobName: kv.Key);
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

                var componentsPerTexel = texture.format == RenderTextureFormat.RFloat ? 1L : 4L;
                var physicalCount = (long)texture.width * texture.height * sliceCount * packs * componentsPerTexel;
                var logicalCount = (long)Mathf.Max(1, logicalShape.w) * Mathf.Max(1, logicalShape.h) * Mathf.Max(1, logicalShape.d) * Mathf.Max(1, logicalShape.c);
                if (logicalCount > physicalCount)
                    throw new InvalidOperationException("command-buffer texture input logical shape exceeds physical storage: " + kv.Key);

                var storageShape = ResolveExternalTextureInputStorageShape(logicalShape, texture.width, texture.height, texture.dimension, depth, texture.format);
                blobs[kv.Key] = CreateCmdTensorRef(texture, logicalShape, storageShape, owned: false, refs: useCount, blobName: kv.Key);
                shapes[kv.Key] = logicalShape;
            }
        }

        private static BufferShape ResolveExternalTextureInputStorageShape(
            BufferShape logicalShape,
            int textureWidth,
            int textureHeight,
            TextureDimension dimension,
            int textureDepth,
            RenderTextureFormat format)
        {
            if (format == RenderTextureFormat.RFloat
                && dimension == TextureDimension.Tex2D
                && textureDepth == 1)
            {
                var linearStorage = ResolveLinearMatStorageShape(logicalShape);
                if (textureWidth == linearStorage.w && textureHeight == linearStorage.h)
                    return linearStorage;
            }

            if ((logicalShape.dims == 1 || logicalShape.dims == 2)
                && dimension == TextureDimension.Tex2DArray
                && textureDepth == 1
                && textureWidth == Mathf.Max(1, logicalShape.w)
                && textureHeight == (logicalShape.dims == 1 ? 1 : Mathf.Max(1, logicalShape.h)))
            {
                // Texture2DArray scalar uploads (for example Qwen causal masks)
                // store one logical scalar in the x lane of each float4 texel.
                // Preserve this scalar-Pack4 contract instead of collapsing its
                // physical descriptor to rank two.
                return new BufferShape(3, logicalShape.w, logicalShape.dims == 1 ? 1 : logicalShape.h, 1, 1);
            }

            if ((logicalShape.dims == 1 || logicalShape.dims == 2)
                && dimension == TextureDimension.Tex2DArray
                && textureDepth == 1
                && textureWidth == Mathf.CeilToInt(Mathf.Max(1, logicalShape.w) / 4f)
                && textureHeight == (logicalShape.dims == 1 ? 1 : Mathf.Max(1, logicalShape.h)))
            {
                return ResolvePack4LinearMatStorageShape(logicalShape);
            }

            return logicalShape;
        }

        internal bool TryGetPack4Texture(
            string name,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            AexisGraphModel.Layer layer,
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
            AexisGraphModel.Layer layer,
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
            AexisGraphModel.Layer layer,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            AexisGraphModel.Layer layer,
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
            public int managedCleanupCount;
            public long managedCleanupMs;
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
            AexisTensorBuffer tensor,
            bool preferTexture,
            Dictionary<string, TensorRef> textureBlobs,
            Dictionary<string, BufferShape> textureShapes,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, BufferRef> bufferRefs,
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            bufferViews[topName] = new AexisTensorBuffer(tensor.buffer, tensor.dims, tensor.w, tensor.h, tensor.d, tensor.c, false);

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
            AexisTensorBuffer tensor,
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

            var shape = logicalShape
                ?? new BufferShape(3, Mathf.Max(1, width), Mathf.Max(1, height), 1, Mathf.Max(1, packs * 4));
            throw CreateDisallowedBufferPathException(
                "CommandBuffer placeholder output is forbidden",
                topName,
                "requested=" + width + "x" + height + "x" + packs + "p logical=" + shape);
        }

        internal static void ResolveCmdTextureLayout(AexisTensorBuffer tensor, out int width, out int height, out int packs)
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
            AexisTensorBuffer tensor,
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

        public static ComputeBuffer UploadImmutableFloatConstants(float[] data, string label)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (data.Length == 0)
                throw new ArgumentException("Immutable constant data is empty.", nameof(data));
            if (string.IsNullOrWhiteSpace(label))
                throw new ArgumentException("Immutable constant label is empty.", nameof(label));
            var buf = new ComputeBuffer(data.Length, sizeof(float), ComputeBufferType.Structured);
            try
            {
                AexisGpuResourceTracker.RegisterBuffer(buf, data.Length, sizeof(float), label);
                buf.SetData(data);
                return buf;
            }
            catch
            {
                try { AexisGpuResourceTracker.ReleaseBuffer(buf, label + ".UploadFailure"); } catch { }
                buf.Dispose();
                throw;
            }
        }

        public static float ReadScalarTexture(RenderTexture texture)
        {
            if (texture == null)
                throw new ArgumentNullException(nameof(texture));
            if (texture.dimension != TextureDimension.Tex2D || texture.width < 1 || texture.height < 1)
                throw new ArgumentException("Scalar readback requires a non-empty Texture2D.", nameof(texture));

            var previousActive = RenderTexture.active;
            Texture2D readback = null;
            try
            {
                readback = new Texture2D(1, 1, TextureFormat.RFloat, false, true);
                RenderTexture.active = texture;
                readback.ReadPixels(new Rect(0, 0, 1, 1), 0, 0, false);
                readback.Apply(false, false);
                return readback.GetPixel(0, 0).r;
            }
            finally
            {
                RenderTexture.active = previousActive;
                if (readback != null)
                    UnityEngine.Object.DestroyImmediate(readback);
            }
        }

        internal static ComputeBuffer NewBuffer(float[] data)
        {
            return UploadImmutableFloatConstants(data, "AexisGraphSession.NewBuffer");
        }

        internal static ComputeBuffer NewFp16Buffer(float[] data, string label)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            var packed = new uint[Mathf.Max(1, (data.Length + 1) / 2)];
            for (var index = 0; index < data.Length; index++)
            {
                var bits = FloatToHalfBits(data[index]);
                var packedIndex = index >> 1;
                packed[packedIndex] |= (uint)bits << ((index & 1) * 16);
            }

            var buffer = new ComputeBuffer(packed.Length, sizeof(uint), ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(buffer, packed.Length, sizeof(uint), label ?? "AexisGraphSession.Fp16Weight");
            buffer.SetData(packed);
            return buffer;
        }

        internal static ComputeBuffer NewFp16Vector4Buffer(Vector4[] data, string label)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            var packed = new PackedFp16x4[Mathf.Max(1, data.Length)];
            for (var index = 0; index < data.Length; index++)
            {
                var value = data[index];
                packed[index].xy = PackHalf2(value.x, value.y);
                packed[index].zw = PackHalf2(value.z, value.w);
            }

            var buffer = new ComputeBuffer(Mathf.Max(1, data.Length), sizeof(uint) * 2, ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(buffer, Mathf.Max(1, data.Length), sizeof(uint) * 2, label ?? "AexisGraphSession.Fp16Weight4");
            buffer.SetData(packed);
            return buffer;
        }

        // D2 stores signed weights as four two's-complement INT8 values per uint.  The
        // original float array is used only while importing the immutable upload, never
        // as a GPU-side expansion source during texture-native inference.
        internal static Int8WeightOnlyUpload NewInt8WeightOnlyUpload(
            float[] data,
            int outputChannels,
            int valuesPerOutputChannel,
            bool outputChannelsAreContiguous,
            string label)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (outputChannels <= 0)
                throw new ArgumentOutOfRangeException(nameof(outputChannels));
            if (valuesPerOutputChannel <= 0 || data.Length != outputChannels * valuesPerOutputChannel)
                throw new ArgumentException("INT8 weight layout must contain exactly outputChannels * valuesPerOutputChannel values.", nameof(data));

            var scales = new float[outputChannels];
            var packed = new uint[Mathf.Max(1, (data.Length + 3) / 4)];
            for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                var maxAbs = 0f;
                for (var valueIndex = 0; valueIndex < valuesPerOutputChannel; valueIndex++)
                {
                    var sourceIndex = outputChannelsAreContiguous
                        ? outputChannel * valuesPerOutputChannel + valueIndex
                        : valueIndex * outputChannels + outputChannel;
                    maxAbs = Mathf.Max(maxAbs, Mathf.Abs(data[sourceIndex]));
                }

                var scale = maxAbs > 0f ? maxAbs / 127f : 1f;
                scales[outputChannel] = scale;
                for (var valueIndex = 0; valueIndex < valuesPerOutputChannel; valueIndex++)
                {
                    var sourceIndex = outputChannelsAreContiguous
                        ? outputChannel * valuesPerOutputChannel + valueIndex
                        : valueIndex * outputChannels + outputChannel;
                    var quantized = Mathf.Clamp(Mathf.RoundToInt(data[sourceIndex] / scale), -127, 127);
                    var packedIndex = sourceIndex >> 2;
                    var bitOffset = (sourceIndex & 3) * 8;
                    packed[packedIndex] |= (uint)(byte)(sbyte)quantized << bitOffset;
                }
            }

            var upload = new Int8WeightOnlyUpload
            {
                packedWeights = new ComputeBuffer(packed.Length, sizeof(uint), ComputeBufferType.Structured),
                scales = new ComputeBuffer(scales.Length, sizeof(float), ComputeBufferType.Structured)
            };
            AexisGpuResourceTracker.RegisterBuffer(upload.packedWeights, packed.Length, sizeof(uint), (label ?? "AexisGraphSession.Int8WeightOnly") + ".PackedInt8");
            AexisGpuResourceTracker.RegisterBuffer(upload.scales, scales.Length, sizeof(float), (label ?? "AexisGraphSession.Int8WeightOnly") + ".PerOutputScale");
            upload.packedWeights.SetData(packed);
            upload.scales.SetData(scales);
            return upload;
        }

        internal static Int8WeightOnlyUpload NewInt8WeightOnlyUpload(
            AexisQuantizedTensor data,
            int expectedOutputChannels,
            string label)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (expectedOutputChannels <= 0 || data.Scales == null || data.Scales.Length != expectedOutputChannels)
                throw new ArgumentException("Direct Q8 upload must contain one scale per output channel.", nameof(data));
            if (data.PackedValues == null || data.PackedValues.Length != Math.Max(1, (data.ElementCount + 3) / 4))
                throw new ArgumentException("Direct Q8 packed payload size mismatch.", nameof(data));
            var upload = new Int8WeightOnlyUpload
            {
                packedWeights = new ComputeBuffer(data.PackedValues.Length, sizeof(uint), ComputeBufferType.Structured),
                scales = new ComputeBuffer(data.Scales.Length, sizeof(float), ComputeBufferType.Structured)
            };
            AexisGpuResourceTracker.RegisterBuffer(upload.packedWeights, data.PackedValues.Length, sizeof(uint), (label ?? "AexisGraphSession.Int8WeightOnly") + ".PackedInt8Direct");
            AexisGpuResourceTracker.RegisterBuffer(upload.scales, data.Scales.Length, sizeof(float), (label ?? "AexisGraphSession.Int8WeightOnly") + ".PerOutputScaleDirect");
            upload.packedWeights.SetData(data.PackedValues);
            upload.scales.SetData(data.Scales);
            return upload;
        }

        // INT4 stores signed weights as eight two's-complement 4-bit values per uint.
        // It is an immutable upload selected by INT4Selective manifests; activations
        // and outputs remain texture-native Pack4 resources during inference.
        internal static Int4WeightOnlyUpload NewInt4WeightOnlyUpload(
            float[] data,
            int outputChannels,
            int valuesPerOutputChannel,
            bool outputChannelsAreContiguous,
            string label)
        {
            if (data == null)
                throw new ArgumentNullException(nameof(data));
            if (outputChannels <= 0)
                throw new ArgumentOutOfRangeException(nameof(outputChannels));
            if (valuesPerOutputChannel <= 0 || data.Length != outputChannels * valuesPerOutputChannel)
                throw new ArgumentException("INT4 weight layout must contain exactly outputChannels * valuesPerOutputChannel values.", nameof(data));

            var scales = new float[outputChannels];
            var packed = new uint[Mathf.Max(1, (data.Length + 7) / 8)];
            for (var outputChannel = 0; outputChannel < outputChannels; outputChannel++)
            {
                var maxAbs = 0f;
                for (var valueIndex = 0; valueIndex < valuesPerOutputChannel; valueIndex++)
                {
                    var sourceIndex = outputChannelsAreContiguous
                        ? outputChannel * valuesPerOutputChannel + valueIndex
                        : valueIndex * outputChannels + outputChannel;
                    maxAbs = Mathf.Max(maxAbs, Mathf.Abs(data[sourceIndex]));
                }

                var scale = maxAbs > 0f ? maxAbs / 7f : 1f;
                scales[outputChannel] = scale;
                for (var valueIndex = 0; valueIndex < valuesPerOutputChannel; valueIndex++)
                {
                    var sourceIndex = outputChannelsAreContiguous
                        ? outputChannel * valuesPerOutputChannel + valueIndex
                        : valueIndex * outputChannels + outputChannel;
                    var quantized = Mathf.Clamp(Mathf.RoundToInt(data[sourceIndex] / scale), -7, 7);
                    var packedIndex = sourceIndex >> 3;
                    var bitOffset = (sourceIndex & 7) * 4;
                    packed[packedIndex] |= ((uint)quantized & 0xfu) << bitOffset;
                }
            }

            var upload = new Int4WeightOnlyUpload
            {
                packedWeights = new ComputeBuffer(packed.Length, sizeof(uint), ComputeBufferType.Structured),
                scales = new ComputeBuffer(scales.Length, sizeof(float), ComputeBufferType.Structured),
                scaleBlockSize = outputChannelsAreContiguous ? valuesPerOutputChannel : 0
            };
            AexisGpuResourceTracker.RegisterBuffer(upload.packedWeights, packed.Length, sizeof(uint), (label ?? "AexisGraphSession.Int4WeightOnly") + ".PackedInt4");
            AexisGpuResourceTracker.RegisterBuffer(upload.scales, scales.Length, sizeof(float), (label ?? "AexisGraphSession.Int4WeightOnly") + ".PerOutputScale");
            upload.packedWeights.SetData(packed);
            upload.scales.SetData(scales);
            return upload;
        }

        internal static Int4WeightOnlyUpload NewInt4WeightOnlyUpload(
            AexisQuantizedTensor data,
            int expectedOutputChannels,
            int valuesPerOutputChannel,
            string label)
        {
            if (data == null) throw new ArgumentNullException(nameof(data));
            if (expectedOutputChannels <= 0 || valuesPerOutputChannel <= 0
                || data.ElementCount != checked(expectedOutputChannels * valuesPerOutputChannel))
            {
                throw new ArgumentException("Direct Q4 upload shape does not match Gemm output channels and K.", nameof(data));
            }
            if (data.BlockSize <= 0 || valuesPerOutputChannel % data.BlockSize != 0)
                throw new ArgumentException("Direct Q4 group size must divide the Gemm K dimension.", nameof(data));
            var expectedScaleCount = checked(data.ElementCount / data.BlockSize);
            if (data.Scales == null || data.Scales.Length != expectedScaleCount)
                throw new ArgumentException("Direct Q4 upload scale count does not match its group size.", nameof(data));
            if (data.PackedValues == null || data.PackedValues.Length != Math.Max(1, (data.ElementCount + 7) / 8))
                throw new ArgumentException("Direct Q4 packed payload size mismatch.", nameof(data));
            var upload = new Int4WeightOnlyUpload
            {
                packedWeights = new ComputeBuffer(data.PackedValues.Length, sizeof(uint), ComputeBufferType.Structured),
                scales = new ComputeBuffer(data.Scales.Length, sizeof(float), ComputeBufferType.Structured),
                scaleBlockSize = data.BlockSize
            };
            AexisGpuResourceTracker.RegisterBuffer(upload.packedWeights, data.PackedValues.Length, sizeof(uint), (label ?? "AexisGraphSession.Int4WeightOnly") + ".PackedInt4Direct");
            AexisGpuResourceTracker.RegisterBuffer(upload.scales, data.Scales.Length, sizeof(float), (label ?? "AexisGraphSession.Int4WeightOnly") + ".PerOutputScaleDirect");
            upload.packedWeights.SetData(data.PackedValues);
            upload.scales.SetData(data.Scales);
            return upload;
        }

        private static uint PackHalf2(float x, float y)
        {
            return (uint)FloatToHalfBits(x) | ((uint)FloatToHalfBits(y) << 16);
        }

        // This is the IEEE-754 round-to-nearest-even conversion used only while uploading
        // immutable weights. Activations remain texture-native throughout execution.
        private static ushort FloatToHalfBits(float value)
        {
            var bits = unchecked((uint)BitConverter.ToInt32(BitConverter.GetBytes(value), 0));
            var sign = (bits >> 16) & 0x8000u;
            var exponent = (int)((bits >> 23) & 0xff) - 127 + 15;
            var mantissa = bits & 0x7fffffu;
            if (exponent <= 0)
            {
                if (exponent < -10)
                    return (ushort)sign;
                mantissa = (mantissa | 0x800000u) >> (1 - exponent);
                return (ushort)(sign | ((mantissa + 0x1000u) >> 13));
            }
            if (exponent >= 31)
            {
                // Preserve infinities and make NaNs quiet in the half representation.
                return (ushort)(sign | 0x7c00u | (mantissa == 0 ? 0u : 0x0200u));
            }
            return (ushort)(sign | ((uint)exponent << 10) | ((mantissa + 0x1000u) >> 13));
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
                try
                {
                    ReturnTempArray(cmd, tensor.texture);
                }
                catch (Exception exception) when (_activeCommandBufferRtArena != null)
                {
                    throw new InvalidOperationException(
                        "Failed to release a planned CommandBuffer Pack4 activation"
                        + " | layer=" + (_currentExecutingLayerName ?? string.Empty)
                        + " | layer_index=" + _currentExecutingLayerIndex.ToString(CultureInfo.InvariantCulture)
                        + " | arena_slot=" + tensor.texture.arenaSlot.ToString(CultureInfo.InvariantCulture)
                        + " | texture=" + (tensor.texture.trackerLabel ?? string.Empty),
                        exception);
                }
                catch
                {
                    // Non-arena legacy cleanup remains best effort. Strict CommandBuffer
                    // inference takes the branch above and must never hide a lifetime fault.
                }
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
            EnsureNcnnTextureTensor(tr);
            return tr;
        }

        internal static AexisTensorBuffer TryGetBufferView(
            string name,
            Dictionary<string, ComputeBuffer> bufferBlobs,
            Dictionary<string, AexisTensorBuffer> bufferViews)
        {
            if (bufferViews.TryGetValue(name, out var view) && view != null && view.buffer != null)
                return view;
            if (bufferBlobs.TryGetValue(name, out var buf) && buf != null)
            {
                view = new AexisTensorBuffer(buf, 1, buf.count, 1, 1, 1, false);
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
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
                bufferViews[name] = new AexisTensorBuffer(convertedLinear, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
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
                bufferViews[name] = new AexisTensorBuffer(convertedExact, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
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
                bufferViews[name] = new AexisTensorBuffer(converted, shape.dims, shape.w, shape.h, shape.d, shape.c, false);
                tempOwned.Add(converted);
                if (emitMaterializeLog)
                    DebugLog("[BufferMaterialize] convert-done | site=" + site + " | name=" + name + " | mode=partial | count=" + converted.count);
                return converted;
            }

            throw new InvalidOperationException("texture logical shape mismatch: " + name + " | physical=" + physicalCount + " logical=" + logicalCount);
        }

        internal static float[] ReadClipArrayAsFloat32(AexisWeightReader br, int count, int loadType)
        {
            if (br == null)
                throw new ArgumentNullException(nameof(br));
            if (count < 0)
                throw new ArgumentOutOfRangeException(nameof(count));
            if (count == 0)
                return Array.Empty<float>();
            return br.ReadTensorAsFloat32(count, 0, 0, 0, loadType);
        }

        internal static float[] ReadClipMatAsFloat32(AexisWeightReader br, int w, int h, int d, int c, int loadType)
        {
            int count;
            if (d != 0) count = checked(w * h * d * c);
            else if (c != 0) count = checked(w * h * c);
            else if (h != 0) count = checked(w * h);
            else if (w != 0) count = w;
            else count = 1;
            if (count == 0) return Array.Empty<float>();
            return br.ReadTensorAsFloat32(w, h, d, c, loadType);
        }

        internal static float[] ReadPackedOrRawWeightArray(AexisWeightReader br, int count, string layerName)
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

        internal static float[] RunGemmCpu(ComputeBuffer aBuf, AexisTensorBuffer aView, GemmPack gp, float[] cDataOverride = null)
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

        internal AexisTensorBuffer RunMatMulLayer(ComputeBuffer aBuf, AexisTensorBuffer aView, ComputeBuffer bBuf, AexisTensorBuffer bView, bool transB)
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

            static void GetMatrixShape(AexisTensorBuffer view, out int rows, out int cols)
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

            static int GetBatchDepth(AexisTensorBuffer view)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                return view.dims == 4 ? view.d : 1;
            }

            static int GetBatchChannels(AexisTensorBuffer view)
            {
                if (view == null)
                    throw new ArgumentNullException(nameof(view));
                if (view.dims == 4 || view.dims == 3)
                    return view.c;
                return 1;
            }

            static int GetMatrixOffset(AexisTensorBuffer view, int depthIndex, int channelIndex)
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
            Dictionary<string, AexisTensorBuffer> bufferViews,
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

        internal static AexisTensorBuffer ResolveReshapeTensor(AexisTensorBuffer src, AexisGraphModel.Layer layer)
        {
            return ResolveReshapeTensor(src, layer, null);
        }

        internal static AexisTensorBuffer ResolveReshapeTensor(AexisTensorBuffer src, AexisGraphModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
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

        internal static BufferShape ResolveReshapeShape(BufferShape src, AexisGraphModel.Layer layer)
        {
            return ResolveReshapeShape(src, layer, null);
        }

        internal static BufferShape ResolveReshapeShape(BufferShape src, AexisGraphModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
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

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, AexisTensorBuffer src, AexisGraphModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            return EvaluateReshapeShapeExpression(expr, new[] { new BufferShape(src.dims, src.w, src.h, src.d, src.c) }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, BufferShape src, AexisGraphModel.Layer layer)
        {
            return EvaluateReshapeShapeExpression(expr, new[] { src }, layer);
        }

        internal static BufferShape EvaluateReshapeShapeExpression(string expr, IReadOnlyList<BufferShape> bottomShapes, AexisGraphModel.Layer layer)
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

        internal static IReadOnlyList<int> EvaluateExpressionList(string expr, IReadOnlyList<BufferShape> bottomShapes, AexisGraphModel.Layer layer)
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

        internal static int[] EvaluateExpressionListOrNull(string expr, IReadOnlyList<BufferShape> bottomShapes, AexisGraphModel.Layer layer)
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

        private static ExprValue PopExpr(Stack<ExprValue> stack, AexisGraphModel.Layer layer)
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

        private static int GetShapeRefValue(string token, IReadOnlyList<BufferShape> bottomShapes, AexisGraphModel.Layer layer)
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

        private static ExprValue ApplyBinaryIntPref(string op, ExprValue a, ExprValue b, AexisGraphModel.Layer layer)
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
                if (layer.type != AexisLayerTypes.Convolution
                    && layer.type != AexisLayerTypes.ConvolutionDepthWise)
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

        internal static BufferShape ResolvePermuteShape(AexisTensorBuffer src, int dims, Vector4Int axes)
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

        internal static BufferShape GetShapeOf(AexisTensorBuffer view)
        {
            if (view == null)
                throw new ArgumentNullException(nameof(view));
            return new BufferShape(view.dims, view.w, view.h, view.d, view.c);
        }

        internal static BufferShape ResolveSqueezeShape(BufferShape src, AexisGraphModel.Layer layer)
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

        internal static AexisTensorBuffer ResolveSqueezeView(AexisTensorBuffer src, AexisGraphModel.Layer layer)
        {
            if (src == null)
                throw new ArgumentNullException(nameof(src));
            var shape = ResolveSqueezeShape(GetShapeOf(src), layer);
            return src.Reshape(shape.dims, shape.w, shape.h, shape.d, shape.c);
        }

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, AexisGraphModel.Layer layer, IReadOnlyList<BufferShape> bottomShapes)
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

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, BufferShape referenceShape, AexisGraphModel.Layer layer)
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

        internal static CropRoi ResolveCropRoi(BufferShape srcShape, int[] paramData, AexisGraphModel.Layer layer)
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

        internal AexisTensorBuffer ApplyCrop(
            ComputeBuffer srcBuf,
            AexisTensorBuffer srcView,
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
                var outView = new AexisTensorBuffer(outBuf, currentView.dims, outW, outH, outD, outC, false);
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

        internal AexisTensorBuffer ShuffleChannelCpu(ComputeBuffer srcBuffer, AexisTensorBuffer srcView, int group, bool reverse)
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
            return new AexisTensorBuffer(outBuffer, srcView.dims, srcView.w, srcView.h, srcView.d, srcView.c, false);
        }

        internal static (int mode, int size, int total, AexisTensorBuffer outputView) ResolveBinaryBroadcast(
            AexisTensorBuffer aView,
            AexisTensorBuffer bView,
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

        private static bool IsChannelVectorBroadcastSource(AexisTensorBuffer vectorView, AexisTensorBuffer tensorView)
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
            AexisTensorBuffer sourceView,
            AexisTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out AexisTensorBuffer expandedView)
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
            expandedView = new AexisTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand1DTo2DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            AexisTensorBuffer sourceView,
            AexisTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out AexisTensorBuffer expandedView)
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
            expandedView = new AexisTensorBuffer(expandedBuffer, 2, targetView.w, targetView.h, 1, 1, false);
            return true;
        }

        internal bool TryExpand3DBroadcastBuffer(
            ComputeBuffer sourceBuffer,
            AexisTensorBuffer sourceView,
            AexisTensorBuffer targetView,
            out ComputeBuffer expandedBuffer,
            out AexisTensorBuffer expandedView)
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
            expandedView = new AexisTensorBuffer(expandedBuffer, 3, targetView.w, targetView.h, 1, targetView.c, false);
            return true;
        }

        public static Dictionary<string, int> BuildBlobUseCount(AexisGraphModel model)
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

        public static float ParseLeakySlope(AexisGraphModel.Layer layer)
        {
            if (layer.intParams == null || !layer.intParams.TryGetValue(-23310, out var s) || string.IsNullOrWhiteSpace(s))
                return 0.2f;
            var parts = s.Split(',');
            if (parts.Length >= 2 && float.TryParse(parts[1], NumberStyles.Float, CultureInfo.InvariantCulture, out var v))
                return v;
            return 0.2f;
        }

        public static float[] ParseActivationParams(AexisGraphModel.Layer layer)
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

        public static (float coeffA, float coeffB) ParseEltwiseCoeff(AexisGraphModel.Layer layer)
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

        public static Vector4[] PackWeightsToO4I4K2D(float[] w, int outC, int inC, int kernelW, int kernelH, int outPacks, int inPacks)
        {
            if (kernelW == kernelH)
                return PackWeightsToO4I4K(w, outC, inC, kernelW, outPacks, inPacks);
            if (w == null || outC <= 0 || inC <= 0 || kernelW <= 0 || kernelH <= 0)
                throw new ArgumentException("Invalid rectangular convolution weight profile.");
            var kernelArea = checked(kernelW * kernelH);
            if (w.Length != checked(outC * inC * kernelArea))
                throw new ArgumentException("Rectangular convolution weight count does not match OIHW.");
            var packed = new Vector4[checked(outPacks * inPacks * kernelArea * 4)];
            for (var op = 0; op < outPacks; op++)
            for (var ip = 0; ip < inPacks; ip++)
            for (var k = 0; k < kernelArea; k++)
            for (var lane = 0; lane < 4; lane++)
            {
                var oc = op * 4 + lane;
                var value = Vector4.zero;
                for (var icLane = 0; icLane < 4; icLane++)
                {
                    var ic = ip * 4 + icLane;
                    if (oc < outC && ic < inC)
                        value[icLane] = w[(oc * inC + ic) * kernelArea + k];
                }
                packed[((op * inPacks + ip) * kernelArea + k) * 4 + lane] = value;
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

        // One-to-one CDHW depthwise weights are stored as one float4 per
        // [output-pack,kz,ky,kx]. Activations stay in Texture2DArray slices;
        // this is an immutable upload format only, never an activation buffer.
        public static Vector4[] PackDepthWiseWeightsToP4KdKhKw(float[] w, int channels, int kernelW, int kernelH, int kernelD, int packs)
        {
            if (w == null) throw new ArgumentNullException(nameof(w));
            if (channels <= 0 || kernelW <= 0 || kernelH <= 0 || kernelD <= 0 || packs <= 0)
                throw new ArgumentOutOfRangeException(nameof(channels));
            var kernelVolume = checked(kernelW * kernelH * kernelD);
            if (w.Length != checked(channels * kernelVolume))
                throw new ArgumentException("Depthwise 3D weight length does not match channels*kernel volume.", nameof(w));
            var packed = new Vector4[checked(packs * kernelVolume)];
            for (var pack = 0; pack < packs; pack++)
            {
                for (var kz = 0; kz < kernelD; kz++)
                {
                    for (var ky = 0; ky < kernelH; ky++)
                    {
                        for (var kx = 0; kx < kernelW; kx++)
                        {
                            var kernelIndex = (kz * kernelH + ky) * kernelW + kx;
                            var value = Vector4.zero;
                            for (var lane = 0; lane < 4; lane++)
                            {
                                var channel = pack * 4 + lane;
                                if (channel < channels)
                                    value[lane] = w[channel * kernelVolume + kernelIndex];
                            }
                            packed[pack * kernelVolume + kernelIndex] = value;
                        }
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
            AexisGpuResourceTracker.RegisterBuffer(pack.rawWeight, weights.Length, sizeof(float), "AexisGraphSession.ConvRawWeight");
            pack.rawBias = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.rawBias, bias.Length, sizeof(float), "AexisGraphSession.ConvRawBias");
            pack.rawWeight.SetData(weights);
            pack.rawBias.SetData(bias);
        }

        internal static void UploadInt8WeightOnlyConvWeights(ConvPack pack, float[] weights, float[] bias, string layerName)
        {
            if (pack == null)
                throw new ArgumentNullException(nameof(pack));
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (bias == null)
                throw new ArgumentNullException(nameof(bias));
            if (pack.outC <= 0 || weights.Length % pack.outC != 0)
                throw new ArgumentException("Convolution INT8 weight data must be divisible by output channels.", nameof(weights));

            var quantized = NewInt8WeightOnlyUpload(
                weights,
                pack.outC,
                weights.Length / pack.outC,
                outputChannelsAreContiguous: true,
                "AexisGraphSession.ConvInt8WeightOnly:" + (layerName ?? string.Empty));
            pack.rawWeightInt8Packed = quantized.packedWeights;
            pack.rawWeightInt8Scales = quantized.scales;
            pack.rawBias = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.rawBias, bias.Length, sizeof(float), "AexisGraphSession.ConvRawBias:" + (layerName ?? string.Empty));
            pack.rawBias.SetData(bias);
        }

        internal static void UploadInt4WeightOnlyConvWeights(ConvPack pack, float[] weights, float[] bias, string layerName)
        {
            if (pack == null)
                throw new ArgumentNullException(nameof(pack));
            if (weights == null)
                throw new ArgumentNullException(nameof(weights));
            if (bias == null)
                throw new ArgumentNullException(nameof(bias));
            if (pack.outC <= 0 || weights.Length % pack.outC != 0)
                throw new ArgumentException("Convolution INT4 weight data must be divisible by output channels.", nameof(weights));

            var quantized = NewInt4WeightOnlyUpload(
                weights,
                pack.outC,
                weights.Length / pack.outC,
                outputChannelsAreContiguous: true,
                "AexisGraphSession.ConvInt4WeightOnly:" + (layerName ?? string.Empty));
            pack.rawWeightInt4Packed = quantized.packedWeights;
            pack.rawWeightInt4Scales = quantized.scales;
            pack.rawBias = new ComputeBuffer(bias.Length, sizeof(float), ComputeBufferType.Structured);
            AexisGpuResourceTracker.RegisterBuffer(pack.rawBias, bias.Length, sizeof(float), "AexisGraphSession.ConvRawBias:" + (layerName ?? string.Empty));
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
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            Dictionary<string, AexisTensorBuffer> bufferViews,
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
            AexisTensorBuffer srcView,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padTop,
            AexisTensorBuffer outValue,
            AexisTensorBuffer outIndex)
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
            AexisTensorBuffer pooledView,
            ComputeBuffer indexBuffer,
            AexisTensorBuffer indexView,
            int outW,
            int outH,
            AexisTensorBuffer outTensor)
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

