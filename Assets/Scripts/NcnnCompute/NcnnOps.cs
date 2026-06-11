using System;
using System.Globalization;
using System.Runtime.InteropServices;
using UnityEngine;
using UnityEngine.Rendering;

namespace UnityEngine
{
    [System.Serializable]
    [StructLayout(LayoutKind.Sequential)]
    public struct Vector4Int : IEquatable<Vector4Int>, IFormattable
    {
        public int x { get { return m_X; } set { m_X = value; } }
        public int y { get { return m_Y; } set { m_Y = value; } }
        public int z { get { return m_Z; } set { m_Z = value; } }
        public int w { get { return m_W; } set { m_W = value; } }

        private int m_X;
        private int m_Y;
        private int m_Z;
        private int m_W;

        public Vector4Int(int x, int y, int z, int w)
        {
            m_X = x;
            m_Y = y;
            m_Z = z;
            m_W = w;
        }

        // Set x, y and z components of an existing Vector.
        public void Set(int x, int y, int z, int w)
        {
            m_X = x;
            m_Y = y;
            m_Z = z;
            m_W = w;
        }

        // Access the /x/, /y/ or /z/ component using [0], [1] or [2] respectively.
        public int this[int index]
        {
            get
            {
                switch (index)
                {
                    case 0: return x;
                    case 1: return y;
                    case 2: return z;
                    case 3: return w;
                    default:
                        throw new IndexOutOfRangeException(string.Format("Invalid Vector4Int index addressed: {0}!", index));
                }
            }

            set
            {
                switch (index)
                {
                    case 0: x = value; break;
                    case 1: y = value; break;
                    case 2: z = value; break;
                    case 3: w = value; break;
                    default:
                        throw new IndexOutOfRangeException(string.Format("Invalid Vector4Int index addressed: {0}!", index));
                }
            }
        }

        // Returns the length of this vector (RO).
        public float magnitude { get { return Mathf.Sqrt((float)(x * x + y * y + z * z + w * w)); } }

        // Returns the squared length of this vector (RO).
        public int sqrMagnitude { get { return x * x + y * y + z * z + w * w; } }

        // Returns the distance between /a/ and /b/.
        public static float Distance(Vector4Int a, Vector4Int b) { return (a - b).magnitude; }

        // Returns a vector that is made from the smallest components of two vectors.
        public static Vector4Int Min(Vector4Int lhs, Vector4Int rhs) { return new Vector4Int(Mathf.Min(lhs.x, rhs.x), Mathf.Min(lhs.y, rhs.y), Mathf.Min(lhs.z, rhs.z), Mathf.Min(lhs.w, rhs.w)); }

        // Returns a vector that is made from the largest components of two vectors.
        public static Vector4Int Max(Vector4Int lhs, Vector4Int rhs) { return new Vector4Int(Mathf.Max(lhs.x, rhs.x), Mathf.Max(lhs.y, rhs.y), Mathf.Max(lhs.z, rhs.z), Mathf.Max(lhs.w, rhs.w)); }

        // Multiplies two vectors component-wise.
        public static Vector4Int Scale(Vector4Int a, Vector4Int b) { return new Vector4Int(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w); }

        // Multiplies every component of this vector by the same component of /scale/.
        public void Scale(Vector4Int scale) { x *= scale.x; y *= scale.y; z *= scale.z; w *= scale.w; }

        public void Clamp(Vector4Int min, Vector4Int max)
        {
            x = Math.Max(min.x, x);
            x = Math.Min(max.x, x);
            y = Math.Max(min.y, y);
            y = Math.Min(max.y, y);
            z = Math.Max(min.z, z);
            z = Math.Min(max.z, z);
            w = Math.Max(min.w, w);
            w = Math.Min(max.w, w);
        }

        // Converts a Vector4Int to a [[Vector4]].
        public static implicit operator Vector4(Vector4Int v)
        {
            return new Vector4(v.x, v.y, v.z, v.w);
        }

        // Converts a Vector4Int to a [[Vector2Int]].
        public static explicit operator Vector3Int(Vector4Int v)
        {
            return new Vector3Int(v.x, v.y, v.z);
        }

        // Converts a Vector4Int to a [[Vector2Int]].
        public static explicit operator Vector2Int(Vector4Int v)
        {
            return new Vector2Int(v.x, v.y);
        }

        public static Vector4Int FloorToInt(Vector4 v)
        {
            return new Vector4Int(
                Mathf.FloorToInt(v.x),
                Mathf.FloorToInt(v.y),
                Mathf.FloorToInt(v.z),
                Mathf.FloorToInt(v.w)
            );
        }

        public static Vector4Int CeilToInt(Vector4 v)
        {
            return new Vector4Int(
                Mathf.CeilToInt(v.x),
                Mathf.CeilToInt(v.y),
                Mathf.CeilToInt(v.z),
                Mathf.CeilToInt(v.w)
            );
        }

        public static Vector4Int RoundToInt(Vector4 v)
        {
            return new Vector4Int(
                Mathf.RoundToInt(v.x),
                Mathf.RoundToInt(v.y),
                Mathf.RoundToInt(v.z),
                Mathf.RoundToInt(v.w)
            );
        }

        public static Vector4Int operator +(Vector4Int a, Vector4Int b)
        {
            return new Vector4Int(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
        }

        public static Vector4Int operator -(Vector4Int a, Vector4Int b)
        {
            return new Vector4Int(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
        }

        public static Vector4Int operator *(Vector4Int a, Vector4Int b)
        {
            return new Vector4Int(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
        }

        public static Vector4Int operator -(Vector4Int a)
        {
            return new Vector4Int(-a.x, -a.y, -a.z, -a.w);
        }

        public static Vector4Int operator *(Vector4Int a, int b)
        {
            return new Vector4Int(a.x * b, a.y * b, a.z * b, a.w * b);
        }

        public static Vector4Int operator *(int a, Vector4Int b)
        {
            return new Vector4Int(a * b.x, a * b.y, a * b.z, a * b.w);
        }

        public static Vector4Int operator /(Vector4Int a, int b)
        {
            return new Vector4Int(a.x / b, a.y / b, a.z / b, a.w / b);
        }

        public static bool operator ==(Vector4Int lhs, Vector4Int rhs)
        {
            return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z && lhs.w == rhs.w;
        }

        public static bool operator !=(Vector4Int lhs, Vector4Int rhs)
        {
            return !(lhs == rhs);
        }

        public override bool Equals(object other)
        {
            if (!(other is Vector4Int)) return false;

            return Equals((Vector4Int)other);
        }

        public bool Equals(Vector4Int other)
        {
            return this == other;
        }

        public override int GetHashCode()
        {
            var yHash = y.GetHashCode();
            var zHash = z.GetHashCode();
            var wHash = w.GetHashCode();
            return x.GetHashCode() ^ (yHash << 8) ^ (yHash >> 24) ^ (zHash << 16) ^ (zHash >> 16) ^ (wHash << 24) ^ (wHash >> 8);
        }

        public override string ToString()
        {
            return ToString(null, CultureInfo.InvariantCulture.NumberFormat);
        }

        public string ToString(string format)
        {
            return ToString(format, CultureInfo.InvariantCulture.NumberFormat);
        }

        public string ToString(string format, IFormatProvider formatProvider)
        {
            return string.Format("({0}, {1}, {2}, {3})", x.ToString(format, formatProvider), y.ToString(format, formatProvider), z.ToString(format, formatProvider), w.ToString(format, formatProvider));
        }

        public static Vector4Int zero { get { return s_Zero; } }
        public static Vector4Int one { get { return s_One; } }

        private static readonly Vector4Int s_Zero = new Vector4Int(0, 0, 0, 0);
        private static readonly Vector4Int s_One = new Vector4Int(1, 1, 1, 1);
    }
}

namespace NcnnCompute
{
    public class ComputeTexture
    {
        public int nameID;
        public int width;
        public int height;
        public int depth;
        public RenderTextureFormat format;
        public string trackerLabel;
    }

    public sealed class NcnnOps
    {
        public enum PointwiseType
        {
            Elu = 0,
            Erf = 1,
            HardSigmoid = 2,
            HardSwish = 3,
            Mish = 4,
            Selu = 5,
            Shrink = 6,
            Softplus = 7,
            Celu = 8,
            ScaleScalar = 9
        }

        private readonly ComputeShader _cs;
        private readonly int _kConv3x3;
        private readonly int _kConv3dBuf;
        private readonly int _kConv3dPack4Cdhw;
        private readonly int _kDeconvolutionBuf;
        private readonly int _kDeconvolution3dPack4Cdhw;
        private readonly int _kDeconvolution3dBuf;
        private readonly int _kTexToBuf3;
        private readonly int _kBufToTex3;
        private readonly int _kLeakyReluBuf;
        private readonly int _kAddWeighted;
        private readonly int _kCopyBuf;
        private readonly int _kCopyBufPartial;
        private readonly int _kBinaryOpBuf;
        private readonly int _kUnaryOpBuf;
        private readonly int _kSigmoidBuf;
        private readonly int _kSwishBuf;
        private readonly int _kGeluBuf;
        private readonly int _kPointwiseBuf;
        private readonly int _kCopyC;
        private readonly int _kInterp2x;
        private readonly int _kBlitTileToDst;
        private readonly int _kPackRgbToPack4;
        private readonly int _kConv3x3Pack4;
        private readonly int _kConvPack4General;
        private readonly int _kDeconvolutionPack4General;
        private readonly int _kConvDepthWisePack4;
        private readonly int _kWinograd23TransformInput;
        private readonly int _kWinograd23Gemm;
        private readonly int _kWinograd23TransformOutput;
        private readonly int _kConv1x1Pack4;

        private ComputeBuffer _winoBottomTm;
        private ComputeBuffer _winoTopTm;
        private int _winoBottomCap;
        private int _winoTopCap;
        private ComputeBuffer _gpuIdleSync;
        private readonly uint[] _gpuIdleScratch = new uint[1];
        private readonly int _kAddPack4;
        private readonly int _kCopyPack4;
        private readonly int _kConcatPack4Cdhw;
        private readonly int _kBuildSdInpaintInput9Pack4;
        private readonly int _kInterpPack4;
        private readonly int _kInterpPack4Cdhw;
        private readonly int _kInterp2xPack4;
        private readonly int _kInterp2xNearestPack4;
        private readonly int _kInterpDown2Pack4;
        private readonly int _kInterpDown2NearestPack4;
        private readonly int _kPack4ToBufferChw;
        private readonly int _kPack4ChannelsToWidth;
        private readonly int _kPack4ToBufferCdhw;
        private readonly int _kInnerProduct;
        private readonly int _kPackRgbToPack4Gfpgan;
        private readonly int _kFillPack4FromBufferChw;
        private readonly int _kFillPack4FromBufferCdhw;
        private readonly int _kScalePack4;
        private readonly int _kAddBiasPack4;
        private readonly int _kBatchNormPack4;
        private readonly int _kLeakyReluPack4;
        private readonly int _kAddNoiseBroadcastPack4;
        private readonly int _kClipPack4;
        private readonly int _kSftPack4;
        private readonly int _kPack4ToRgb01;
        private readonly int _kProbeTilePack4;
        private readonly int _kProbeSeams;
        private readonly int _kPaddingPack4;
        private readonly int _kPoolingPack4;
        private readonly int _kPoolingPack4Cdhw;
        private readonly int _kMaxPoolingIndPack4;
        private readonly int _kMaxPoolingIndicesFromValuePack4;
        private readonly int _kMaxUnPoolingPack4;
        private readonly int _kSoftmaxChannelPack4;
        private readonly int _kUnaryOpPack4;
        private readonly int _kBinaryOpPack4;
        private readonly int _kBinaryOpPack4Broadcast;
        private readonly int _kBinaryOpPack4BufferScalar;
        private readonly int _kShuffleChannelPack4;
        private readonly int _kCropPack4;
        private readonly int _kSlicePack4;
        private readonly int _kSlicePack4Cdhw;
        private readonly int _kPermutePack4;
        private readonly int _kPermutePack4Cdhw;
        private readonly int _kWindowPartitionPack4;
        private readonly int _kWindowUnpartitionPack4;
        private readonly int _kReshapePack4ToScalar2D;
        private readonly int _kReshapePack4ToPack4;
        private readonly int _kReshapeScalar2DToPack4;
        private readonly int _kAttentionReshapePack4;
        private readonly int _kSwishPack4;
        private readonly int _kSigmoidPack4;
        private readonly int _kGeluPack4;
        private readonly int _kMatMul2D;
        private readonly int _kMatMulPack4Cdhw;
        private readonly int _kVistaTailPromptDotPack4;
        private readonly int _kGemm2DTextureA;
        private readonly int _kGemm2D;
        private readonly int _kGemm2D16;
        private readonly int _kLayerNorm2D;
        private readonly int _kSoftmax2D;
        private readonly int _kSoftmaxPack4Cdhw;
        private readonly int _kEmbed;
        private readonly int _kPermute;
        private readonly int _kSlice;
        private readonly int _kTile;
        private readonly int _kReduceSum256;
        private readonly int _kMulScalarBuf;
        private readonly int _kGroupNormStats;
        private readonly int _kGroupNormApply;
        private readonly int _kGroupNormMean;
        private readonly int _kGroupNormVariance;
        private readonly int _kGroupNormApplyMeanVar;
        private readonly int _kGroupNormPack4Mean;
        private readonly int _kGroupNormPack4Variance;
        private readonly int _kGroupNormPack4ApplyMeanVar;
        private readonly int _kGroupNormPack4MeanTex;
        private readonly int _kGroupNormPack4VarianceTex;
        private readonly int _kGroupNormPack4ApplyMeanVarTex;
        private readonly int _kTouchU32;
        private readonly int _kInnerProduct2D;
        private readonly int _kMhaAttention;
        private readonly int _kMhaAttentionFast;
        private readonly int _kMhaAttentionQkvFast;
        private readonly int _kMhaProjectQkv2D;
        private readonly int _kReorgPack4;
        private readonly int _kPointwisePack4;
        private readonly int _kPixelShufflePack4;
        private readonly int _kCastBuf;
        private readonly int _kScaleBuf;
        private readonly int _kPReluBuf;
        private readonly int _kReorgBuf;
        private readonly int _kReductionRowsBuf;
        private readonly int _kConv1dBuf;
        private readonly int _kConvDepthWise;
        private readonly int _kQuantizeBuf;
        private readonly int _kDequantizeBuf;
        private readonly int _kRequantizeBuf;
        private readonly int _kPixelShuffleBuf;
        private readonly int _kRotaryEmbedBuf;
        private readonly int _kNormalizeBuf;
        private readonly int _kLrnBuf;
        private readonly int _kRmsNormBuf;
        private readonly int _kUnfoldBuf;
        private readonly int _kSdpaQkBuf;
        private readonly int _kSdpaSoftmaxBuf;
        private readonly int _kSdpaQkvBuf;
        private readonly int _kSdpaAttentionFast;

        private static int ResolveRenderTextureDispatchDepth(RenderTexture output, int fallbackPacks)
        {
            if (output == null)
                return Mathf.Max(1, fallbackPacks);
            return Mathf.Max(1, output.volumeDepth > 0 ? output.volumeDepth : fallbackPacks);
        }

        private static int ResolveComputeTextureDispatchDepth(ComputeTexture output, int fallbackPacks)
        {
            if (output == null)
                return Mathf.Max(1, fallbackPacks);
            return Mathf.Max(1, output.depth > 0 ? output.depth : fallbackPacks);
        }

        public NcnnOps()
        {
            _cs = Resources.Load<ComputeShader>("NcnnCompute");
            if (_cs == null)
                throw new InvalidOperationException("ComputeShader not found: Resources/NcnnCompute.compute");
            _kConv3x3 = _cs.FindKernel("NcnnConv3x3");
            _kConv3dBuf = _cs.FindKernel("NcnnConv3dBuf");
            _kConv3dPack4Cdhw = _cs.FindKernel("NcnnConv3dPack4CDHW");
            _kDeconvolutionBuf = _cs.FindKernel("NcnnDeconvolutionBuf");
            _kDeconvolution3dPack4Cdhw = _cs.FindKernel("NcnnDeconvolution3dPack4CDHW");
            _kDeconvolution3dBuf = _cs.FindKernel("NcnnDeconvolution3dBuf");
            _kConvDepthWise = _cs.FindKernel("NcnnConvDepthWise");
            _kTexToBuf3 = _cs.FindKernel("NcnnTexToBuf3");
            _kBufToTex3 = _cs.FindKernel("NcnnBufToTex3");
            _kLeakyReluBuf = _cs.FindKernel("NcnnLeakyReluBuf");
            _kAddWeighted = _cs.FindKernel("NcnnAddWeighted");
            _kCopyBuf = _cs.FindKernel("NcnnCopyBuf");
            _kCopyBufPartial = _cs.FindKernel("NcnnCopyBufPartial");
            _kBinaryOpBuf = _cs.FindKernel("NcnnBinaryOpBuf");
            _kUnaryOpBuf = _cs.FindKernel("NcnnUnaryOpBuf");
            _kSigmoidBuf = _cs.FindKernel("NcnnSigmoidBuf");
            _kSwishBuf = _cs.FindKernel("NcnnSwishBuf");
            _kGeluBuf = _cs.FindKernel("NcnnGeluBuf");
            _kPointwiseBuf = _cs.FindKernel("NcnnPointwiseBuf");
            _kCopyC = _cs.FindKernel("NcnnCopyC");
            _kInterp2x = _cs.FindKernel("NcnnInterp2x");
            _kBlitTileToDst = _cs.FindKernel("NcnnBlitTileToDst");
            _kPackRgbToPack4 = _cs.FindKernel("NcnnPackRgbToPack4");
            _kConv3x3Pack4 = _cs.FindKernel("NcnnConv3x3Pack4");
            _kConvPack4General = _cs.FindKernel("NcnnConvPack4General");
            _kDeconvolutionPack4General = _cs.FindKernel("NcnnDeconvolutionPack4General");
            _kConvDepthWisePack4 = _cs.FindKernel("NcnnConvDepthWisePack4");
            _kWinograd23TransformInput = _cs.FindKernel("NcnnWinograd23TransformInputPack4");
            _kWinograd23Gemm = _cs.FindKernel("NcnnWinograd23GemmPack4");
            _kWinograd23TransformOutput = _cs.FindKernel("NcnnWinograd23TransformOutputPack4");
            _kConv1x1Pack4 = _cs.FindKernel("NcnnConv1x1Pack4");
            _kAddPack4 = _cs.FindKernel("NcnnAddPack4");
            _kCopyPack4 = _cs.FindKernel("NcnnCopyPack4");
            _kConcatPack4Cdhw = _cs.FindKernel("NcnnConcatPack4CDHW");
            _kBuildSdInpaintInput9Pack4 = _cs.FindKernel("NcnnBuildSdInpaintInput9Pack4");
            _kInterpPack4 = _cs.FindKernel("NcnnInterpPack4");
            _kInterpPack4Cdhw = _cs.FindKernel("NcnnInterpPack4CDHW");
            _kInterp2xPack4 = _cs.FindKernel("NcnnInterp2xPack4");
            _kInterp2xNearestPack4 = _cs.FindKernel("NcnnInterp2xNearestPack4");
            _kInterpDown2Pack4 = _cs.FindKernel("NcnnInterpDown2Pack4");
            _kInterpDown2NearestPack4 = _cs.FindKernel("NcnnInterpDown2NearestPack4");
            _kPack4ToBufferChw = _cs.FindKernel("NcnnPack4ToBufferCHW");
            _kPack4ChannelsToWidth = _cs.FindKernel("NcnnPack4ChannelsToWidth");
            _kPack4ToBufferCdhw = _cs.FindKernel("NcnnPack4ToBufferCDHW");
            _kInnerProduct = _cs.FindKernel("NcnnInnerProduct");
            _kPackRgbToPack4Gfpgan = _cs.FindKernel("NcnnPackRgbToPack4Gfpgan");
            _kFillPack4FromBufferChw = _cs.FindKernel("NcnnFillPack4FromBufferCHW");
            _kFillPack4FromBufferCdhw = _cs.FindKernel("NcnnFillPack4FromBufferCDHW");
            _kScalePack4 = _cs.FindKernel("NcnnScalePack4");
            _kAddBiasPack4 = _cs.FindKernel("NcnnAddBiasPack4");
            _kBatchNormPack4 = _cs.FindKernel("NcnnBatchNormPack4");
            _kLeakyReluPack4 = _cs.FindKernel("NcnnLeakyReluPack4");
            _kAddNoiseBroadcastPack4 = _cs.FindKernel("NcnnAddNoiseBroadcastPack4");
            _kClipPack4 = _cs.FindKernel("NcnnClipPack4");
            _kSftPack4 = _cs.FindKernel("NcnnSftPack4");
            _kPack4ToRgb01 = _cs.FindKernel("NcnnPack4ToRgb01");
            _kProbeTilePack4 = _cs.FindKernel("NcnnProbeTilePack4");
            _kProbeSeams = _cs.FindKernel("NcnnProbeSeams");
            _kPaddingPack4 = _cs.FindKernel("NcnnPaddingPack4");
            _kPoolingPack4 = _cs.FindKernel("NcnnPoolingPack4");
            _kPoolingPack4Cdhw = _cs.FindKernel("NcnnPoolingPack4CDHW");
            _kMaxPoolingIndPack4 = _cs.FindKernel("NcnnMaxPoolingIndPack4");
            _kMaxPoolingIndicesFromValuePack4 = _cs.FindKernel("NcnnMaxPoolingIndicesFromValuePack4");
            _kMaxUnPoolingPack4 = _cs.FindKernel("NcnnMaxUnPoolingPack4");
            _kSoftmaxChannelPack4 = _cs.FindKernel("NcnnSoftmaxChannelPack4");
            _kUnaryOpPack4 = _cs.FindKernel("NcnnUnaryOpPack4");
            _kBinaryOpPack4 = _cs.FindKernel("NcnnBinaryOpPack4");
            _kBinaryOpPack4Broadcast = _cs.FindKernel("NcnnBinaryOpPack4Broadcast");
            _kBinaryOpPack4BufferScalar = _cs.FindKernel("NcnnBinaryOpPack4BufferScalar");
            _kShuffleChannelPack4 = _cs.FindKernel("NcnnShuffleChannelPack4");
            _kCropPack4 = _cs.FindKernel("NcnnCropPack4");
            _kSlicePack4 = _cs.FindKernel("NcnnSlicePack4");
            _kSlicePack4Cdhw = _cs.FindKernel("NcnnSlicePack4CDHW");
            _kPermutePack4 = _cs.FindKernel("NcnnPermutePack4");
            _kPermutePack4Cdhw = _cs.FindKernel("NcnnPermutePack4CDHW");
            _kWindowPartitionPack4 = _cs.FindKernel("NcnnWindowPartitionPack4");
            _kWindowUnpartitionPack4 = _cs.FindKernel("NcnnWindowUnpartitionPack4");
            _kReshapePack4ToScalar2D = _cs.FindKernel("NcnnReshapePack4ToScalar2D");
            _kReshapePack4ToPack4 = _cs.FindKernel("NcnnReshapePack4ToPack4");
            _kReshapeScalar2DToPack4 = _cs.FindKernel("NcnnReshapeScalar2DToPack4");
            _kAttentionReshapePack4 = _cs.FindKernel("NcnnAttentionReshapePack4");
            _kSwishPack4 = _cs.FindKernel("NcnnSwishPack4");
            _kSigmoidPack4 = _cs.FindKernel("NcnnSigmoidPack4");
            _kGeluPack4 = _cs.FindKernel("NcnnGeluPack4");
            _kMatMul2D = _cs.FindKernel("NcnnMatMul2D");
            _kMatMulPack4Cdhw = _cs.FindKernel("NcnnMatMulPack4CDHW");
            _kVistaTailPromptDotPack4 = _cs.FindKernel("NcnnVistaTailPromptDotPack4");
            _kGemm2DTextureA = _cs.FindKernel("NcnnGemm2DTextureA");
            _kGemm2D = _cs.FindKernel("NcnnGemm2D");
            _kGemm2D16 = _cs.FindKernel("NcnnGemm2D16");
            _kLayerNorm2D = _cs.FindKernel("NcnnLayerNorm2D");
            _kSoftmax2D = _cs.FindKernel("NcnnSoftmax2D");
            _kSoftmaxPack4Cdhw = _cs.FindKernel("NcnnSoftmaxPack4CDHW");
            _kEmbed = _cs.FindKernel("NcnnEmbed");
            _kPermute = _cs.FindKernel("NcnnPermute");
            _kSlice = _cs.FindKernel("NcnnSlice");
            _kTile = _cs.FindKernel("NcnnTile");
            _kReduceSum256 = _cs.FindKernel("NcnnReduceSum256");
            _kMulScalarBuf = _cs.FindKernel("NcnnMulScalarBuf");
            _kGroupNormStats = _cs.FindKernel("NcnnGroupNormStats");
            _kGroupNormApply = _cs.FindKernel("NcnnGroupNormApply");
            _kGroupNormMean = _cs.FindKernel("NcnnGroupNormMean");
            _kGroupNormVariance = _cs.FindKernel("NcnnGroupNormVariance");
            _kGroupNormApplyMeanVar = _cs.FindKernel("NcnnGroupNormApplyMeanVar");
            _kGroupNormPack4Mean = _cs.FindKernel("NcnnGroupNormPack4Mean");
            _kGroupNormPack4Variance = _cs.FindKernel("NcnnGroupNormPack4Variance");
            _kGroupNormPack4ApplyMeanVar = _cs.FindKernel("NcnnGroupNormPack4ApplyMeanVar");
            _kGroupNormPack4MeanTex = _cs.FindKernel("NcnnGroupNormPack4MeanTex");
            _kGroupNormPack4VarianceTex = _cs.FindKernel("NcnnGroupNormPack4VarianceTex");
            _kGroupNormPack4ApplyMeanVarTex = _cs.FindKernel("NcnnGroupNormPack4ApplyMeanVarTex");
            _kTouchU32 = _cs.FindKernel("NcnnTouchU32");
            _kInnerProduct2D = _cs.FindKernel("NcnnInnerProduct2D");
            _kMhaAttention = _cs.FindKernel("NcnnMhaAttention");
            _kMhaAttentionFast = _cs.FindKernel("NcnnMhaAttentionFast");
            _kMhaAttentionQkvFast = _cs.FindKernel("NcnnMhaAttentionQkvFast");
            _kMhaProjectQkv2D = _cs.FindKernel("NcnnMhaProjectQkv2D");
            _kReorgPack4 = _cs.FindKernel("NcnnReorgPack4");
            _kPointwisePack4 = _cs.FindKernel("NcnnPointwisePack4");
            _kPixelShufflePack4 = _cs.FindKernel("NcnnPixelShufflePack4");
            _kCastBuf = _cs.FindKernel("NcnnCastBuf");
            _kScaleBuf = _cs.FindKernel("NcnnScaleBuf");
            _kPReluBuf = _cs.FindKernel("NcnnPReluBuf");
            _kReorgBuf = _cs.FindKernel("NcnnReorgBuf");
            _kReductionRowsBuf = _cs.FindKernel("NcnnReductionRowsBuf");
            _kConv1dBuf = _cs.FindKernel("NcnnConv1dBuf");
            _kQuantizeBuf = _cs.FindKernel("NcnnQuantizeBuf");
            _kDequantizeBuf = _cs.FindKernel("NcnnDequantizeBuf");
            _kRequantizeBuf = _cs.FindKernel("NcnnRequantizeBuf");
            _kPixelShuffleBuf = _cs.FindKernel("NcnnPixelShuffleBuf");
            _kRotaryEmbedBuf = _cs.FindKernel("NcnnRotaryEmbedBuf");
            _kNormalizeBuf = _cs.FindKernel("NcnnNormalizeBuf");
            _kLrnBuf = _cs.FindKernel("NcnnLrnBuf");
            _kRmsNormBuf = _cs.FindKernel("NcnnRmsNormBuf");
            _kUnfoldBuf = _cs.FindKernel("NcnnUnfoldBuf");
            _kSdpaQkBuf = _cs.FindKernel("NcnnSdpaQkBuf");
            _kSdpaSoftmaxBuf = _cs.FindKernel("NcnnSdpaSoftmaxBuf");
            _kSdpaQkvBuf = _cs.FindKernel("NcnnSdpaQkvBuf");
            _kSdpaAttentionFast = _cs.FindKernel("NcnnSdpaAttentionFast");
        }

        public void TextureToBuffer3(Texture src, int offsetX, int offsetY, NcnnTensorBuffer output)
        {
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (output.c != 3) throw new ArgumentOutOfRangeException(nameof(output), "output.c must be 3");

            _cs.SetInt("_InW", output.w);
            _cs.SetInt("_InH", output.h);
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetTexture(_kTexToBuf3, "_NcnnIn", src);
            _cs.SetBuffer(_kTexToBuf3, "_BufOut", output.buffer);

            var n = output.w * output.h;
            Dispatch1D(_kTexToBuf3, n, 256);
        }

        public void BufferToTexture3(NcnnTensorBuffer input, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (input.c != 3) throw new ArgumentOutOfRangeException(nameof(input), "input.c must be 3");
            if (input.w != output.width || input.h != output.height)
                throw new ArgumentOutOfRangeException(nameof(output), "output dimensions must match input");

            _cs.SetBuffer(_kBufToTex3, "_BufA", input.buffer);
            _cs.SetTexture(_kBufToTex3, "_NcnnOut", output);
            Dispatch2D(_kBufToTex3, output.width, output.height, 8, 8);
        }

        public void BlitTileToDst(RenderTexture tileOut, RenderTexture dst, int dstX, int dstY, int dstW, int dstH, int tilePadX, int tilePadY, int tileCoreW, int tileCoreH, int tileW, int tileH)
        {
            if (tileOut == null) throw new ArgumentNullException(nameof(tileOut));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dstW <= 0 || dstH <= 0) return;
            if (dstX < 0 || dstY < 0 || dstX + dstW > dst.width || dstY + dstH > dst.height)
                throw new ArgumentOutOfRangeException(nameof(dstX), "dst rect out of range");

            _cs.SetInt("_BlitW", dstW);
            _cs.SetInt("_BlitH", dstH);
            _cs.SetInt("_DstX", dstX);
            _cs.SetInt("_DstY", dstY);
            _cs.SetInt("_TilePadX", tilePadX);
            _cs.SetInt("_TilePadY", tilePadY);
            _cs.SetInt("_TileCoreW", tileCoreW);
            _cs.SetInt("_TileCoreH", tileCoreH);
            _cs.SetInt("_TileW", tileW);
            _cs.SetInt("_TileH", tileH);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnInArr", tileOut);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnOut", dst);
            Dispatch2D(_cs, _kBlitTileToDst, dstW, dstH, 32, 32);
        }

        public void BlitTileToDst(CommandBuffer cmd, ComputeTexture tileOut, RenderTexture dst, int dstX, int dstY, int dstW, int dstH, int tilePadX, int tilePadY, int tileCoreW, int tileCoreH, int tileW, int tileH)
        {
            if (tileOut == null) throw new ArgumentNullException(nameof(tileOut));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (dstW <= 0 || dstH <= 0) return;
            if (dstX < 0 || dstY < 0 || dstX + dstW > dst.width || dstY + dstH > dst.height)
                throw new ArgumentOutOfRangeException(nameof(dstX), "dst rect out of range");

            cmd.SetComputeIntParam(_cs, "_BlitW", dstW);
            cmd.SetComputeIntParam(_cs, "_BlitH", dstH);
            cmd.SetComputeIntParam(_cs, "_DstX", dstX);
            cmd.SetComputeIntParam(_cs, "_DstY", dstY);
            cmd.SetComputeIntParam(_cs, "_TilePadX", tilePadX);
            cmd.SetComputeIntParam(_cs, "_TilePadY", tilePadY);
            cmd.SetComputeIntParam(_cs, "_TileCoreW", tileCoreW);
            cmd.SetComputeIntParam(_cs, "_TileCoreH", tileCoreH);
            cmd.SetComputeIntParam(_cs, "_TileW", tileW);
            cmd.SetComputeIntParam(_cs, "_TileH", tileH);
            cmd.SetComputeTextureParam(_cs, _kBlitTileToDst, "_NcnnInArr", tileOut.nameID);
            cmd.SetComputeTextureParam(_cs, _kBlitTileToDst, "_NcnnOut", dst);
            Dispatch2D(cmd, _cs, _kBlitTileToDst, dstW, dstH, 32, 32);
        }


        public void PackRgbToPack4(Texture src, int offsetX, int offsetY, float sx, float sy, RenderTexture dstPack4, bool flipY = false)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetFloat("_ScaleX", sx);
            _cs.SetFloat("_ScaleY", sy);
            _cs.SetInt("_FlipY", flipY ? 1 : 0);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnOutArr", dstPack4);
            Dispatch2D(_cs, _kPackRgbToPack4, dstPack4.width, dstPack4.height, 32, 32);
        }


        public void PackRgbToPack4(CommandBuffer cmd, Texture src, int offsetX, int offsetY, float sx, float sy, ComputeTexture dstPack4, bool flipY = false)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            cmd.SetComputeIntParams(_cs, "_OffsetX", offsetX);
            cmd.SetComputeIntParams(_cs, "_OffsetY", offsetY);
            cmd.SetComputeFloatParam(_cs, "_ScaleX", sx);
            cmd.SetComputeFloatParam(_cs, "_ScaleY", sy);
            cmd.SetComputeIntParam(_cs, "_FlipY", flipY ? 1 : 0);
            cmd.SetComputeTextureParam(_cs, _kPackRgbToPack4, "_NcnnIn", src);
            cmd.SetComputeTextureParam(_cs, _kPackRgbToPack4, "_NcnnOutArr", dstPack4.nameID);
            Dispatch2D(cmd, _cs, _kPackRgbToPack4, dstPack4.width, dstPack4.height, 32, 32);
        }

  

        public void PackRgbToPack4Gfpgan(Texture src, int offsetX, int offsetY, float sx, float sy, RenderTexture dstPack4, bool flipY = false)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetFloat("_ScaleX", sx);
            _cs.SetFloat("_ScaleY", sy);
            _cs.SetInt("_FlipY", flipY ? 1 : 0);
            _cs.SetTexture(_kPackRgbToPack4Gfpgan, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4Gfpgan, "_NcnnOutArr", dstPack4);
            Dispatch2D(_kPackRgbToPack4Gfpgan, dstPack4.width, dstPack4.height, 32, 32);
        }

        public void FillPack4FromBufferCHW(ComputeBuffer input, int w, int h, int c, RenderTexture outputPack4)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputPack4 == null) throw new ArgumentNullException(nameof(outputPack4));
            _cs.SetInt("_FillW", w);
            _cs.SetInt("_FillH", h);
            _cs.SetInt("_FillD", 1);
            _cs.SetInt("_FillC", c);
            _cs.SetBuffer(_kFillPack4FromBufferChw, "_FillIn", input);
            _cs.SetTexture(_kFillPack4FromBufferChw, "_FillOutArr", outputPack4);
            Dispatch3D(_kFillPack4FromBufferChw, w, h, outputPack4.volumeDepth, 8, 8);
        }

        public void FillPack4FromBufferCHW(CommandBuffer cmd, ComputeBuffer input, int w, int h, int c, ComputeTexture outputPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputPack4 == null) throw new ArgumentNullException(nameof(outputPack4));
            cmd.SetComputeIntParam(_cs, "_FillW", w);
            cmd.SetComputeIntParam(_cs, "_FillH", h);
            cmd.SetComputeIntParam(_cs, "_FillD", 1);
            cmd.SetComputeIntParam(_cs, "_FillC", c);
            cmd.SetComputeBufferParam(_cs, _kFillPack4FromBufferChw, "_FillIn", input);
            cmd.SetComputeTextureParam(_cs, _kFillPack4FromBufferChw, "_FillOutArr", outputPack4.nameID);
            Dispatch3D(cmd, _kFillPack4FromBufferChw, w, h, Mathf.Max(1, Mathf.CeilToInt(c / 4f)), 8, 8);
        }

        public void FillPack4FromBufferCDHW(ComputeBuffer input, int w, int h, int d, int c, RenderTexture outputPack4)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputPack4 == null) throw new ArgumentNullException(nameof(outputPack4));
            _cs.SetInt("_FillW", w);
            _cs.SetInt("_FillH", h);
            _cs.SetInt("_FillD", d);
            _cs.SetInt("_FillC", c);
            _cs.SetBuffer(_kFillPack4FromBufferCdhw, "_FillIn", input);
            _cs.SetTexture(_kFillPack4FromBufferCdhw, "_FillOutArr", outputPack4);
            Dispatch3D(_kFillPack4FromBufferCdhw, w, h, outputPack4.volumeDepth, 8, 8);
        }

        public void FillPack4FromBufferCDHW(CommandBuffer cmd, ComputeBuffer input, int w, int h, int d, int c, ComputeTexture outputPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputPack4 == null) throw new ArgumentNullException(nameof(outputPack4));
            cmd.SetComputeIntParam(_cs, "_FillW", w);
            cmd.SetComputeIntParam(_cs, "_FillH", h);
            cmd.SetComputeIntParam(_cs, "_FillD", d);
            cmd.SetComputeIntParam(_cs, "_FillC", c);
            cmd.SetComputeBufferParam(_cs, _kFillPack4FromBufferCdhw, "_FillIn", input);
            cmd.SetComputeTextureParam(_cs, _kFillPack4FromBufferCdhw, "_FillOutArr", outputPack4.nameID);
            Dispatch3D(cmd, _kFillPack4FromBufferCdhw, w, h, outputPack4.depth, 8, 8);
        }

        public void ScalePack4(RenderTexture input, float k, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_ScaleK", k);
            _cs.SetTexture(_kScalePack4, "_ScaleInArr", input);
            _cs.SetTexture(_kScalePack4, "_ScaleOutArr", output);
            Dispatch3D(_kScalePack4, output.width, output.height, packs, 8, 8);
        }

        public void AddBiasPack4(RenderTexture input, ComputeBuffer bias4, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (bias4 == null) throw new ArgumentNullException(nameof(bias4));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetBuffer(_kAddBiasPack4, "_Bias4", bias4);
            _cs.SetTexture(_kAddBiasPack4, "_BiasInArr", input);
            _cs.SetTexture(_kAddBiasPack4, "_BiasOutArr", output);
            Dispatch3D(_kAddBiasPack4, output.width, output.height, packs, 8, 8);
        }

        public void BatchNormPack4(RenderTexture input, ComputeBuffer biasA4, ComputeBuffer scaleB4, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (biasA4 == null) throw new ArgumentNullException(nameof(biasA4));
            if (scaleB4 == null) throw new ArgumentNullException(nameof(scaleB4));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetBuffer(_kBatchNormPack4, "_BatchNormA4", biasA4);
            _cs.SetBuffer(_kBatchNormPack4, "_BatchNormB4", scaleB4);
            _cs.SetTexture(_kBatchNormPack4, "_BatchNormInArr", input);
            _cs.SetTexture(_kBatchNormPack4, "_BatchNormOutArr", output);
            Dispatch3D(_kBatchNormPack4, output.width, output.height, packs, 8, 8);
        }

        public void BatchNormPack4(CommandBuffer cmd, ComputeTexture input, ComputeBuffer biasA4, ComputeBuffer scaleB4, int packs, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (biasA4 == null) throw new ArgumentNullException(nameof(biasA4));
            if (scaleB4 == null) throw new ArgumentNullException(nameof(scaleB4));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeBufferParam(_cs, _kBatchNormPack4, "_BatchNormA4", biasA4);
            cmd.SetComputeBufferParam(_cs, _kBatchNormPack4, "_BatchNormB4", scaleB4);
            cmd.SetComputeTextureParam(_cs, _kBatchNormPack4, "_BatchNormInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kBatchNormPack4, "_BatchNormOutArr", output.nameID);
            Dispatch3D(cmd, _kBatchNormPack4, output.width, output.height, packs, 8, 8);
        }

        public void ConvDepthWisePack4(CommandBuffer cmd, ComputeTexture srcPack4, ComputeBuffer w4, ComputeBuffer b4, int inputChannels, int outputChannels, int group, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int dilationW, int dilationH, int activationType, float activationParam, ComputeTexture dstPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
            if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));

            cmd.SetComputeIntParam(_cs, "_InW", srcPack4.width);
            cmd.SetComputeIntParam(_cs, "_InH", srcPack4.height);
            cmd.SetComputeIntParam(_cs, "_InC", inputChannels);
            cmd.SetComputeIntParam(_cs, "_OutC", outputChannels);
            cmd.SetComputeIntParam(_cs, "_OutW", dstPack4.width);
            cmd.SetComputeIntParam(_cs, "_OutH", dstPack4.height);
            cmd.SetComputeIntParam(_cs, "_ConvGroup", group);
            cmd.SetComputeIntParam(_cs, "_KernelWVar", kernelW);
            cmd.SetComputeIntParam(_cs, "_KernelHVar", kernelH);
            cmd.SetComputeIntParam(_cs, "_StrideWVar", Mathf.Max(1, strideW));
            cmd.SetComputeIntParam(_cs, "_StrideHVar", Mathf.Max(1, strideH));
            cmd.SetComputeIntParam(_cs, "_PadLeftVar", Mathf.Max(0, padLeft));
            cmd.SetComputeIntParam(_cs, "_PadTopVar", Mathf.Max(0, padTop));
            cmd.SetComputeIntParam(_cs, "_DilationWVar", Mathf.Max(1, dilationW));
            cmd.SetComputeIntParam(_cs, "_DilationHVar", Mathf.Max(1, dilationH));
            cmd.SetComputeIntParam(_cs, "_OutPacks", packs);
            cmd.SetComputeIntParam(_cs, "_ActType", activationType);
            cmd.SetComputeFloatParam(_cs, "_ActParam", activationParam);
            cmd.SetComputeBufferParam(_cs, _kConvDepthWisePack4, "_DwConvW4", w4);
            cmd.SetComputeBufferParam(_cs, _kConvDepthWisePack4, "_DwConvB4", b4);
            cmd.SetComputeTextureParam(_cs, _kConvDepthWisePack4, "_ConvInArr", srcPack4.nameID);
            cmd.SetComputeTextureParam(_cs, _kConvDepthWisePack4, "_ConvOutArr", dstPack4.nameID);
            Dispatch3D(cmd, _kConvDepthWisePack4, dstPack4.width, dstPack4.height, packs, 8, 8);
        }

        public void LeakyReluPack4(RenderTexture input, float slope, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_LreluSlope", slope);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluInArr", input);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluOutArr", output);
            Dispatch3D(_kLeakyReluPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void LeakyReluPack4(CommandBuffer cmd, ComputeTexture input, float slope, int packs, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeFloatParam(_cs, "_LreluSlope", slope);
            cmd.SetComputeTextureParam(_cs, _kLeakyReluPack4, "_LreluInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kLeakyReluPack4, "_LreluOutArr", output.nameID);
            Dispatch3D(cmd, _kLeakyReluPack4, output.width, output.height, ResolveComputeTextureDispatchDepth(output, packs), 8, 8);
        }

        public void AddNoiseBroadcastPack4(RenderTexture inOut, ComputeBuffer noise, float weight, int packs)
        {
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (noise == null) throw new ArgumentNullException(nameof(noise));
            _cs.SetFloat("_NoiseWeight", weight);
            _cs.SetBuffer(_kAddNoiseBroadcastPack4, "_Noise", noise);
            _cs.SetTexture(_kAddNoiseBroadcastPack4, "_NoiseInOutArr", inOut);
            Dispatch3D(_kAddNoiseBroadcastPack4, inOut.width, inOut.height, packs, 8, 8);
        }

        public void ClipPack4(RenderTexture input, float min, float max, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_ClipMin", min);
            _cs.SetFloat("_ClipMax", max);
            _cs.SetTexture(_kClipPack4, "_ClipInArr", input);
            _cs.SetTexture(_kClipPack4, "_ClipOutArr", output);
            Dispatch3D(_kClipPack4, output.width, output.height, packs, 8, 8);
        }

        public void ClipPack4(CommandBuffer cmd, ComputeTexture input, float min, float max, int packs, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeFloatParam(_cs, "_ClipMin", min);
            cmd.SetComputeFloatParam(_cs, "_ClipMax", max);
            cmd.SetComputeTextureParam(_cs, _kClipPack4, "_ClipInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kClipPack4, "_ClipOutArr", output.nameID);
            Dispatch3D(cmd, _kClipPack4, output.width, output.height, packs, 8, 8);
        }

        public void ShuffleChannelPack4(RenderTexture input, int packs, int channels, int group, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            _cs.SetInt("_ShuffleChannels", channels);
            _cs.SetInt("_ShuffleGroup", group);
            _cs.SetTexture(_kShuffleChannelPack4, "_ShuffleInArr", input);
            _cs.SetTexture(_kShuffleChannelPack4, "_ShuffleOutArr", output);
            Dispatch3D(_kShuffleChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void ShuffleChannelPack4(CommandBuffer cmd, ComputeTexture input, int packs, int channels, int group, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            cmd.SetComputeIntParam(_cs, "_ShuffleChannels", channels);
            cmd.SetComputeIntParam(_cs, "_ShuffleGroup", group);
            cmd.SetComputeTextureParam(_cs, _kShuffleChannelPack4, "_ShuffleInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kShuffleChannelPack4, "_ShuffleOutArr", output.nameID);
            Dispatch3D(cmd, _kShuffleChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void CropPack4(RenderTexture input, int inW, int inH, int inC, int offsetW, int offsetH, int offsetC, int outW, int outH, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_CropPack4InW", inW);
            _cs.SetInt("_CropPack4InH", inH);
            _cs.SetInt("_CropPack4InC", inC);
            _cs.SetInt("_CropPack4OffsetW", offsetW);
            _cs.SetInt("_CropPack4OffsetH", offsetH);
            _cs.SetInt("_CropPack4OffsetC", offsetC);
            _cs.SetInt("_CropPack4OutW", outW);
            _cs.SetInt("_CropPack4OutH", outH);
            _cs.SetInt("_CropPack4OutC", outC);
            _cs.SetTexture(_kCropPack4, "_CropPack4InArr", input);
            _cs.SetTexture(_kCropPack4, "_CropPack4OutArr", output);
            Dispatch3D(_kCropPack4, output.width, output.height, output.volumeDepth > 0 ? output.volumeDepth : 1, 8, 8);
        }

        public void CropPack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inC, int offsetW, int offsetH, int offsetC, int outW, int outH, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_CropPack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_CropPack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_CropPack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_CropPack4OffsetW", offsetW);
            cmd.SetComputeIntParam(_cs, "_CropPack4OffsetH", offsetH);
            cmd.SetComputeIntParam(_cs, "_CropPack4OffsetC", offsetC);
            cmd.SetComputeIntParam(_cs, "_CropPack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_CropPack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_CropPack4OutC", outC);
            cmd.SetComputeTextureParam(_cs, _kCropPack4, "_CropPack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kCropPack4, "_CropPack4OutArr", output.nameID);
            Dispatch3D(cmd, _kCropPack4, output.width, output.height, Mathf.Max(1, Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void SlicePack4(RenderTexture input, int inW, int inH, int inC, int axis, int begin, int outW, int outH, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SlicePack4InW", inW);
            _cs.SetInt("_SlicePack4InH", inH);
            _cs.SetInt("_SlicePack4InC", inC);
            _cs.SetInt("_SlicePack4Axis", axis);
            _cs.SetInt("_SlicePack4Begin", begin);
            _cs.SetInt("_SlicePack4OutW", outW);
            _cs.SetInt("_SlicePack4OutH", outH);
            _cs.SetInt("_SlicePack4OutC", outC);
            _cs.SetTexture(_kSlicePack4, "_SlicePack4InArr", input);
            _cs.SetTexture(_kSlicePack4, "_SlicePack4OutArr", output);
            Dispatch3D(_kSlicePack4, output.width, output.height, output.volumeDepth > 0 ? output.volumeDepth : 1, 8, 8);
        }

        public void SlicePack4Cdhw(RenderTexture input, int inW, int inH, int inD, int inC, int axis, int begin, int outW, int outH, int outD, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SlicePack4CDHWInW", inW);
            _cs.SetInt("_SlicePack4CDHWInH", inH);
            _cs.SetInt("_SlicePack4CDHWInD", inD);
            _cs.SetInt("_SlicePack4CDHWInC", inC);
            _cs.SetInt("_SlicePack4CDHWAxis", axis);
            _cs.SetInt("_SlicePack4CDHWBegin", begin);
            _cs.SetInt("_SlicePack4CDHWOutW", outW);
            _cs.SetInt("_SlicePack4CDHWOutH", outH);
            _cs.SetInt("_SlicePack4CDHWOutD", outD);
            _cs.SetInt("_SlicePack4CDHWOutC", outC);
            _cs.SetTexture(_kSlicePack4Cdhw, "_SlicePack4CDHWInArr", input);
            _cs.SetTexture(_kSlicePack4Cdhw, "_SlicePack4CDHWOutArr", output);
            Dispatch3D(_kSlicePack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void SlicePack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inC, int axis, int begin, int outW, int outH, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_SlicePack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_SlicePack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_SlicePack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_SlicePack4Axis", axis);
            cmd.SetComputeIntParam(_cs, "_SlicePack4Begin", begin);
            cmd.SetComputeIntParam(_cs, "_SlicePack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_SlicePack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_SlicePack4OutC", outC);
            cmd.SetComputeTextureParam(_cs, _kSlicePack4, "_SlicePack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSlicePack4, "_SlicePack4OutArr", output.nameID);
            Dispatch3D(cmd, _kSlicePack4, output.width, output.height, Mathf.Max(1, Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void SlicePack4Cdhw(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inD, int inC, int axis, int begin, int outW, int outH, int outD, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWInW", inW);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWInH", inH);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWInD", inD);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWInC", inC);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWAxis", axis);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWBegin", begin);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWOutW", outW);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWOutH", outH);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWOutD", outD);
            cmd.SetComputeIntParam(_cs, "_SlicePack4CDHWOutC", outC);
            cmd.SetComputeTextureParam(_cs, _kSlicePack4Cdhw, "_SlicePack4CDHWInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSlicePack4Cdhw, "_SlicePack4CDHWOutArr", output.nameID);
            Dispatch3D(cmd, _kSlicePack4Cdhw, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void PermutePack4(RenderTexture input, int inW, int inH, int inC, Vector4Int axes, int outW, int outH, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PermutePack4InW", inW);
            _cs.SetInt("_PermutePack4InH", inH);
            _cs.SetInt("_PermutePack4InC", inC);
            _cs.SetInt("_PermutePack4OutW", outW);
            _cs.SetInt("_PermutePack4OutH", outH);
            _cs.SetInt("_PermutePack4OutC", outC);
            _cs.SetInt("_PermutePack4Axis0", axes.x);
            _cs.SetInt("_PermutePack4Axis1", axes.y);
            _cs.SetInt("_PermutePack4Axis2", axes.z);
            _cs.SetTexture(_kPermutePack4, "_PermutePack4InArr", input);
            _cs.SetTexture(_kPermutePack4, "_PermutePack4OutArr", output);
            Dispatch3D(_kPermutePack4, output.width, output.height, output.volumeDepth > 0 ? output.volumeDepth : 1, 8, 8);
        }

        public void PermutePack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inC, Vector4Int axes, int outW, int outH, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PermutePack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_PermutePack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_PermutePack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_PermutePack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_PermutePack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_PermutePack4OutC", outC);
            cmd.SetComputeIntParam(_cs, "_PermutePack4Axis0", axes.x);
            cmd.SetComputeIntParam(_cs, "_PermutePack4Axis1", axes.y);
            cmd.SetComputeIntParam(_cs, "_PermutePack4Axis2", axes.z);
            cmd.SetComputeTextureParam(_cs, _kPermutePack4, "_PermutePack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPermutePack4, "_PermutePack4OutArr", output.nameID);
            Dispatch3D(cmd, _kPermutePack4, output.width, output.height, Mathf.Max(1, Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void PermutePack4Cdhw(RenderTexture input, int inW, int inH, int inD, int inC, Vector4Int axes, int outW, int outH, int outD, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PermutePack4CDHWInW", inW);
            _cs.SetInt("_PermutePack4CDHWInH", inH);
            _cs.SetInt("_PermutePack4CDHWInD", inD);
            _cs.SetInt("_PermutePack4CDHWInC", inC);
            _cs.SetInt("_PermutePack4CDHWOutW", outW);
            _cs.SetInt("_PermutePack4CDHWOutH", outH);
            _cs.SetInt("_PermutePack4CDHWOutD", outD);
            _cs.SetInt("_PermutePack4CDHWOutC", outC);
            _cs.SetInt("_PermutePack4CDHWAxis0", axes.x);
            _cs.SetInt("_PermutePack4CDHWAxis1", axes.y);
            _cs.SetInt("_PermutePack4CDHWAxis2", axes.z);
            _cs.SetInt("_PermutePack4CDHWAxis3", axes.w);
            _cs.SetTexture(_kPermutePack4Cdhw, "_PermutePack4CDHWInArr", input);
            _cs.SetTexture(_kPermutePack4Cdhw, "_PermutePack4CDHWOutArr", output);
            Dispatch3D(_kPermutePack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void PermutePack4Cdhw(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inD, int inC, Vector4Int axes, int outW, int outH, int outD, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWInW", inW);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWInH", inH);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWInD", inD);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWInC", inC);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWOutW", outW);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWOutH", outH);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWOutD", outD);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWOutC", outC);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWAxis0", axes.x);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWAxis1", axes.y);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWAxis2", axes.z);
            cmd.SetComputeIntParam(_cs, "_PermutePack4CDHWAxis3", axes.w);
            cmd.SetComputeTextureParam(_cs, _kPermutePack4Cdhw, "_PermutePack4CDHWInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPermutePack4Cdhw, "_PermutePack4CDHWOutArr", output.nameID);
            Dispatch3D(cmd, _kPermutePack4Cdhw, output.width, output.height, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void WindowPartitionPack4(RenderTexture input, int inW, int inH, int inD, int inC, int outW, int outH, int outC, int groupsA, int groupsB, int groupsC, int tokensA, int tokensB, int tokensC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_WindowPartitionPack4InW", inW);
            _cs.SetInt("_WindowPartitionPack4InH", inH);
            _cs.SetInt("_WindowPartitionPack4InD", inD);
            _cs.SetInt("_WindowPartitionPack4InC", inC);
            _cs.SetInt("_WindowPartitionPack4OutW", outW);
            _cs.SetInt("_WindowPartitionPack4OutH", outH);
            _cs.SetInt("_WindowPartitionPack4OutC", outC);
            _cs.SetInt("_WindowPartitionPack4GroupsA", groupsA);
            _cs.SetInt("_WindowPartitionPack4GroupsB", groupsB);
            _cs.SetInt("_WindowPartitionPack4GroupsC", groupsC);
            _cs.SetInt("_WindowPartitionPack4TokensA", tokensA);
            _cs.SetInt("_WindowPartitionPack4TokensB", tokensB);
            _cs.SetInt("_WindowPartitionPack4TokensC", tokensC);
            _cs.SetTexture(_kWindowPartitionPack4, "_WindowPartitionPack4InArr", input);
            _cs.SetTexture(_kWindowPartitionPack4, "_WindowPartitionPack4OutArr", output);
            Dispatch3D(_kWindowPartitionPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void WindowPartitionPack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inD, int inC, int outW, int outH, int outC, int groupsA, int groupsB, int groupsC, int tokensA, int tokensB, int tokensC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4InD", inD);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4OutC", outC);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4GroupsA", groupsA);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4GroupsB", groupsB);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4GroupsC", groupsC);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4TokensA", tokensA);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4TokensB", tokensB);
            cmd.SetComputeIntParam(_cs, "_WindowPartitionPack4TokensC", tokensC);
            cmd.SetComputeTextureParam(_cs, _kWindowPartitionPack4, "_WindowPartitionPack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kWindowPartitionPack4, "_WindowPartitionPack4OutArr", output.nameID);
            Dispatch3D(cmd, _kWindowPartitionPack4, output.width, output.height, Mathf.Max(1, Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void WindowUnpartitionPack4(RenderTexture input, int inW, int inH, int inC, int outW, int outH, int outD, int outC, int groupsA, int groupsB, int groupsC, int tokensA, int tokensB, int tokensC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_WindowUnpartitionPack4InW", inW);
            _cs.SetInt("_WindowUnpartitionPack4InH", inH);
            _cs.SetInt("_WindowUnpartitionPack4InC", inC);
            _cs.SetInt("_WindowUnpartitionPack4OutW", outW);
            _cs.SetInt("_WindowUnpartitionPack4OutH", outH);
            _cs.SetInt("_WindowUnpartitionPack4OutD", outD);
            _cs.SetInt("_WindowUnpartitionPack4OutC", outC);
            _cs.SetInt("_WindowUnpartitionPack4GroupsA", groupsA);
            _cs.SetInt("_WindowUnpartitionPack4GroupsB", groupsB);
            _cs.SetInt("_WindowUnpartitionPack4GroupsC", groupsC);
            _cs.SetInt("_WindowUnpartitionPack4TokensA", tokensA);
            _cs.SetInt("_WindowUnpartitionPack4TokensB", tokensB);
            _cs.SetInt("_WindowUnpartitionPack4TokensC", tokensC);
            _cs.SetTexture(_kWindowUnpartitionPack4, "_WindowUnpartitionPack4InArr", input);
            _cs.SetTexture(_kWindowUnpartitionPack4, "_WindowUnpartitionPack4OutArr", output);
            Dispatch3D(_kWindowUnpartitionPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void WindowUnpartitionPack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inC, int outW, int outH, int outD, int outC, int groupsA, int groupsB, int groupsC, int tokensA, int tokensB, int tokensC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4OutD", outD);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4OutC", outC);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4GroupsA", groupsA);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4GroupsB", groupsB);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4GroupsC", groupsC);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4TokensA", tokensA);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4TokensB", tokensB);
            cmd.SetComputeIntParam(_cs, "_WindowUnpartitionPack4TokensC", tokensC);
            cmd.SetComputeTextureParam(_cs, _kWindowUnpartitionPack4, "_WindowUnpartitionPack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kWindowUnpartitionPack4, "_WindowUnpartitionPack4OutArr", output.nameID);
            Dispatch3D(cmd, _kWindowUnpartitionPack4, output.width, output.height, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f)), 8, 8);
        }

        public void ReshapePack4ToScalar2D(RenderTexture input, int inW, int inH, int inD, int inC, int inDims, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_ReshapePack4ToScalar2DInW", inW);
            _cs.SetInt("_ReshapePack4ToScalar2DInH", inH);
            _cs.SetInt("_ReshapePack4ToScalar2DInD", inD);
            _cs.SetInt("_ReshapePack4ToScalar2DInC", inC);
            _cs.SetInt("_ReshapePack4ToScalar2DInDims", inDims);
            _cs.SetTexture(_kReshapePack4ToScalar2D, "_ReshapePack4ToScalar2DInArr", input);
            _cs.SetTexture(_kReshapePack4ToScalar2D, "_ReshapePack4ToScalar2DOutArr", output);
            Dispatch3D(_kReshapePack4ToScalar2D, output.width, output.height, ResolveRenderTextureDispatchDepth(output, 1), 8, 8);
        }

        public void ReshapePack4ToScalar2D(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inD, int inC, int inDims, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToScalar2DInW", inW);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToScalar2DInH", inH);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToScalar2DInD", inD);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToScalar2DInC", inC);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToScalar2DInDims", inDims);
            cmd.SetComputeTextureParam(_cs, _kReshapePack4ToScalar2D, "_ReshapePack4ToScalar2DInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kReshapePack4ToScalar2D, "_ReshapePack4ToScalar2DOutArr", output.nameID);
            Dispatch3D(cmd, _kReshapePack4ToScalar2D, output.width, output.height, ResolveComputeTextureDispatchDepth(output, 1), 8, 8);
        }

        public void ReshapePack4ToPack4(RenderTexture input, int inW, int inH, int inD, int inC, int inDims, int outW, int outH, int outD, int outC, int outDims, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_ReshapePack4ToPack4InW", inW);
            _cs.SetInt("_ReshapePack4ToPack4InH", inH);
            _cs.SetInt("_ReshapePack4ToPack4InD", inD);
            _cs.SetInt("_ReshapePack4ToPack4InC", inC);
            _cs.SetInt("_ReshapePack4ToPack4InDims", inDims);
            _cs.SetInt("_ReshapePack4ToPack4OutW", outW);
            _cs.SetInt("_ReshapePack4ToPack4OutH", outH);
            _cs.SetInt("_ReshapePack4ToPack4OutD", outD);
            _cs.SetInt("_ReshapePack4ToPack4OutC", outC);
            _cs.SetInt("_ReshapePack4ToPack4OutDims", outDims);
            _cs.SetTexture(_kReshapePack4ToPack4, "_ReshapePack4ToPack4InArr", input);
            _cs.SetTexture(_kReshapePack4ToPack4, "_ReshapePack4ToPack4OutArr", output);
            Dispatch3D(_kReshapePack4ToPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outDims >= 4 ? outD * Mathf.CeilToInt(outC / 4f) : Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void ReshapePack4ToPack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inD, int inC, int inDims, int outW, int outH, int outD, int outC, int outDims, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4InW", inW);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4InH", inH);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4InD", inD);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4InC", inC);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4InDims", inDims);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4OutW", outW);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4OutH", outH);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4OutD", outD);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4OutC", outC);
            cmd.SetComputeIntParam(_cs, "_ReshapePack4ToPack4OutDims", outDims);
            cmd.SetComputeTextureParam(_cs, _kReshapePack4ToPack4, "_ReshapePack4ToPack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kReshapePack4ToPack4, "_ReshapePack4ToPack4OutArr", output.nameID);
            Dispatch3D(cmd, _kReshapePack4ToPack4, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, outDims >= 4 ? outD * Mathf.CeilToInt(outC / 4f) : Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void ReshapeScalar2DToPack4(RenderTexture input, int inW, int inH, int outW, int outH, int outD, int outC, int outDims, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_ReshapeScalar2DInW", inW);
            _cs.SetInt("_ReshapeScalar2DInH", inH);
            _cs.SetInt("_ReshapeScalar2DOutW", outW);
            _cs.SetInt("_ReshapeScalar2DOutH", outH);
            _cs.SetInt("_ReshapeScalar2DOutD", outD);
            _cs.SetInt("_ReshapeScalar2DOutC", outC);
            _cs.SetInt("_ReshapeScalar2DOutDims", outDims);
            _cs.SetTexture(_kReshapeScalar2DToPack4, "_ReshapeScalar2DInArr", input);
            _cs.SetTexture(_kReshapeScalar2DToPack4, "_ReshapeScalar2DOutArr", output);
            Dispatch3D(_kReshapeScalar2DToPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outDims >= 4 ? outD * Mathf.CeilToInt(outC / 4f) : Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void ReshapeScalar2DToPack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int outW, int outH, int outD, int outC, int outDims, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DInW", inW);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DInH", inH);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DOutW", outW);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DOutH", outH);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DOutD", outD);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DOutC", outC);
            cmd.SetComputeIntParam(_cs, "_ReshapeScalar2DOutDims", outDims);
            cmd.SetComputeTextureParam(_cs, _kReshapeScalar2DToPack4, "_ReshapeScalar2DInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kReshapeScalar2DToPack4, "_ReshapeScalar2DOutArr", output.nameID);
            Dispatch3D(cmd, _kReshapeScalar2DToPack4, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, outDims >= 4 ? outD * Mathf.CeilToInt(outC / 4f) : Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void AttentionReshapePack4(RenderTexture input, int inW, int inH, int inC, int headDim, int outD, int outC, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_AttentionReshapeInW", inW);
            _cs.SetInt("_AttentionReshapeInH", inH);
            _cs.SetInt("_AttentionReshapeInC", inC);
            _cs.SetInt("_AttentionReshapeHeadDim", headDim);
            _cs.SetInt("_AttentionReshapeOutD", outD);
            _cs.SetInt("_AttentionReshapeOutC", outC);
            _cs.SetTexture(_kAttentionReshapePack4, "_AttentionReshapeInArr", input);
            _cs.SetTexture(_kAttentionReshapePack4, "_AttentionReshapeOutArr", output);
            Dispatch3D(_kAttentionReshapePack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void AttentionReshapePack4(CommandBuffer cmd, ComputeTexture input, int inW, int inH, int inC, int headDim, int outD, int outC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeInW", inW);
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeInH", inH);
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeInC", inC);
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeHeadDim", headDim);
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeOutD", outD);
            cmd.SetComputeIntParam(_cs, "_AttentionReshapeOutC", outC);
            cmd.SetComputeTextureParam(_cs, _kAttentionReshapePack4, "_AttentionReshapeInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kAttentionReshapePack4, "_AttentionReshapeOutArr", output.nameID);
            Dispatch3D(cmd, _kAttentionReshapePack4, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void Gemm2DTextureA(RenderTexture a, ComputeBuffer b, ComputeBuffer c, int m, int n, int k, bool transB, float alpha, float beta, bool useC, int broadcastTypeC, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (useC && c == null) throw new ArgumentNullException(nameof(c));

            _cs.SetInt("_MatM", m);
            _cs.SetInt("_MatN", n);
            _cs.SetInt("_MatK", k);
            _cs.SetInt("_MatTransB", transB ? 1 : 0);
            _cs.SetInt("_MatUseC", useC ? 1 : 0);
            _cs.SetInt("_MatBroadcastTypeC", broadcastTypeC);
            _cs.SetFloat("_MatAlpha", alpha);
            _cs.SetFloat("_MatBeta", beta);
            _cs.SetInt("_GemmTexAInW", k);
            _cs.SetInt("_GemmTexAInH", m);
            _cs.SetTexture(_kGemm2DTextureA, "_GemmTexAInArr", a);
            _cs.SetBuffer(_kGemm2DTextureA, "_MatB", b);
            _cs.SetBuffer(_kGemm2DTextureA, "_MatC", useC ? c : b);
            _cs.SetTexture(_kGemm2DTextureA, "_GemmTexOutArr", output);
            Dispatch3D(_kGemm2DTextureA, n, m, ResolveRenderTextureDispatchDepth(output, 1), 8, 8);
        }

        public void Gemm2DTextureA(CommandBuffer cmd, ComputeTexture a, ComputeBuffer b, ComputeBuffer c, int m, int n, int k, bool transB, float alpha, float beta, bool useC, int broadcastTypeC, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (useC && c == null) throw new ArgumentNullException(nameof(c));

            cmd.SetComputeIntParam(_cs, "_MatM", m);
            cmd.SetComputeIntParam(_cs, "_MatN", n);
            cmd.SetComputeIntParam(_cs, "_MatK", k);
            cmd.SetComputeIntParam(_cs, "_MatTransB", transB ? 1 : 0);
            cmd.SetComputeIntParam(_cs, "_MatUseC", useC ? 1 : 0);
            cmd.SetComputeIntParam(_cs, "_MatBroadcastTypeC", broadcastTypeC);
            cmd.SetComputeFloatParam(_cs, "_MatAlpha", alpha);
            cmd.SetComputeFloatParam(_cs, "_MatBeta", beta);
            cmd.SetComputeIntParam(_cs, "_GemmTexAInW", k);
            cmd.SetComputeIntParam(_cs, "_GemmTexAInH", m);
            cmd.SetComputeTextureParam(_cs, _kGemm2DTextureA, "_GemmTexAInArr", a.nameID);
            cmd.SetComputeBufferParam(_cs, _kGemm2DTextureA, "_MatB", b);
            cmd.SetComputeBufferParam(_cs, _kGemm2DTextureA, "_MatC", useC ? c : b);
            cmd.SetComputeTextureParam(_cs, _kGemm2DTextureA, "_GemmTexOutArr", output.nameID);
            Dispatch3D(cmd, _kGemm2DTextureA, n, m, ResolveComputeTextureDispatchDepth(output, 1), 8, 8);
        }

        public void SftPack4(RenderTexture input, RenderTexture condMul, RenderTexture condAdd, int outPacks, int halfPacks, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (condMul == null) throw new ArgumentNullException(nameof(condMul));
            if (condAdd == null) throw new ArgumentNullException(nameof(condAdd));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SftHalfPacks", halfPacks);
            _cs.SetTexture(_kSftPack4, "_SftInArr", input);
            _cs.SetTexture(_kSftPack4, "_SftCondMulArr", condMul);
            _cs.SetTexture(_kSftPack4, "_SftCondAddArr", condAdd);
            _cs.SetTexture(_kSftPack4, "_SftOutArr", output);
            Dispatch3D(_kSftPack4, output.width, output.height, outPacks, 8, 8);
        }

        public void Pack4ToRgb01(RenderTexture inputPack4, RenderTexture outputRgb, bool flipY = false)
        {
            if (inputPack4 == null) throw new ArgumentNullException(nameof(inputPack4));
            if (outputRgb == null) throw new ArgumentNullException(nameof(outputRgb));
            _cs.SetInt("_FlipY", flipY ? 1 : 0);
            _cs.SetTexture(_kPack4ToRgb01, "_RgbInArr", inputPack4);
            _cs.SetTexture(_kPack4ToRgb01, "_RgbOut", outputRgb);
            Dispatch2D(_kPack4ToRgb01, outputRgb.width, outputRgb.height, 32, 32);
        }

        public void ProbeTilePack4(CommandBuffer cmd, ComputeTexture tileOutPack4, int probeIndex, int pad, int coreW, int coreH, ComputeBuffer probeOut)
        {
            if (probeOut == null) throw new ArgumentNullException(nameof(probeOut));
            cmd.SetComputeIntParam(_cs, "_ProbeIndex", probeIndex);
            cmd.SetComputeIntParam(_cs, "_ProbePad", pad);
            cmd.SetComputeIntParam(_cs, "_ProbeCoreW", coreW);
            cmd.SetComputeIntParam(_cs, "_ProbeCoreH", coreH);
            cmd.SetComputeTextureParam(_cs, _kProbeTilePack4, "_ProbeInArr", tileOutPack4.nameID);
            cmd.SetComputeBufferParam(_cs, _kProbeTilePack4, "_ProbeOut", probeOut);
            cmd.DispatchCompute(_cs, _kProbeTilePack4, 1, 1, 1);

        }

        public void ProbeTilePack4(RenderTexture tileOutPack4, int probeIndex, int pad, int coreW, int coreH, ComputeBuffer probeOut)
        {
            if (tileOutPack4 == null) throw new ArgumentNullException(nameof(tileOutPack4));
            if (probeOut == null) throw new ArgumentNullException(nameof(probeOut));
            _cs.SetInt("_ProbeIndex", probeIndex);
            _cs.SetInt("_ProbePad", pad);
            _cs.SetInt("_ProbeCoreW", coreW);
            _cs.SetInt("_ProbeCoreH", coreH);
            _cs.SetTexture(_kProbeTilePack4, "_ProbeInArr", tileOutPack4);
            _cs.SetBuffer(_kProbeTilePack4, "_ProbeOut", probeOut);
            _cs.Dispatch(_kProbeTilePack4, 1, 1, 1);
        }

        public void ProbeSeams(RenderTexture tex, int tilesX, int tilesY, int stepX, int stepY, int samples, ComputeBuffer seamOut4)
        {
            if (tex == null) throw new ArgumentNullException(nameof(tex));
            if (seamOut4 == null) throw new ArgumentNullException(nameof(seamOut4));
            _cs.SetTexture(_kProbeSeams, "_SeamTex", tex);
            _cs.SetBuffer(_kProbeSeams, "_SeamOut4", seamOut4);
            _cs.SetInt("_SeamW", tex.width);
            _cs.SetInt("_SeamH", tex.height);
            _cs.SetInt("_SeamTilesX", tilesX);
            _cs.SetInt("_SeamTilesY", tilesY);
            _cs.SetInt("_SeamStepX", stepX);
            _cs.SetInt("_SeamStepY", stepY);
            _cs.SetInt("_SeamSamples", samples);
            var seamCount = Mathf.Max(0, tilesX - 1) + Mathf.Max(0, tilesY - 1);
            if (seamCount <= 0) return;
            var groups = Mathf.CeilToInt(seamCount / 64f);
            _cs.Dispatch(_kProbeSeams, groups, 1, 1);
        }

        public void TouchU32(ComputeBuffer outU32, uint value)
        {
            if (outU32 == null) throw new ArgumentNullException(nameof(outU32));
            _cs.SetInt("_TouchValue", unchecked((int)value));
            _cs.SetBuffer(_kTouchU32, "_TouchOut", outU32);
            _cs.Dispatch(_kTouchU32, 1, 1, 1);
        }

        public void PaddingPack4(RenderTexture input, int packs, int padLeft, int padRight, int padTop, int padBottom, int padType, Vector4 padValue, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PadLeft", padLeft);
            _cs.SetInt("_PadRight", padRight);
            _cs.SetInt("_PadTop", padTop);
            _cs.SetInt("_PadBottom", padBottom);
            _cs.SetInt("_PadType", padType);
            _cs.SetVector("_PadValue4", padValue);
            _cs.SetTexture(_kPaddingPack4, "_PadInArr", input);
            _cs.SetTexture(_kPaddingPack4, "_PadOutArr", output);
            Dispatch3D(_kPaddingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PaddingPack4(CommandBuffer cmd, ComputeTexture input, int packs, int padLeft, int padRight, int padTop, int padBottom, int padType, Vector4 padValue, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PadRight", padRight);
            cmd.SetComputeIntParam(_cs, "_PadTop", padTop);
            cmd.SetComputeIntParam(_cs, "_PadBottom", padBottom);
            cmd.SetComputeIntParam(_cs, "_PadType", padType);
            cmd.SetComputeVectorParam(_cs, "_PadValue4", padValue);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, "_PadInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, "_PadOutArr", output.nameID);
            Dispatch3D(cmd, _kPaddingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PoolingPack4(RenderTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int poolType, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PoolKernelW", kernelW);
            _cs.SetInt("_PoolKernelH", kernelH);
            _cs.SetInt("_PoolStrideW", strideW);
            _cs.SetInt("_PoolStrideH", strideH);
            _cs.SetInt("_PoolPadLeft", padLeft);
            _cs.SetInt("_PoolPadTop", padTop);
            _cs.SetInt("_PoolType", poolType);
            _cs.SetTexture(_kPoolingPack4, "_PoolInArr", input);
            _cs.SetTexture(_kPoolingPack4, "_PoolOutArr", output);
            Dispatch3D(_kPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PoolingPack4Cdhw(
            RenderTexture input,
            int inW,
            int inH,
            int inD,
            int inC,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padTop,
            int padFront,
            int poolType,
            bool includePad,
            bool adaptive,
            bool global,
            int outW,
            int outH,
            int outD,
            int outC,
            RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_Pool4DInW", inW);
            _cs.SetInt("_Pool4DInH", inH);
            _cs.SetInt("_Pool4DInD", inD);
            _cs.SetInt("_Pool4DInC", inC);
            _cs.SetInt("_Pool4DKernelW", kernelW);
            _cs.SetInt("_Pool4DKernelH", kernelH);
            _cs.SetInt("_Pool4DKernelD", kernelD);
            _cs.SetInt("_Pool4DStrideW", strideW);
            _cs.SetInt("_Pool4DStrideH", strideH);
            _cs.SetInt("_Pool4DStrideD", strideD);
            _cs.SetInt("_Pool4DPadLeft", padLeft);
            _cs.SetInt("_Pool4DPadTop", padTop);
            _cs.SetInt("_Pool4DPadFront", padFront);
            _cs.SetInt("_Pool4DPoolType", poolType);
            _cs.SetInt("_Pool4DIncludePad", includePad ? 1 : 0);
            _cs.SetInt("_Pool4DAdaptive", adaptive ? 1 : 0);
            _cs.SetInt("_Pool4DGlobal", global ? 1 : 0);
            _cs.SetInt("_Pool4DOutW", outW);
            _cs.SetInt("_Pool4DOutH", outH);
            _cs.SetInt("_Pool4DOutD", outD);
            _cs.SetInt("_Pool4DOutC", outC);
            _cs.SetTexture(_kPoolingPack4Cdhw, "_Pool4DInArr", input);
            _cs.SetTexture(_kPoolingPack4Cdhw, "_Pool4DOutArr", output);
            Dispatch3D(_kPoolingPack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outD * Mathf.CeilToInt(outC / 4f))), 8, 8);
        }

        public void MaxPoolingIndPack4(RenderTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, RenderTexture output, RenderTexture indices)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            _cs.SetInt("_PoolKernelW", kernelW);
            _cs.SetInt("_PoolKernelH", kernelH);
            _cs.SetInt("_PoolStrideW", strideW);
            _cs.SetInt("_PoolStrideH", strideH);
            _cs.SetInt("_PoolPadLeft", padLeft);
            _cs.SetInt("_PoolPadTop", padTop);
            _cs.SetTexture(_kMaxPoolingIndPack4, "_MaxPoolInArr", input);
            _cs.SetTexture(_kMaxPoolingIndPack4, "_MaxPoolOutArr", output);
            _cs.SetTexture(_kMaxPoolingIndPack4, "_MaxPoolIndicesArr", indices);
            Dispatch3D(_kMaxPoolingIndPack4, output.width, output.height, packs, 8, 8);
        }

        public void MaxPoolingIndPack4(CommandBuffer cmd, ComputeTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, ComputeTexture output, ComputeTexture indices)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            cmd.SetComputeIntParam(_cs, "_PoolKernelW", kernelW);
            cmd.SetComputeIntParam(_cs, "_PoolKernelH", kernelH);
            cmd.SetComputeIntParam(_cs, "_PoolStrideW", strideW);
            cmd.SetComputeIntParam(_cs, "_PoolStrideH", strideH);
            cmd.SetComputeIntParam(_cs, "_PoolPadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PoolPadTop", padTop);
            cmd.SetComputeTextureParam(_cs, _kMaxPoolingIndPack4, "_MaxPoolInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kMaxPoolingIndPack4, "_MaxPoolOutArr", output.nameID);
            cmd.SetComputeTextureParam(_cs, _kMaxPoolingIndPack4, "_MaxPoolIndicesArr", indices.nameID);
            Dispatch3D(cmd, _kMaxPoolingIndPack4, output.width, output.height, packs, 8, 8);
        }

        public void MaxPoolingIndicesFromValuePack4(RenderTexture input, RenderTexture pooled, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, RenderTexture indices)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (pooled == null) throw new ArgumentNullException(nameof(pooled));
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            _cs.SetInt("_PoolKernelW", kernelW);
            _cs.SetInt("_PoolKernelH", kernelH);
            _cs.SetInt("_PoolStrideW", strideW);
            _cs.SetInt("_PoolStrideH", strideH);
            _cs.SetInt("_PoolPadLeft", padLeft);
            _cs.SetInt("_PoolPadTop", padTop);
            _cs.SetTexture(_kMaxPoolingIndicesFromValuePack4, "_MaxPoolInArr", input);
            _cs.SetTexture(_kMaxPoolingIndicesFromValuePack4, "_MaxPoolPooledArr", pooled);
            _cs.SetTexture(_kMaxPoolingIndicesFromValuePack4, "_MaxPoolIndicesArr", indices);
            Dispatch3D(_kMaxPoolingIndicesFromValuePack4, pooled.width, pooled.height, packs, 8, 8);
        }

        public void MaxUnPoolingPack4(RenderTexture input, RenderTexture indices, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PoolKernelW", kernelW);
            _cs.SetInt("_PoolKernelH", kernelH);
            _cs.SetInt("_PoolStrideW", strideW);
            _cs.SetInt("_PoolStrideH", strideH);
            _cs.SetInt("_PoolPadLeft", padLeft);
            _cs.SetInt("_PoolPadTop", padTop);
            _cs.SetTexture(_kMaxUnPoolingPack4, "_MaxUnpoolInArr", input);
            _cs.SetTexture(_kMaxUnPoolingPack4, "_MaxUnpoolIndicesArr", indices);
            _cs.SetTexture(_kMaxUnPoolingPack4, "_MaxUnpoolOutArr", output);
            Dispatch3D(_kMaxUnPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void MaxUnPoolingPack4(CommandBuffer cmd, ComputeTexture input, ComputeTexture indices, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PoolKernelW", kernelW);
            cmd.SetComputeIntParam(_cs, "_PoolKernelH", kernelH);
            cmd.SetComputeIntParam(_cs, "_PoolStrideW", strideW);
            cmd.SetComputeIntParam(_cs, "_PoolStrideH", strideH);
            cmd.SetComputeIntParam(_cs, "_PoolPadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PoolPadTop", padTop);
            cmd.SetComputeTextureParam(_cs, _kMaxUnPoolingPack4, "_MaxUnpoolInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kMaxUnPoolingPack4, "_MaxUnpoolIndicesArr", indices.nameID);
            cmd.SetComputeTextureParam(_cs, _kMaxUnPoolingPack4, "_MaxUnpoolOutArr", output.nameID);
            Dispatch3D(cmd, _kMaxUnPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PoolingPack4(CommandBuffer cmd, ComputeTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int poolType, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParams(_cs, "_PoolKernelW", kernelW);
            cmd.SetComputeIntParams(_cs, "_PoolKernelH", kernelH);
            cmd.SetComputeIntParams(_cs, "_PoolStrideW", strideW);
            cmd.SetComputeIntParams(_cs, "_PoolStrideH", strideH);
            cmd.SetComputeIntParams(_cs, "_PoolPadLeft", padLeft);
            cmd.SetComputeIntParams(_cs, "_PoolPadTop", padTop);
            cmd.SetComputeIntParams(_cs, "_PoolType", poolType);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, "_PoolInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, "_PoolOutArr", output.nameID);
            Dispatch3D(cmd, _kPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void SoftmaxChannelPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SoftmaxPacks", packs);
            _cs.SetTexture(_kSoftmaxChannelPack4, "_SoftmaxInArr", input);
            _cs.SetTexture(_kSoftmaxChannelPack4, "_SoftmaxOutArr", output);
            Dispatch3D(_kSoftmaxChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void SoftmaxPack4Cdhw(RenderTexture input, int w, int h, int d, int c, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_SoftmaxPack4CDHWW", w);
            _cs.SetInt("_SoftmaxPack4CDHWH", h);
            _cs.SetInt("_SoftmaxPack4CDHWD", d);
            _cs.SetInt("_SoftmaxPack4CDHWC", c);
            _cs.SetTexture(_kSoftmaxPack4Cdhw, "_SoftmaxPack4CDHWInArr", input);
            _cs.SetTexture(_kSoftmaxPack4Cdhw, "_SoftmaxPack4CDHWOutArr", output);
            Dispatch3D(_kSoftmaxPack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, d * Mathf.CeilToInt(c / 4f))), 8, 8);
        }

        public void SoftmaxPack4Cdhw(CommandBuffer cmd, ComputeTexture input, int w, int h, int d, int c, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_SoftmaxPack4CDHWW", w);
            cmd.SetComputeIntParam(_cs, "_SoftmaxPack4CDHWH", h);
            cmd.SetComputeIntParam(_cs, "_SoftmaxPack4CDHWD", d);
            cmd.SetComputeIntParam(_cs, "_SoftmaxPack4CDHWC", c);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxPack4Cdhw, "_SoftmaxPack4CDHWInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxPack4Cdhw, "_SoftmaxPack4CDHWOutArr", output.nameID);
            Dispatch3D(cmd, _kSoftmaxPack4Cdhw, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, d * Mathf.CeilToInt(c / 4f))), 8, 8);
        }

        public void SoftmaxChannelPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_SoftmaxPacks", packs);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, "_SoftmaxInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, "_SoftmaxOutArr", output.nameID);
            Dispatch3D(cmd, _kSoftmaxChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void UnaryOpPack4(RenderTexture input, int packs, int opType, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_UnaryOpType", opType);
            _cs.SetTexture(_kUnaryOpPack4, "_UnaryInArr", input);
            _cs.SetTexture(_kUnaryOpPack4, "_UnaryOutArr", output);
            Dispatch3D(_kUnaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void UnaryOpPack4(CommandBuffer cmd, ComputeTexture input, int packs, int opType, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_UnaryOpType", opType);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, "_UnaryInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, "_UnaryOutArr", output.nameID);
            Dispatch3D(cmd, _kUnaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpPack4(RenderTexture a, RenderTexture b, int packs, int opType, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 0);
            _cs.SetFloat("_BinaryScalar", 0f);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryA", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryB", b);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void BinaryOpPack4Broadcast(RenderTexture a, RenderTexture b, int packs, int opType, int broadcastMode, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (broadcastMode != 1 && broadcastMode != 2)
                throw new ArgumentOutOfRangeException(nameof(broadcastMode));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 0);
            _cs.SetFloat("_BinaryScalar", 0f);
            _cs.SetInt("_BinaryPack4BroadcastMode", broadcastMode);
            _cs.SetTexture(_kBinaryOpPack4Broadcast, "_BinaryA", a);
            _cs.SetTexture(_kBinaryOpPack4Broadcast, "_BinaryB", b);
            _cs.SetTexture(_kBinaryOpPack4Broadcast, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4Broadcast, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void BinaryOpPack4BufferScalar(RenderTexture texture, ComputeBuffer scalar, int packs, int opType, bool scalarIsA, RenderTexture output)
        {
            if (texture == null) throw new ArgumentNullException(nameof(texture));
            if (scalar == null) throw new ArgumentNullException(nameof(scalar));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (scalar.count < 1) throw new ArgumentOutOfRangeException(nameof(scalar));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryPack4BufferScalarMode", scalarIsA ? 1 : 2);
            _cs.SetTexture(_kBinaryOpPack4BufferScalar, "_BinaryA", texture);
            _cs.SetBuffer(_kBinaryOpPack4BufferScalar, "_BufB", scalar);
            _cs.SetTexture(_kBinaryOpPack4BufferScalar, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4BufferScalar, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void BinaryOpPack4(CommandBuffer cmd, ComputeTexture a, ComputeTexture b, int packs, int opType, ComputeTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 0);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", 0f);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryA", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryB", b.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryOutArr", output.nameID);
            Dispatch3D(cmd, _kBinaryOpPack4, output.width, output.height, ResolveComputeTextureDispatchDepth(output, packs), 8, 8);
        }

        public void BinaryOpPack4Broadcast(CommandBuffer cmd, ComputeTexture a, ComputeTexture b, int packs, int opType, int broadcastMode, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (broadcastMode != 1 && broadcastMode != 2)
                throw new ArgumentOutOfRangeException(nameof(broadcastMode));

            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 0);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", 0f);
            cmd.SetComputeIntParam(_cs, "_BinaryPack4BroadcastMode", broadcastMode);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4Broadcast, "_BinaryA", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4Broadcast, "_BinaryB", b.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4Broadcast, "_BinaryOutArr", output.nameID);
            Dispatch3D(cmd, _kBinaryOpPack4Broadcast, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpPack4BufferScalar(CommandBuffer cmd, ComputeTexture texture, ComputeBuffer scalar, int packs, int opType, bool scalarIsA, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (texture == null) throw new ArgumentNullException(nameof(texture));
            if (scalar == null) throw new ArgumentNullException(nameof(scalar));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (scalar.count < 1) throw new ArgumentOutOfRangeException(nameof(scalar));

            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryPack4BufferScalarMode", scalarIsA ? 1 : 2);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4BufferScalar, "_BinaryA", texture.nameID);
            cmd.SetComputeBufferParam(_cs, _kBinaryOpPack4BufferScalar, "_BufB", scalar);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4BufferScalar, "_BinaryOutArr", output.nameID);
            Dispatch3D(cmd, _kBinaryOpPack4BufferScalar, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpScalarPack4(RenderTexture a, float b, int packs, int opType, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 1);
            _cs.SetFloat("_BinaryScalar", b);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryA", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryB", a);
            _cs.SetTexture(_kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void BinaryOpScalarPack4(CommandBuffer cmd, ComputeTexture a, float b, int packs, int opType, ComputeTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 1);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", b);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryA", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryB", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryOutArr", output.nameID);
            Dispatch3D(cmd, _kBinaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void SwishPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kSwishPack4, "_ActInArr", input);
            _cs.SetTexture(_kSwishPack4, "_ActOutArr", output);
            Dispatch3D(_kSwishPack4, output.width, output.height, packs, 8, 8);
        }

        public void SwishPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, "_ActInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, "_ActOutArr", output.nameID);
            Dispatch3D(cmd, _kSwishPack4, output.width, output.height, packs, 8, 8);
        }

        public void SigmoidPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kSigmoidPack4, "_ActInArr", input);
            _cs.SetTexture(_kSigmoidPack4, "_ActOutArr", output);
            Dispatch3D(_kSigmoidPack4, output.width, output.height, packs, 8, 8);
        }

        public void SigmoidPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, "_ActInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, "_ActOutArr", output.nameID);
            Dispatch3D(cmd, _kSigmoidPack4, output.width, output.height, packs, 8, 8);
        }

        public void GeluPack4(RenderTexture input, int packs, bool fastGelu, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_GeluFast", fastGelu ? 1 : 0);
            _cs.SetTexture(_kGeluPack4, "_ActInArr", input);
            _cs.SetTexture(_kGeluPack4, "_ActOutArr", output);
            Dispatch3D(_kGeluPack4, output.width, output.height, packs, 8, 8);
        }

        public void GeluPack4(CommandBuffer cmd, ComputeTexture input, int packs, bool fastGelu, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParams(_cs, "_GeluFast", fastGelu ? 1 : 0);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, "_ActInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, "_ActOutArr", output.nameID);
            Dispatch3D(cmd, _kGeluPack4, output.width, output.height, packs, 8, 8);
        }

        public void Conv3x3Pack4(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int pad, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_Pad", Mathf.Max(0, pad));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3x3Pack4, "_ConvW4", w4);
            _cs.SetBuffer(_kConv3x3Pack4, "_ConvB4", b4);
            _cs.SetTexture(_kConv3x3Pack4, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConv3x3Pack4, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConv3x3Pack4, (dstPack4.width + 1) / 2, (dstPack4.height + 1) / 2, (outPacks + 1) / 2, 8, 8);
        }

        public void ConvPack4General(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int dilationW, int dilationH, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));
            if (kernelW <= 0 || kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            _cs.SetInt("_InW", srcPack4.width);
            _cs.SetInt("_InH", srcPack4.height);
            _cs.SetInt("_OutW", dstPack4.width);
            _cs.SetInt("_OutH", dstPack4.height);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConvPack4General, "_ConvW4", w4);
            _cs.SetBuffer(_kConvPack4General, "_ConvB4", b4);
            _cs.SetTexture(_kConvPack4General, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConvPack4General, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConvPack4General, dstPack4.width, dstPack4.height, outPacks, 8, 8);
        }

        public void DeconvolutionPack4General(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int dilationW, int dilationH, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));
            if (kernelW <= 0 || kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            _cs.SetInt("_InW", srcPack4.width);
            _cs.SetInt("_InH", srcPack4.height);
            _cs.SetInt("_OutW", dstPack4.width);
            _cs.SetInt("_OutH", dstPack4.height);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kDeconvolutionPack4General, "_ConvW4", w4);
            _cs.SetBuffer(_kDeconvolutionPack4General, "_ConvB4", b4);
            _cs.SetTexture(_kDeconvolutionPack4General, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kDeconvolutionPack4General, "_ConvOutArr", dstPack4);
            Dispatch3D(_kDeconvolutionPack4General, dstPack4.width, dstPack4.height, outPacks, 8, 8);
        }

        public void ConvDepthWisePack4(RenderTexture srcPack4, ComputeBuffer w4, ComputeBuffer b4, int inputChannels, int outputChannels, int group, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int dilationW, int dilationH, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(inputChannels));
            if (outputChannels <= 0) throw new ArgumentOutOfRangeException(nameof(outputChannels));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));

            _cs.SetInt("_InW", srcPack4.width);
            _cs.SetInt("_InH", srcPack4.height);
            _cs.SetInt("_InC", inputChannels);
            _cs.SetInt("_OutC", outputChannels);
            _cs.SetInt("_OutW", dstPack4.width);
            _cs.SetInt("_OutH", dstPack4.height);
            _cs.SetInt("_ConvGroup", group);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_OutPacks", packs);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConvDepthWisePack4, "_DwConvW4", w4);
            _cs.SetBuffer(_kConvDepthWisePack4, "_DwConvB4", b4);
            _cs.SetTexture(_kConvDepthWisePack4, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConvDepthWisePack4, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConvDepthWisePack4, dstPack4.width, dstPack4.height, packs, 8, 8);
        }

        public void Conv3x3Pack4Winograd23(RenderTexture srcPack4, int inPacks, ComputeBuffer wTm23, ComputeBuffer b4, int outPacks, int biasTerm, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (wTm23 == null) throw new ArgumentNullException(nameof(wTm23));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            var w = srcPack4.width;
            var h = srcPack4.height;
            var blockX = NcnnWinograd23.BlockX(w);
            var blockY = NcnnWinograd23.BlockY(h);
            var tiles = blockX * blockY;
            var bottomCount = NcnnWinograd23.BottomTmCount(w, h, inPacks);
            var topCount = NcnnWinograd23.TopTmCount(w, h, outPacks);
            EnsureWinogradWorkspace(bottomCount, topCount);

            
            SetWinograd23Params(w, h, inPacks, outPacks, activationType, activationParam, biasTerm);
            _cs.SetTexture(_kWinograd23TransformInput, "_WinoInArr", srcPack4);
            _cs.SetBuffer(_kWinograd23TransformInput, "_WinoBottomTm", _winoBottomTm);
            DispatchWinograd23TransformInput(null, blockX, blockY, inPacks);

            _cs.SetBuffer(_kWinograd23Gemm, "_WinoBottomTm", _winoBottomTm);
            _cs.SetBuffer(_kWinograd23Gemm, "_WinoTopTm", _winoTopTm);
            _cs.SetBuffer(_kWinograd23Gemm, "_WinoWeightTm", wTm23);
            DispatchWinograd23Gemm(null, tiles, outPacks);

            _cs.SetBuffer(_kWinograd23TransformOutput, "_WinoTopTm", _winoTopTm);
            _cs.SetBuffer(_kWinograd23TransformOutput, "_WinoBias4", b4);
            _cs.SetTexture(_kWinograd23TransformOutput, "_WinoOutArr", dstPack4);
            DispatchWinograd23TransformOutput(null, blockX, blockY, outPacks);
            

            //WaitGpuIdle();
        }

        public void Conv3x3Pack4Winograd23(CommandBuffer cmd, ComputeTexture srcPack4, int inPacks, ComputeBuffer wTm23, ComputeBuffer b4, int outPacks, int biasTerm, int activationType, float activationParam, ComputeTexture dstPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            var w = srcPack4.width;
            var h = srcPack4.height;
            var blockX = NcnnWinograd23.BlockX(w);
            var blockY = NcnnWinograd23.BlockY(h);
            var tiles = blockX * blockY;
            var bottomCount = NcnnWinograd23.BottomTmCount(w, h, inPacks);
            var topCount = NcnnWinograd23.TopTmCount(w, h, outPacks);
            EnsureWinogradWorkspace(bottomCount, topCount);

            SetWinograd23Params(cmd, w, h, inPacks, outPacks, activationType, activationParam, biasTerm);
            cmd.SetComputeTextureParam(_cs, _kWinograd23TransformInput, "_WinoInArr", srcPack4.nameID);
            cmd.SetComputeBufferParam(_cs, _kWinograd23TransformInput, "_WinoBottomTm", _winoBottomTm);
            DispatchWinograd23TransformInput(cmd, blockX, blockY, inPacks);

            cmd.SetComputeBufferParam(_cs, _kWinograd23Gemm, "_WinoBottomTm", _winoBottomTm);
            cmd.SetComputeBufferParam(_cs, _kWinograd23Gemm, "_WinoTopTm", _winoTopTm);
            cmd.SetComputeBufferParam(_cs, _kWinograd23Gemm, "_WinoWeightTm", wTm23);
            DispatchWinograd23Gemm(cmd, tiles, outPacks);

            cmd.SetComputeBufferParam(_cs, _kWinograd23TransformOutput, "_WinoTopTm", _winoTopTm);
            cmd.SetComputeBufferParam(_cs, _kWinograd23TransformOutput, "_WinoBias4", b4);
            cmd.SetComputeTextureParam(_cs, _kWinograd23TransformOutput, "_WinoOutArr", dstPack4.nameID);
            DispatchWinograd23TransformOutput(cmd, blockX, blockY, outPacks);

            //WaitGpuIdle();
        }

        public void ReleaseWinogradWorkspace()
        {
            if (_winoBottomTm != null)
            {
                try { _winoBottomTm.Release(); } catch { }
                _winoBottomTm = null;
            }
            if (_winoTopTm != null)
            {
                try { _winoTopTm.Release(); } catch { }
                _winoTopTm = null;
            }
            if (_gpuIdleSync != null)
            {
                try { _gpuIdleSync.Release(); } catch { }
                _gpuIdleSync = null;
            }
            _winoBottomCap = 0;
            _winoTopCap = 0;
        }

        #region debug-point B:gpu-layer-sync
        public void DebugSyncGpu()
        {
            WaitGpuIdle();
        }
        #endregion

        private void WaitGpuIdle()
        {
            if (_gpuIdleSync == null)
                _gpuIdleSync = new ComputeBuffer(1, sizeof(uint), ComputeBufferType.Structured);
            TouchU32(_gpuIdleSync, 1u);
            _gpuIdleSync.GetData(_gpuIdleScratch);
        }

        private void EnsureWinogradWorkspace(int bottomCount, int topCount)
        {
            if (bottomCount <= 0 || topCount <= 0)
                throw new ArgumentOutOfRangeException(nameof(bottomCount));

            if (_winoBottomTm == null || _winoBottomCap < bottomCount)
            {
                if (_winoBottomTm != null)
                {
                    try { _winoBottomTm.Release(); } catch { }
                }
                _winoBottomCap = bottomCount;
                _winoBottomTm = new ComputeBuffer(bottomCount, sizeof(float) * 4, ComputeBufferType.Structured);
            }

            if (_winoTopTm == null || _winoTopCap < topCount)
            {
                if (_winoTopTm != null)
                {
                    try { _winoTopTm.Release(); } catch { }
                }
                _winoTopCap = topCount;
                _winoTopTm = new ComputeBuffer(topCount, sizeof(float) * 4, ComputeBufferType.Structured);
            }
        }


        private void DispatchWinograd23TransformInput(CommandBuffer cmd, int blockX, int blockY, int inPacks)
        {
            var groupsX = Mathf.Max(1, Mathf.CeilToInt(blockX / 8f));
            var groupsY = Mathf.Max(1, Mathf.CeilToInt(blockY / 8f));
            var groupsZ = Mathf.Max(1, inPacks);
            if (cmd != null)
                cmd.DispatchCompute(_cs, _kWinograd23TransformInput, groupsX, groupsY, groupsZ);
            else
                _cs.Dispatch(_kWinograd23TransformInput, groupsX, groupsY, groupsZ);
        }

        private void DispatchWinograd23Gemm(CommandBuffer cmd, int tiles, int outPacks)
        {
            var tileBlocks = Mathf.Max(1, (tiles + 3) / 4);
            var groupsX = Mathf.Max(1, Mathf.CeilToInt(tileBlocks / 8f));
            var groupsY = Mathf.Max(1, Mathf.CeilToInt(outPacks / 8f));
            var groupsZ = 16;
            if (cmd != null)
                cmd.DispatchCompute(_cs, _kWinograd23Gemm, groupsX, groupsY, groupsZ);
            else
                _cs.Dispatch(_kWinograd23Gemm, groupsX, groupsY, groupsZ);
        }

        private void DispatchWinograd23TransformOutput(CommandBuffer cmd, int blockX, int blockY, int outPacks)
        {
            var groupsX = Mathf.Max(1, Mathf.CeilToInt(blockX / 8f));
            var groupsY = Mathf.Max(1, Mathf.CeilToInt(blockY / 8f));
            var groupsZ = Mathf.Max(1, outPacks);
            if (cmd != null)
                cmd.DispatchCompute(_cs, _kWinograd23TransformOutput, groupsX, groupsY, groupsZ);
            else
                _cs.Dispatch(_kWinograd23TransformOutput, groupsX, groupsY, groupsZ);
        }

        private void SetWinograd23Params(int w, int h, int inPacks, int outPacks, int activationType, float activationParam, int biasTerm)
        {
            var blockX = NcnnWinograd23.BlockX(w);
            var blockY = NcnnWinograd23.BlockY(h);
            var tiles = blockX * blockY;
            _cs.SetInt("_WinoW", w);
            _cs.SetInt("_WinoH", h);
            _cs.SetInt("_WinoOutW", w);
            _cs.SetInt("_WinoOutH", h);
            _cs.SetInt("_WinoInCstep", tiles);
            _cs.SetInt("_WinoOutCstep", tiles);
            _cs.SetInt("_WinoBlockX", blockX);
            _cs.SetInt("_WinoBlockY", blockY);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetInt("_WinoBiasTerm", biasTerm);
        }

        private void SetWinograd23Params(CommandBuffer cmd, int w, int h, int inPacks, int outPacks, int activationType, float activationParam, int biasTerm)
        {
            var blockX = NcnnWinograd23.BlockX(w);
            var blockY = NcnnWinograd23.BlockY(h);
            var tiles = blockX * blockY;
            cmd.SetComputeIntParam(_cs, "_WinoW", w);
            cmd.SetComputeIntParam(_cs, "_WinoH", h);
            cmd.SetComputeIntParam(_cs, "_WinoOutW", w);
            cmd.SetComputeIntParam(_cs, "_WinoOutH", h);
            cmd.SetComputeIntParam(_cs, "_WinoInCstep", tiles);
            cmd.SetComputeIntParam(_cs, "_WinoOutCstep", tiles);
            cmd.SetComputeIntParam(_cs, "_WinoBlockX", blockX);
            cmd.SetComputeIntParam(_cs, "_WinoBlockY", blockY);
            cmd.SetComputeIntParam(_cs, "_InPacks", inPacks);
            cmd.SetComputeIntParam(_cs, "_OutPacks", outPacks);
            cmd.SetComputeIntParam(_cs, "_ActType", activationType);
            cmd.SetComputeFloatParam(_cs, "_ActParam", activationParam);
            cmd.SetComputeIntParam(_cs, "_WinoBiasTerm", biasTerm);
        }

        public void Conv1x1Pack4(RenderTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int activationType, float activationParam, RenderTexture dstPack4)
        {
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv1x1Pack4, "_ConvW4", w4);
            _cs.SetBuffer(_kConv1x1Pack4, "_ConvB4", b4);
            _cs.SetTexture(_kConv1x1Pack4, "_ConvInArr", srcPack4);
            _cs.SetTexture(_kConv1x1Pack4, "_ConvOutArr", dstPack4);
            Dispatch3D(_kConv1x1Pack4, (dstPack4.width + 1) / 2, (dstPack4.height + 1) / 2, (outPacks + 1) / 2, 8, 8);
        }

        public void Conv1x1Pack4(CommandBuffer cmd, ComputeTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int activationType, float activationParam, ComputeTexture dstPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            cmd.SetComputeIntParam(_cs, "_InPacks", inPacks);
            cmd.SetComputeIntParam(_cs, "_OutPacks", outPacks);
            cmd.SetComputeIntParam(_cs, "_ActType", activationType);
            cmd.SetComputeFloatParam(_cs, "_ActParam", activationParam);
            cmd.SetComputeBufferParam(_cs, _kConv1x1Pack4, "_ConvW4", w4);
            cmd.SetComputeBufferParam(_cs, _kConv1x1Pack4, "_ConvB4", b4);
            cmd.SetComputeTextureParam(_cs, _kConv1x1Pack4, "_ConvInArr", srcPack4.nameID);
            cmd.SetComputeTextureParam(_cs, _kConv1x1Pack4, "_ConvOutArr", dstPack4.nameID);
            Dispatch3D(cmd, _kConv1x1Pack4, (dstPack4.width + 1) / 2, (dstPack4.height + 1) / 2, (outPacks + 1) / 2, 8, 8);
        }

        public void AddPack4(RenderTexture a, RenderTexture b, float coeffA, float coeffB, int packs, RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_CoeffA", coeffA);
            _cs.SetFloat("_CoeffB", coeffB);
            _cs.SetTexture(_kAddPack4, "_AddA", a);
            _cs.SetTexture(_kAddPack4, "_AddB", b);
            _cs.SetTexture(_kAddPack4, "_AddOutArr", output);
            Dispatch3D(_kAddPack4, output.width, output.height, packs, 8, 8);
        }

        public void AddPack4(CommandBuffer cmd, ComputeTexture a, ComputeTexture b, float coeffA, float coeffB, int packs, ComputeTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeFloatParam(_cs, "_CoeffA", coeffA);
            cmd.SetComputeFloatParam(_cs, "_CoeffB", coeffB);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddA", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddB", b.nameID);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddOutArr", output.nameID);
            Dispatch3D(cmd, _kAddPack4, output.width, output.height, packs, 8, 8);
        }

        public void CopyPack4(RenderTexture src, int srcPackOffset, RenderTexture dst, int dstPackOffset, int packs)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (packs <= 0) return;
            if (src.width != dst.width || src.height != dst.height)
                throw new InvalidOperationException("CopyPack4 requires same width/height");
            _cs.SetInt("_CopyInOffset", srcPackOffset);
            _cs.SetInt("_CopyOutOffset", dstPackOffset);
            _cs.SetInt("_CopyPacks", packs);
            _cs.SetTexture(_kCopyPack4, "_CopyInArr", src);
            _cs.SetTexture(_kCopyPack4, "_CopyOutArr", dst);
            Dispatch3D(_kCopyPack4, dst.width, dst.height, packs, 8, 8);
        }

        public void ConcatPack4Cdhw(
            RenderTexture a,
            RenderTexture b,
            int w,
            int h,
            int d,
            int aChannels,
            int bChannels,
            int outChannels,
            RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_ConcatPack4CDHWW", w);
            _cs.SetInt("_ConcatPack4CDHWH", h);
            _cs.SetInt("_ConcatPack4CDHWD", d);
            _cs.SetInt("_ConcatPack4CDHWAC", aChannels);
            _cs.SetInt("_ConcatPack4CDHWBC", bChannels);
            _cs.SetInt("_ConcatPack4CDHWOutC", outChannels);
            _cs.SetTexture(_kConcatPack4Cdhw, "_ConcatPack4CDHWAInArr", a);
            _cs.SetTexture(_kConcatPack4Cdhw, "_ConcatPack4CDHWBInArr", b);
            _cs.SetTexture(_kConcatPack4Cdhw, "_ConcatPack4CDHWOutArr", output);
            Dispatch3D(_kConcatPack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, d * Mathf.CeilToInt(outChannels / 4f))), 8, 8);
        }

        public void BuildSdInpaintInput9Pack4(RenderTexture latents, RenderTexture mask, RenderTexture maskedLatents, RenderTexture output)
        {
            if (latents == null) throw new ArgumentNullException(nameof(latents));
            if (mask == null) throw new ArgumentNullException(nameof(mask));
            if (maskedLatents == null) throw new ArgumentNullException(nameof(maskedLatents));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (latents.width != output.width || latents.height != output.height)
                throw new InvalidOperationException("BuildSdInpaintInput9Pack4 requires latents/output same width/height");
            if (mask.width != output.width || mask.height != output.height)
                throw new InvalidOperationException("BuildSdInpaintInput9Pack4 requires mask/output same width/height");
            if (maskedLatents.width != output.width || maskedLatents.height != output.height)
                throw new InvalidOperationException("BuildSdInpaintInput9Pack4 requires maskedLatents/output same width/height");
            if (output.volumeDepth < 3)
                throw new InvalidOperationException("BuildSdInpaintInput9Pack4 requires output volumeDepth >= 3");

            _cs.SetTexture(_kBuildSdInpaintInput9Pack4, "_SdInpaintLatentsArr", latents);
            _cs.SetTexture(_kBuildSdInpaintInput9Pack4, "_SdInpaintMaskArr", mask);
            _cs.SetTexture(_kBuildSdInpaintInput9Pack4, "_SdInpaintMaskedLatentsArr", maskedLatents);
            _cs.SetTexture(_kBuildSdInpaintInput9Pack4, "_SdInpaintInputOutArr", output);
            Dispatch3D(_kBuildSdInpaintInput9Pack4, output.width, output.height, 3, 8, 8);
        }

        public void CopyPack4(CommandBuffer cmd, ComputeTexture src, int srcPackOffset, ComputeTexture dst, int dstPackOffset, int packs)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (packs <= 0) return;
            if (src.width != dst.width || src.height != dst.height)
                throw new InvalidOperationException("CopyPack4 requires same width/height");
            cmd.SetComputeIntParam(_cs, "_CopyInOffset", srcPackOffset);
            cmd.SetComputeIntParam(_cs, "_CopyOutOffset", dstPackOffset);
            cmd.SetComputeIntParam(_cs, "_CopyPacks", packs);
            cmd.SetComputeTextureParam(_cs, _kCopyPack4, "_CopyInArr", src.nameID);
            cmd.SetComputeTextureParam(_cs, _kCopyPack4, "_CopyOutArr", dst.nameID);
            Dispatch3D(cmd, _kCopyPack4, dst.width, dst.height, packs, 8, 8);
        }

        public void Interp2xPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xPack4, "_InterpInArr", input);
            _cs.SetTexture(_kInterp2xPack4, "_InterpOutArr", output);
            Dispatch3D(_kInterp2xPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpPack4(RenderTexture input, int packs, float scaleX, float scaleY, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_InterpScaleFactorX", scaleX);
            _cs.SetFloat("_InterpScaleFactorY", scaleY);
            _cs.SetTexture(_kInterpPack4, "_InterpInArr", input);
            _cs.SetTexture(_kInterpPack4, "_InterpOutArr", output);
            Dispatch3D(_kInterpPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpPack4CDHW(
            RenderTexture input,
            int inW,
            int inH,
            int inD,
            int inPacks,
            int outW,
            int outH,
            int outD,
            int outPacks,
            float scaleX,
            float scaleY,
            float scaleZ,
            int resizeType,
            RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inW <= 0 || inH <= 0 || inD <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (outW <= 0 || outH <= 0 || outD <= 0) throw new ArgumentOutOfRangeException(nameof(outW));
            if (inPacks <= 0 || outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));

            _cs.SetInt("_InW", inW);
            _cs.SetInt("_InH", inH);
            _cs.SetInt("_InD", inD);
            _cs.SetInt("_OutW", outW);
            _cs.SetInt("_OutH", outH);
            _cs.SetInt("_OutD", outD);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetFloat("_InterpScaleFactorX", scaleX);
            _cs.SetFloat("_InterpScaleFactorY", scaleY);
            _cs.SetFloat("_InterpScaleFactorZ", scaleZ);
            _cs.SetInt("_InterpResizeType", resizeType);
            _cs.SetTexture(_kInterpPack4Cdhw, "_InterpInArr", input);
            _cs.SetTexture(_kInterpPack4Cdhw, "_InterpOutArr", output);
            Dispatch3D(_kInterpPack4Cdhw, outW, outH, ResolveRenderTextureDispatchDepth(output, outD * outPacks), 8, 8);
        }

        public void InterpPack4(CommandBuffer cmd, ComputeTexture input, int packs, float scaleX, float scaleY, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeFloatParam(_cs, "_InterpScaleFactorX", scaleX);
            cmd.SetComputeFloatParam(_cs, "_InterpScaleFactorY", scaleY);
            cmd.SetComputeTextureParam(_cs, _kInterpPack4, "_InterpInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kInterpPack4, "_InterpOutArr", output.nameID);
            Dispatch3D(cmd, _kInterpPack4, output.width, output.height, packs, 8, 8);
        }
         
        public void Interp2xPack4(CommandBuffer cmd,  ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, "_InterpInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, "_InterpOutArr", output.nameID);
            Dispatch3D(cmd, _kInterp2xPack4, output.width, output.height, packs, 8, 8);
        }

        public void Interp2xNearestPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xNearestPack4, "_InterpNnInArr", input);
            _cs.SetTexture(_kInterp2xNearestPack4, "_InterpNnOutArr", output);
            Dispatch3D(_kInterp2xNearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void Interp2xNearestPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, "_InterpNnInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, "_InterpNnOutArr", output.nameID);
            Dispatch3D(cmd, _kInterp2xNearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2Pack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterpDown2Pack4, "_InterpDownInArr", input);
            _cs.SetTexture(_kInterpDown2Pack4, "_InterpDownOutArr", output);
            Dispatch3D(_kInterpDown2Pack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2Pack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, "_InterpDownInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, "_InterpDownOutArr", output.nameID);
            Dispatch3D(cmd, _kInterpDown2Pack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2NearestPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnInArr", input);
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnOutArr", output);
            Dispatch3D(_kInterpDown2NearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2NearestPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, "_InterpDownNnInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, "_InterpDownNnOutArr", output.nameID);
            Dispatch3D(cmd, _kInterpDown2NearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void Conv3x3Pack4(CommandBuffer cmd, ComputeTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int pad, int activationType, float activationParam, ComputeTexture dstPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));

            cmd.SetComputeIntParam(_cs, "_InPacks", inPacks);
            cmd.SetComputeIntParam(_cs, "_OutPacks", outPacks);
            cmd.SetComputeIntParam(_cs, "_Pad", Mathf.Max(0, pad));
            cmd.SetComputeIntParam(_cs, "_ActType", activationType);
            cmd.SetComputeFloatParam(_cs, "_ActParam", activationParam);
            cmd.SetComputeBufferParam(_cs, _kConv3x3Pack4, "_ConvW4", w4);
            cmd.SetComputeBufferParam(_cs, _kConv3x3Pack4, "_ConvB4", b4);
            cmd.SetComputeTextureParam(_cs, _kConv3x3Pack4, "_ConvInArr", srcPack4.nameID);
            cmd.SetComputeTextureParam(_cs, _kConv3x3Pack4, "_ConvOutArr", dstPack4.nameID);
            Dispatch3D(cmd, _kConv3x3Pack4, (dstPack4.width + 1) / 2, (dstPack4.height + 1) / 2, (outPacks + 1) / 2, 8, 8);
        }

        public void ConvPack4General(CommandBuffer cmd, ComputeTexture srcPack4, int inPacks, ComputeBuffer w4, ComputeBuffer b4, int outPacks, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int dilationW, int dilationH, int activationType, float activationParam, ComputeTexture dstPack4)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (srcPack4 == null) throw new ArgumentNullException(nameof(srcPack4));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            if (w4 == null) throw new ArgumentNullException(nameof(w4));
            if (b4 == null) throw new ArgumentNullException(nameof(b4));
            if (inPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));
            if (outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(outPacks));
            if (kernelW <= 0 || kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            cmd.SetComputeIntParam(_cs, "_InW", srcPack4.width);
            cmd.SetComputeIntParam(_cs, "_InH", srcPack4.height);
            cmd.SetComputeIntParam(_cs, "_OutW", dstPack4.width);
            cmd.SetComputeIntParam(_cs, "_OutH", dstPack4.height);
            cmd.SetComputeIntParam(_cs, "_InPacks", inPacks);
            cmd.SetComputeIntParam(_cs, "_OutPacks", outPacks);
            cmd.SetComputeIntParam(_cs, "_KernelWVar", kernelW);
            cmd.SetComputeIntParam(_cs, "_KernelHVar", kernelH);
            cmd.SetComputeIntParam(_cs, "_StrideWVar", Mathf.Max(1, strideW));
            cmd.SetComputeIntParam(_cs, "_StrideHVar", Mathf.Max(1, strideH));
            cmd.SetComputeIntParam(_cs, "_PadLeftVar", Mathf.Max(0, padLeft));
            cmd.SetComputeIntParam(_cs, "_PadTopVar", Mathf.Max(0, padTop));
            cmd.SetComputeIntParam(_cs, "_DilationWVar", Mathf.Max(1, dilationW));
            cmd.SetComputeIntParam(_cs, "_DilationHVar", Mathf.Max(1, dilationH));
            cmd.SetComputeIntParam(_cs, "_ActType", activationType);
            cmd.SetComputeFloatParam(_cs, "_ActParam", activationParam);
            cmd.SetComputeBufferParam(_cs, _kConvPack4General, "_ConvW4", w4);
            cmd.SetComputeBufferParam(_cs, _kConvPack4General, "_ConvB4", b4);
            cmd.SetComputeTextureParam(_cs, _kConvPack4General, "_ConvInArr", srcPack4.nameID);
            cmd.SetComputeTextureParam(_cs, _kConvPack4General, "_ConvOutArr", dstPack4.nameID);
            Dispatch3D(cmd, _kConvPack4General, dstPack4.width, dstPack4.height, outPacks, 8, 8);
        }



        public void AddPack4(CommandBuffer cmd, RenderTexture a, RenderTexture b, float coeffA, float coeffB, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeFloatParam(_cs, "_CoeffA", coeffA);
            cmd.SetComputeFloatParam(_cs, "_CoeffB", coeffB);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddA", a);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddB", b);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, "_AddOutArr", output);
            Dispatch3D(cmd, _kAddPack4, output.width, output.height, packs, 8, 8);
        }

        public void AddPack4(CommandBuffer cmd, int aID, int bID, float coeffA, float coeffB, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeFloatParam(_cs, "_CoeffA", coeffA);
            cmd.SetComputeFloatParam(_cs, "_CoeffB", coeffB);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, aID, (RenderTargetIdentifier)aID);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, bID, (RenderTargetIdentifier)bID);
            cmd.SetComputeTextureParam(_cs, _kAddPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kAddPack4, width, height, packs, 8, 8);
        }



        public void Interp2xPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, "_InterpInArr", input);
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, "_InterpOutArr", output);
            Dispatch3D(cmd, _kInterp2xPack4, output.width, output.height, packs, 8, 8);
        }

        public void Interp2xPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kInterp2xPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kInterp2xPack4, width, height, packs, 8, 8);
        }

        public void Interp2xNearestPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, "_InterpNnInArr", input);
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, "_InterpNnOutArr", output);
            Dispatch3D(cmd, _kInterp2xNearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void Interp2xNearestPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kInterp2xNearestPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kInterp2xNearestPack4, width, height, packs, 8, 8);
        }

        public void InterpDown2Pack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, "_InterpDownInArr", input);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, "_InterpDownOutArr", output);
            Dispatch3D(cmd, _kInterpDown2Pack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2Pack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2Pack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kInterpDown2Pack4, width, height, packs, 8, 8);
        }

        public void InterpDown2NearestPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, "_InterpDownNnInArr", input);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, "_InterpDownNnOutArr", output);
            Dispatch3D(cmd, _kInterpDown2NearestPack4, output.width, output.height, packs, 8, 8);
        }

        public void InterpDown2NearestPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kInterpDown2NearestPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kInterpDown2NearestPack4, width, height, packs, 8, 8);
        }

        public void PaddingPack4(CommandBuffer cmd, RenderTexture input, int packs, int padLeft, int padRight, int padTop, int padBottom, int padType, Vector4 padValue, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PadRight", padRight);
            cmd.SetComputeIntParam(_cs, "_PadTop", padTop);
            cmd.SetComputeIntParam(_cs, "_PadBottom", padBottom);
            cmd.SetComputeIntParam(_cs, "_PadType", padType);
            cmd.SetComputeVectorParam(_cs, "_PadValue4", padValue);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, "_PadInArr", input);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, "_PadOutArr", output);
            Dispatch3D(cmd, _kPaddingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PaddingPack4(CommandBuffer cmd, int inputID, int packs, int padLeft, int padRight, int padTop, int padBottom, int padType, Vector4 padValue, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_PadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PadRight", padRight);
            cmd.SetComputeIntParam(_cs, "_PadTop", padTop);
            cmd.SetComputeIntParam(_cs, "_PadBottom", padBottom);
            cmd.SetComputeIntParam(_cs, "_PadType", padType);
            cmd.SetComputeVectorParam(_cs, "_PadValue4", padValue);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kPaddingPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kPaddingPack4, width, height, packs, 8, 8);
        }

        public void PoolingPack4(CommandBuffer cmd, RenderTexture input, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int poolType, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PoolKernelW", kernelW);
            cmd.SetComputeIntParam(_cs, "_PoolKernelH", kernelH);
            cmd.SetComputeIntParam(_cs, "_PoolStrideW", strideW);
            cmd.SetComputeIntParam(_cs, "_PoolStrideH", strideH);
            cmd.SetComputeIntParam(_cs, "_PoolPadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PoolPadTop", padTop);
            cmd.SetComputeIntParam(_cs, "_PoolType", poolType);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, "_PoolInArr", input);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, "_PoolOutArr", output);
            Dispatch3D(cmd, _kPoolingPack4, output.width, output.height, packs, 8, 8);
        }

        public void PoolingPack4(CommandBuffer cmd, int inputID, int packs, int kernelW, int kernelH, int strideW, int strideH, int padLeft, int padTop, int poolType, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_PoolKernelW", kernelW);
            cmd.SetComputeIntParam(_cs, "_PoolKernelH", kernelH);
            cmd.SetComputeIntParam(_cs, "_PoolStrideW", strideW);
            cmd.SetComputeIntParam(_cs, "_PoolStrideH", strideH);
            cmd.SetComputeIntParam(_cs, "_PoolPadLeft", padLeft);
            cmd.SetComputeIntParam(_cs, "_PoolPadTop", padTop);
            cmd.SetComputeIntParam(_cs, "_PoolType", poolType);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kPoolingPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kPoolingPack4, width, height, packs, 8, 8);
        }

        public void SoftmaxChannelPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_SoftmaxPacks", packs);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, "_SoftmaxInArr", input);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, "_SoftmaxOutArr", output);
            Dispatch3D(cmd, _kSoftmaxChannelPack4, output.width, output.height, packs, 8, 8);
        }

        public void SoftmaxChannelPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_SoftmaxPacks", packs);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kSoftmaxChannelPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kSoftmaxChannelPack4, width, height, packs, 8, 8);
        }

        public void UnaryOpPack4(CommandBuffer cmd, RenderTexture input, int packs, int opType, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_UnaryOpType", opType);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, "_UnaryInArr", input);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, "_UnaryOutArr", output);
            Dispatch3D(cmd, _kUnaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void UnaryOpPack4(CommandBuffer cmd, int inputID, int packs, int opType, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_UnaryOpType", opType);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kUnaryOpPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kUnaryOpPack4, width, height, packs, 8, 8);
        }

        public void BinaryOpPack4(CommandBuffer cmd, RenderTexture a, RenderTexture b, int packs, int opType, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 0);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", 0f);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryA", a);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryB", b);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(cmd, _kBinaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpPack4(CommandBuffer cmd, int aID, int bID, int packs, int opType, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 0);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", 0f);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, aID, (RenderTargetIdentifier)aID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, bID, (RenderTargetIdentifier)bID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kBinaryOpPack4, width, height, packs, 8, 8);
        }

        public void BinaryOpScalarPack4(CommandBuffer cmd, RenderTexture a, float scalarB, int packs, int opType, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 1);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", scalarB);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryA", a);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryB", a);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, "_BinaryOutArr", output);
            Dispatch3D(cmd, _kBinaryOpPack4, output.width, output.height, packs, 8, 8);
        }

        public void BinaryOpScalarPack4(CommandBuffer cmd, int aID, float scalarB, int packs, int opType, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_BinaryOpType", opType);
            cmd.SetComputeIntParam(_cs, "_BinaryWithScalar", 1);
            cmd.SetComputeFloatParam(_cs, "_BinaryScalar", scalarB);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, aID, (RenderTargetIdentifier)aID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, aID, (RenderTargetIdentifier)aID);
            cmd.SetComputeTextureParam(_cs, _kBinaryOpPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kBinaryOpPack4, width, height, packs, 8, 8);
        }

        public void SwishPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, "_ActInArr", input);
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, "_ActOutArr", output);
            Dispatch3D(cmd, _kSwishPack4, output.width, output.height, packs, 8, 8);
        }

        public void SwishPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kSwishPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kSwishPack4, width, height, packs, 8, 8);
        }

        public void SigmoidPack4(CommandBuffer cmd, RenderTexture input, int packs, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, "_ActInArr", input);
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, "_ActOutArr", output);
            Dispatch3D(cmd, _kSigmoidPack4, output.width, output.height, packs, 8, 8);
        }

        public void SigmoidPack4(CommandBuffer cmd, int inputID, int packs, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kSigmoidPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kSigmoidPack4, width, height, packs, 8, 8);
        }

        public void GeluPack4(CommandBuffer cmd, RenderTexture input, int packs, bool fast, RenderTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_GeluFast", fast ? 1 : 0);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, "_ActInArr", input);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, "_ActOutArr", output);
            Dispatch3D(cmd, _kGeluPack4, output.width, output.height, packs, 8, 8);
        }

        public void GeluPack4(CommandBuffer cmd, int inputID, int packs, bool fast, int outputID, int width, int height)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            cmd.SetComputeIntParam(_cs, "_GeluFast", fast ? 1 : 0);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, inputID, (RenderTargetIdentifier)inputID);
            cmd.SetComputeTextureParam(_cs, _kGeluPack4, outputID, (RenderTargetIdentifier)outputID);
            Dispatch3D(cmd, _kGeluPack4, width, height, packs, 8, 8);
        }

        public void CopyBuf(ComputeBuffer src, ComputeBuffer dst, int total)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetBuffer(_kCopyBuf, "_BufA", src);
            _cs.SetBuffer(_kCopyBuf, "_BufOut", dst);
            Dispatch1D(_kCopyBuf, total, 256);
        }

        public void CopyBufPartial(ComputeBuffer src, int srcOffset, ComputeBuffer dst, int total, int dstOffset = 0)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_SrcOffset", srcOffset);
            _cs.SetInt("_DstOffset", dstOffset);
            _cs.SetBuffer(_kCopyBufPartial, "_BufA", src);
            _cs.SetBuffer(_kCopyBufPartial, "_BufOut", dst);
            Dispatch1D(_kCopyBufPartial, total, 256);
        }

        public void ReductionBuf(ComputeBuffer input, int elemCount, int outCount, int redType, float coeff, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (elemCount <= 0) throw new ArgumentOutOfRangeException(nameof(elemCount));
            if (outCount <= 0) throw new ArgumentOutOfRangeException(nameof(outCount));

            if (outCount == 1)
            {
                ReduceAllSumOrMean(input, elemCount, redType == 3 || redType == 5, output);
                if (coeff != 1f)
                    MulScalarInplace(output, coeff, 1);
            }
            else
            {
                var host = new float[input.count];
                input.GetData(host);
                var reduced = new float[outCount];
                var useMean = redType == 3 || redType == 5;
                for (var i = 0; i < outCount; i++)
                {
                    var offset = i * elemCount;
                    double sum = 0.0;
                    for (var j = 0; j < elemCount; j++)
                    {
                        sum += host[offset + j];
                    }
                    var value = useMean ? (float)(sum / Math.Max(1, elemCount)) : (float)sum;
                    reduced[i] = value * coeff;
                }
                output.SetData(reduced);
            }
        }

        public void BinaryOpBuf(ComputeBuffer a, ComputeBuffer b, int total, int opType, ComputeBuffer output, int broadcastMode = 0, int broadcastSize = 0)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 0);
            _cs.SetFloat("_BinaryScalar", 0f);
            _cs.SetInt("_BinaryBroadcastMode", broadcastMode);
            _cs.SetInt("_BinaryBroadcastSize", broadcastSize);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufA", a);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufB", b);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufOut", output);
            Dispatch1D(_kBinaryOpBuf, total, 256);
        }

        public void BinaryOpScalarBuf(ComputeBuffer a, float scalarB, int total, int opType, ComputeBuffer output, int broadcastMode = 0, int broadcastSize = 0)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 1);
            _cs.SetFloat("_BinaryScalar", scalarB);
            _cs.SetInt("_BinaryBroadcastMode", broadcastMode);
            _cs.SetInt("_BinaryBroadcastSize", broadcastSize);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufA", a);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufB", a);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufOut", output);
            Dispatch1D(_kBinaryOpBuf, total, 256);
        }

        public void UnaryOpBuf(ComputeBuffer input, int total, int opType, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_UnaryOpType", opType);
            _cs.SetBuffer(_kUnaryOpBuf, "_BufA", input);
            _cs.SetBuffer(_kUnaryOpBuf, "_BufOut", output);
            Dispatch1D(_kUnaryOpBuf, total, 256);
        }

        public void SigmoidBuf(ComputeBuffer input, int total, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetBuffer(_kSigmoidBuf, "_BufA", input);
            _cs.SetBuffer(_kSigmoidBuf, "_BufOut", output);
            Dispatch1D(_kSigmoidBuf, total, 256);
        }

        public void SwishBuf(ComputeBuffer input, int total, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetBuffer(_kSwishBuf, "_BufA", input);
            _cs.SetBuffer(_kSwishBuf, "_BufOut", output);
            Dispatch1D(_kSwishBuf, total, 256);
        }

        public void GeluBuf(ComputeBuffer input, int total, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetBuffer(_kGeluBuf, "_BufA", input);
            _cs.SetBuffer(_kGeluBuf, "_BufOut", output);
            Dispatch1D(_kGeluBuf, total, 256);
        }

        public void PointwiseBuf(ComputeBuffer input, int total, PointwiseType type, float a, float b, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_PointwiseType", (int)type);
            _cs.SetFloat("_PointwiseA", a);
            _cs.SetFloat("_PointwiseB", b);
            _cs.SetBuffer(_kPointwiseBuf, "_BufA", input);
            _cs.SetBuffer(_kPointwiseBuf, "_BufOut", output);
            Dispatch1D(_kPointwiseBuf, total, 256);
        }

        public void CastBuf(ComputeBuffer input, int total, int typeFrom, int typeTo, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_CastTypeFrom", typeFrom);
            _cs.SetInt("_CastTypeTo", typeTo);
            _cs.SetBuffer(_kCastBuf, "_QuantIn", input);
            _cs.SetBuffer(_kCastBuf, "_QuantOut", output);
            Dispatch1D(_kCastBuf, total, 256);
        }

        public void ScaleBuf(
            ComputeBuffer input,
            NcnnTensorBuffer view,
            ComputeBuffer scale,
            int scaleDataSize,
            bool biasTerm,
            ComputeBuffer bias,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_ScaleDims", view.dims);
            _cs.SetInt("_ScaleW", view.w);
            _cs.SetInt("_ScaleH", view.h);
            _cs.SetInt("_ScaleD", view.d);
            _cs.SetInt("_ScaleC", view.c);
            _cs.SetInt("_ScaleDataSize", scaleDataSize);
            _cs.SetInt("_ScaleBiasTerm", biasTerm ? 1 : 0);
            _cs.SetBuffer(_kScaleBuf, "_ScaleInBuf", input);
            _cs.SetBuffer(_kScaleBuf, "_ScaleOutBuf", output);
            _cs.SetBuffer(_kScaleBuf, "_ScaleScaleBuf", scale ?? input);
            _cs.SetBuffer(_kScaleBuf, "_ScaleBiasBuf", bias ?? scale ?? input);
            Dispatch1D(_kScaleBuf, total, 256);
        }

        public void LeakyReluBuf(ComputeBuffer input, int total, float slope, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            if (!ReferenceEquals(input, output))
                CopyBuf(input, output, total);

            _cs.SetInt("_Total", total);
            _cs.SetFloat("_CoeffA", slope);
            _cs.SetBuffer(_kLeakyReluBuf, "_BufOut", output);
            Dispatch1D(_kLeakyReluBuf, total, 256);
        }

        public void PReluBuf(ComputeBuffer input, NcnnTensorBuffer view, ComputeBuffer slope, int slopeCount, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_PReluDims", view.dims);
            _cs.SetInt("_PReluW", view.w);
            _cs.SetInt("_PReluH", view.h);
            _cs.SetInt("_PReluD", view.d);
            _cs.SetInt("_PReluC", view.c);
            _cs.SetInt("_PReluSlopeCount", slopeCount);
            _cs.SetBuffer(_kPReluBuf, "_PReluInBuf", input);
            _cs.SetBuffer(_kPReluBuf, "_PReluOutBuf", output);
            _cs.SetBuffer(_kPReluBuf, "_PReluSlopeBuf", slope ?? input);
            Dispatch1D(_kPReluBuf, total, 256);
        }

        public void ReorgBuf(ComputeBuffer input, int inW, int inH, int inC, int stride, int mode, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inW <= 0 || inH <= 0 || inC <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (stride <= 0) throw new ArgumentOutOfRangeException(nameof(stride));

            var outW = inW / stride;
            var outH = inH / stride;
            var outC = inC * stride * stride;
            var total = outW * outH * outC;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_ReorgInW", inW);
            _cs.SetInt("_ReorgInH", inH);
            _cs.SetInt("_ReorgInC", inC);
            _cs.SetInt("_ReorgStride", stride);
            _cs.SetInt("_ReorgMode", mode);
            _cs.SetBuffer(_kReorgBuf, "_ReorgInBuf", input);
            _cs.SetBuffer(_kReorgBuf, "_ReorgOutBuf", output);
            Dispatch1D(_kReorgBuf, total, 256);
        }

        public void ReductionRowsBuf(ComputeBuffer input, int reduceElems, int outCount, int opType, float coeff, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (reduceElems <= 0) throw new ArgumentOutOfRangeException(nameof(reduceElems));
            if (outCount <= 0) throw new ArgumentOutOfRangeException(nameof(outCount));

            _cs.SetInt("_ReductionRowsReduceElems", reduceElems);
            _cs.SetInt("_ReductionRowsOutCount", outCount);
            _cs.SetInt("_ReductionRowsOpType", opType);
            _cs.SetFloat("_ReductionRowsCoeff", coeff);
            _cs.SetBuffer(_kReductionRowsBuf, "_ReductionRowsIn", input);
            _cs.SetBuffer(_kReductionRowsBuf, "_ReductionRowsOut", output);
            _cs.Dispatch(_kReductionRowsBuf, Mathf.Max(1, Mathf.CeilToInt(outCount / 256f)), 1, 1);
        }

        public void Conv1dBuf(
            ComputeBuffer input,
            ComputeBuffer weights,
            ComputeBuffer bias,
            int inW,
            int inC,
            int outW,
            int outC,
            int kernelW,
            int strideW,
            int dilationW,
            int padLeft,
            int activationType,
            float activationParam,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = outW * outC;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_Conv1dInW", inW);
            _cs.SetInt("_Conv1dInC", inC);
            _cs.SetInt("_Conv1dOutW", outW);
            _cs.SetInt("_Conv1dOutC", outC);
            _cs.SetInt("_Conv1dKernelW", kernelW);
            _cs.SetInt("_Conv1dStrideW", strideW);
            _cs.SetInt("_Conv1dDilationW", dilationW);
            _cs.SetInt("_Conv1dPadLeft", padLeft);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv1dBuf, "_Conv1dIn", input);
            _cs.SetBuffer(_kConv1dBuf, "_Conv1dW", weights);
            _cs.SetBuffer(_kConv1dBuf, "_Conv1dB", bias);
            _cs.SetBuffer(_kConv1dBuf, "_Conv1dOut", output);
            Dispatch1D(_kConv1dBuf, total, 256);
        }

        public void QuantizeBuf(ComputeBuffer input, NcnnTensorBuffer view, ComputeBuffer scale, int scaleDataSize, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_QuantDims", view.dims);
            _cs.SetInt("_QuantW", view.w);
            _cs.SetInt("_QuantH", view.h);
            _cs.SetInt("_QuantD", view.d);
            _cs.SetInt("_QuantC", view.c);
            _cs.SetInt("_QuantScaleInSize", scaleDataSize);
            _cs.SetBuffer(_kQuantizeBuf, "_QuantIn", input);
            _cs.SetBuffer(_kQuantizeBuf, "_QuantOut", output);
            _cs.SetBuffer(_kQuantizeBuf, "_QuantScale", scale);
            Dispatch1D(_kQuantizeBuf, total, 256);
        }

        public void DequantizeBuf(ComputeBuffer input, NcnnTensorBuffer view, ComputeBuffer scale, int scaleDataSize, ComputeBuffer bias, int biasDataSize, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_QuantDims", view.dims);
            _cs.SetInt("_QuantW", view.w);
            _cs.SetInt("_QuantH", view.h);
            _cs.SetInt("_QuantD", view.d);
            _cs.SetInt("_QuantC", view.c);
            _cs.SetInt("_QuantScaleInSize", scaleDataSize);
            _cs.SetInt("_QuantBiasSize", biasDataSize);
            _cs.SetBuffer(_kDequantizeBuf, "_QuantIn", input);
            _cs.SetBuffer(_kDequantizeBuf, "_QuantOut", output);
            _cs.SetBuffer(_kDequantizeBuf, "_QuantScale", scale);
            _cs.SetBuffer(_kDequantizeBuf, "_QuantBias", bias ?? scale ?? input);
            Dispatch1D(_kDequantizeBuf, total, 256);
        }

        public void RequantizeBuf(
            ComputeBuffer input,
            NcnnTensorBuffer view,
            ComputeBuffer scaleIn,
            int scaleInDataSize,
            ComputeBuffer scaleOut,
            int scaleOutDataSize,
            ComputeBuffer bias,
            int biasDataSize,
            int activationType,
            float activationParam0,
            float activationParam1,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_QuantDims", view.dims);
            _cs.SetInt("_QuantW", view.w);
            _cs.SetInt("_QuantH", view.h);
            _cs.SetInt("_QuantD", view.d);
            _cs.SetInt("_QuantC", view.c);
            _cs.SetInt("_QuantScaleInSize", scaleInDataSize);
            _cs.SetInt("_QuantScaleOutSize", scaleOutDataSize);
            _cs.SetInt("_QuantBiasSize", biasDataSize);
            _cs.SetInt("_QuantActType", activationType);
            _cs.SetFloat("_QuantActParam0", activationParam0);
            _cs.SetFloat("_QuantActParam1", activationParam1);
            _cs.SetBuffer(_kRequantizeBuf, "_QuantIn", input);
            _cs.SetBuffer(_kRequantizeBuf, "_QuantOut", output);
            _cs.SetBuffer(_kRequantizeBuf, "_QuantScale", scaleIn);
            _cs.SetBuffer(_kRequantizeBuf, "_QuantScaleOut", scaleOut);
            _cs.SetBuffer(_kRequantizeBuf, "_QuantBias", bias ?? scaleIn ?? input);
            Dispatch1D(_kRequantizeBuf, total, 256);
        }

        public void PixelShuffleBuf(ComputeBuffer input, int inW, int inH, int inC, int upscaleFactor, int mode, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inW <= 0 || inH <= 0 || inC <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (upscaleFactor <= 0) throw new ArgumentOutOfRangeException(nameof(upscaleFactor));

            var divisor = upscaleFactor * upscaleFactor;
            if (inC % divisor != 0)
                throw new ArgumentOutOfRangeException(nameof(inC), "channel count must be divisible by upscale_factor^2");

            var outW = inW * upscaleFactor;
            var outH = inH * upscaleFactor;
            var outC = inC / divisor;
            var total = outW * outH * outC;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_PixelShuffleInW", inW);
            _cs.SetInt("_PixelShuffleInH", inH);
            _cs.SetInt("_PixelShuffleOutW", outW);
            _cs.SetInt("_PixelShuffleOutH", outH);
            _cs.SetInt("_PixelShuffleOutC", outC);
            _cs.SetInt("_PixelShuffleScale", upscaleFactor);
            _cs.SetInt("_PixelShuffleMode", mode);
            _cs.SetBuffer(_kPixelShuffleBuf, "_PixelShuffleIn", input);
            _cs.SetBuffer(_kPixelShuffleBuf, "_PixelShuffleOut", output);
            Dispatch1D(_kPixelShuffleBuf, total, 256);
        }

        public void RotaryEmbedBuf(ComputeBuffer input, int embedDim, int seqLen, int numHeads, bool interleaved, ComputeBuffer cos, ComputeBuffer sin, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (cos == null) throw new ArgumentNullException(nameof(cos));
            if (sin == null) throw new ArgumentNullException(nameof(sin));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (embedDim <= 0 || seqLen <= 0 || numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));

            var halfDim = embedDim / 2;
            if (halfDim <= 0)
                throw new ArgumentOutOfRangeException(nameof(embedDim), "embedDim must be at least 2");

            CopyBuf(input, output, Mathf.Min(input.count, output.count));
            var total = halfDim * seqLen * numHeads;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_RotaryEmbedDim", embedDim);
            _cs.SetInt("_RotarySeqLen", seqLen);
            _cs.SetInt("_RotaryNumHeads", numHeads);
            _cs.SetInt("_RotaryInterleaved", interleaved ? 1 : 0);
            _cs.SetBuffer(_kRotaryEmbedBuf, "_RotaryIn", input);
            _cs.SetBuffer(_kRotaryEmbedBuf, "_RotaryCos", cos);
            _cs.SetBuffer(_kRotaryEmbedBuf, "_RotarySin", sin);
            _cs.SetBuffer(_kRotaryEmbedBuf, "_RotaryOut", output);
            Dispatch1D(_kRotaryEmbedBuf, total, 256);
        }

        public void NormalizeBuf(
            ComputeBuffer input,
            NcnnTensorBuffer view,
            ComputeBuffer scale,
            int scaleDataSize,
            bool acrossSpatial,
            bool acrossChannel,
            bool channelShared,
            float eps,
            int epsMode,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_NormDims", view.dims);
            _cs.SetInt("_NormW", view.w);
            _cs.SetInt("_NormH", view.h);
            _cs.SetInt("_NormD", view.d);
            _cs.SetInt("_NormC", view.c);
            _cs.SetInt("_NormScaleSize", scaleDataSize);
            _cs.SetInt("_NormAcrossSpatial", acrossSpatial ? 1 : 0);
            _cs.SetInt("_NormAcrossChannel", acrossChannel ? 1 : 0);
            _cs.SetInt("_NormChannelShared", channelShared ? 1 : 0);
            _cs.SetFloat("_NormEps", eps);
            _cs.SetInt("_NormEpsMode", epsMode);
            _cs.SetBuffer(_kNormalizeBuf, "_NormIn", input);
            _cs.SetBuffer(_kNormalizeBuf, "_NormOut", output);
            _cs.SetBuffer(_kNormalizeBuf, "_NormScale", scale ?? input);
            Dispatch1D(_kNormalizeBuf, total, 256);
        }

        public void LrnBuf(
            ComputeBuffer input,
            int w,
            int h,
            int c,
            int regionType,
            int localSize,
            float alpha,
            float beta,
            float bias,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (w <= 0 || h <= 0 || c <= 0) throw new ArgumentOutOfRangeException(nameof(w));

            var total = w * h * c;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_LrnW", w);
            _cs.SetInt("_LrnH", h);
            _cs.SetInt("_LrnC", c);
            _cs.SetInt("_LrnRegionType", regionType);
            _cs.SetInt("_LrnLocalSize", localSize);
            _cs.SetFloat("_LrnAlpha", alpha);
            _cs.SetFloat("_LrnBeta", beta);
            _cs.SetFloat("_LrnBias", bias);
            _cs.SetBuffer(_kLrnBuf, "_LrnIn", input);
            _cs.SetBuffer(_kLrnBuf, "_LrnOut", output);
            Dispatch1D(_kLrnBuf, total, 256);
        }

        public void RmsNormBuf(
            ComputeBuffer input,
            NcnnTensorBuffer view,
            ComputeBuffer gamma,
            int affineSize,
            bool affine,
            float eps,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (view == null) throw new ArgumentNullException(nameof(view));
            if (output == null) throw new ArgumentNullException(nameof(output));

            var total = view.elementCount;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_RmsNormDims", view.dims);
            _cs.SetInt("_RmsNormW", view.w);
            _cs.SetInt("_RmsNormH", view.h);
            _cs.SetInt("_RmsNormD", view.d);
            _cs.SetInt("_RmsNormC", view.c);
            _cs.SetInt("_RmsNormAffineSize", affineSize);
            _cs.SetInt("_RmsNormAffine", affine ? 1 : 0);
            _cs.SetFloat("_RmsNormEps", eps);
            _cs.SetBuffer(_kRmsNormBuf, "_RmsNormIn", input);
            _cs.SetBuffer(_kRmsNormBuf, "_RmsNormOut", output);
            _cs.SetBuffer(_kRmsNormBuf, "_RmsNormGamma", gamma ?? input);
            Dispatch1D(_kRmsNormBuf, total, 256);
        }

        public void UnfoldBuf(
            ComputeBuffer input,
            int inW,
            int inH,
            int inC,
            int outW,
            int outH,
            int kernelW,
            int kernelH,
            int dilationW,
            int dilationH,
            int strideW,
            int strideH,
            int padLeft,
            int padTop,
            float padValue,
            ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inW <= 0 || inH <= 0 || inC <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (outW <= 0 || outH <= 0) throw new ArgumentOutOfRangeException(nameof(outW));

            var total = outW * outH * kernelW * kernelH * inC;
            _cs.SetInt("_Total", total);
            _cs.SetInt("_UnfoldInW", inW);
            _cs.SetInt("_UnfoldInH", inH);
            _cs.SetInt("_UnfoldInC", inC);
            _cs.SetInt("_UnfoldOutW", outW);
            _cs.SetInt("_UnfoldOutH", outH);
            _cs.SetInt("_UnfoldKernelW", kernelW);
            _cs.SetInt("_UnfoldKernelH", kernelH);
            _cs.SetInt("_UnfoldDilationW", dilationW);
            _cs.SetInt("_UnfoldDilationH", dilationH);
            _cs.SetInt("_UnfoldStrideW", strideW);
            _cs.SetInt("_UnfoldStrideH", strideH);
            _cs.SetInt("_UnfoldPadLeft", padLeft);
            _cs.SetInt("_UnfoldPadTop", padTop);
            _cs.SetFloat("_UnfoldPadValue", padValue);
            _cs.SetBuffer(_kUnfoldBuf, "_UnfoldIn", input);
            _cs.SetBuffer(_kUnfoldBuf, "_UnfoldOut", output);
            Dispatch1D(_kUnfoldBuf, total, 256);
        }

        public void SdpaQkBuf(
            ComputeBuffer query,
            ComputeBuffer key,
            ComputeBuffer mask,
            int srcLen,
            int dstLen,
            int embedDim,
            int numHeads,
            int numGroups,
            int maskDims,
            int maskW,
            int maskH,
            int maskC,
            float scale,
            ComputeBuffer scores)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (key == null) throw new ArgumentNullException(nameof(key));
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (srcLen <= 0 || dstLen <= 0 || embedDim <= 0 || numHeads <= 0 || numGroups <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen > 4096)
                throw new ArgumentOutOfRangeException(nameof(dstLen), "SDPA dstLen exceeds shared-memory kernel limit: " + dstLen);

            _cs.SetInt("_SdpaSrcLen", srcLen);
            _cs.SetInt("_SdpaDstLen", dstLen);
            _cs.SetInt("_SdpaEmbedDim", embedDim);
            _cs.SetInt("_SdpaNumHeads", numHeads);
            _cs.SetInt("_SdpaNumGroups", numGroups);
            _cs.SetInt("_SdpaNumHeadsPerGroup", Mathf.Max(1, numHeads / numGroups));
            _cs.SetInt("_SdpaMaskDims", maskDims);
            _cs.SetInt("_SdpaMaskW", maskW);
            _cs.SetInt("_SdpaMaskH", maskH);
            _cs.SetInt("_SdpaMaskC", maskC);
            _cs.SetFloat("_SdpaScale", scale);
            _cs.SetBuffer(_kSdpaQkBuf, "_SdpaQ", query);
            _cs.SetBuffer(_kSdpaQkBuf, "_SdpaK", key);
            _cs.SetBuffer(_kSdpaQkBuf, "_SdpaMask", mask ?? query);
            _cs.SetBuffer(_kSdpaQkBuf, "_SdpaScores", scores);
            _cs.Dispatch(_kSdpaQkBuf, srcLen, numHeads, 1);
        }

        public void SdpaSoftmaxBuf(ComputeBuffer scores, int srcLen, int dstLen, int numHeads)
        {
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (srcLen <= 0 || dstLen <= 0 || numHeads <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen > 4096)
                throw new ArgumentOutOfRangeException(nameof(dstLen), "SDPA dstLen exceeds shared-memory kernel limit: " + dstLen);

            _cs.SetInt("_SdpaSrcLen", srcLen);
            _cs.SetInt("_SdpaDstLen", dstLen);
            _cs.SetInt("_SdpaNumHeads", numHeads);
            _cs.SetBuffer(_kSdpaSoftmaxBuf, "_SdpaScores", scores);
            _cs.Dispatch(_kSdpaSoftmaxBuf, srcLen, numHeads, 1);
        }

        public void SdpaQkvBuf(
            ComputeBuffer scores,
            ComputeBuffer value,
            int srcLen,
            int dstLen,
            int outEmbedDim,
            int numHeads,
            int numGroups,
            ComputeBuffer output)
        {
            if (scores == null) throw new ArgumentNullException(nameof(scores));
            if (value == null) throw new ArgumentNullException(nameof(value));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (srcLen <= 0 || dstLen <= 0 || outEmbedDim <= 0 || numHeads <= 0 || numGroups <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen > 4096)
                throw new ArgumentOutOfRangeException(nameof(dstLen), "SDPA dstLen exceeds shared-memory kernel limit: " + dstLen);

            _cs.SetInt("_SdpaSrcLen", srcLen);
            _cs.SetInt("_SdpaDstLen", dstLen);
            _cs.SetInt("_SdpaOutEmbedDim", outEmbedDim);
            _cs.SetInt("_SdpaNumHeads", numHeads);
            _cs.SetInt("_SdpaNumGroups", numGroups);
            _cs.SetInt("_SdpaNumHeadsPerGroup", Mathf.Max(1, numHeads / numGroups));
            _cs.SetBuffer(_kSdpaQkvBuf, "_SdpaScores", scores);
            _cs.SetBuffer(_kSdpaQkvBuf, "_SdpaV", value);
            _cs.SetBuffer(_kSdpaQkvBuf, "_SdpaOut", output);
            _cs.Dispatch(_kSdpaQkvBuf, srcLen, numHeads, 1);
        }

        public void SdpaAttentionFast(
            ComputeBuffer query,
            ComputeBuffer key,
            ComputeBuffer value,
            ComputeBuffer mask,
            int srcLen,
            int dstLen,
            int embedDim,
            int outEmbedDim,
            int numHeads,
            int numGroups,
            int maskDims,
            int maskW,
            int maskH,
            int maskC,
            float scale,
            ComputeBuffer output)
        {
            if (query == null) throw new ArgumentNullException(nameof(query));
            if (key == null) throw new ArgumentNullException(nameof(key));
            if (value == null) throw new ArgumentNullException(nameof(value));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (srcLen <= 0 || dstLen <= 0 || embedDim <= 0 || outEmbedDim <= 0 || numHeads <= 0 || numGroups <= 0)
                throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen > 4096)
                throw new ArgumentOutOfRangeException(nameof(dstLen), "SDPA dstLen exceeds shared-memory kernel limit: " + dstLen);

            _cs.SetInt("_SdpaSrcLen", srcLen);
            _cs.SetInt("_SdpaDstLen", dstLen);
            _cs.SetInt("_SdpaEmbedDim", embedDim);
            _cs.SetInt("_SdpaOutEmbedDim", outEmbedDim);
            _cs.SetInt("_SdpaNumHeads", numHeads);
            _cs.SetInt("_SdpaNumGroups", numGroups);
            _cs.SetInt("_SdpaNumHeadsPerGroup", Mathf.Max(1, numHeads / numGroups));
            _cs.SetInt("_SdpaMaskDims", maskDims);
            _cs.SetInt("_SdpaMaskW", maskW);
            _cs.SetInt("_SdpaMaskH", maskH);
            _cs.SetInt("_SdpaMaskC", maskC);
            _cs.SetFloat("_SdpaScale", scale);
            _cs.SetBuffer(_kSdpaAttentionFast, "_SdpaQ", query);
            _cs.SetBuffer(_kSdpaAttentionFast, "_SdpaK", key);
            _cs.SetBuffer(_kSdpaAttentionFast, "_SdpaV", value);
            _cs.SetBuffer(_kSdpaAttentionFast, "_SdpaMask", mask ?? query);
            _cs.SetBuffer(_kSdpaAttentionFast, "_SdpaOut", output);
            _cs.Dispatch(_kSdpaAttentionFast, srcLen, numHeads, 1);
        }

        public void InnerProduct2D(ComputeBuffer input, int rows, int inFeatures, ComputeBuffer weightsOxi, ComputeBuffer biasO, int outFeatures, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weightsOxi == null) throw new ArgumentNullException(nameof(weightsOxi));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
            if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
            if (rows == 0) return;

            if (inFeatures <= 8192)
            {
                Gemm2D(input, weightsOxi, biasO, rows, outFeatures, inFeatures, true, 1f, 1f, true, 4, output);
                return;
            }

            _cs.SetInt("_IP2Rows", rows);
            _cs.SetInt("_IP2InFeatures", inFeatures);
            _cs.SetInt("_IP2OutFeatures", outFeatures);
            _cs.SetBuffer(_kInnerProduct2D, "_IP2In", input);
            _cs.SetBuffer(_kInnerProduct2D, "_IP2W", weightsOxi);
            _cs.SetBuffer(_kInnerProduct2D, "_IP2B", biasO);
            _cs.SetBuffer(_kInnerProduct2D, "_IP2Out", output);

            var gx = (outFeatures + 7) / 8;
            var gy = (rows + 7) / 8;
            _cs.Dispatch(_kInnerProduct2D, gx, gy, 1);
        }

        public void MhaAttention(
            ComputeBuffer q,
            ComputeBuffer k,
            ComputeBuffer v,
            ComputeBuffer mask,
            int srcLen,
            int dstLen,
            int embedDim,
            int numHeads,
            float scale,
            int maskDims,
            int maskW,
            int maskH,
            int maskC,
            ComputeBuffer outContext,
            bool parallelSoftmax = false)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (outContext == null) throw new ArgumentNullException(nameof(outContext));
            if (srcLen <= 0) throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen <= 0) throw new ArgumentOutOfRangeException(nameof(dstLen));
            if (srcLen > 65535) throw new ArgumentOutOfRangeException(nameof(srcLen), "srcLen exceeds Unity compute dispatch limit: " + srcLen);
            if (numHeads > 65535) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads exceeds Unity compute dispatch limit: " + numHeads);
            if (dstLen > 4096) throw new ArgumentOutOfRangeException(nameof(dstLen), "dstLen exceeds current shader shared-memory limit: " + dstLen);
            if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if ((embedDim % numHeads) != 0) throw new ArgumentOutOfRangeException(nameof(embedDim), "embedDim must be divisible by numHeads");

            _cs.SetInt("_MhaSrcLen", srcLen);
            _cs.SetInt("_MhaDstLen", dstLen);
            _cs.SetInt("_MhaEmbedDim", embedDim);
            _cs.SetInt("_MhaNumHeads", numHeads);
            _cs.SetInt("_MhaHeadDim", embedDim / numHeads);
            _cs.SetFloat("_MhaScale", scale);
            _cs.SetInt("_SdpaMaskDims", maskDims);
            _cs.SetInt("_SdpaMaskW", maskW);
            _cs.SetInt("_SdpaMaskH", maskH);
            _cs.SetInt("_SdpaMaskC", maskC);
            var kernel = parallelSoftmax ? _kMhaAttentionFast : _kMhaAttention;
            _cs.SetBuffer(kernel, "_MhaQ", q);
            _cs.SetBuffer(kernel, "_MhaK", k);
            _cs.SetBuffer(kernel, "_MhaV", v);
            _cs.SetBuffer(kernel, "_SdpaMask", mask ?? q);
            _cs.SetBuffer(kernel, "_MhaOut", outContext);

            _cs.Dispatch(kernel, srcLen, numHeads, 1);
        }

        public void MhaProjectQkv2D(ComputeBuffer input, int rows, int inFeatures, ComputeBuffer qW, ComputeBuffer qB, ComputeBuffer kW, ComputeBuffer kB, ComputeBuffer vW, ComputeBuffer vB, int outFeatures, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (qW == null) throw new ArgumentNullException(nameof(qW));
            if (qB == null) throw new ArgumentNullException(nameof(qB));
            if (kW == null) throw new ArgumentNullException(nameof(kW));
            if (kB == null) throw new ArgumentNullException(nameof(kB));
            if (vW == null) throw new ArgumentNullException(nameof(vW));
            if (vB == null) throw new ArgumentNullException(nameof(vB));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (rows < 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
            if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));
            if (rows == 0) return;

            _cs.SetInt("_MhaProjectRows", rows);
            _cs.SetInt("_MhaProjectInFeatures", inFeatures);
            _cs.SetInt("_MhaProjectOutFeatures", outFeatures);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectIn", input);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectQW", qW);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectQB", qB);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectKW", kW);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectKB", kB);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectVW", vW);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectVB", vB);
            _cs.SetBuffer(_kMhaProjectQkv2D, "_MhaProjectOut", output);

            _cs.Dispatch(_kMhaProjectQkv2D, (outFeatures + 15) / 16, (rows + 15) / 16, 3);
        }

        public void MhaAttentionQkv(ComputeBuffer qkv, int srcLen, int embedDim, int numHeads, float scale, ComputeBuffer outContext)
        {
            if (qkv == null) throw new ArgumentNullException(nameof(qkv));
            if (outContext == null) throw new ArgumentNullException(nameof(outContext));
            if (srcLen <= 0) throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (srcLen > 65535) throw new ArgumentOutOfRangeException(nameof(srcLen), "srcLen exceeds Unity compute dispatch limit: " + srcLen);
            if (srcLen > 4096) throw new ArgumentOutOfRangeException(nameof(srcLen), "srcLen exceeds current shader shared-memory limit: " + srcLen);
            if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if (numHeads > 65535) throw new ArgumentOutOfRangeException(nameof(numHeads), "numHeads exceeds Unity compute dispatch limit: " + numHeads);
            if ((embedDim % numHeads) != 0) throw new ArgumentOutOfRangeException(nameof(embedDim), "embedDim must be divisible by numHeads");

            _cs.SetInt("_MhaSrcLen", srcLen);
            _cs.SetInt("_MhaDstLen", srcLen);
            _cs.SetInt("_MhaEmbedDim", embedDim);
            _cs.SetInt("_MhaNumHeads", numHeads);
            _cs.SetInt("_MhaHeadDim", embedDim / numHeads);
            _cs.SetFloat("_MhaScale", scale);
            _cs.SetBuffer(_kMhaAttentionQkvFast, "_MhaQkv", qkv);
            _cs.SetBuffer(_kMhaAttentionQkvFast, "_MhaOut", outContext);

            _cs.Dispatch(_kMhaAttentionQkvFast, srcLen, numHeads, 1);
        }

        public void Pack4ToBufferCHW(RenderTexture input, int w, int h, int c, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * c;
            if (total <= 0) return;
            _cs.SetInt("_Pack4W", w);
            _cs.SetInt("_Pack4H", h);
            _cs.SetInt("_Pack4D", 1);
            _cs.SetInt("_Pack4C", c);
            _cs.SetTexture(_kPack4ToBufferChw, "_Pack4InArr", input);
            _cs.SetBuffer(_kPack4ToBufferChw, "_Pack4Out", output);
            Dispatch1D(_kPack4ToBufferChw, total, 256);
        }

        public void Pack4ToBufferCHW(CommandBuffer cmd, ComputeTexture input, int w, int h, int c, ComputeBuffer output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * c;
            if (total <= 0) return;
            cmd.SetComputeIntParam(_cs, "_Pack4W", w);
            cmd.SetComputeIntParam(_cs, "_Pack4H", h);
            cmd.SetComputeIntParam(_cs, "_Pack4D", 1);
            cmd.SetComputeIntParam(_cs, "_Pack4C", c);
            cmd.SetComputeTextureParam(_cs, _kPack4ToBufferChw, "_Pack4InArr", input.nameID);
            cmd.SetComputeBufferParam(_cs, _kPack4ToBufferChw, "_Pack4Out", output);
            Dispatch1D(cmd, _kPack4ToBufferChw, total, 256);
        }

        public void Pack4ToBufferCDHW(RenderTexture input, int w, int h, int d, int c, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * d * c;
            if (total <= 0) return;
            _cs.SetInt("_Pack4W", w);
            _cs.SetInt("_Pack4H", h);
            _cs.SetInt("_Pack4D", d);
            _cs.SetInt("_Pack4C", c);
            _cs.SetTexture(_kPack4ToBufferCdhw, "_Pack4InArr", input);
            _cs.SetBuffer(_kPack4ToBufferCdhw, "_Pack4Out", output);
            Dispatch1D(_kPack4ToBufferCdhw, total, 256);
        }

        public void Pack4ToBufferCDHW(CommandBuffer cmd, ComputeTexture input, int w, int h, int d, int c, ComputeBuffer output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * d * c;
            if (total <= 0) return;
            cmd.SetComputeIntParam(_cs, "_Pack4W", w);
            cmd.SetComputeIntParam(_cs, "_Pack4H", h);
            cmd.SetComputeIntParam(_cs, "_Pack4D", d);
            cmd.SetComputeIntParam(_cs, "_Pack4C", c);
            cmd.SetComputeTextureParam(_cs, _kPack4ToBufferCdhw, "_Pack4InArr", input.nameID);
            cmd.SetComputeBufferParam(_cs, _kPack4ToBufferCdhw, "_Pack4Out", output);
            Dispatch1D(cmd, _kPack4ToBufferCdhw, total, 256);
        }

        public void Pack4ChannelsToWidth(RenderTexture input, int channels, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));

            _cs.SetInt("_Pack4ChannelCount", channels);
            _cs.SetTexture(_kPack4ChannelsToWidth, "_Pack4ChannelInArr", input);
            _cs.SetTexture(_kPack4ChannelsToWidth, "_Pack4ChannelOutArr", output);
            Dispatch3D(_kPack4ChannelsToWidth, output.width, output.height, ResolveRenderTextureDispatchDepth(output, 1), 8, 8);
        }

        public void Pack4ChannelsToWidth(CommandBuffer cmd, ComputeTexture input, int channels, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (channels <= 0) throw new ArgumentOutOfRangeException(nameof(channels));

            cmd.SetComputeIntParam(_cs, "_Pack4ChannelCount", channels);
            cmd.SetComputeTextureParam(_cs, _kPack4ChannelsToWidth, "_Pack4ChannelInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPack4ChannelsToWidth, "_Pack4ChannelOutArr", output.nameID);
            Dispatch3D(cmd, _kPack4ChannelsToWidth, output.width, output.height, ResolveComputeTextureDispatchDepth(output, 1), 8, 8);
        }

        public void InnerProduct(ComputeBuffer input, int inFeatures, ComputeBuffer weights, ComputeBuffer bias, int outFeatures, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (weights == null) throw new ArgumentNullException(nameof(weights));
            if (bias == null) throw new ArgumentNullException(nameof(bias));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (inFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(inFeatures));
            if (outFeatures <= 0) throw new ArgumentOutOfRangeException(nameof(outFeatures));

            _cs.SetInt("_IPInFeatures", inFeatures);
            _cs.SetInt("_IPOutFeatures", outFeatures);
            _cs.SetBuffer(_kInnerProduct, "_IPIn", input);
            _cs.SetBuffer(_kInnerProduct, "_IPW", weights);
            _cs.SetBuffer(_kInnerProduct, "_IPB", bias);
            _cs.SetBuffer(_kInnerProduct, "_IPOut", output);
            Dispatch1D(_kInnerProduct, outFeatures, 64);
        }

        public void MatMul2D(ComputeBuffer a, ComputeBuffer b, int m, int n, int k, bool transB, ComputeBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (k > 8192) throw new ArgumentOutOfRangeException(nameof(k), "k too large for current tiled kernel (MATK_MAX=8192): " + k);

            _cs.SetInt("_MatM", m);
            _cs.SetInt("_MatN", n);
            _cs.SetInt("_MatK", k);
            _cs.SetInt("_MatTransB", transB ? 1 : 0);
            _cs.SetInt("_MatUseC", 0);
            _cs.SetInt("_MatBroadcastTypeC", 0);
            _cs.SetFloat("_MatAlpha", 1f);
            _cs.SetFloat("_MatBeta", 0f);
            _cs.SetBuffer(_kMatMul2D, "_MatA", a);
            _cs.SetBuffer(_kMatMul2D, "_MatB", b);
            _cs.SetBuffer(_kMatMul2D, "_MatC", a);
            _cs.SetBuffer(_kMatMul2D, "_MatOut", output);

            Dispatch2D(_cs, _kMatMul2D, n, m, 8, 8);
        }

        public void MatMulPack4Cdhw(
            RenderTexture a,
            int aRows,
            int aCols,
            int aBatchD,
            int aBatchC,
            RenderTexture b,
            int bRows,
            int bCols,
            int bBatchD,
            int bBatchC,
            bool transB,
            int outBatchD,
            int outBatchC,
            RenderTexture output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (aRows <= 0) throw new ArgumentOutOfRangeException(nameof(aRows));
            if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
            if (bRows <= 0) throw new ArgumentOutOfRangeException(nameof(bRows));
            if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));

            var n = transB ? bRows : bCols;
            _cs.SetInt("_MatM", aRows);
            _cs.SetInt("_MatN", n);
            _cs.SetInt("_MatK", aCols);
            _cs.SetInt("_MatTransB", transB ? 1 : 0);
            _cs.SetInt("_MatPack4ABatchD", aBatchD);
            _cs.SetInt("_MatPack4ABatchC", aBatchC);
            _cs.SetInt("_MatPack4BBatchD", bBatchD);
            _cs.SetInt("_MatPack4BBatchC", bBatchC);
            _cs.SetInt("_MatPack4OutBatchD", outBatchD);
            _cs.SetInt("_MatPack4OutBatchC", outBatchC);
            _cs.SetTexture(_kMatMulPack4Cdhw, "_MatPack4AInArr", a);
            _cs.SetTexture(_kMatMulPack4Cdhw, "_MatPack4BInArr", b);
            _cs.SetTexture(_kMatMulPack4Cdhw, "_MatPack4OutArr", output);
            Dispatch3D(_kMatMulPack4Cdhw, output.width, output.height, ResolveRenderTextureDispatchDepth(output, Mathf.Max(1, outBatchD * Mathf.CeilToInt(outBatchC / 4f))), 8, 8);
        }

        public void MatMulPack4Cdhw(
            CommandBuffer cmd,
            ComputeTexture a,
            int aRows,
            int aCols,
            int aBatchD,
            int aBatchC,
            ComputeTexture b,
            int bRows,
            int bCols,
            int bBatchD,
            int bBatchC,
            bool transB,
            int outBatchD,
            int outBatchC,
            ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (aRows <= 0) throw new ArgumentOutOfRangeException(nameof(aRows));
            if (aCols <= 0) throw new ArgumentOutOfRangeException(nameof(aCols));
            if (bRows <= 0) throw new ArgumentOutOfRangeException(nameof(bRows));
            if (bCols <= 0) throw new ArgumentOutOfRangeException(nameof(bCols));

            var n = transB ? bRows : bCols;
            cmd.SetComputeIntParam(_cs, "_MatM", aRows);
            cmd.SetComputeIntParam(_cs, "_MatN", n);
            cmd.SetComputeIntParam(_cs, "_MatK", aCols);
            cmd.SetComputeIntParam(_cs, "_MatTransB", transB ? 1 : 0);
            cmd.SetComputeIntParam(_cs, "_MatPack4ABatchD", aBatchD);
            cmd.SetComputeIntParam(_cs, "_MatPack4ABatchC", aBatchC);
            cmd.SetComputeIntParam(_cs, "_MatPack4BBatchD", bBatchD);
            cmd.SetComputeIntParam(_cs, "_MatPack4BBatchC", bBatchC);
            cmd.SetComputeIntParam(_cs, "_MatPack4OutBatchD", outBatchD);
            cmd.SetComputeIntParam(_cs, "_MatPack4OutBatchC", outBatchC);
            cmd.SetComputeTextureParam(_cs, _kMatMulPack4Cdhw, "_MatPack4AInArr", a.nameID);
            cmd.SetComputeTextureParam(_cs, _kMatMulPack4Cdhw, "_MatPack4BInArr", b.nameID);
            cmd.SetComputeTextureParam(_cs, _kMatMulPack4Cdhw, "_MatPack4OutArr", output.nameID);
            Dispatch3D(cmd, _kMatMulPack4Cdhw, output.width, output.height, ResolveComputeTextureDispatchDepth(output, Mathf.Max(1, outBatchD * Mathf.CeilToInt(outBatchC / 4f))), 8, 8);
        }

        public void VistaTailPromptDotPack4(RenderTexture featureTex, int width, int height, int depth, int packs, ComputeBuffer prompt, RenderTexture output)
        {
            if (featureTex == null) throw new ArgumentNullException(nameof(featureTex));
            if (prompt == null) throw new ArgumentNullException(nameof(prompt));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (width <= 0 || height <= 0 || depth <= 0) throw new ArgumentOutOfRangeException(nameof(width));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (prompt.count < packs * 4) throw new ArgumentOutOfRangeException(nameof(prompt), "prompt vector is smaller than required packed channel count");

            _cs.SetInt("_VistaTailW", width);
            _cs.SetInt("_VistaTailH", height);
            _cs.SetInt("_VistaTailD", depth);
            _cs.SetInt("_VistaTailPacks", packs);
            _cs.SetTexture(_kVistaTailPromptDotPack4, "_VistaTailInArr", featureTex);
            _cs.SetBuffer(_kVistaTailPromptDotPack4, "_VistaTailPrompt", prompt);
            _cs.SetTexture(_kVistaTailPromptDotPack4, "_VistaTailOutArr", output);
            Dispatch3D(_kVistaTailPromptDotPack4, output.width, output.height, ResolveRenderTextureDispatchDepth(output, depth), 8, 8);
        }

        public void Gemm2D(ComputeBuffer a, ComputeBuffer b, ComputeBuffer c, int m, int n, int k, bool transB, float alpha, float beta, bool useC, int broadcastTypeC, ComputeBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (k > 8192) throw new ArgumentOutOfRangeException(nameof(k), "k too large for current tiled kernel (MATK_MAX=8192): " + k);
            if (useC && c == null) throw new ArgumentNullException(nameof(c));

            _cs.SetInt("_MatM", m);
            _cs.SetInt("_MatN", n);
            _cs.SetInt("_MatK", k);
            _cs.SetInt("_MatTransB", transB ? 1 : 0);
            _cs.SetInt("_MatUseC", useC ? 1 : 0);
            _cs.SetInt("_MatBroadcastTypeC", broadcastTypeC);
            _cs.SetFloat("_MatAlpha", alpha);
            _cs.SetFloat("_MatBeta", beta);
            var useLargeTile = m >= 16 && n >= 128 && k >= 128;
            var kernel = useLargeTile ? _kGemm2D16 : _kGemm2D;
            _cs.SetBuffer(kernel, "_MatA", a);
            _cs.SetBuffer(kernel, "_MatB", b);
            _cs.SetBuffer(kernel, "_MatC", useC ? c : a);
            _cs.SetBuffer(kernel, "_MatOut", output);

            Dispatch2D(_cs, kernel, n, m, useLargeTile ? 16 : 8, useLargeTile ? 16 : 8);
        }

        public void LayerNorm2DInplace(ComputeBuffer inOut, int rows, int cols, float eps, bool affine, ComputeBuffer gamma, ComputeBuffer beta)
        {
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));
            if (affine && (gamma == null || beta == null)) throw new ArgumentNullException(nameof(gamma));

            _cs.SetInt("_LnRows", rows);
            _cs.SetInt("_LnCols", cols);
            _cs.SetInt("_LnAffine", affine ? 1 : 0);
            _cs.SetFloat("_LnEps", eps);
            _cs.SetBuffer(_kLayerNorm2D, "_LnInOut", inOut);
            _cs.SetBuffer(_kLayerNorm2D, "_LnGamma", affine ? gamma : inOut);
            _cs.SetBuffer(_kLayerNorm2D, "_LnBeta", affine ? beta : inOut);
            _cs.Dispatch(_kLayerNorm2D, Mathf.Max(1, rows), 1, 1);
        }

        public void Softmax2D(ComputeBuffer input, ComputeBuffer output, int rows, int cols)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (rows <= 0) throw new ArgumentOutOfRangeException(nameof(rows));
            if (cols <= 0) throw new ArgumentOutOfRangeException(nameof(cols));

            _cs.SetInt("_SoftRows", rows);
            _cs.SetInt("_SoftCols", cols);
            _cs.SetBuffer(_kSoftmax2D, "_SoftIn", input);
            _cs.SetBuffer(_kSoftmax2D, "_SoftOut", output);
            _cs.Dispatch(_kSoftmax2D, Mathf.Max(1, rows), 1, 1);
        }

        public void Embed(ComputeBuffer indices, int words, ComputeBuffer weight, ComputeBuffer bias, int numOutput, int inputDim, bool biasTerm, ComputeBuffer output)
        {
            if (indices == null) throw new ArgumentNullException(nameof(indices));
            if (weight == null) throw new ArgumentNullException(nameof(weight));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (words <= 0) throw new ArgumentOutOfRangeException(nameof(words));
            if (numOutput <= 0) throw new ArgumentOutOfRangeException(nameof(numOutput));
            if (inputDim <= 0) throw new ArgumentOutOfRangeException(nameof(inputDim));
            if (biasTerm && bias == null) throw new ArgumentNullException(nameof(bias));

            _cs.SetInt("_EmbedWords", words);
            _cs.SetInt("_EmbedNumOutput", numOutput);
            _cs.SetInt("_EmbedInputDim", inputDim);
            _cs.SetInt("_EmbedBiasTerm", biasTerm ? 1 : 0);
            _cs.SetBuffer(_kEmbed, "_EmbedIdx", indices);
            _cs.SetBuffer(_kEmbed, "_EmbedW", weight);
            _cs.SetBuffer(_kEmbed, "_EmbedB", biasTerm ? bias : weight);
            _cs.SetBuffer(_kEmbed, "_EmbedOut", output);
            Dispatch2D(_cs, _kEmbed, numOutput, words, 8, 8);
        }

        public void Permute(ComputeBuffer input, int dims, int inW, int inH, int inD, int inC, int orderType, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (dims < 2 || dims > 4) throw new ArgumentOutOfRangeException(nameof(dims));
            if (inW <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (inH <= 0) throw new ArgumentOutOfRangeException(nameof(inH));
            if (dims >= 3 && inC <= 0) throw new ArgumentOutOfRangeException(nameof(inC));
            if (dims == 4 && inD <= 0) throw new ArgumentOutOfRangeException(nameof(inD));
            if (dims != 4) inD = 1;
            if (dims == 2) inC = 1;

            var axes = GetPermuteAxes(dims, orderType);
            var outW = GetAxisSize(dims, inW, inH, inD, inC, axes.x);
            var outH = GetAxisSize(dims, inW, inH, inD, inC, axes.y);
            var outD = dims == 4 ? GetAxisSize(dims, inW, inH, inD, inC, axes.z) : 1;
            var outC = dims == 2 ? 1 : GetAxisSize(dims, inW, inH, inD, inC, dims == 4 ? axes.w : axes.z);

            var outCount = checked(outW * outH * outD * outC);
            if (output.count != outCount)
                throw new ArgumentOutOfRangeException(nameof(output), "output.count mismatch, expect " + outCount);

            _cs.SetInt("_PermuteDims", dims);
            _cs.SetInt("_PermuteInW", inW);
            _cs.SetInt("_PermuteInH", inH);
            _cs.SetInt("_PermuteInD", inD);
            _cs.SetInt("_PermuteInC", inC);
            _cs.SetInt("_PermuteOutW", outW);
            _cs.SetInt("_PermuteOutH", outH);
            _cs.SetInt("_PermuteOutD", outD);
            _cs.SetInt("_PermuteOutC", outC);
            _cs.SetInts("_PermuteAxes", axes.x, axes.y, axes.z, axes.w);
            _cs.SetBuffer(_kPermute, "_PermuteIn", input);
            _cs.SetBuffer(_kPermute, "_PermuteOut", output);
            Dispatch1D(_kPermute, outCount, 256);
        }

        public void Slice(ComputeBuffer input, int dims, int inW, int inH, int inD, int inC, int axis, int begin, int outW, int outH, int outD, int outC, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (dims < 2 || dims > 4) throw new ArgumentOutOfRangeException(nameof(dims));
            if (axis < 0) axis += dims;
            if (axis < 0 || axis >= dims) throw new ArgumentOutOfRangeException(nameof(axis));
            if (begin < 0) throw new ArgumentOutOfRangeException(nameof(begin));
            if (inW <= 0 || inH <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (dims == 3 && inC <= 0) throw new ArgumentOutOfRangeException(nameof(inC));
            if (dims == 4 && (inD <= 0 || inC <= 0)) throw new ArgumentOutOfRangeException(nameof(inD));
            if (dims != 4) inD = 1;
            if (dims == 2) inC = 1;

            if (outW <= 0 || outH <= 0) throw new ArgumentOutOfRangeException(nameof(outW));
            if (dims == 3 && outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (dims == 4 && (outD <= 0 || outC <= 0)) throw new ArgumentOutOfRangeException(nameof(outD));
            if (dims != 4) outD = 1;
            if (dims == 2) outC = 1;

            var outCount = checked(outW * outH * outD * outC);
            if (output.count != outCount)
                throw new ArgumentOutOfRangeException(nameof(output), "output.count mismatch, expect " + outCount);

            _cs.SetInt("_SliceDims", dims);
            _cs.SetInt("_SliceInW", inW);
            _cs.SetInt("_SliceInH", inH);
            _cs.SetInt("_SliceInD", inD);
            _cs.SetInt("_SliceInC", inC);
            _cs.SetInt("_SliceAxis", axis);
            _cs.SetInt("_SliceBegin", begin);
            _cs.SetInt("_SliceOutW", outW);
            _cs.SetInt("_SliceOutH", outH);
            _cs.SetInt("_SliceOutD", outD);
            _cs.SetInt("_SliceOutC", outC);
            _cs.SetBuffer(_kSlice, "_SliceIn", input);
            _cs.SetBuffer(_kSlice, "_SliceOut", output);
            Dispatch1D(_kSlice, outCount, 256);
        }

        public void Tile(ComputeBuffer input, int dims, int inW, int inH, int inD, int inC, int axis, int tiles, int outW, int outH, int outD, int outC, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (dims < 1 || dims > 4) throw new ArgumentOutOfRangeException(nameof(dims));
            if (axis < 0) axis += dims;
            if (axis < 0 || axis >= dims) throw new ArgumentOutOfRangeException(nameof(axis));
            if (tiles <= 0) throw new ArgumentOutOfRangeException(nameof(tiles));

            if (inW <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (dims >= 2 && inH <= 0) throw new ArgumentOutOfRangeException(nameof(inH));
            if (dims == 3 && inC <= 0) throw new ArgumentOutOfRangeException(nameof(inC));
            if (dims == 4 && (inD <= 0 || inC <= 0)) throw new ArgumentOutOfRangeException(nameof(inD));
            if (dims < 2) inH = 1;
            if (dims < 3) inC = 1;
            if (dims < 4) inD = 1;

            if (outW <= 0) throw new ArgumentOutOfRangeException(nameof(outW));
            if (dims >= 2 && outH <= 0) throw new ArgumentOutOfRangeException(nameof(outH));
            if (dims == 3 && outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (dims == 4 && (outD <= 0 || outC <= 0)) throw new ArgumentOutOfRangeException(nameof(outD));
            if (dims < 2) outH = 1;
            if (dims < 3) outC = 1;
            if (dims < 4) outD = 1;

            var outCount = checked(outW * outH * outD * outC);
            if (output.count != outCount)
                throw new ArgumentOutOfRangeException(nameof(output), "output.count mismatch, expect " + outCount);

            _cs.SetInt("_TileDims", dims);
            _cs.SetInt("_TileInW", inW);
            _cs.SetInt("_TileInH", inH);
            _cs.SetInt("_TileInD", inD);
            _cs.SetInt("_TileInC", inC);
            _cs.SetInt("_TileAxis", axis);
            _cs.SetInt("_TileTiles", tiles);
            _cs.SetInt("_TileOutW", outW);
            _cs.SetInt("_TileOutH", outH);
            _cs.SetInt("_TileOutD", outD);
            _cs.SetInt("_TileOutC", outC);
            _cs.SetBuffer(_kTile, "_TileIn", input);
            _cs.SetBuffer(_kTile, "_TileOut", output);
            Dispatch1D(_kTile, outCount, 256);
        }

        public void ReduceAllSumOrMean(ComputeBuffer input, int total, bool mean, ComputeBuffer outputScalar)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (outputScalar == null) throw new ArgumentNullException(nameof(outputScalar));
            if (total <= 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (outputScalar.count != 1) throw new ArgumentOutOfRangeException(nameof(outputScalar), "outputScalar.count must be 1");

            var n = total;
            var groups = Mathf.CeilToInt(n / 256f);
            var cur = new ComputeBuffer(groups, sizeof(float), ComputeBufferType.Structured);
            try
            {
                _cs.SetInt("_ReduceTotal", n);
                _cs.SetBuffer(_kReduceSum256, "_ReduceIn", input);
                _cs.SetBuffer(_kReduceSum256, "_ReduceOut", cur);
                _cs.Dispatch(_kReduceSum256, Mathf.Max(1, groups), 1, 1);

                while (groups > 1)
                {
                    n = groups;
                    groups = Mathf.CeilToInt(n / 256f);
                    var next = new ComputeBuffer(groups, sizeof(float), ComputeBufferType.Structured);
                    _cs.SetInt("_ReduceTotal", n);
                    _cs.SetBuffer(_kReduceSum256, "_ReduceIn", cur);
                    _cs.SetBuffer(_kReduceSum256, "_ReduceOut", next);
                    _cs.Dispatch(_kReduceSum256, Mathf.Max(1, groups), 1, 1);
                    cur.Dispose();
                    cur = next;
                }

                _cs.SetInt("_MulTotal", 1);
                _cs.SetFloat("_MulK", 1f);
                _cs.SetBuffer(_kMulScalarBuf, "_MulIn", cur);
                _cs.SetBuffer(_kMulScalarBuf, "_MulOut", outputScalar);
                _cs.Dispatch(_kMulScalarBuf, 1, 1, 1);

                if (mean)
                    MulScalarInplace(outputScalar, 1f / total, 1);
            }
            finally
            {
                try { cur?.Dispose(); } catch { }
            }
        }

        public void MulScalarInplace(ComputeBuffer inOut, float k, int total)
        {
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (total <= 0) throw new ArgumentOutOfRangeException(nameof(total));
            _cs.SetInt("_MulTotal", total);
            _cs.SetFloat("_MulK", k);
            _cs.SetBuffer(_kMulScalarBuf, "_MulIn", inOut);
            _cs.SetBuffer(_kMulScalarBuf, "_MulOut", inOut);
            Dispatch1D(_kMulScalarBuf, total, 256);
        }

        public void GroupNormInplace(ComputeBuffer inOut, int w, int h, int c, int group, float eps, bool affine, ComputeBuffer gamma, ComputeBuffer beta)
        {
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            using var stats = new ComputeBuffer(group, sizeof(float) * 4, ComputeBufferType.Structured);
            GroupNormInplace(inOut, w, h, c, group, eps, affine, gamma, beta, stats, false);
        }

        public void GroupNormInplace(ComputeBuffer inOut, int w, int h, int c, int group, float eps, bool affine, ComputeBuffer gamma, ComputeBuffer beta, bool ncnnStyleVariance)
        {
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            using var stats = new ComputeBuffer(group, sizeof(float) * 4, ComputeBufferType.Structured);
            GroupNormInplace(inOut, w, h, c, group, eps, affine, gamma, beta, stats, ncnnStyleVariance);
        }

        public void GroupNormInplace(ComputeBuffer inOut, int w, int h, int c, int group, float eps, bool affine, ComputeBuffer gamma, ComputeBuffer beta, ComputeBuffer stats, bool ncnnStyleVariance)
        {
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (c % group != 0) throw new ArgumentOutOfRangeException(nameof(group), "c must be divisible by group");
            if (affine && (gamma == null || beta == null)) throw new ArgumentNullException(nameof(gamma));
            if (stats == null) throw new ArgumentNullException(nameof(stats));

            var channelsG = c / group;

            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", affine ? 1 : 0);
            if (ncnnStyleVariance)
            {
                _cs.SetBuffer(_kGroupNormMean, "_GnInOut", inOut);
                _cs.SetBuffer(_kGroupNormMean, "_GnGamma", affine ? gamma : inOut);
                _cs.SetBuffer(_kGroupNormMean, "_GnBeta", affine ? beta : inOut);
                _cs.SetBuffer(_kGroupNormMean, "_GnStatsOut", stats);
                _cs.Dispatch(_kGroupNormMean, Mathf.Max(1, group), 1, 1);

                _cs.SetBuffer(_kGroupNormVariance, "_GnInOut", inOut);
                _cs.SetBuffer(_kGroupNormVariance, "_GnGamma", affine ? gamma : inOut);
                _cs.SetBuffer(_kGroupNormVariance, "_GnBeta", affine ? beta : inOut);
                _cs.SetBuffer(_kGroupNormVariance, "_GnStatsOut", stats);
                _cs.Dispatch(_kGroupNormVariance, Mathf.Max(1, group), 1, 1);
            }
            else
            {
                _cs.SetBuffer(_kGroupNormStats, "_GnInOut", inOut);
                _cs.SetBuffer(_kGroupNormStats, "_GnGamma", affine ? gamma : inOut);
                _cs.SetBuffer(_kGroupNormStats, "_GnBeta", affine ? beta : inOut);
                _cs.SetBuffer(_kGroupNormStats, "_GnStatsOut", stats);
                _cs.Dispatch(_kGroupNormStats, Mathf.Max(1, group), 1, 1);
            }

            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", affine ? 1 : 0);
            var applyKernel = ncnnStyleVariance ? _kGroupNormApplyMeanVar : _kGroupNormApply;
            _cs.SetBuffer(applyKernel, "_GnInOut", inOut);
            _cs.SetBuffer(applyKernel, "_GnGamma", affine ? gamma : inOut);
            _cs.SetBuffer(applyKernel, "_GnBeta", affine ? beta : inOut);
            _cs.SetBuffer(applyKernel, "_GnStatsOut", stats);
            _cs.Dispatch(applyKernel, Mathf.Max(1, group), 1, 1);
        }

        public void GroupNormPack4(RenderTexture input, int w, int h, int c, int packs, int group, float eps, ComputeBuffer gamma, ComputeBuffer beta, ComputeBuffer stats, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));
            if (stats == null) throw new ArgumentNullException(nameof(stats));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (c % group != 0) throw new ArgumentOutOfRangeException(nameof(group), "c must be divisible by group");

            var channelsG = c / group;
            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", 1);

            _cs.SetTexture(_kGroupNormPack4Mean, "_GnTexInArr", input);
            _cs.SetBuffer(_kGroupNormPack4Mean, "_GnStatsOut", stats);
            _cs.Dispatch(_kGroupNormPack4Mean, Mathf.Max(1, group), 1, 1);

            _cs.SetTexture(_kGroupNormPack4Variance, "_GnTexInArr", input);
            _cs.SetBuffer(_kGroupNormPack4Variance, "_GnStatsOut", stats);
            _cs.Dispatch(_kGroupNormPack4Variance, Mathf.Max(1, group), 1, 1);

            _cs.SetTexture(_kGroupNormPack4ApplyMeanVar, "_GnTexInArr", input);
            _cs.SetTexture(_kGroupNormPack4ApplyMeanVar, "_GnTexOutArr", output);
            _cs.SetBuffer(_kGroupNormPack4ApplyMeanVar, "_GnGamma", gamma);
            _cs.SetBuffer(_kGroupNormPack4ApplyMeanVar, "_GnBeta", beta);
            _cs.SetBuffer(_kGroupNormPack4ApplyMeanVar, "_GnStatsOut", stats);
            Dispatch3D(_kGroupNormPack4ApplyMeanVar, output.width, output.height, packs, 8, 8);
        }

        public void GroupNormPack4Tex(RenderTexture input, int w, int h, int d, int c, int packs, int group, float eps, ComputeBuffer gamma, ComputeBuffer beta, RenderTexture statsA, RenderTexture statsB, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (statsA == null) throw new ArgumentNullException(nameof(statsA));
            if (statsB == null) throw new ArgumentNullException(nameof(statsB));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (c % group != 0) throw new ArgumentOutOfRangeException(nameof(group), "c must be divisible by group");

            var channelsG = c / group;
            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnD", d);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", 1);

            _cs.SetTexture(_kGroupNormPack4MeanTex, "_GnTexInArr", input);
            _cs.SetTexture(_kGroupNormPack4MeanTex, "_GnStatsTexOut", statsA);
            _cs.Dispatch(_kGroupNormPack4MeanTex, Mathf.Max(1, group), 1, 1);

            _cs.SetTexture(_kGroupNormPack4VarianceTex, "_GnTexInArr", input);
            _cs.SetTexture(_kGroupNormPack4VarianceTex, "_GnStatsTexIn", statsA);
            _cs.SetTexture(_kGroupNormPack4VarianceTex, "_GnStatsTexOut", statsB);
            _cs.Dispatch(_kGroupNormPack4VarianceTex, Mathf.Max(1, group), 1, 1);

            _cs.SetTexture(_kGroupNormPack4ApplyMeanVarTex, "_GnTexInArr", input);
            _cs.SetTexture(_kGroupNormPack4ApplyMeanVarTex, "_GnTexOutArr", output);
            _cs.SetTexture(_kGroupNormPack4ApplyMeanVarTex, "_GnStatsTexIn", statsB);
            _cs.SetBuffer(_kGroupNormPack4ApplyMeanVarTex, "_GnGamma", gamma);
            _cs.SetBuffer(_kGroupNormPack4ApplyMeanVarTex, "_GnBeta", beta);
            Dispatch3D(_kGroupNormPack4ApplyMeanVarTex, output.width, output.height, ResolveRenderTextureDispatchDepth(output, packs), 8, 8);
        }

        public void GroupNormPack4Tex(CommandBuffer cmd, ComputeTexture input, int w, int h, int d, int c, int packs, int group, float eps, ComputeBuffer gamma, ComputeBuffer beta, ComputeTexture statsA, ComputeTexture statsB, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (statsA == null) throw new ArgumentNullException(nameof(statsA));
            if (statsB == null) throw new ArgumentNullException(nameof(statsB));
            if (gamma == null) throw new ArgumentNullException(nameof(gamma));
            if (beta == null) throw new ArgumentNullException(nameof(beta));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (packs <= 0) throw new ArgumentOutOfRangeException(nameof(packs));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (c % group != 0) throw new ArgumentOutOfRangeException(nameof(group), "c must be divisible by group");

            var channelsG = c / group;
            cmd.SetComputeIntParam(_cs, "_GnW", w);
            cmd.SetComputeIntParam(_cs, "_GnH", h);
            cmd.SetComputeIntParam(_cs, "_GnD", d);
            cmd.SetComputeIntParam(_cs, "_GnC", c);
            cmd.SetComputeIntParam(_cs, "_GnGroup", group);
            cmd.SetComputeIntParam(_cs, "_GnChannelsG", channelsG);
            cmd.SetComputeFloatParam(_cs, "_GnEps", eps);
            cmd.SetComputeIntParam(_cs, "_GnAffine", 1);

            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4MeanTex, "_GnTexInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4MeanTex, "_GnStatsTexOut", statsA.nameID);
            cmd.DispatchCompute(_cs, _kGroupNormPack4MeanTex, Mathf.Max(1, group), 1, 1);

            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4VarianceTex, "_GnTexInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4VarianceTex, "_GnStatsTexIn", statsA.nameID);
            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4VarianceTex, "_GnStatsTexOut", statsB.nameID);
            cmd.DispatchCompute(_cs, _kGroupNormPack4VarianceTex, Mathf.Max(1, group), 1, 1);

            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4ApplyMeanVarTex, "_GnTexInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4ApplyMeanVarTex, "_GnTexOutArr", output.nameID);
            cmd.SetComputeTextureParam(_cs, _kGroupNormPack4ApplyMeanVarTex, "_GnStatsTexIn", statsB.nameID);
            cmd.SetComputeBufferParam(_cs, _kGroupNormPack4ApplyMeanVarTex, "_GnGamma", gamma);
            cmd.SetComputeBufferParam(_cs, _kGroupNormPack4ApplyMeanVarTex, "_GnBeta", beta);
            Dispatch3D(cmd, _kGroupNormPack4ApplyMeanVarTex, output.width, output.height, ResolveComputeTextureDispatchDepth(output, packs), 8, 8);
        }

        private static Vector4Int GetPermuteAxes(int dims, int orderType)
        {
            if (dims == 2)
            {
                if (orderType == 0) return new Vector4Int(0, 1, 0, 0);
                if (orderType == 1) return new Vector4Int(1, 0, 0, 0);
                throw new ArgumentOutOfRangeException(nameof(orderType), "unsupported orderType for dims=2: " + orderType);
            }

            if (dims == 3)
            {
                if (orderType == 0) return new Vector4Int(0, 1, 2, 0);
                if (orderType == 1) return new Vector4Int(1, 0, 2, 0);
                if (orderType == 2) return new Vector4Int(0, 2, 1, 0);
                if (orderType == 3) return new Vector4Int(2, 0, 1, 0);
                if (orderType == 4) return new Vector4Int(1, 2, 0, 0);
                if (orderType == 5) return new Vector4Int(2, 1, 0, 0);
                throw new ArgumentOutOfRangeException(nameof(orderType), "unsupported orderType for dims=3: " + orderType);
            }

            switch (orderType)
            {
                case 0: return new Vector4Int(0, 1, 2, 3);
                case 1: return new Vector4Int(1, 0, 2, 3);
                case 2: return new Vector4Int(0, 2, 1, 3);
                case 3: return new Vector4Int(2, 0, 1, 3);
                case 4: return new Vector4Int(1, 2, 0, 3);
                case 5: return new Vector4Int(2, 1, 0, 3);
                case 6: return new Vector4Int(0, 1, 3, 2);
                case 7: return new Vector4Int(1, 0, 3, 2);
                case 8: return new Vector4Int(0, 3, 1, 2);
                case 9: return new Vector4Int(3, 0, 1, 2);
                case 10: return new Vector4Int(1, 3, 0, 2);
                case 11: return new Vector4Int(3, 1, 0, 2);
                case 12: return new Vector4Int(0, 2, 3, 1);
                case 13: return new Vector4Int(2, 0, 3, 1);
                case 14: return new Vector4Int(0, 3, 2, 1);
                case 15: return new Vector4Int(3, 0, 2, 1);
                case 16: return new Vector4Int(2, 3, 0, 1);
                case 17: return new Vector4Int(3, 2, 0, 1);
                case 18: return new Vector4Int(1, 2, 3, 0);
                case 19: return new Vector4Int(2, 1, 3, 0);
                case 20: return new Vector4Int(1, 3, 2, 0);
                case 21: return new Vector4Int(3, 1, 2, 0);
                case 22: return new Vector4Int(2, 3, 1, 0);
                case 23: return new Vector4Int(3, 2, 1, 0);
                default: throw new ArgumentOutOfRangeException(nameof(orderType), "unsupported orderType for dims=4: " + orderType);
            }
        }

        private static int GetAxisSize(int dims, int w, int h, int d, int c, int axis)
        {
            if (axis == 0) return w;
            if (axis == 1) return h;
            if (axis == 2) return dims == 4 ? d : c;
            if (axis == 3) return c;
            throw new ArgumentOutOfRangeException(nameof(axis));
        }

        public void Conv3x3(NcnnTensorBuffer input, ComputeBuffer weightsOihw, ComputeBuffer biasO, int outC, int stride, int pad, int activationType, float activationParam, NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOihw == null) throw new ArgumentNullException(nameof(weightsOihw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_OutW", output.w);
            _cs.SetInt("_OutH", output.h);
            _cs.SetInt("_Stride", Mathf.Max(1, stride));
            _cs.SetInt("_Pad", Mathf.Max(0, pad));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3x3, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kConv3x3, "_ConvW", weightsOihw);
            _cs.SetBuffer(_kConv3x3, "_ConvB", biasO);
            _cs.SetBuffer(_kConv3x3, "_ConvOut", output.buffer);

            Dispatch3D(_kConv3x3, output.w, output.h, outC, 8, 8);
        }

        public void ConvDepthWise(
            NcnnTensorBuffer input,
            ComputeBuffer weightsOihw,
            ComputeBuffer biasO,
            int outC,
            int group,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padTop,
            int dilationW,
            int dilationH,
            int activationType,
            float activationParam,
            NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOihw == null) throw new ArgumentNullException(nameof(weightsOihw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (kernelW <= 0 || kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_OutW", output.w);
            _cs.SetInt("_OutH", output.h);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_ConvGroup", group);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConvDepthWise, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kConvDepthWise, "_ConvW", weightsOihw);
            _cs.SetBuffer(_kConvDepthWise, "_ConvB", biasO);
            _cs.SetBuffer(_kConvDepthWise, "_ConvOut", output.buffer);
            Dispatch3D(_kConvDepthWise, output.w, output.h, outC, 8, 8);
        }

        public void Conv3dBuf(
            NcnnTensorBuffer input,
            ComputeBuffer weightsOidhw,
            ComputeBuffer biasO,
            int outC,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int dilationW,
            int dilationH,
            int dilationD,
            int activationType,
            float activationParam,
            NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOidhw == null) throw new ArgumentNullException(nameof(weightsOidhw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (input.dims != 4) throw new ArgumentOutOfRangeException(nameof(input), "Conv3dBuf expects dims=4 input");
            if (output.dims != 4) throw new ArgumentOutOfRangeException(nameof(output), "Conv3dBuf expects dims=4 output");
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (kernelW <= 0 || kernelH <= 0 || kernelD <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InD", input.d);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_OutW", output.w);
            _cs.SetInt("_OutH", output.h);
            _cs.SetInt("_OutD", output.d);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_KernelDVar", kernelD);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_StrideDVar", Mathf.Max(1, strideD));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadRightVar", Mathf.Max(0, padRight));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_PadBottomVar", Mathf.Max(0, padBottom));
            _cs.SetInt("_PadFrontVar", Mathf.Max(0, padFront));
            _cs.SetInt("_PadBehindVar", Mathf.Max(0, padBehind));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_DilationDVar", Mathf.Max(1, dilationD));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3dBuf, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kConv3dBuf, "_ConvW", weightsOidhw);
            _cs.SetBuffer(_kConv3dBuf, "_ConvB", biasO);
            _cs.SetBuffer(_kConv3dBuf, "_ConvOut", output.buffer);
            Dispatch3D(_kConv3dBuf, output.w, output.h, output.d * outC, 8, 8);
        }

        public void Conv3dPack4CDHW(
            RenderTexture input,
            int inW,
            int inH,
            int inD,
            int inPacks,
            ComputeBuffer weightsO4I4K3,
            ComputeBuffer biasO4,
            int outW,
            int outH,
            int outD,
            int outPacks,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int dilationW,
            int dilationH,
            int dilationD,
            int activationType,
            float activationParam,
            RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsO4I4K3 == null) throw new ArgumentNullException(nameof(weightsO4I4K3));
            if (biasO4 == null) throw new ArgumentNullException(nameof(biasO4));
            if (inW <= 0 || inH <= 0 || inD <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (outW <= 0 || outH <= 0 || outD <= 0) throw new ArgumentOutOfRangeException(nameof(outW));
            if (inPacks <= 0 || outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));

            _cs.SetInt("_InW", inW);
            _cs.SetInt("_InH", inH);
            _cs.SetInt("_InD", inD);
            _cs.SetInt("_OutW", outW);
            _cs.SetInt("_OutH", outH);
            _cs.SetInt("_OutD", outD);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_KernelDVar", kernelD);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_StrideDVar", Mathf.Max(1, strideD));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadRightVar", Mathf.Max(0, padRight));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_PadBottomVar", Mathf.Max(0, padBottom));
            _cs.SetInt("_PadFrontVar", Mathf.Max(0, padFront));
            _cs.SetInt("_PadBehindVar", Mathf.Max(0, padBehind));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_DilationDVar", Mathf.Max(1, dilationD));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kConv3dPack4Cdhw, "_ConvW4", weightsO4I4K3);
            _cs.SetBuffer(_kConv3dPack4Cdhw, "_ConvB4", biasO4);
            _cs.SetTexture(_kConv3dPack4Cdhw, "_ConvInArr", input);
            _cs.SetTexture(_kConv3dPack4Cdhw, "_ConvOutArr", output);
            Dispatch3D(_kConv3dPack4Cdhw, outW, outH, ResolveRenderTextureDispatchDepth(output, outD * outPacks), 8, 8);
        }

        public void Deconvolution(
            NcnnTensorBuffer input,
            ComputeBuffer weightsOihw,
            ComputeBuffer biasO,
            int outC,
            int group,
            int kernelW,
            int kernelH,
            int strideW,
            int strideH,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int outputPadRight,
            int outputPadBottom,
            int dilationW,
            int dilationH,
            int activationType,
            float activationParam,
            NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOihw == null) throw new ArgumentNullException(nameof(weightsOihw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (kernelW <= 0 || kernelH <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            var kernelExtentW = Mathf.Max(1, dilationW) * (kernelW - 1) + 1;
            var kernelExtentH = Mathf.Max(1, dilationH) * (kernelH - 1) + 1;
            var borderedOutW = (input.w - 1) * Mathf.Max(1, strideW) + kernelExtentW + Mathf.Max(0, outputPadRight);
            var borderedOutH = (input.h - 1) * Mathf.Max(1, strideH) + kernelExtentH + Mathf.Max(0, outputPadBottom);
            var expectedOutW = borderedOutW - Mathf.Max(0, padLeft) - Mathf.Max(0, padRight);
            var expectedOutH = borderedOutH - Mathf.Max(0, padTop) - Mathf.Max(0, padBottom);
            if (output.w != expectedOutW || output.h != expectedOutH)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(output),
                    "deconvolution output shape mismatch expected "
                    + expectedOutW.ToString(CultureInfo.InvariantCulture) + "x"
                    + expectedOutH.ToString(CultureInfo.InvariantCulture) + " but got "
                    + output.w.ToString(CultureInfo.InvariantCulture) + "x"
                    + output.h.ToString(CultureInfo.InvariantCulture));
            }

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_OutW", output.w);
            _cs.SetInt("_OutH", output.h);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_ConvGroup", group);
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kDeconvolutionBuf, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kDeconvolutionBuf, "_ConvW", weightsOihw);
            _cs.SetBuffer(_kDeconvolutionBuf, "_ConvB", biasO);
            _cs.SetBuffer(_kDeconvolutionBuf, "_ConvOut", output.buffer);
            Dispatch3D(_kDeconvolutionBuf, output.w, output.h, outC, 8, 8);
        }

        public void Deconvolution3D(
            NcnnTensorBuffer input,
            ComputeBuffer weightsOidhw,
            ComputeBuffer biasO,
            int outC,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int outputPadRight,
            int outputPadBottom,
            int outputPadBehind,
            int dilationW,
            int dilationH,
            int dilationD,
            int activationType,
            float activationParam,
            NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsOidhw == null) throw new ArgumentNullException(nameof(weightsOidhw));
            if (biasO == null) throw new ArgumentNullException(nameof(biasO));
            if (input.dims != 4) throw new ArgumentOutOfRangeException(nameof(input), "Deconvolution3D expects dims=4 input");
            if (output.dims != 4) throw new ArgumentOutOfRangeException(nameof(output), "Deconvolution3D expects dims=4 output");
            if (outC <= 0) throw new ArgumentOutOfRangeException(nameof(outC));
            if (kernelW <= 0 || kernelH <= 0 || kernelD <= 0) throw new ArgumentOutOfRangeException(nameof(kernelW));

            var kernelExtentW = Mathf.Max(1, dilationW) * (kernelW - 1) + 1;
            var kernelExtentH = Mathf.Max(1, dilationH) * (kernelH - 1) + 1;
            var kernelExtentD = Mathf.Max(1, dilationD) * (kernelD - 1) + 1;
            var borderedOutW = (input.w - 1) * Mathf.Max(1, strideW) + kernelExtentW + Mathf.Max(0, outputPadRight);
            var borderedOutH = (input.h - 1) * Mathf.Max(1, strideH) + kernelExtentH + Mathf.Max(0, outputPadBottom);
            var borderedOutD = (input.d - 1) * Mathf.Max(1, strideD) + kernelExtentD + Mathf.Max(0, outputPadBehind);
            var expectedOutW = borderedOutW - Mathf.Max(0, padLeft) - Mathf.Max(0, padRight);
            var expectedOutH = borderedOutH - Mathf.Max(0, padTop) - Mathf.Max(0, padBottom);
            var expectedOutD = borderedOutD - Mathf.Max(0, padFront) - Mathf.Max(0, padBehind);
            if (output.w != expectedOutW || output.h != expectedOutH || output.d != expectedOutD)
            {
                throw new ArgumentOutOfRangeException(
                    nameof(output),
                    "deconvolution3d output shape mismatch expected "
                    + expectedOutW.ToString(CultureInfo.InvariantCulture) + "x"
                    + expectedOutH.ToString(CultureInfo.InvariantCulture) + "x"
                    + expectedOutD.ToString(CultureInfo.InvariantCulture) + " but got "
                    + output.w.ToString(CultureInfo.InvariantCulture) + "x"
                    + output.h.ToString(CultureInfo.InvariantCulture) + "x"
                    + output.d.ToString(CultureInfo.InvariantCulture));
            }

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InD", input.d);
            _cs.SetInt("_InC", input.c);
            _cs.SetInt("_OutC", outC);
            _cs.SetInt("_OutW", output.w);
            _cs.SetInt("_OutH", output.h);
            _cs.SetInt("_OutD", output.d);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_KernelDVar", kernelD);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_StrideDVar", Mathf.Max(1, strideD));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadRightVar", Mathf.Max(0, padRight));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_PadBottomVar", Mathf.Max(0, padBottom));
            _cs.SetInt("_PadFrontVar", Mathf.Max(0, padFront));
            _cs.SetInt("_PadBehindVar", Mathf.Max(0, padBehind));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_DilationDVar", Mathf.Max(1, dilationD));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kDeconvolution3dBuf, "_ConvIn", input.buffer);
            _cs.SetBuffer(_kDeconvolution3dBuf, "_ConvW", weightsOidhw);
            _cs.SetBuffer(_kDeconvolution3dBuf, "_ConvB", biasO);
            _cs.SetBuffer(_kDeconvolution3dBuf, "_ConvOut", output.buffer);
            Dispatch3D(_kDeconvolution3dBuf, output.w, output.h, output.d * outC, 8, 8);
        }

        public void Deconvolution3dPack4CDHW(
            RenderTexture input,
            int inW,
            int inH,
            int inD,
            int inPacks,
            ComputeBuffer weightsO4I4K3,
            ComputeBuffer biasO4,
            int outW,
            int outH,
            int outD,
            int outPacks,
            int kernelW,
            int kernelH,
            int kernelD,
            int strideW,
            int strideH,
            int strideD,
            int padLeft,
            int padRight,
            int padTop,
            int padBottom,
            int padFront,
            int padBehind,
            int dilationW,
            int dilationH,
            int dilationD,
            int activationType,
            float activationParam,
            RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (weightsO4I4K3 == null) throw new ArgumentNullException(nameof(weightsO4I4K3));
            if (biasO4 == null) throw new ArgumentNullException(nameof(biasO4));
            if (inW <= 0 || inH <= 0 || inD <= 0) throw new ArgumentOutOfRangeException(nameof(inW));
            if (outW <= 0 || outH <= 0 || outD <= 0) throw new ArgumentOutOfRangeException(nameof(outW));
            if (inPacks <= 0 || outPacks <= 0) throw new ArgumentOutOfRangeException(nameof(inPacks));

            _cs.SetInt("_InW", inW);
            _cs.SetInt("_InH", inH);
            _cs.SetInt("_InD", inD);
            _cs.SetInt("_OutW", outW);
            _cs.SetInt("_OutH", outH);
            _cs.SetInt("_OutD", outD);
            _cs.SetInt("_InPacks", inPacks);
            _cs.SetInt("_OutPacks", outPacks);
            _cs.SetInt("_KernelWVar", kernelW);
            _cs.SetInt("_KernelHVar", kernelH);
            _cs.SetInt("_KernelDVar", kernelD);
            _cs.SetInt("_StrideWVar", Mathf.Max(1, strideW));
            _cs.SetInt("_StrideHVar", Mathf.Max(1, strideH));
            _cs.SetInt("_StrideDVar", Mathf.Max(1, strideD));
            _cs.SetInt("_PadLeftVar", Mathf.Max(0, padLeft));
            _cs.SetInt("_PadRightVar", Mathf.Max(0, padRight));
            _cs.SetInt("_PadTopVar", Mathf.Max(0, padTop));
            _cs.SetInt("_PadBottomVar", Mathf.Max(0, padBottom));
            _cs.SetInt("_PadFrontVar", Mathf.Max(0, padFront));
            _cs.SetInt("_PadBehindVar", Mathf.Max(0, padBehind));
            _cs.SetInt("_DilationWVar", Mathf.Max(1, dilationW));
            _cs.SetInt("_DilationHVar", Mathf.Max(1, dilationH));
            _cs.SetInt("_DilationDVar", Mathf.Max(1, dilationD));
            _cs.SetInt("_ActType", activationType);
            _cs.SetFloat("_ActParam", activationParam);
            _cs.SetBuffer(_kDeconvolution3dPack4Cdhw, "_ConvW4", weightsO4I4K3);
            _cs.SetBuffer(_kDeconvolution3dPack4Cdhw, "_ConvB4", biasO4);
            _cs.SetTexture(_kDeconvolution3dPack4Cdhw, "_ConvInArr", input);
            _cs.SetTexture(_kDeconvolution3dPack4Cdhw, "_ConvOutArr", output);
            Dispatch3D(_kDeconvolution3dPack4Cdhw, outW, outH, ResolveRenderTextureDispatchDepth(output, outD * outPacks), 8, 8);
        }

        public void LeakyReluInplace(NcnnTensorBuffer t, float slope)
        {
            if (t == null) throw new ArgumentNullException(nameof(t));
            var total = t.w * t.h * t.c;
            _cs.SetInt("_Total", total);
            _cs.SetFloat("_CoeffA", slope);
            _cs.SetBuffer(_kLeakyReluBuf, "_BufOut", t.buffer);
            Dispatch1D(_kLeakyReluBuf, total, 256);
        }

        public void AddWeighted(NcnnTensorBuffer a, NcnnTensorBuffer b, float coeffA, float coeffB, NcnnTensorBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (a.w != b.w || a.h != b.h || a.c != b.c || a.w != output.w || a.h != output.h || a.c != output.c)
                throw new ArgumentOutOfRangeException(nameof(output), "shape mismatch");

            var total = output.w * output.h * output.c;
            _cs.SetInt("_Total", total);
            _cs.SetFloat("_CoeffA", coeffA);
            _cs.SetFloat("_CoeffB", coeffB);
            _cs.SetBuffer(_kAddWeighted, "_BufA", a.buffer);
            _cs.SetBuffer(_kAddWeighted, "_BufB", b.buffer);
            _cs.SetBuffer(_kAddWeighted, "_BufOut", output.buffer);
            Dispatch1D(_kAddWeighted, total, 256);
        }

        public void CopyToConcat(NcnnTensorBuffer src, NcnnTensorBuffer dst, int dstChannelOffset)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (src.w != dst.w || src.h != dst.h) throw new ArgumentOutOfRangeException(nameof(dst), "shape mismatch");
            if (dstChannelOffset < 0 || dstChannelOffset + src.c > dst.c) throw new ArgumentOutOfRangeException(nameof(dstChannelOffset));

            var total = src.w * src.h * src.c;
            _cs.SetInt("_InW", src.w);
            _cs.SetInt("_InH", src.h);
            _cs.SetInt("_ChanOffset", dstChannelOffset);
            _cs.SetInt("_CopyTotal", total);
            _cs.SetBuffer(_kCopyC, "_BufA", src.buffer);
            _cs.SetBuffer(_kCopyC, "_BufOut", dst.buffer);
            Dispatch1D(_kCopyC, total, 256);
        }

        public void Interp2x(NcnnTensorBuffer input, NcnnTensorBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (output.w != input.w * 2 || output.h != input.h * 2 || output.c != input.c)
                throw new ArgumentOutOfRangeException(nameof(output), "output shape must be 2x");

            _cs.SetInt("_InW", input.w);
            _cs.SetInt("_InH", input.h);
            _cs.SetInt("_InC", input.c);
            _cs.SetBuffer(_kInterp2x, "_BufA", input.buffer);
            _cs.SetBuffer(_kInterp2x, "_BufOut", output.buffer);
            var total = output.w * output.h * output.c;
            Dispatch1D(_kInterp2x, total, 64);
        }

        private static void Dispatch2D(ComputeShader cs, int kernel, int w, int h, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cs.Dispatch(kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);
        }

        private static void Dispatch2D(CommandBuffer cmd, ComputeShader cs, int kernel, int w, int h, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cmd.DispatchCompute(cs, kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), 1);
        }

        private static void Dispatch3D(ComputeShader cs, int kernel, int w, int h, int z, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cs.Dispatch(kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), Mathf.Max(1, z));
        }

        private void Dispatch2D(int kernel, int w, int h, int tx, int ty)
        {
            Dispatch2D(_cs, kernel, w, h, tx, ty);
        }

        private void Dispatch3D(int kernel, int w, int h, int z, int tx, int ty)
        {
            Dispatch3D(_cs, kernel, w, h, z, tx, ty);
        }

        private void Dispatch3D(CommandBuffer cmd, int kernel, int w, int h, int z, int tx, int ty)
        {
            var gx = Mathf.CeilToInt(w / (float)tx);
            var gy = Mathf.CeilToInt(h / (float)ty);
            cmd.DispatchCompute(_cs, kernel, Mathf.Max(1, gx), Mathf.Max(1, gy), Mathf.Max(1, z));
        }

        public void ReorgPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var outPacks = packs * 4;
            _cs.SetTexture(_kReorgPack4, "_ReorgInArr", input);
            _cs.SetTexture(_kReorgPack4, "_ReorgOutArr", output);
            Dispatch3D(_kReorgPack4, output.width, output.height, outPacks, 8, 8);
        }

        public void ReorgPack4(CommandBuffer cmd, ComputeTexture input, int packs, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var outPacks = packs * 4;
            cmd.SetComputeTextureParam(_cs, _kReorgPack4, "_ReorgInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kReorgPack4, "_ReorgOutArr", output.nameID);
            Dispatch3D(cmd, _kReorgPack4, output.width, output.height, outPacks, 8, 8);
        }

        public void PointwisePack4(RenderTexture input, int packs, PointwiseType type, float a, float b, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PointwiseType", (int)type);
            _cs.SetFloat("_PointwiseA", a);
            _cs.SetFloat("_PointwiseB", b);
            _cs.SetTexture(_kPointwisePack4, "_PointwiseInArr", input);
            _cs.SetTexture(_kPointwisePack4, "_PointwiseOutArr", output);
            Dispatch3D(_kPointwisePack4, output.width, output.height, packs, 8, 8);
        }

        public void PointwisePack4(CommandBuffer cmd, ComputeTexture input, int packs, PointwiseType type, float a, float b, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PointwiseType", (int)type);
            cmd.SetComputeFloatParam(_cs, "_PointwiseA", a);
            cmd.SetComputeFloatParam(_cs, "_PointwiseB", b);
            cmd.SetComputeTextureParam(_cs, _kPointwisePack4, "_PointwiseInArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPointwisePack4, "_PointwiseOutArr", output.nameID);
            Dispatch3D(cmd, _kPointwisePack4, output.width, output.height, packs, 8, 8);
        }

        public void PixelShufflePack4(RenderTexture input, int outChannels, int scale, int mode, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetInt("_PixelShufflePack4OutC", outChannels);
            _cs.SetInt("_PixelShufflePack4Scale", scale);
            _cs.SetInt("_PixelShufflePack4Mode", mode);
            _cs.SetTexture(_kPixelShufflePack4, "_PixelShufflePack4InArr", input);
            _cs.SetTexture(_kPixelShufflePack4, "_PixelShufflePack4OutArr", output);
            Dispatch3D(_kPixelShufflePack4, output.width, output.height, output.volumeDepth, 8, 8);
        }

        public void PixelShufflePack4(CommandBuffer cmd, ComputeTexture input, int outChannels, int scale, int mode, ComputeTexture output)
        {
            if (cmd == null) throw new ArgumentNullException(nameof(cmd));
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            cmd.SetComputeIntParam(_cs, "_PixelShufflePack4OutC", outChannels);
            cmd.SetComputeIntParam(_cs, "_PixelShufflePack4Scale", scale);
            cmd.SetComputeIntParam(_cs, "_PixelShufflePack4Mode", mode);
            cmd.SetComputeTextureParam(_cs, _kPixelShufflePack4, "_PixelShufflePack4InArr", input.nameID);
            cmd.SetComputeTextureParam(_cs, _kPixelShufflePack4, "_PixelShufflePack4OutArr", output.nameID);
            Dispatch3D(cmd, _kPixelShufflePack4, output.width, output.height, Mathf.Max(1, Mathf.CeilToInt(outChannels / 4f)), 8, 8);
        }

        private void Dispatch1D(int kernel, int total, int threadsPerGroup)
        {
            if (total <= 0)
                return;

            const int maxGroups = 65535;
            var maxThreads = threadsPerGroup * maxGroups;
            var baseIndex = 0;
            while (baseIndex < total)
            {
                var remaining = total - baseIndex;
                var dispatchCount = Mathf.Min(remaining, maxThreads);
                var groups = Mathf.CeilToInt(dispatchCount / (float)threadsPerGroup);
                _cs.SetInt("_BaseIndex", baseIndex);
                _cs.Dispatch(kernel, Mathf.Max(1, groups), 1, 1);
                baseIndex += groups * threadsPerGroup;
            }
        }

        private void Dispatch1D(CommandBuffer cmd, int kernel, int total, int threadsPerGroup)
        {
            if (total <= 0)
                return;

            const int maxGroups = 65535;
            var maxThreads = threadsPerGroup * maxGroups;
            var baseIndex = 0;
            while (baseIndex < total)
            {
                var remaining = total - baseIndex;
                var dispatchCount = Mathf.Min(remaining, maxThreads);
                var groups = Mathf.CeilToInt(dispatchCount / (float)threadsPerGroup);
                cmd.SetComputeIntParam(_cs, "_BaseIndex", baseIndex);
                cmd.DispatchCompute(_cs, kernel, Mathf.Max(1, groups), 1, 1);
                baseIndex += groups * threadsPerGroup;
            }
        }
    }
}
