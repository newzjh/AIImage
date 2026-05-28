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
    }

    public sealed class NcnnOps
    {
        private readonly ComputeShader _cs;
        private readonly int _kConv3x3;
        private readonly int _kTexToBuf3;
        private readonly int _kBufToTex3;
        private readonly int _kLeakyReluBuf;
        private readonly int _kAddWeighted;
        private readonly int _kCopyBuf;
        private readonly int _kBinaryOpBuf;
        private readonly int _kUnaryOpBuf;
        private readonly int _kSigmoidBuf;
        private readonly int _kSwishBuf;
        private readonly int _kGeluBuf;
        private readonly int _kCopyC;
        private readonly int _kInterp2x;
        private readonly int _kBlitTileToDst;
        private readonly int _kPackRgbToPack4;
        private readonly int _kConv3x3Pack4;
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
        private readonly int _kInterp2xPack4;
        private readonly int _kInterp2xNearestPack4;
        private readonly int _kInterpDown2Pack4;
        private readonly int _kInterpDown2NearestPack4;
        private readonly int _kPack4ToBufferChw;
        private readonly int _kInnerProduct;
        private readonly int _kPackRgbToPack4Gfpgan;
        private readonly int _kFillPack4FromBufferChw;
        private readonly int _kScalePack4;
        private readonly int _kAddBiasPack4;
        private readonly int _kLeakyReluPack4;
        private readonly int _kAddNoiseBroadcastPack4;
        private readonly int _kClipPack4;
        private readonly int _kSftPack4;
        private readonly int _kPack4ToRgb01;
        private readonly int _kProbeTilePack4;
        private readonly int _kProbeSeams;
        private readonly int _kPaddingPack4;
        private readonly int _kPoolingPack4;
        private readonly int _kSoftmaxChannelPack4;
        private readonly int _kUnaryOpPack4;
        private readonly int _kBinaryOpPack4;
        private readonly int _kSwishPack4;
        private readonly int _kSigmoidPack4;
        private readonly int _kGeluPack4;
        private readonly int _kMatMul2D;
        private readonly int _kGemm2D;
        private readonly int _kLayerNorm2D;
        private readonly int _kSoftmax2D;
        private readonly int _kEmbed;
        private readonly int _kPermute;
        private readonly int _kSlice;
        private readonly int _kTile;
        private readonly int _kReduceSum256;
        private readonly int _kMulScalarBuf;
        private readonly int _kGroupNormStats;
        private readonly int _kGroupNormApply;
        private readonly int _kTouchU32;
        private readonly int _kInnerProduct2D;
        private readonly int _kMhaAttention;

        public NcnnOps()
        {
            _cs = Resources.Load<ComputeShader>("NcnnCompute");
            if (_cs == null)
                throw new InvalidOperationException("ComputeShader not found: Resources/NcnnCompute.compute");
            _kConv3x3 = _cs.FindKernel("NcnnConv3x3");
            _kTexToBuf3 = _cs.FindKernel("NcnnTexToBuf3");
            _kBufToTex3 = _cs.FindKernel("NcnnBufToTex3");
            _kLeakyReluBuf = _cs.FindKernel("NcnnLeakyReluBuf");
            _kAddWeighted = _cs.FindKernel("NcnnAddWeighted");
            _kCopyBuf = _cs.FindKernel("NcnnCopyBuf");
            _kBinaryOpBuf = _cs.FindKernel("NcnnBinaryOpBuf");
            _kUnaryOpBuf = _cs.FindKernel("NcnnUnaryOpBuf");
            _kSigmoidBuf = _cs.FindKernel("NcnnSigmoidBuf");
            _kSwishBuf = _cs.FindKernel("NcnnSwishBuf");
            _kGeluBuf = _cs.FindKernel("NcnnGeluBuf");
            _kCopyC = _cs.FindKernel("NcnnCopyC");
            _kInterp2x = _cs.FindKernel("NcnnInterp2x");
            _kBlitTileToDst = _cs.FindKernel("NcnnBlitTileToDst");
            _kPackRgbToPack4 = _cs.FindKernel("NcnnPackRgbToPack4");
            _kConv3x3Pack4 = _cs.FindKernel("NcnnConv3x3Pack4");
            _kWinograd23TransformInput = _cs.FindKernel("NcnnWinograd23TransformInputPack4");
            _kWinograd23Gemm = _cs.FindKernel("NcnnWinograd23GemmPack4");
            _kWinograd23TransformOutput = _cs.FindKernel("NcnnWinograd23TransformOutputPack4");
            _kConv1x1Pack4 = _cs.FindKernel("NcnnConv1x1Pack4");
            _kAddPack4 = _cs.FindKernel("NcnnAddPack4");
            _kCopyPack4 = _cs.FindKernel("NcnnCopyPack4");
            _kInterp2xPack4 = _cs.FindKernel("NcnnInterp2xPack4");
            _kInterp2xNearestPack4 = _cs.FindKernel("NcnnInterp2xNearestPack4");
            _kInterpDown2Pack4 = _cs.FindKernel("NcnnInterpDown2Pack4");
            _kInterpDown2NearestPack4 = _cs.FindKernel("NcnnInterpDown2NearestPack4");
            _kPack4ToBufferChw = _cs.FindKernel("NcnnPack4ToBufferCHW");
            _kInnerProduct = _cs.FindKernel("NcnnInnerProduct");
            _kPackRgbToPack4Gfpgan = _cs.FindKernel("NcnnPackRgbToPack4Gfpgan");
            _kFillPack4FromBufferChw = _cs.FindKernel("NcnnFillPack4FromBufferCHW");
            _kScalePack4 = _cs.FindKernel("NcnnScalePack4");
            _kAddBiasPack4 = _cs.FindKernel("NcnnAddBiasPack4");
            _kLeakyReluPack4 = _cs.FindKernel("NcnnLeakyReluPack4");
            _kAddNoiseBroadcastPack4 = _cs.FindKernel("NcnnAddNoiseBroadcastPack4");
            _kClipPack4 = _cs.FindKernel("NcnnClipPack4");
            _kSftPack4 = _cs.FindKernel("NcnnSftPack4");
            _kPack4ToRgb01 = _cs.FindKernel("NcnnPack4ToRgb01");
            _kProbeTilePack4 = _cs.FindKernel("NcnnProbeTilePack4");
            _kProbeSeams = _cs.FindKernel("NcnnProbeSeams");
            _kPaddingPack4 = _cs.FindKernel("NcnnPaddingPack4");
            _kPoolingPack4 = _cs.FindKernel("NcnnPoolingPack4");
            _kSoftmaxChannelPack4 = _cs.FindKernel("NcnnSoftmaxChannelPack4");
            _kUnaryOpPack4 = _cs.FindKernel("NcnnUnaryOpPack4");
            _kBinaryOpPack4 = _cs.FindKernel("NcnnBinaryOpPack4");
            _kSwishPack4 = _cs.FindKernel("NcnnSwishPack4");
            _kSigmoidPack4 = _cs.FindKernel("NcnnSigmoidPack4");
            _kGeluPack4 = _cs.FindKernel("NcnnGeluPack4");
            _kMatMul2D = _cs.FindKernel("NcnnMatMul2D");
            _kGemm2D = _cs.FindKernel("NcnnGemm2D");
            _kLayerNorm2D = _cs.FindKernel("NcnnLayerNorm2D");
            _kSoftmax2D = _cs.FindKernel("NcnnSoftmax2D");
            _kEmbed = _cs.FindKernel("NcnnEmbed");
            _kPermute = _cs.FindKernel("NcnnPermute");
            _kSlice = _cs.FindKernel("NcnnSlice");
            _kTile = _cs.FindKernel("NcnnTile");
            _kReduceSum256 = _cs.FindKernel("NcnnReduceSum256");
            _kMulScalarBuf = _cs.FindKernel("NcnnMulScalarBuf");
            _kGroupNormStats = _cs.FindKernel("NcnnGroupNormStats");
            _kGroupNormApply = _cs.FindKernel("NcnnGroupNormApply");
            _kTouchU32 = _cs.FindKernel("NcnnTouchU32");
            _kInnerProduct2D = _cs.FindKernel("NcnnInnerProduct2D");
            _kMhaAttention = _cs.FindKernel("NcnnMhaAttention");
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

        public void BlitTileToDst(RenderTexture tileOut, RenderTexture dst, int dstX, int dstY, int tileOutOriginX, int tileOutOriginY, int w, int h, float dstToSrcScale)
        {
            if (tileOut == null) throw new ArgumentNullException(nameof(tileOut));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (w <= 0 || h <= 0) return;
            if (dstX < 0 || dstY < 0 || dstX + w > dst.width || dstY + h > dst.height)
                throw new ArgumentOutOfRangeException(nameof(dstX), "dst rect out of range");

            _cs.SetInt("_BlitW", w);
            _cs.SetInt("_BlitH", h);
            _cs.SetInt("_DstX", dstX);
            _cs.SetInt("_DstY", dstY);
            _cs.SetInt("_TileOutX", tileOutOriginX);
            _cs.SetInt("_TileOutY", tileOutOriginY);
            _cs.SetFloat("_BlitScale", dstToSrcScale);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnInArr", tileOut);
            _cs.SetTexture(_kBlitTileToDst, "_NcnnOut", dst);
            Dispatch2D(_cs, _kBlitTileToDst, w, h, 32, 32);
        }

        public void BlitTileToDst(CommandBuffer cmd, ComputeTexture tileOut, RenderTexture dst, int dstX, int dstY, int tileOutOriginX, int tileOutOriginY, int w, int h, float dstToSrcScale)
        {
            if (tileOut == null) throw new ArgumentNullException(nameof(tileOut));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (w <= 0 || h <= 0) return;
            if (dstX < 0 || dstY < 0 || dstX + w > dst.width || dstY + h > dst.height)
                throw new ArgumentOutOfRangeException(nameof(dstX), "dst rect out of range");

            cmd.SetComputeIntParam(_cs, "_BlitW", w);
            cmd.SetComputeIntParam(_cs, "_BlitH", h);
            cmd.SetComputeIntParam(_cs, "_DstX", dstX);
            cmd.SetComputeIntParam(_cs, "_DstY", dstY);
            cmd.SetComputeIntParam(_cs, "_TileOutX", tileOutOriginX);
            cmd.SetComputeIntParam(_cs, "_TileOutY", tileOutOriginY);
            cmd.SetComputeFloatParam(_cs, "_BlitScale", dstToSrcScale);
            cmd.SetComputeTextureParam(_cs, _kBlitTileToDst, "_NcnnInArr", tileOut.nameID);
            cmd.SetComputeTextureParam(_cs, _kBlitTileToDst, "_NcnnOut", dst);
            Dispatch2D(cmd, _cs, _kBlitTileToDst, w, h, 32, 32);
        }


        public void PackRgbToPack4(Texture src, int offsetX, int offsetY, float sx, float sy, RenderTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetFloat("_ScaleX", sx);
            _cs.SetFloat("_ScaleY", sy);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnIn", src);
            _cs.SetTexture(_kPackRgbToPack4, "_NcnnOutArr", dstPack4);
            Dispatch2D(_cs, _kPackRgbToPack4, dstPack4.width, dstPack4.height, 32, 32);
        }


        public void PackRgbToPack4(CommandBuffer cmd, Texture src, int offsetX, int offsetY, float sx, float sy, ComputeTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            cmd.SetComputeIntParams(_cs, "_OffsetX", offsetX);
            cmd.SetComputeIntParams(_cs, "_OffsetY", offsetY);
            cmd.SetComputeFloatParam(_cs, "_ScaleX", sx);
            cmd.SetComputeFloatParam(_cs, "_ScaleY", sy);
            cmd.SetComputeTextureParam(_cs, _kPackRgbToPack4, "_NcnnIn", src);
            cmd.SetComputeTextureParam(_cs, _kPackRgbToPack4, "_NcnnOutArr", dstPack4.nameID);
            Dispatch2D(cmd, _cs, _kPackRgbToPack4, dstPack4.width, dstPack4.height, 32, 32);
        }

  

        public void PackRgbToPack4Gfpgan(Texture src, int offsetX, int offsetY, float sx, float sy, RenderTexture dstPack4)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dstPack4 == null) throw new ArgumentNullException(nameof(dstPack4));
            _cs.SetInt("_OffsetX", offsetX);
            _cs.SetInt("_OffsetY", offsetY);
            _cs.SetFloat("_ScaleX", sx);
            _cs.SetFloat("_ScaleY", sy);
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
            _cs.SetInt("_FillC", c);
            _cs.SetBuffer(_kFillPack4FromBufferChw, "_FillIn", input);
            _cs.SetTexture(_kFillPack4FromBufferChw, "_FillOutArr", outputPack4);
            Dispatch3D(_kFillPack4FromBufferChw, w, h, outputPack4.volumeDepth, 8, 8);
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

        public void LeakyReluPack4(RenderTexture input, float slope, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetFloat("_LreluSlope", slope);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluInArr", input);
            _cs.SetTexture(_kLeakyReluPack4, "_LreluOutArr", output);
            Dispatch3D(_kLeakyReluPack4, output.width, output.height, packs, 8, 8);
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

        public void Pack4ToRgb01(RenderTexture inputPack4, RenderTexture outputRgb)
        {
            if (inputPack4 == null) throw new ArgumentNullException(nameof(inputPack4));
            if (outputRgb == null) throw new ArgumentNullException(nameof(outputRgb));
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
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, packs, 8, 8);
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
            Dispatch3D(cmd, _kBinaryOpPack4, output.width, output.height, packs, 8, 8);
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
            Dispatch3D(_kBinaryOpPack4, output.width, output.height, packs, 8, 8);
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
            Dispatch3D(_kConv1x1Pack4, dstPack4.width, dstPack4.height, outPacks, 8, 8);
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

            for (var p = 0; p < packs; p++)
            {
                var sp = srcPackOffset + p;
                var dp = dstPackOffset + p;
                Graphics.CopyTexture(src, sp, 0, 0, 0, src.width, src.height, dst, dp, 0, 0, 0);
            }
        }

        public void CopyPack4(CommandBuffer cmd, ComputeTexture src, int srcPackOffset, ComputeTexture dst, int dstPackOffset, int packs)
        {
            if (src == null) throw new ArgumentNullException(nameof(src));
            if (dst == null) throw new ArgumentNullException(nameof(dst));
            if (packs <= 0) return;
            if (src.width != dst.width || src.height != dst.height)
                throw new InvalidOperationException("CopyPack4 requires same width/height");

            for (var p = 0; p < packs; p++)
            {
                var sp = srcPackOffset + p;
                var dp = dstPackOffset + p;
                cmd.CopyTexture(src.nameID, sp, 0, 0, 0, src.width, src.height, dst.nameID, dp, 0, 0, 0);
            }
        }

        public void Interp2xPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterp2xPack4, "_InterpInArr", input);
            _cs.SetTexture(_kInterp2xPack4, "_InterpOutArr", output);
            Dispatch3D(_kInterp2xPack4, output.width, output.height, packs, 8, 8);
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

        public void InterpDown2NearestPack4(RenderTexture input, int packs, RenderTexture output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnInArr", input);
            _cs.SetTexture(_kInterpDown2NearestPack4, "_InterpDownNnOutArr", output);
            Dispatch3D(_kInterpDown2NearestPack4, output.width, output.height, packs, 8, 8);
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

        public void BinaryOpBuf(ComputeBuffer a, ComputeBuffer b, int total, int opType, ComputeBuffer output)
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
            _cs.SetBuffer(_kBinaryOpBuf, "_BufA", a);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufB", b);
            _cs.SetBuffer(_kBinaryOpBuf, "_BufOut", output);
            Dispatch1D(_kBinaryOpBuf, total, 256);
        }

        public void BinaryOpScalarBuf(ComputeBuffer a, float scalarB, int total, int opType, ComputeBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (total < 0) throw new ArgumentOutOfRangeException(nameof(total));
            if (total == 0) return;

            _cs.SetInt("_Total", total);
            _cs.SetInt("_BinaryOpType", opType);
            _cs.SetInt("_BinaryWithScalar", 1);
            _cs.SetFloat("_BinaryScalar", scalarB);
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

        public void MhaAttention(ComputeBuffer q, ComputeBuffer k, ComputeBuffer v, int srcLen, int dstLen, int embedDim, int numHeads, float scale, ComputeBuffer outContext)
        {
            if (q == null) throw new ArgumentNullException(nameof(q));
            if (k == null) throw new ArgumentNullException(nameof(k));
            if (v == null) throw new ArgumentNullException(nameof(v));
            if (outContext == null) throw new ArgumentNullException(nameof(outContext));
            if (srcLen <= 0) throw new ArgumentOutOfRangeException(nameof(srcLen));
            if (dstLen <= 0) throw new ArgumentOutOfRangeException(nameof(dstLen));
            if (embedDim <= 0) throw new ArgumentOutOfRangeException(nameof(embedDim));
            if (numHeads <= 0) throw new ArgumentOutOfRangeException(nameof(numHeads));
            if ((embedDim % numHeads) != 0) throw new ArgumentOutOfRangeException(nameof(embedDim), "embedDim must be divisible by numHeads");

            _cs.SetInt("_MhaSrcLen", srcLen);
            _cs.SetInt("_MhaDstLen", dstLen);
            _cs.SetInt("_MhaEmbedDim", embedDim);
            _cs.SetInt("_MhaNumHeads", numHeads);
            _cs.SetInt("_MhaHeadDim", embedDim / numHeads);
            _cs.SetFloat("_MhaScale", scale);
            _cs.SetBuffer(_kMhaAttention, "_MhaQ", q);
            _cs.SetBuffer(_kMhaAttention, "_MhaK", k);
            _cs.SetBuffer(_kMhaAttention, "_MhaV", v);
            _cs.SetBuffer(_kMhaAttention, "_MhaOut", outContext);

            _cs.Dispatch(_kMhaAttention, srcLen, numHeads, 1);
        }

        public void Pack4ToBufferCHW(RenderTexture input, int w, int h, int c, ComputeBuffer output)
        {
            if (input == null) throw new ArgumentNullException(nameof(input));
            if (output == null) throw new ArgumentNullException(nameof(output));
            var total = w * h * c;
            if (total <= 0) return;
            _cs.SetInt("_Pack4W", w);
            _cs.SetInt("_Pack4H", h);
            _cs.SetInt("_Pack4C", c);
            _cs.SetTexture(_kPack4ToBufferChw, "_Pack4InArr", input);
            _cs.SetBuffer(_kPack4ToBufferChw, "_Pack4Out", output);
            Dispatch1D(_kPack4ToBufferChw, total, 256);
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
            if (k > 2048) throw new ArgumentOutOfRangeException(nameof(k), "k too large for current tiled kernel (MATK_MAX=2048): " + k);

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

        public void Gemm2D(ComputeBuffer a, ComputeBuffer b, ComputeBuffer c, int m, int n, int k, bool transB, float alpha, float beta, bool useC, int broadcastTypeC, ComputeBuffer output)
        {
            if (a == null) throw new ArgumentNullException(nameof(a));
            if (b == null) throw new ArgumentNullException(nameof(b));
            if (output == null) throw new ArgumentNullException(nameof(output));
            if (m <= 0) throw new ArgumentOutOfRangeException(nameof(m));
            if (n <= 0) throw new ArgumentOutOfRangeException(nameof(n));
            if (k <= 0) throw new ArgumentOutOfRangeException(nameof(k));
            if (k > 2048) throw new ArgumentOutOfRangeException(nameof(k), "k too large for current tiled kernel (MATK_MAX=2048): " + k);
            if (useC && c == null) throw new ArgumentNullException(nameof(c));

            _cs.SetInt("_MatM", m);
            _cs.SetInt("_MatN", n);
            _cs.SetInt("_MatK", k);
            _cs.SetInt("_MatTransB", transB ? 1 : 0);
            _cs.SetInt("_MatUseC", useC ? 1 : 0);
            _cs.SetInt("_MatBroadcastTypeC", broadcastTypeC);
            _cs.SetFloat("_MatAlpha", alpha);
            _cs.SetFloat("_MatBeta", beta);
            _cs.SetBuffer(_kGemm2D, "_MatA", a);
            _cs.SetBuffer(_kGemm2D, "_MatB", b);
            _cs.SetBuffer(_kGemm2D, "_MatC", useC ? c : a);
            _cs.SetBuffer(_kGemm2D, "_MatOut", output);

            Dispatch2D(_cs, _kGemm2D, n, m, 8, 8);
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
            if (inOut == null) throw new ArgumentNullException(nameof(inOut));
            if (w <= 0) throw new ArgumentOutOfRangeException(nameof(w));
            if (h <= 0) throw new ArgumentOutOfRangeException(nameof(h));
            if (c <= 0) throw new ArgumentOutOfRangeException(nameof(c));
            if (group <= 0) throw new ArgumentOutOfRangeException(nameof(group));
            if (c % group != 0) throw new ArgumentOutOfRangeException(nameof(group), "c must be divisible by group");
            if (affine && (gamma == null || beta == null)) throw new ArgumentNullException(nameof(gamma));

            var channelsG = c / group;
            using var stats = new ComputeBuffer(group, sizeof(float) * 4, ComputeBufferType.Structured);

            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", affine ? 1 : 0);
            _cs.SetBuffer(_kGroupNormStats, "_GnInOut", inOut);
            _cs.SetBuffer(_kGroupNormStats, "_GnGamma", affine ? gamma : inOut);
            _cs.SetBuffer(_kGroupNormStats, "_GnBeta", affine ? beta : inOut);
            _cs.SetBuffer(_kGroupNormStats, "_GnStatsOut", stats);
            _cs.Dispatch(_kGroupNormStats, Mathf.Max(1, group), 1, 1);

            _cs.SetInt("_GnW", w);
            _cs.SetInt("_GnH", h);
            _cs.SetInt("_GnC", c);
            _cs.SetInt("_GnGroup", group);
            _cs.SetInt("_GnChannelsG", channelsG);
            _cs.SetFloat("_GnEps", eps);
            _cs.SetInt("_GnAffine", affine ? 1 : 0);
            _cs.SetBuffer(_kGroupNormApply, "_GnInOut", inOut);
            _cs.SetBuffer(_kGroupNormApply, "_GnGamma", affine ? gamma : inOut);
            _cs.SetBuffer(_kGroupNormApply, "_GnBeta", affine ? beta : inOut);
            _cs.SetBuffer(_kGroupNormApply, "_GnStatsOut", stats);
            _cs.Dispatch(_kGroupNormApply, Mathf.Max(1, group), 1, 1);
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
    }
}
