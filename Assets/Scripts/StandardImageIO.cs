using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;
using ICSharpCode.SharpZipLib.Zip.Compression;
using ICSharpCode.SharpZipLib.Zip.Compression.Streams;
using UnityEngine;

public static class StandardImageIO
{
    private const string LegacyPngExifKeyword = "Raw profile type exif";
    private const string MetadataCaptureTimeKey = "AIIMAGE_CAPTURE_TIME";
    private const string MetadataCameraKey = "AIIMAGE_CAMERA";
    private const string MetadataApertureKey = "AIIMAGE_APERTURE";
    private const string MetadataLocationKey = "AIIMAGE_LOCATION";
    private const string LegacyTgaExifMarker = "AIIMAGE_EXIF_HEX=";
    private static readonly byte[] PngSignatureBytes = { 137, 80, 78, 71, 13, 10, 26, 10 };
    private static readonly byte[] TgaFooterSignature = Encoding.ASCII.GetBytes("TRUEVISION-XFILE.\0");

    private sealed class TiffEntrySpec
    {
        public ushort Tag;
        public ushort Type;
        public uint Count;
        public byte[] Data;
        public string PatchKey;
    }

    private sealed class DeferredBlob
    {
        public long PatchPosition;
        public byte[] Data;
    }

    public static bool TryLoadDisplayBytes(string filePath, out byte[] imageBytes, out RawPhotoParser.RawPhotoData photoData)
    {
        imageBytes = null;
        photoData = null;
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return false;

        try
        {
            var bytes = File.ReadAllBytes(filePath);
            if (bytes == null || bytes.Length == 0)
                return false;

            photoData = TryReadMetadataFromBytes(filePath, bytes);
            imageBytes = TryConvertToDisplayBytes(filePath, bytes);
            return imageBytes != null && imageBytes.Length > 0;
        }
        catch
        {
            return false;
        }
    }

    public static bool TryReadMetadata(string filePath, out RawPhotoParser.RawPhotoData result)
    {
        result = null;
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return false;

        try
        {
            var bytes = File.ReadAllBytes(filePath);
            result = TryReadMetadataFromBytes(filePath, bytes);
            return HasMetadata(result);
        }
        catch
        {
            return false;
        }
    }

    public static bool TryEncodeTextureWithMetadata(
        Texture2D texture,
        string destinationPath,
        string sourcePath,
        int jpgQuality,
        out byte[] outputBytes,
        out string error)
    {
        outputBytes = null;
        error = null;
        if (texture == null)
        {
            error = "Texture is null";
            return false;
        }

        if (string.IsNullOrWhiteSpace(destinationPath))
        {
            error = "Destination path is empty";
            return false;
        }

        byte[] sourceBytes = null;
        try
        {
            if (!string.IsNullOrWhiteSpace(sourcePath) && File.Exists(sourcePath))
                sourceBytes = File.ReadAllBytes(sourcePath);
        }
        catch
        {
            sourceBytes = null;
        }

        var sourceMetadata = sourceBytes != null && sourceBytes.Length > 0
            ? TryReadMetadataFromBytes(sourcePath, sourceBytes)
            : null;

        var ext = (Path.GetExtension(destinationPath) ?? string.Empty).ToLowerInvariant();
        try
        {
            switch (ext)
            {
                case ".png":
                    outputBytes = SavePng(texture, sourcePath, sourceBytes, sourceMetadata);
                    break;
                case ".jpg":
                case ".jpeg":
                    outputBytes = SaveJpeg(texture, sourcePath, sourceBytes, sourceMetadata, jpgQuality);
                    break;
                case ".tif":
                case ".tiff":
                    outputBytes = SaveTiff(texture, sourceMetadata);
                    break;
                case ".tga":
                    outputBytes = SaveTga(texture, sourceMetadata);
                    break;
                case ".hdr":
                    outputBytes = SaveRadianceHdr(texture, sourceMetadata);
                    break;
                case ".exr":
                    outputBytes = texture.EncodeToEXR(Texture2D.EXRFlags.CompressZIP);
                    break;
                default:
                    error = "Unsupported output format: " + ext;
                    return false;
            }
        }
        catch (Exception e)
        {
            error = e.Message;
            return false;
        }

        if (outputBytes == null || outputBytes.Length == 0)
        {
            error = "Encoded image is empty";
            return false;
        }

        return true;
    }

    public static bool TrySaveTextureWithMetadata(
        Texture2D texture,
        string destinationPath,
        string sourcePath,
        int jpgQuality,
        out string error)
    {
        error = null;
        if (!TryEncodeTextureWithMetadata(texture, destinationPath, sourcePath, jpgQuality, out var outputBytes, out error))
            return false;

        try
        {
            File.WriteAllBytes(destinationPath, outputBytes);
            return true;
        }
        catch (Exception e)
        {
            error = e.Message;
            return false;
        }
    }

    private static RawPhotoParser.RawPhotoData TryReadMetadataFromBytes(string filePath, byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
            return null;

        var ext = (Path.GetExtension(filePath) ?? string.Empty).ToLowerInvariant();
        switch (ext)
        {
            case ".jpg":
            case ".jpeg":
            case ".tif":
            case ".tiff":
                if (RawPhotoParser.TryReadJpegOrTiffMetadataFromBytes(bytes, out var jpegOrTiff))
                    return jpegOrTiff;
                break;
            case ".png":
                if (TryReadPngMetadata(bytes, out var png))
                    return png;
                break;
            case ".tga":
                if (TryReadTgaMetadata(bytes, out var tga))
                    return tga;
                break;
            case ".hdr":
                if (TryReadHdrMetadata(bytes, out var hdr))
                    return hdr;
                break;
        }

        if (LooksLikeJpeg(bytes) || LooksLikeTiff(bytes))
        {
            if (RawPhotoParser.TryReadJpegOrTiffMetadataFromBytes(bytes, out var fallbackExif))
                return fallbackExif;
        }

        if (LooksLikePng(bytes) && TryReadPngMetadata(bytes, out var fallbackPng))
            return fallbackPng;

        if (LooksLikeHdr(bytes) && TryReadHdrMetadata(bytes, out var fallbackHdr))
            return fallbackHdr;

        if (TryReadTgaMetadata(bytes, out var fallbackTga))
            return fallbackTga;

        return null;
    }

    private static byte[] TryConvertToDisplayBytes(string filePath, byte[] bytes)
    {
        if (bytes == null || bytes.Length == 0)
            return null;

        var ext = (Path.GetExtension(filePath) ?? string.Empty).ToLowerInvariant();
        if (ext == ".tga")
            return ConvertTgaToPng(bytes);
        if (ext == ".hdr")
            return ConvertHdrToPng(bytes);
        if (ext == ".tif" || ext == ".tiff")
            return ConvertTiffToPng(bytes);
        return bytes;
    }

    private static byte[] SavePng(
        Texture2D texture,
        string sourcePath,
        byte[] sourceBytes,
        RawPhotoParser.RawPhotoData sourceMetadata)
    {
        var pngBytes = texture.EncodeToPNG();
        if (pngBytes == null || pngBytes.Length == 0)
            return pngBytes;

        var exifPayload = BuildBestExifPayload(sourcePath, sourceBytes, sourceMetadata);
        return InjectPngMetadataChunks(pngBytes, exifPayload, sourceMetadata);
    }

    private static byte[] SaveJpeg(
        Texture2D texture,
        string sourcePath,
        byte[] sourceBytes,
        RawPhotoParser.RawPhotoData sourceMetadata,
        int jpgQuality)
    {
        var jpgBytes = texture.EncodeToJPG(Mathf.Clamp(jpgQuality, 1, 100));
        if (jpgBytes == null || jpgBytes.Length == 0)
            return jpgBytes;

        var exifPayload = BuildBestExifPayload(sourcePath, sourceBytes, sourceMetadata);
        if (exifPayload == null || exifPayload.Length == 0)
            return jpgBytes;

        return InjectJpegExifSegment(jpgBytes, exifPayload);
    }

    private static byte[] SaveTiff(Texture2D texture, RawPhotoParser.RawPhotoData sourceMetadata)
    {
        var pixels = texture.GetPixels32();
        return BuildTiffFromPixels(texture.width, texture.height, pixels, sourceMetadata);
    }

    private static byte[] SaveTga(Texture2D texture, RawPhotoParser.RawPhotoData sourceMetadata)
    {
        var tgaBytes = texture.EncodeToTGA();
        if (tgaBytes == null || tgaBytes.Length == 0 || !HasMetadata(sourceMetadata))
            return tgaBytes;

        return InjectTgaMetadataExtension(tgaBytes, sourceMetadata);
    }

    private static byte[] SaveRadianceHdr(Texture2D texture, RawPhotoParser.RawPhotoData sourceMetadata)
    {
        return BuildRadianceHdr(texture, HasMetadata(sourceMetadata) ? sourceMetadata : null);
    }

    private static byte[] BuildBestExifPayload(
        string sourcePath,
        byte[] sourceBytes,
        RawPhotoParser.RawPhotoData sourceMetadata)
    {
        if (TryExtractNativeExifPayload(sourcePath, sourceBytes, out var nativeExif) &&
            nativeExif != null &&
            nativeExif.Length > 0)
        {
            return nativeExif;
        }

        return HasMetadata(sourceMetadata)
            ? BuildStandaloneExifPayload(sourceMetadata)
            : null;
    }

    private static bool TryReadPngMetadata(byte[] pngBytes, out RawPhotoParser.RawPhotoData result)
    {
        result = null;
        if (!LooksLikePng(pngBytes))
            return false;

        if (TryExtractPngExifPayload(pngBytes, out var exifPayload) &&
            RawPhotoParser.TryReadTiffMetadataFromBytes(exifPayload, out result))
        {
            return true;
        }

        if (TryReadPngTextMetadata(pngBytes, out result))
            return true;

        return false;
    }

    private static bool TryReadTgaMetadata(byte[] tgaBytes, out RawPhotoParser.RawPhotoData result)
    {
        result = null;
        if (tgaBytes == null || tgaBytes.Length < 26)
            return false;

        if (TryExtractLegacyTgaExifPayload(tgaBytes, out var exifPayload) &&
            RawPhotoParser.TryReadTiffMetadataFromBytes(exifPayload, out result))
        {
            return true;
        }

        if (!TryExtractTgaMetadataText(tgaBytes, out var metadataText))
            return false;

        result = ParseAiImageMetadataText(metadataText);
        return HasMetadata(result);
    }

    private static bool TryReadHdrMetadata(byte[] hdrBytes, out RawPhotoParser.RawPhotoData result)
    {
        result = null;
        if (!LooksLikeHdr(hdrBytes))
            return false;

        var headerText = ReadHdrHeader(hdrBytes);
        if (string.IsNullOrWhiteSpace(headerText))
            return false;

        result = ParseAiImageMetadataText(headerText);
        return HasMetadata(result);
    }

    private static byte[] ConvertTgaToPng(byte[] tgaBytes)
    {
        if (!TryDecodeTga(tgaBytes, out var width, out var height, out var pixels))
            return tgaBytes;

        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        try
        {
            tex.SetPixels32(pixels);
            tex.Apply(false, false);
            return tex.EncodeToPNG();
        }
        finally
        {
            UnityEngine.Object.Destroy(tex);
        }
    }

    private static byte[] ConvertHdrToPng(byte[] hdrBytes)
    {
        if (!TryDecodeRadianceHdr(hdrBytes, out var width, out var height, out var pixels))
            return hdrBytes;

        var tex = new Texture2D(width, height, TextureFormat.RGBAHalf, false, true);
        try
        {
            tex.SetPixels(pixels);
            tex.Apply(false, false);

            var readable = new Texture2D(width, height, TextureFormat.RGBA32, false);
            try
            {
                readable.SetPixels(tex.GetPixels());
                readable.Apply(false, false);
                return readable.EncodeToPNG();
            }
            finally
            {
                UnityEngine.Object.Destroy(readable);
            }
        }
        finally
        {
            UnityEngine.Object.Destroy(tex);
        }
    }

    private static byte[] ConvertTiffToPng(byte[] tiffBytes)
    {
        if (!TryDecodeTiff(tiffBytes, out var width, out var height, out var pixels))
            return tiffBytes;

        var tex = new Texture2D(width, height, TextureFormat.RGBA32, false);
        try
        {
            tex.SetPixels32(pixels);
            tex.Apply(false, false);
            return tex.EncodeToPNG();
        }
        finally
        {
            UnityEngine.Object.Destroy(tex);
        }
    }

    private static bool TryDecodeTga(byte[] bytes, out int width, out int height, out Color32[] pixels)
    {
        width = 0;
        height = 0;
        pixels = null;
        if (bytes == null || bytes.Length < 18)
            return false;

        var idLength = bytes[0];
        var colorMapType = bytes[1];
        var imageType = bytes[2];
        if (colorMapType != 0 || (imageType != 2 && imageType != 10))
            return false;

        width = bytes[12] | (bytes[13] << 8);
        height = bytes[14] | (bytes[15] << 8);
        var pixelDepth = bytes[16];
        var descriptor = bytes[17];
        if (width <= 0 || height <= 0 || (pixelDepth != 24 && pixelDepth != 32))
            return false;

        var topOrigin = (descriptor & 0x20) != 0;
        var bytesPerPixel = pixelDepth / 8;
        var offset = 18 + idLength;
        var pixelCount = width * height;
        pixels = new Color32[pixelCount];

        if (imageType == 2)
        {
            for (var y = 0; y < height; y++)
            {
                var dstY = topOrigin ? y : (height - 1 - y);
                for (var x = 0; x < width; x++)
                {
                    if (offset + bytesPerPixel > bytes.Length)
                        return false;

                    var b = bytes[offset++];
                    var g = bytes[offset++];
                    var r = bytes[offset++];
                    var a = bytesPerPixel == 4 ? bytes[offset++] : (byte)255;
                    pixels[dstY * width + x] = new Color32(r, g, b, a);
                }
            }

            return true;
        }

        var index = 0;
        while (index < pixelCount && offset < bytes.Length)
        {
            var packet = bytes[offset++];
            var count = (packet & 0x7F) + 1;
            if ((packet & 0x80) != 0)
            {
                if (offset + bytesPerPixel > bytes.Length)
                    return false;

                var b = bytes[offset++];
                var g = bytes[offset++];
                var r = bytes[offset++];
                var a = bytesPerPixel == 4 ? bytes[offset++] : (byte)255;
                for (var i = 0; i < count && index < pixelCount; i++, index++)
                    AssignTgaPixel(pixels, width, height, topOrigin, index, new Color32(r, g, b, a));
            }
            else
            {
                for (var i = 0; i < count && index < pixelCount; i++, index++)
                {
                    if (offset + bytesPerPixel > bytes.Length)
                        return false;

                    var b = bytes[offset++];
                    var g = bytes[offset++];
                    var r = bytes[offset++];
                    var a = bytesPerPixel == 4 ? bytes[offset++] : (byte)255;
                    AssignTgaPixel(pixels, width, height, topOrigin, index, new Color32(r, g, b, a));
                }
            }
        }

        return index == pixelCount;
    }

    private static bool TryDecodeRadianceHdr(byte[] bytes, out int width, out int height, out Color[] pixels)
    {
        width = 0;
        height = 0;
        pixels = null;
        if (bytes == null || bytes.Length < 32)
            return false;

        var offset = FindHdrDataOffset(bytes, out width, out height);
        if (offset <= 0 || width <= 0 || height <= 0)
            return false;

        var rgbe = new byte[width * height * 4];
        if (width < 8 || width > 0x7fff)
        {
            var remaining = bytes.Length - offset;
            if (remaining < rgbe.Length)
                return false;

            Buffer.BlockCopy(bytes, offset, rgbe, 0, rgbe.Length);
        }
        else
        {
            var scanline = new byte[4 * width];
            for (var y = 0; y < height; y++)
            {
                if (offset + 4 > bytes.Length)
                    return false;

                if (bytes[offset++] != 2 || bytes[offset++] != 2)
                    return false;

                var scanWidth = (bytes[offset++] << 8) | bytes[offset++];
                if (scanWidth != width)
                    return false;

                for (var channel = 0; channel < 4; channel++)
                {
                    var x = 0;
                    while (x < width && offset < bytes.Length)
                    {
                        var count = bytes[offset++];
                        if (count > 128)
                        {
                            var run = count - 128;
                            if (run <= 0 || offset >= bytes.Length || x + run > width)
                                return false;

                            var value = bytes[offset++];
                            for (var i = 0; i < run; i++)
                                scanline[channel * width + x++] = value;
                        }
                        else
                        {
                            if (count <= 0 || offset + count > bytes.Length || x + count > width)
                                return false;

                            Buffer.BlockCopy(bytes, offset, scanline, channel * width + x, count);
                            offset += count;
                            x += count;
                        }
                    }
                }

                for (var x = 0; x < width; x++)
                {
                    var baseIndex = (y * width + x) * 4;
                    rgbe[baseIndex + 0] = scanline[x];
                    rgbe[baseIndex + 1] = scanline[width + x];
                    rgbe[baseIndex + 2] = scanline[2 * width + x];
                    rgbe[baseIndex + 3] = scanline[3 * width + x];
                }
            }
        }

        pixels = new Color[width * height];
        for (var i = 0; i < pixels.Length; i++)
        {
            var r = rgbe[i * 4 + 0];
            var g = rgbe[i * 4 + 1];
            var b = rgbe[i * 4 + 2];
            var e = rgbe[i * 4 + 3];
            pixels[i] = DecodeRgbe(r, g, b, e);
        }

        return true;
    }

    private static bool TryDecodeTiff(byte[] bytes, out int width, out int height, out Color32[] pixels)
    {
        width = 0;
        height = 0;
        pixels = null;
        if (!LooksLikeTiff(bytes))
            return false;

        var littleEndian = bytes[0] == 0x49;
        var ifdOffset = (int)ReadUInt32Tiff(bytes, 4, littleEndian);
        if (ifdOffset <= 0 || ifdOffset + 2 > bytes.Length)
            return false;

        uint imageWidth = 0;
        uint imageHeight = 0;
        uint compression = 1;
        uint photometric = 2;
        uint samplesPerPixel = 0;
        uint rowsPerStrip = 0;
        uint planarConfiguration = 1;
        uint orientation = 1;
        ushort[] bitsPerSample = null;
        uint[] stripOffsets = null;
        uint[] stripByteCounts = null;

        var entryCount = ReadUInt16Tiff(bytes, ifdOffset, littleEndian);
        if (entryCount == 0 || entryCount > 512)
            return false;

        for (var i = 0; i < entryCount; i++)
        {
            var entryOffset = ifdOffset + 2 + i * 12;
            if (entryOffset + 12 > bytes.Length)
                return false;

            var tag = ReadUInt16Tiff(bytes, entryOffset, littleEndian);
            var type = ReadUInt16Tiff(bytes, entryOffset + 2, littleEndian);
            var count = ReadUInt32Tiff(bytes, entryOffset + 4, littleEndian);
            if (!TryResolveTiffValueDataOffset(bytes, 0, entryOffset, littleEndian, type, count, out var dataOffset, out _))
                continue;

            switch (tag)
            {
                case 256:
                    imageWidth = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 257:
                    imageHeight = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 258:
                    bitsPerSample = ReadTiffUShortArray(bytes, dataOffset, littleEndian, type, count);
                    break;
                case 259:
                    compression = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 262:
                    photometric = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 273:
                    stripOffsets = ReadTiffUIntArray(bytes, dataOffset, littleEndian, type, count);
                    break;
                case 274:
                    orientation = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 277:
                    samplesPerPixel = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 278:
                    rowsPerStrip = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
                case 279:
                    stripByteCounts = ReadTiffUIntArray(bytes, dataOffset, littleEndian, type, count);
                    break;
                case 284:
                    planarConfiguration = ReadTiffScalarUInt(bytes, dataOffset, littleEndian, type);
                    break;
            }
        }

        if (imageWidth == 0 || imageHeight == 0 || imageWidth > int.MaxValue || imageHeight > int.MaxValue)
            return false;
        if (compression != 1 || planarConfiguration != 1 || orientation != 1)
            return false;
        if (stripOffsets == null || stripByteCounts == null || stripOffsets.Length == 0 || stripOffsets.Length != stripByteCounts.Length)
            return false;

        width = (int)imageWidth;
        height = (int)imageHeight;
        if (width <= 0 || height <= 0)
            return false;

        if (samplesPerPixel == 0)
            samplesPerPixel = bitsPerSample != null ? (uint)bitsPerSample.Length : 3u;

        if (bitsPerSample == null || bitsPerSample.Length == 0)
        {
            bitsPerSample = new ushort[samplesPerPixel];
            for (var i = 0; i < bitsPerSample.Length; i++)
                bitsPerSample[i] = 8;
        }

        for (var i = 0; i < bitsPerSample.Length; i++)
        {
            if (bitsPerSample[i] != 8)
                return false;
        }

        var channelCount = (int)samplesPerPixel;
        if (channelCount != 1 && channelCount != 2 && channelCount != 3 && channelCount != 4)
            return false;

        if (rowsPerStrip == 0)
            rowsPerStrip = (uint)height;

        var rowBytes = width * channelCount;
        pixels = new Color32[width * height];
        var decodedRows = 0;
        for (var stripIndex = 0; stripIndex < stripOffsets.Length && decodedRows < height; stripIndex++)
        {
            var stripOffset = (int)stripOffsets[stripIndex];
            var stripRowCount = Mathf.Min((int)rowsPerStrip, height - decodedRows);
            var requiredBytes = stripRowCount * rowBytes;
            if (requiredBytes < 0 || stripByteCounts[stripIndex] < requiredBytes)
                return false;
            if (stripOffset < 0 || stripOffset + requiredBytes > bytes.Length)
                return false;

            for (var localRow = 0; localRow < stripRowCount; localRow++, decodedRows++)
            {
                var srcOffset = stripOffset + localRow * rowBytes;
                var dstY = height - 1 - decodedRows;
                for (var x = 0; x < width; x++)
                {
                    var pixelOffset = srcOffset + x * channelCount;
                    byte r;
                    byte g;
                    byte b;
                    byte a;

                    if (photometric == 2)
                    {
                        r = bytes[pixelOffset + 0];
                        g = channelCount > 1 ? bytes[pixelOffset + 1] : bytes[pixelOffset + 0];
                        b = channelCount > 2 ? bytes[pixelOffset + 2] : bytes[pixelOffset + 0];
                        a = channelCount > 3 ? bytes[pixelOffset + 3] : (byte)255;
                    }
                    else if (photometric == 0 || photometric == 1)
                    {
                        var gray = bytes[pixelOffset + 0];
                        if (photometric == 0)
                            gray = (byte)(255 - gray);
                        r = g = b = gray;
                        a = channelCount > 1 ? bytes[pixelOffset + 1] : (byte)255;
                    }
                    else
                    {
                        return false;
                    }

                    pixels[dstY * width + x] = new Color32(r, g, b, a);
                }
            }
        }

        return decodedRows == height;
    }

    private static void AssignTgaPixel(Color32[] pixels, int width, int height, bool topOrigin, int index, Color32 pixel)
    {
        var x = index % width;
        var y = index / width;
        var dstY = topOrigin ? y : (height - 1 - y);
        pixels[dstY * width + x] = pixel;
    }

    private static Color DecodeRgbe(byte r, byte g, byte b, byte e)
    {
        if (e == 0)
            return Color.clear;

        var f = Mathf.Pow(2f, e - (128 + 8));
        return new Color(r * f, g * f, b * f, 1f);
    }

    private static int FindHdrDataOffset(byte[] bytes, out int width, out int height)
    {
        width = 0;
        height = 0;
        var header = ReadHdrHeader(bytes);
        if (string.IsNullOrWhiteSpace(header))
            return -1;

        var lines = header.Split(new[] { '\n' }, StringSplitOptions.RemoveEmptyEntries);
        foreach (var line in lines)
        {
            var trimmed = line.Trim();
            if (!trimmed.StartsWith("-Y ", StringComparison.OrdinalIgnoreCase) &&
                !trimmed.StartsWith("+Y ", StringComparison.OrdinalIgnoreCase))
            {
                continue;
            }

            var tokens = trimmed.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
            if (tokens.Length >= 4 &&
                int.TryParse(tokens[1], NumberStyles.Integer, CultureInfo.InvariantCulture, out height) &&
                int.TryParse(tokens[3], NumberStyles.Integer, CultureInfo.InvariantCulture, out width))
            {
                var marker = Encoding.ASCII.GetBytes(trimmed + "\n");
                var idx = IndexOf(bytes, marker, 0);
                return idx >= 0 ? idx + marker.Length : -1;
            }
        }

        return -1;
    }

    private static string ReadHdrHeader(byte[] bytes)
    {
        if (bytes == null || bytes.Length < 8)
            return null;

        var end = -1;
        for (var i = 0; i < bytes.Length - 1; i++)
        {
            if (bytes[i] == '\n' && bytes[i + 1] == '\n')
            {
                end = i + 2;
                break;
            }
        }

        if (end <= 0)
            return null;

        return Encoding.ASCII.GetString(bytes, 0, end);
    }

    private static byte[] BuildRadianceHdr(Texture2D texture, RawPhotoParser.RawPhotoData metadata)
    {
        var colors = texture.GetPixels();
        var width = texture.width;
        var height = texture.height;

        using var ms = new MemoryStream();
        using var writer = new StreamWriter(ms, Encoding.ASCII, 1024, true);
        writer.WriteLine("#?RADIANCE");
        writer.WriteLine("FORMAT=32-bit_rle_rgbe");
        WriteHdrMetadataLines(writer, metadata);
        writer.WriteLine();
        writer.WriteLine("-Y " + height.ToString(CultureInfo.InvariantCulture) + " +X " + width.ToString(CultureInfo.InvariantCulture));
        writer.Flush();

        var scanline = new byte[width * 4];
        for (var y = 0; y < height; y++)
        {
            ms.WriteByte(2);
            ms.WriteByte(2);
            ms.WriteByte((byte)(width >> 8));
            ms.WriteByte((byte)(width & 0xFF));

            for (var x = 0; x < width; x++)
            {
                var c = colors[(height - 1 - y) * width + x];
                EncodeRgbe(c, out var r, out var g, out var b, out var e);
                scanline[x] = r;
                scanline[width + x] = g;
                scanline[2 * width + x] = b;
                scanline[3 * width + x] = e;
            }

            for (var channel = 0; channel < 4; channel++)
                WriteHdrRle(ms, scanline, channel * width, width);
        }

        return ms.ToArray();
    }

    private static void WriteHdrMetadataLines(StreamWriter writer, RawPhotoParser.RawPhotoData metadata)
    {
        if (!HasMetadata(metadata))
            return;

        if (metadata.captureTime.HasValue)
            writer.WriteLine(MetadataCaptureTimeKey + "=" + metadata.captureTime.Value.ToString("o", CultureInfo.InvariantCulture));
        if (!string.IsNullOrWhiteSpace(metadata.cameraText))
            writer.WriteLine(MetadataCameraKey + "=" + SanitizeHdrText(metadata.cameraText));
        if (!string.IsNullOrWhiteSpace(metadata.apertureText))
            writer.WriteLine(MetadataApertureKey + "=" + SanitizeHdrText(metadata.apertureText));
        if (!string.IsNullOrWhiteSpace(metadata.locationText))
            writer.WriteLine(MetadataLocationKey + "=" + SanitizeHdrText(metadata.locationText));
    }

    private static void EncodeRgbe(Color color, out byte r, out byte g, out byte b, out byte e)
    {
        var max = Mathf.Max(color.r, Mathf.Max(color.g, color.b));
        if (max < 1e-9f)
        {
            r = g = b = e = 0;
            return;
        }

        var exponent = Mathf.CeilToInt(Mathf.Log(max, 2f));
        var scale = Mathf.Pow(2f, exponent - 8);
        r = (byte)Mathf.Clamp(Mathf.RoundToInt(color.r / scale), 0, 255);
        g = (byte)Mathf.Clamp(Mathf.RoundToInt(color.g / scale), 0, 255);
        b = (byte)Mathf.Clamp(Mathf.RoundToInt(color.b / scale), 0, 255);
        e = (byte)(exponent + 128);
    }

    private static void WriteHdrRle(Stream stream, byte[] data, int offset, int count)
    {
        var i = 0;
        while (i < count)
        {
            var runLength = 1;
            var value = data[offset + i];
            while (i + runLength < count && runLength < 127 && data[offset + i + runLength] == value)
                runLength++;

            if (runLength >= 4)
            {
                stream.WriteByte((byte)(128 + runLength));
                stream.WriteByte(value);
                i += runLength;
                continue;
            }

            var literalStart = i;
            i += runLength;
            while (i < count)
            {
                var lookahead = 1;
                var nextValue = data[offset + i];
                while (i + lookahead < count && lookahead < 127 && data[offset + i + lookahead] == nextValue)
                    lookahead++;
                if (lookahead >= 4 || i - literalStart >= 127)
                    break;
                i += lookahead;
            }

            var literalCount = i - literalStart;
            stream.WriteByte((byte)literalCount);
            stream.Write(data, offset + literalStart, literalCount);
        }
    }

    private static string SanitizeHdrText(string text)
    {
        return SanitizeAscii(text)?.Replace("\r", " ").Replace("\n", " ");
    }

    private static byte[] BuildTiffFromPixels(int width, int height, Color32[] pixels, RawPhotoParser.RawPhotoData metadata)
    {
        if (width <= 0 || height <= 0 || pixels == null || pixels.Length < width * height)
            return null;

        var rootEntries = new List<TiffEntrySpec>
        {
            CreateLongEntry(256, (uint)width),
            CreateLongEntry(257, (uint)height),
            CreateShortArrayEntry(258, 8, 8, 8, 8),
            CreateShortEntry(259, 1),
            CreateShortEntry(262, 2),
            CreateLongPatchEntry(273, "IMAGE_DATA_OFFSET"),
            CreateShortEntry(274, 1),
            CreateShortEntry(277, 4),
            CreateLongEntry(278, (uint)height),
            CreateLongEntry(279, (uint)(width * height * 4)),
            CreateRationalEntry(282, 72, 1),
            CreateRationalEntry(283, 72, 1),
            CreateShortEntry(284, 1),
            CreateShortEntry(296, 2),
            CreateShortEntry(338, 2)
        };

        BuildMetadataEntrySets(metadata, out var metadataRootEntries, out var exifEntries, out var gpsEntries);
        rootEntries.AddRange(metadataRootEntries);

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write((byte)'I');
        writer.Write((byte)'I');
        writer.Write((ushort)42);
        writer.Write((uint)8);

        var rootPatches = WriteIfd(writer, rootEntries);

        if (exifEntries.Count > 0 && rootPatches.TryGetValue("EXIF_IFD", out var exifPatchPos))
        {
            AlignToWord(writer);
            PatchUInt32(writer, exifPatchPos, (uint)writer.BaseStream.Position);
            WriteIfd(writer, exifEntries);
        }

        if (gpsEntries.Count > 0 && rootPatches.TryGetValue("GPS_IFD", out var gpsPatchPos))
        {
            AlignToWord(writer);
            PatchUInt32(writer, gpsPatchPos, (uint)writer.BaseStream.Position);
            WriteIfd(writer, gpsEntries);
        }

        AlignToWord(writer);
        if (rootPatches.TryGetValue("IMAGE_DATA_OFFSET", out var imagePatchPos))
            PatchUInt32(writer, imagePatchPos, (uint)writer.BaseStream.Position);

        for (var y = height - 1; y >= 0; y--)
        {
            for (var x = 0; x < width; x++)
            {
                var c = pixels[y * width + x];
                writer.Write(c.r);
                writer.Write(c.g);
                writer.Write(c.b);
                writer.Write(c.a);
            }
        }

        return ms.ToArray();
    }

    private static byte[] BuildStandaloneExifPayload(RawPhotoParser.RawPhotoData metadata)
    {
        if (!HasMetadata(metadata))
            return null;

        BuildMetadataEntrySets(metadata, out var rootEntries, out var exifEntries, out var gpsEntries);
        if (rootEntries.Count == 0)
            return null;

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);

        writer.Write((byte)'I');
        writer.Write((byte)'I');
        writer.Write((ushort)42);
        writer.Write((uint)8);

        var rootPatches = WriteIfd(writer, rootEntries);

        if (exifEntries.Count > 0 && rootPatches.TryGetValue("EXIF_IFD", out var exifPatchPos))
        {
            AlignToWord(writer);
            PatchUInt32(writer, exifPatchPos, (uint)writer.BaseStream.Position);
            WriteIfd(writer, exifEntries);
        }

        if (gpsEntries.Count > 0 && rootPatches.TryGetValue("GPS_IFD", out var gpsPatchPos))
        {
            AlignToWord(writer);
            PatchUInt32(writer, gpsPatchPos, (uint)writer.BaseStream.Position);
            WriteIfd(writer, gpsEntries);
        }

        return ms.ToArray();
    }

    private static void BuildMetadataEntrySets(
        RawPhotoParser.RawPhotoData metadata,
        out List<TiffEntrySpec> rootEntries,
        out List<TiffEntrySpec> exifEntries,
        out List<TiffEntrySpec> gpsEntries)
    {
        rootEntries = new List<TiffEntrySpec>();
        exifEntries = new List<TiffEntrySpec>();
        gpsEntries = new List<TiffEntrySpec>();
        if (!HasMetadata(metadata))
            return;

        var modelData = CreateAsciiData(metadata.cameraText);
        if (modelData != null)
            rootEntries.Add(new TiffEntrySpec { Tag = 0x0110, Type = 2, Count = (uint)modelData.Length, Data = modelData });

        if (metadata.captureTime.HasValue)
        {
            var exifDate = CreateExifDateData(metadata.captureTime.Value);
            if (exifDate != null)
            {
                rootEntries.Add(new TiffEntrySpec { Tag = 0x0132, Type = 2, Count = (uint)exifDate.Length, Data = exifDate });
                exifEntries.Add(new TiffEntrySpec { Tag = 0x9003, Type = 2, Count = (uint)exifDate.Length, Data = exifDate });
            }
        }

        if (TryParseApertureValue(metadata.apertureText, out var apertureValue))
            exifEntries.Add(CreateRationalEntry(0x829D, apertureValue.Item1, apertureValue.Item2));

        if (TryParseGps(metadata.locationText, out var latitude, out var longitude))
        {
            gpsEntries.Add(new TiffEntrySpec
            {
                Tag = 0x0001,
                Type = 2,
                Count = 2,
                Data = CreateAsciiRefData(latitude >= 0d ? "N" : "S")
            });
            gpsEntries.Add(new TiffEntrySpec
            {
                Tag = 0x0002,
                Type = 5,
                Count = 3,
                Data = CreateGpsCoordinateData(latitude)
            });
            gpsEntries.Add(new TiffEntrySpec
            {
                Tag = 0x0003,
                Type = 2,
                Count = 2,
                Data = CreateAsciiRefData(longitude >= 0d ? "E" : "W")
            });
            gpsEntries.Add(new TiffEntrySpec
            {
                Tag = 0x0004,
                Type = 5,
                Count = 3,
                Data = CreateGpsCoordinateData(longitude)
            });
        }

        if (exifEntries.Count > 0)
            rootEntries.Add(CreateLongPatchEntry(0x8769, "EXIF_IFD"));
        if (gpsEntries.Count > 0)
            rootEntries.Add(CreateLongPatchEntry(0x8825, "GPS_IFD"));
    }

    private static Dictionary<string, long> WriteIfd(BinaryWriter writer, List<TiffEntrySpec> entries)
    {
        entries.Sort((a, b) => a.Tag.CompareTo(b.Tag));
        writer.Write((ushort)entries.Count);

        var patchPositions = new Dictionary<string, long>(StringComparer.Ordinal);
        var deferredBlobs = new List<DeferredBlob>();
        for (var i = 0; i < entries.Count; i++)
        {
            var entry = entries[i];
            writer.Write(entry.Tag);
            writer.Write(entry.Type);
            writer.Write(entry.Count);

            if (!string.IsNullOrEmpty(entry.PatchKey))
            {
                patchPositions[entry.PatchKey] = writer.BaseStream.Position;
                writer.Write((uint)0);
                continue;
            }

            var data = entry.Data ?? Array.Empty<byte>();
            if (data.Length <= 4)
            {
                WriteInlineTiffValue(writer, data);
            }
            else
            {
                deferredBlobs.Add(new DeferredBlob
                {
                    PatchPosition = writer.BaseStream.Position,
                    Data = data
                });
                writer.Write((uint)0);
            }
        }

        writer.Write((uint)0);

        for (var i = 0; i < deferredBlobs.Count; i++)
        {
            var blob = deferredBlobs[i];
            AlignToWord(writer);
            PatchUInt32(writer, blob.PatchPosition, (uint)writer.BaseStream.Position);
            writer.Write(blob.Data);
        }

        return patchPositions;
    }

    private static void WriteInlineTiffValue(BinaryWriter writer, byte[] data)
    {
        var length = data != null ? data.Length : 0;
        for (var i = 0; i < 4; i++)
            writer.Write(i < length ? data[i] : (byte)0);
    }

    private static void PatchUInt32(BinaryWriter writer, long position, uint value)
    {
        var returnPosition = writer.BaseStream.Position;
        writer.BaseStream.Position = position;
        writer.Write(value);
        writer.BaseStream.Position = returnPosition;
    }

    private static void AlignToWord(BinaryWriter writer)
    {
        if ((writer.BaseStream.Position & 1) != 0)
            writer.Write((byte)0);
    }

    private static TiffEntrySpec CreateShortEntry(ushort tag, ushort value)
    {
        return new TiffEntrySpec
        {
            Tag = tag,
            Type = 3,
            Count = 1,
            Data = CreateUInt16Data(value)
        };
    }

    private static TiffEntrySpec CreateShortArrayEntry(ushort tag, params ushort[] values)
    {
        return new TiffEntrySpec
        {
            Tag = tag,
            Type = 3,
            Count = (uint)(values?.Length ?? 0),
            Data = CreateUInt16ArrayData(values)
        };
    }

    private static TiffEntrySpec CreateLongEntry(ushort tag, uint value)
    {
        return new TiffEntrySpec
        {
            Tag = tag,
            Type = 4,
            Count = 1,
            Data = CreateUInt32Data(value)
        };
    }

    private static TiffEntrySpec CreateLongPatchEntry(ushort tag, string patchKey)
    {
        return new TiffEntrySpec
        {
            Tag = tag,
            Type = 4,
            Count = 1,
            PatchKey = patchKey
        };
    }

    private static TiffEntrySpec CreateRationalEntry(ushort tag, uint numerator, uint denominator)
    {
        return new TiffEntrySpec
        {
            Tag = tag,
            Type = 5,
            Count = 1,
            Data = CreateRationalData(numerator, denominator)
        };
    }

    private static byte[] CreateUInt16Data(ushort value)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(value);
        return ms.ToArray();
    }

    private static byte[] CreateUInt16ArrayData(ushort[] values)
    {
        if (values == null || values.Length == 0)
            return null;

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        for (var i = 0; i < values.Length; i++)
            writer.Write(values[i]);
        return ms.ToArray();
    }

    private static byte[] CreateUInt32Data(uint value)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(value);
        return ms.ToArray();
    }

    private static byte[] CreateRationalData(uint numerator, uint denominator)
    {
        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write(numerator);
        writer.Write(denominator == 0 ? 1u : denominator);
        return ms.ToArray();
    }

    private static byte[] CreateExifDateData(DateTime value)
    {
        return CreateAsciiData(value.ToString("yyyy:MM:dd HH:mm:ss", CultureInfo.InvariantCulture));
    }

    private static byte[] CreateAsciiData(string text)
    {
        var sanitized = SanitizeAscii(text);
        if (string.IsNullOrWhiteSpace(sanitized))
            return null;

        return Encoding.ASCII.GetBytes(sanitized + "\0");
    }

    private static byte[] CreateAsciiRefData(string text)
    {
        var sanitized = SanitizeAscii(text);
        if (string.IsNullOrWhiteSpace(sanitized))
            return null;

        var bytes = new byte[2];
        bytes[0] = (byte)sanitized[0];
        bytes[1] = 0;
        return bytes;
    }

    private static byte[] CreateGpsCoordinateData(double decimalDegrees)
    {
        var absolute = Math.Abs(decimalDegrees);
        var degrees = Math.Floor(absolute);
        var minutesTotal = (absolute - degrees) * 60d;
        var minutes = Math.Floor(minutesTotal);
        var seconds = (minutesTotal - minutes) * 60d;
        var secondNumerator = (uint)Math.Max(0d, Math.Round(seconds * 10000d, MidpointRounding.AwayFromZero));

        using var ms = new MemoryStream();
        using var writer = new BinaryWriter(ms);
        writer.Write((uint)degrees);
        writer.Write(1u);
        writer.Write((uint)minutes);
        writer.Write(1u);
        writer.Write(secondNumerator);
        writer.Write(10000u);
        return ms.ToArray();
    }

    private static bool TryParseApertureValue(string apertureText, out Tuple<uint, uint> value)
    {
        value = null;
        if (string.IsNullOrWhiteSpace(apertureText))
            return false;

        var text = apertureText.Trim();
        if (text.StartsWith("f/", StringComparison.OrdinalIgnoreCase))
            text = text.Substring(2).Trim();

        if (!double.TryParse(text, NumberStyles.Float, CultureInfo.InvariantCulture, out var aperture) || aperture <= 0.01d)
            return false;

        const uint denominator = 1000;
        var numerator = (uint)Math.Max(1d, Math.Round(aperture * denominator, MidpointRounding.AwayFromZero));
        value = Tuple.Create(numerator, denominator);
        return true;
    }

    private static bool TryParseGps(string locationText, out double latitude, out double longitude)
    {
        latitude = 0d;
        longitude = 0d;
        if (string.IsNullOrWhiteSpace(locationText))
            return false;

        var text = locationText.Trim();
        if (text.StartsWith("GPS ", StringComparison.OrdinalIgnoreCase))
            text = text.Substring(4).Trim();

        var parts = text.Split(',');
        if (parts.Length < 2)
            return false;

        return double.TryParse(parts[0].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out latitude) &&
               double.TryParse(parts[1].Trim(), NumberStyles.Float, CultureInfo.InvariantCulture, out longitude);
    }

    private static bool TryExtractNativeExifPayload(string sourcePath, byte[] sourceBytes, out byte[] exifPayload)
    {
        exifPayload = null;
        if (sourceBytes == null || sourceBytes.Length == 0)
            return false;

        var ext = (Path.GetExtension(sourcePath) ?? string.Empty).ToLowerInvariant();
        switch (ext)
        {
            case ".jpg":
            case ".jpeg":
                return TryExtractJpegExifPayload(sourceBytes, out exifPayload);
            case ".png":
                return TryExtractPngExifPayload(sourceBytes, out exifPayload);
            case ".tga":
                return TryExtractLegacyTgaExifPayload(sourceBytes, out exifPayload);
        }

        if (LooksLikeJpeg(sourceBytes))
            return TryExtractJpegExifPayload(sourceBytes, out exifPayload);
        if (LooksLikePng(sourceBytes))
            return TryExtractPngExifPayload(sourceBytes, out exifPayload);

        return TryExtractLegacyTgaExifPayload(sourceBytes, out exifPayload);
    }

    private static bool TryExtractJpegExifPayload(byte[] jpegBytes, out byte[] exifPayload)
    {
        exifPayload = null;
        if (!LooksLikeJpeg(jpegBytes))
            return false;

        var offset = 2;
        while (offset + 4 < jpegBytes.Length)
        {
            if (jpegBytes[offset] != 0xFF)
                break;

            var marker = jpegBytes[offset + 1];
            if (marker == 0xD9 || marker == 0xDA)
                break;

            var segmentLength = ReadUInt16BigEndian(jpegBytes, offset + 2);
            if (segmentLength < 2 || offset + 2 + segmentLength > jpegBytes.Length)
                break;

            if (marker == 0xE1 &&
                segmentLength >= 8 &&
                jpegBytes[offset + 4] == (byte)'E' &&
                jpegBytes[offset + 5] == (byte)'x' &&
                jpegBytes[offset + 6] == (byte)'i' &&
                jpegBytes[offset + 7] == (byte)'f' &&
                jpegBytes[offset + 8] == 0 &&
                jpegBytes[offset + 9] == 0)
            {
                var payloadOffset = offset + 10;
                var payloadLength = segmentLength - 8;
                exifPayload = new byte[payloadLength];
                Buffer.BlockCopy(jpegBytes, payloadOffset, exifPayload, 0, payloadLength);
                return exifPayload.Length > 0;
            }

            offset += 2 + segmentLength;
        }

        return false;
    }

    private static bool TryExtractPngExifPayload(byte[] pngBytes, out byte[] exifPayload)
    {
        exifPayload = null;
        if (!LooksLikePng(pngBytes))
            return false;

        var offset = 8;
        while (offset + 12 <= pngBytes.Length)
        {
            var length = ReadUInt32BigEndian(pngBytes, offset);
            var type = Encoding.ASCII.GetString(pngBytes, offset + 4, 4);
            var dataOffset = offset + 8;
            if (dataOffset + length + 4 > pngBytes.Length)
                return false;

            if (string.Equals(type, "eXIf", StringComparison.Ordinal))
            {
                exifPayload = new byte[length];
                Buffer.BlockCopy(pngBytes, dataOffset, exifPayload, 0, (int)length);
                return exifPayload.Length > 0;
            }

            if (string.Equals(type, "zTXt", StringComparison.Ordinal) &&
                TryExtractExifFromCompressedPngText(pngBytes, dataOffset, (int)length, out exifPayload))
            {
                return true;
            }

            offset = dataOffset + (int)length + 4;
            if (string.Equals(type, "IEND", StringComparison.Ordinal))
                break;
        }

        return false;
    }

    private static bool TryExtractExifFromCompressedPngText(byte[] pngBytes, int dataOffset, int dataLength, out byte[] exifPayload)
    {
        exifPayload = null;
        if (pngBytes == null || dataLength <= 0 || dataOffset < 0 || dataOffset + dataLength > pngBytes.Length)
            return false;

        try
        {
            var keywordEnd = -1;
            for (var i = 0; i < dataLength; i++)
            {
                if (pngBytes[dataOffset + i] == 0)
                {
                    keywordEnd = dataOffset + i;
                    break;
                }
            }

            if (keywordEnd < 0 || keywordEnd + 2 >= dataOffset + dataLength)
                return false;

            var keyword = Encoding.ASCII.GetString(pngBytes, dataOffset, keywordEnd - dataOffset);
            if (!string.Equals(keyword, LegacyPngExifKeyword, StringComparison.OrdinalIgnoreCase))
                return false;

            var compressionMethod = pngBytes[keywordEnd + 1];
            if (compressionMethod != 0)
                return false;

            var compressedOffset = keywordEnd + 2;
            var compressedLength = dataOffset + dataLength - compressedOffset;
            if (compressedLength <= 0)
                return false;

            using var input = new MemoryStream(pngBytes, compressedOffset, compressedLength, false);
            using var zlib = new InflaterInputStream(input, new Inflater(true));
            using var output = new MemoryStream();
            zlib.CopyTo(output);
            var text = Encoding.ASCII.GetString(output.ToArray()).Trim();
            if (string.IsNullOrWhiteSpace(text) || (text.Length & 1) != 0)
                return false;

            exifPayload = HexToBytes(text);
            return exifPayload != null && exifPayload.Length > 0;
        }
        catch
        {
            return false;
        }
    }

    private static bool TryExtractLegacyTgaExifPayload(byte[] tgaBytes, out byte[] exifPayload)
    {
        exifPayload = null;
        if (!TryExtractTgaMetadataText(tgaBytes, out var metadataText) || string.IsNullOrWhiteSpace(metadataText))
            return false;

        var markerIndex = metadataText.IndexOf(LegacyTgaExifMarker, StringComparison.OrdinalIgnoreCase);
        if (markerIndex < 0)
            return false;

        var hex = metadataText.Substring(markerIndex + LegacyTgaExifMarker.Length).Trim();
        exifPayload = HexToBytes(hex);
        return exifPayload != null && exifPayload.Length > 0;
    }

    private static bool TryExtractTgaMetadataText(byte[] tgaBytes, out string metadataText)
    {
        metadataText = null;
        if (!HasTgaFooter(tgaBytes))
            return false;

        var footerOffset = tgaBytes.Length - 26;
        var extensionOffset = ReadInt32LittleEndian(tgaBytes, footerOffset);
        if (extensionOffset <= 0 || extensionOffset + 495 > tgaBytes.Length)
            return false;

        var commentOffset = extensionOffset + 2 + 41;
        var commentsBytes = new byte[324];
        Buffer.BlockCopy(tgaBytes, commentOffset, commentsBytes, 0, commentsBytes.Length);
        metadataText = Encoding.ASCII.GetString(commentsBytes).Trim('\0', ' ', '\r', '\n', '\t');
        return !string.IsNullOrWhiteSpace(metadataText);
    }

    private static byte[] InjectPngMetadataChunks(byte[] pngBytes, byte[] exifPayload, RawPhotoParser.RawPhotoData metadata)
    {
        var entries = BuildAiImageMetadataEntries(metadata);
        if ((exifPayload == null || exifPayload.Length == 0) && entries.Count == 0)
            return pngBytes;

        using var ms = new MemoryStream();
        ms.Write(pngBytes, 0, 8);
        var offset = 8;
        var inserted = false;
        while (offset + 12 <= pngBytes.Length)
        {
            var length = ReadUInt32BigEndian(pngBytes, offset);
            var type = Encoding.ASCII.GetString(pngBytes, offset + 4, 4);
            var totalLength = 12 + (int)length;
            if (offset + totalLength > pngBytes.Length)
                return pngBytes;

            if (!inserted &&
                (string.Equals(type, "IDAT", StringComparison.Ordinal) || string.Equals(type, "IEND", StringComparison.Ordinal)))
            {
                if (exifPayload != null && exifPayload.Length > 0)
                    WritePngChunk(ms, "eXIf", exifPayload);
                for (var i = 0; i < entries.Count; i++)
                    WritePngTextChunk(ms, entries[i].Key, entries[i].Value);
                inserted = true;
            }

            ms.Write(pngBytes, offset, totalLength);
            offset += totalLength;
        }

        if (!inserted)
        {
            if (exifPayload != null && exifPayload.Length > 0)
                WritePngChunk(ms, "eXIf", exifPayload);
            for (var i = 0; i < entries.Count; i++)
                WritePngTextChunk(ms, entries[i].Key, entries[i].Value);
        }

        return ms.ToArray();
    }

    private static byte[] InjectJpegExifSegment(byte[] jpegBytes, byte[] exifPayload)
    {
        if (!LooksLikeJpeg(jpegBytes) || exifPayload == null || exifPayload.Length == 0)
            return jpegBytes;

        const int exifHeaderLength = 6;
        var segmentLength = exifPayload.Length + exifHeaderLength + 2;
        if (segmentLength > ushort.MaxValue)
            return jpegBytes;

        var insertOffset = 2;
        while (insertOffset + 4 < jpegBytes.Length && jpegBytes[insertOffset] == 0xFF)
        {
            var marker = jpegBytes[insertOffset + 1];
            if (marker == 0xDA || marker == 0xD9)
                break;

            var blockLength = ReadUInt16BigEndian(jpegBytes, insertOffset + 2);
            if (blockLength < 2 || insertOffset + 2 + blockLength > jpegBytes.Length)
                break;

            if (marker != 0xE0 && marker != 0xE1 && marker != 0xE2 && marker != 0xE3 &&
                marker != 0xE4 && marker != 0xE5 && marker != 0xE6 && marker != 0xE7 &&
                marker != 0xE8 && marker != 0xE9 && marker != 0xEA && marker != 0xEB &&
                marker != 0xEC && marker != 0xED && marker != 0xEE && marker != 0xEF &&
                marker != 0xFE)
            {
                break;
            }

            insertOffset += 2 + blockLength;
        }

        using var ms = new MemoryStream(jpegBytes.Length + exifPayload.Length + 16);
        ms.Write(jpegBytes, 0, insertOffset);
        ms.WriteByte(0xFF);
        ms.WriteByte(0xE1);
        WriteUInt16BigEndian(ms, (ushort)segmentLength);
        ms.Write(Encoding.ASCII.GetBytes("Exif\0\0"), 0, exifHeaderLength);
        ms.Write(exifPayload, 0, exifPayload.Length);
        ms.Write(jpegBytes, insertOffset, jpegBytes.Length - insertOffset);
        return ms.ToArray();
    }

    private static byte[] InjectTgaMetadataExtension(byte[] tgaBytes, RawPhotoParser.RawPhotoData metadata)
    {
        var comment = BuildTgaMetadataComment(metadata);
        if (string.IsNullOrWhiteSpace(comment))
            return tgaBytes;

        var baseLength = HasTgaFooter(tgaBytes) ? tgaBytes.Length - 26 : tgaBytes.Length;
        using var ms = new MemoryStream(baseLength + 495 + 26);
        ms.Write(tgaBytes, 0, baseLength);

        var extensionOffset = (int)ms.Position;
        using (var writer = new BinaryWriter(ms, Encoding.ASCII, true))
        {
            writer.Write((ushort)495);
            WriteFixedAscii(writer, string.Empty, 41);
            WriteFixedAscii(writer, comment, 324);
            writer.Write(new byte[12]);
            WriteFixedAscii(writer, string.Empty, 41);
            writer.Write(new byte[6]);
            WriteFixedAscii(writer, "AIImage", 41);
            writer.Write((ushort)1);
            writer.Write((byte)'0');
            writer.Write(new byte[4]);
            writer.Write(new byte[4]);
            writer.Write(new byte[4]);
            writer.Write((uint)0);
            writer.Write((uint)0);
            writer.Write((uint)0);
            writer.Write((byte)0);
        }

        using (var writer = new BinaryWriter(ms, Encoding.ASCII, true))
        {
            writer.Write(extensionOffset);
            writer.Write(0);
            writer.Write(TgaFooterSignature);
        }

        return ms.ToArray();
    }

    private static void WriteFixedAscii(BinaryWriter writer, string text, int byteCount)
    {
        var bytes = Encoding.ASCII.GetBytes(SanitizeAscii(text) ?? string.Empty);
        var length = Mathf.Min(bytes.Length, byteCount);
        writer.Write(bytes, 0, length);
        for (var i = length; i < byteCount; i++)
            writer.Write((byte)0);
    }

    private static string BuildTgaMetadataComment(RawPhotoParser.RawPhotoData metadata)
    {
        var metadataText = BuildAiImageMetadataText(metadata);
        if (string.IsNullOrWhiteSpace(metadataText))
            return null;

        var asciiBytes = Encoding.ASCII.GetBytes(metadataText);
        var length = Mathf.Min(324, asciiBytes.Length);
        return Encoding.ASCII.GetString(asciiBytes, 0, length);
    }

    private static void WritePngChunk(Stream stream, string type, byte[] data)
    {
        var typeBytes = Encoding.ASCII.GetBytes(type);
        var lengthBytes = GetUInt32BigEndian((uint)(data?.Length ?? 0));
        stream.Write(lengthBytes, 0, 4);
        stream.Write(typeBytes, 0, 4);
        if (data != null && data.Length > 0)
            stream.Write(data, 0, data.Length);

        var crcBuffer = new byte[typeBytes.Length + (data?.Length ?? 0)];
        Buffer.BlockCopy(typeBytes, 0, crcBuffer, 0, typeBytes.Length);
        if (data != null && data.Length > 0)
            Buffer.BlockCopy(data, 0, crcBuffer, typeBytes.Length, data.Length);
        var crc = Crc32(crcBuffer, 0, crcBuffer.Length);
        var crcBytes = GetUInt32BigEndian(crc);
        stream.Write(crcBytes, 0, 4);
    }

    private static void WritePngTextChunk(Stream stream, string keyword, string value)
    {
        if (string.IsNullOrWhiteSpace(keyword) || string.IsNullOrWhiteSpace(value))
            return;

        var keywordBytes = Encoding.ASCII.GetBytes(SanitizeAscii(keyword) ?? string.Empty);
        var valueBytes = Encoding.ASCII.GetBytes(SanitizeAscii(value) ?? string.Empty);
        var data = new byte[keywordBytes.Length + 1 + valueBytes.Length];
        Buffer.BlockCopy(keywordBytes, 0, data, 0, keywordBytes.Length);
        data[keywordBytes.Length] = 0;
        Buffer.BlockCopy(valueBytes, 0, data, keywordBytes.Length + 1, valueBytes.Length);
        WritePngChunk(stream, "tEXt", data);
    }

    private static uint Crc32(byte[] bytes, int offset, int count)
    {
        var crc = 0xFFFFFFFFu;
        for (var i = 0; i < count; i++)
        {
            crc ^= bytes[offset + i];
            for (var bit = 0; bit < 8; bit++)
                crc = (crc & 1) != 0 ? (crc >> 1) ^ 0xEDB88320u : crc >> 1;
        }

        return ~crc;
    }

    private static bool TryReadPngTextMetadata(byte[] pngBytes, out RawPhotoParser.RawPhotoData result)
    {
        result = new RawPhotoParser.RawPhotoData();
        if (!LooksLikePng(pngBytes))
            return false;

        var offset = 8;
        while (offset + 12 <= pngBytes.Length)
        {
            var length = ReadUInt32BigEndian(pngBytes, offset);
            var type = Encoding.ASCII.GetString(pngBytes, offset + 4, 4);
            var dataOffset = offset + 8;
            if (dataOffset + length + 4 > pngBytes.Length)
                break;

            if (string.Equals(type, "tEXt", StringComparison.Ordinal))
                ApplyPngTextEntry(pngBytes, dataOffset, (int)length, result);
            else if (string.Equals(type, "zTXt", StringComparison.Ordinal))
                ApplyCompressedPngTextEntry(pngBytes, dataOffset, (int)length, result);

            offset = dataOffset + (int)length + 4;
            if (string.Equals(type, "IEND", StringComparison.Ordinal))
                break;
        }

        return HasMetadata(result);
    }

    private static void ApplyPngTextEntry(byte[] bytes, int offset, int length, RawPhotoParser.RawPhotoData result)
    {
        var keywordEnd = IndexOf(bytes, new byte[] { 0 }, offset, offset + length);
        if (keywordEnd < 0)
            return;

        var keyword = Encoding.ASCII.GetString(bytes, offset, keywordEnd - offset);
        var text = Encoding.ASCII.GetString(bytes, keywordEnd + 1, offset + length - keywordEnd - 1);
        ApplyAiImageMetadataKey(keyword, text, result);
    }

    private static void ApplyCompressedPngTextEntry(byte[] bytes, int offset, int length, RawPhotoParser.RawPhotoData result)
    {
        var keywordEnd = IndexOf(bytes, new byte[] { 0 }, offset, offset + length);
        if (keywordEnd < 0 || keywordEnd + 2 > offset + length)
            return;

        var keyword = Encoding.ASCII.GetString(bytes, offset, keywordEnd - offset);
        var compressionMethod = bytes[keywordEnd + 1];
        if (compressionMethod != 0)
            return;

        try
        {
            using var input = new MemoryStream(bytes, keywordEnd + 2, offset + length - keywordEnd - 2, false);
            using var zlib = new InflaterInputStream(input, new Inflater(true));
            using var output = new MemoryStream();
            zlib.CopyTo(output);
            var text = Encoding.UTF8.GetString(output.ToArray());
            ApplyAiImageMetadataKey(keyword, text, result);
        }
        catch
        {
        }
    }

    private static RawPhotoParser.RawPhotoData ParseAiImageMetadataText(string text)
    {
        var result = new RawPhotoParser.RawPhotoData();
        if (string.IsNullOrWhiteSpace(text))
            return result;

        var lines = text.Split(new[] { '\r', '\n', '|' }, StringSplitOptions.RemoveEmptyEntries);
        for (var i = 0; i < lines.Length; i++)
        {
            var line = lines[i].Trim();
            if (!line.StartsWith("AIIMAGE_", StringComparison.OrdinalIgnoreCase))
                continue;

            var idx = line.IndexOf('=');
            if (idx <= 0 || idx >= line.Length - 1)
                continue;

            var key = line.Substring(0, idx).Trim();
            var value = line.Substring(idx + 1).Trim();
            ApplyAiImageMetadataKey(key, value, result);
        }

        return result;
    }

    private static List<KeyValuePair<string, string>> BuildAiImageMetadataEntries(RawPhotoParser.RawPhotoData metadata)
    {
        var entries = new List<KeyValuePair<string, string>>();
        if (!HasMetadata(metadata))
            return entries;

        if (metadata.captureTime.HasValue)
        {
            entries.Add(new KeyValuePair<string, string>(
                MetadataCaptureTimeKey,
                metadata.captureTime.Value.ToString("o", CultureInfo.InvariantCulture)));
        }

        if (!string.IsNullOrWhiteSpace(metadata.cameraText))
            entries.Add(new KeyValuePair<string, string>(MetadataCameraKey, metadata.cameraText));
        if (!string.IsNullOrWhiteSpace(metadata.apertureText))
            entries.Add(new KeyValuePair<string, string>(MetadataApertureKey, metadata.apertureText));
        if (!string.IsNullOrWhiteSpace(metadata.locationText))
            entries.Add(new KeyValuePair<string, string>(MetadataLocationKey, metadata.locationText));

        return entries;
    }

    private static string BuildAiImageMetadataText(RawPhotoParser.RawPhotoData metadata)
    {
        var entries = BuildAiImageMetadataEntries(metadata);
        if (entries.Count == 0)
            return null;

        var sb = new StringBuilder();
        for (var i = 0; i < entries.Count; i++)
        {
            if (sb.Length > 0)
                sb.Append('\n');
            sb.Append(entries[i].Key);
            sb.Append('=');
            sb.Append(SanitizeAscii(entries[i].Value));
        }

        return sb.ToString();
    }

    private static void ApplyAiImageMetadataKey(string keyword, string text, RawPhotoParser.RawPhotoData result)
    {
        if (result == null || string.IsNullOrWhiteSpace(keyword))
            return;

        var normalized = keyword.Trim().ToUpperInvariant();
        var value = NormalizeMetadataText(text);
        switch (normalized)
        {
            case MetadataCaptureTimeKey:
                result.captureTime = ParseIsoLikeDate(value);
                break;
            case MetadataCameraKey:
                result.cameraText = value;
                break;
            case MetadataApertureKey:
                result.apertureText = value;
                break;
            case MetadataLocationKey:
                result.locationText = value;
                break;
        }
    }

    private static string NormalizeMetadataText(string text)
    {
        return string.IsNullOrWhiteSpace(text) ? null : text.Trim();
    }

    private static DateTime? ParseIsoLikeDate(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        if (DateTime.TryParseExact(
                text.Trim(),
                new[] { "o", "yyyy-MM-ddTHH:mm:ss", "yyyy-MM-dd HH:mm:ss", "yyyy:MM:dd HH:mm:ss" },
                CultureInfo.InvariantCulture,
                DateTimeStyles.AssumeLocal | DateTimeStyles.AllowWhiteSpaces,
                out var value))
        {
            return value;
        }

        return null;
    }

    private static bool HasMetadata(RawPhotoParser.RawPhotoData data)
    {
        return data != null &&
               (data.captureTime.HasValue ||
                !string.IsNullOrWhiteSpace(data.locationText) ||
                !string.IsNullOrWhiteSpace(data.cameraText) ||
                !string.IsNullOrWhiteSpace(data.apertureText));
    }

    private static bool LooksLikePng(byte[] bytes)
    {
        return bytes != null && bytes.Length >= 8 && MatchesBytes(bytes, 0, PngSignatureBytes);
    }

    private static bool LooksLikeJpeg(byte[] bytes)
    {
        return bytes != null && bytes.Length >= 4 && bytes[0] == 0xFF && bytes[1] == 0xD8;
    }

    private static bool LooksLikeTiff(byte[] bytes)
    {
        return bytes != null && bytes.Length >= 8 &&
               ((bytes[0] == 0x49 && bytes[1] == 0x49 && bytes[2] == 0x2A && bytes[3] == 0x00) ||
                (bytes[0] == 0x4D && bytes[1] == 0x4D && bytes[2] == 0x00 && bytes[3] == 0x2A));
    }

    private static bool LooksLikeHdr(byte[] bytes)
    {
        return StartsWithAscii(bytes, "#?RADIANCE") || StartsWithAscii(bytes, "#?RGBE");
    }

    private static bool HasTgaFooter(byte[] bytes)
    {
        return bytes != null &&
               bytes.Length >= 26 &&
               MatchesBytes(bytes, bytes.Length - 18, TgaFooterSignature);
    }

    private static bool MatchesBytes(byte[] bytes, int offset, byte[] expected)
    {
        if (bytes == null || expected == null || offset < 0 || offset + expected.Length > bytes.Length)
            return false;

        for (var i = 0; i < expected.Length; i++)
        {
            if (bytes[offset + i] != expected[i])
                return false;
        }

        return true;
    }

    private static bool StartsWithAscii(byte[] bytes, string text)
    {
        if (bytes == null || string.IsNullOrEmpty(text) || bytes.Length < text.Length)
            return false;

        for (var i = 0; i < text.Length; i++)
        {
            if (bytes[i] != (byte)text[i])
                return false;
        }

        return true;
    }

    private static int IndexOf(byte[] haystack, byte[] needle, int startIndex, int endExclusive = -1)
    {
        if (haystack == null || needle == null || needle.Length == 0)
            return -1;
        if (endExclusive < 0 || endExclusive > haystack.Length)
            endExclusive = haystack.Length;

        for (var i = startIndex; i <= endExclusive - needle.Length; i++)
        {
            var match = true;
            for (var j = 0; j < needle.Length; j++)
            {
                if (haystack[i + j] != needle[j])
                {
                    match = false;
                    break;
                }
            }

            if (match)
                return i;
        }

        return -1;
    }

    private static bool TryResolveTiffValueDataOffset(
        byte[] bytes,
        int tiffOffset,
        int entryOffset,
        bool littleEndian,
        ushort type,
        uint count,
        out int dataOffset,
        out int byteCount)
    {
        dataOffset = -1;
        byteCount = GetTiffValueByteCount(type, count);
        if (byteCount <= 0)
            return false;

        if (byteCount <= 4)
        {
            dataOffset = entryOffset + 8;
            return dataOffset + byteCount <= bytes.Length;
        }

        var valueOffset = (int)ReadUInt32Tiff(bytes, entryOffset + 8, littleEndian);
        dataOffset = tiffOffset + valueOffset;
        return valueOffset >= 0 && dataOffset >= 0 && dataOffset + byteCount <= bytes.Length;
    }

    private static uint ReadTiffScalarUInt(byte[] bytes, int offset, bool littleEndian, ushort type)
    {
        return type switch
        {
            3 => ReadUInt16Tiff(bytes, offset, littleEndian),
            4 => ReadUInt32Tiff(bytes, offset, littleEndian),
            _ => 0u
        };
    }

    private static ushort[] ReadTiffUShortArray(byte[] bytes, int offset, bool littleEndian, ushort type, uint count)
    {
        if (count == 0 || count > 256)
            return null;

        var values = new ushort[count];
        switch (type)
        {
            case 3:
                for (var i = 0; i < count; i++)
                    values[i] = ReadUInt16Tiff(bytes, offset + i * 2, littleEndian);
                return values;
            case 4:
                for (var i = 0; i < count; i++)
                {
                    var value = ReadUInt32Tiff(bytes, offset + i * 4, littleEndian);
                    if (value > ushort.MaxValue)
                        return null;
                    values[i] = (ushort)value;
                }
                return values;
            default:
                return null;
        }
    }

    private static uint[] ReadTiffUIntArray(byte[] bytes, int offset, bool littleEndian, ushort type, uint count)
    {
        if (count == 0 || count > 4096)
            return null;

        var values = new uint[count];
        switch (type)
        {
            case 3:
                for (var i = 0; i < count; i++)
                    values[i] = ReadUInt16Tiff(bytes, offset + i * 2, littleEndian);
                return values;
            case 4:
                for (var i = 0; i < count; i++)
                    values[i] = ReadUInt32Tiff(bytes, offset + i * 4, littleEndian);
                return values;
            default:
                return null;
        }
    }

    private static int GetTiffValueByteCount(ushort type, uint count)
    {
        var typeSize = type switch
        {
            1 => 1,
            2 => 1,
            3 => 2,
            4 => 4,
            5 => 8,
            7 => 1,
            9 => 4,
            10 => 8,
            _ => 0
        };

        if (typeSize <= 0)
            return 0;

        var total = (long)typeSize * count;
        return total > int.MaxValue ? 0 : (int)total;
    }

    private static ushort ReadUInt16BigEndian(byte[] bytes, int offset)
    {
        return (ushort)((bytes[offset] << 8) | bytes[offset + 1]);
    }

    private static uint ReadUInt32BigEndian(byte[] bytes, int offset)
    {
        return (uint)((bytes[offset] << 24) |
                      (bytes[offset + 1] << 16) |
                      (bytes[offset + 2] << 8) |
                      bytes[offset + 3]);
    }

    private static ushort ReadUInt16LittleEndian(byte[] bytes, int offset)
    {
        return (ushort)(bytes[offset] | (bytes[offset + 1] << 8));
    }

    private static int ReadInt32LittleEndian(byte[] bytes, int offset)
    {
        return bytes[offset] |
               (bytes[offset + 1] << 8) |
               (bytes[offset + 2] << 16) |
               (bytes[offset + 3] << 24);
    }

    private static ushort ReadUInt16Tiff(byte[] bytes, int offset, bool littleEndian)
    {
        return littleEndian
            ? (ushort)(bytes[offset] | (bytes[offset + 1] << 8))
            : (ushort)((bytes[offset] << 8) | bytes[offset + 1]);
    }

    private static uint ReadUInt32Tiff(byte[] bytes, int offset, bool littleEndian)
    {
        if (littleEndian)
        {
            return (uint)(bytes[offset] |
                          (bytes[offset + 1] << 8) |
                          (bytes[offset + 2] << 16) |
                          (bytes[offset + 3] << 24));
        }

        return (uint)((bytes[offset] << 24) |
                      (bytes[offset + 1] << 16) |
                      (bytes[offset + 2] << 8) |
                      bytes[offset + 3]);
    }

    private static byte[] GetUInt32BigEndian(uint value)
    {
        return new[]
        {
            (byte)(value >> 24),
            (byte)(value >> 16),
            (byte)(value >> 8),
            (byte)value
        };
    }

    private static void WriteUInt16BigEndian(Stream stream, ushort value)
    {
        stream.WriteByte((byte)(value >> 8));
        stream.WriteByte((byte)value);
    }

    private static string SanitizeAscii(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        var trimmed = text.Trim();
        var sb = new StringBuilder(trimmed.Length);
        for (var i = 0; i < trimmed.Length; i++)
        {
            var c = trimmed[i];
            if (c == '\0')
                continue;
            if (c == '\r' || c == '\n' || c == '\t')
            {
                sb.Append(' ');
                continue;
            }

            sb.Append(c <= 0x7F ? c : '?');
        }

        var result = sb.ToString().Trim();
        return string.IsNullOrWhiteSpace(result) ? null : result;
    }

    private static byte[] HexToBytes(string hex)
    {
        if (string.IsNullOrWhiteSpace(hex))
            return null;

        hex = hex.Trim();
        if ((hex.Length & 1) != 0)
            return null;

        var bytes = new byte[hex.Length / 2];
        for (var i = 0; i < bytes.Length; i++)
        {
            if (!byte.TryParse(hex.Substring(i * 2, 2), NumberStyles.HexNumber, CultureInfo.InvariantCulture, out bytes[i]))
                return null;
        }

        return bytes;
    }
}
