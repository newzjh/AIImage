using System;
using System.Globalization;
using System.IO;
using System.Collections.Generic;
using UnityEngine;

public static class RawPhotoParser
{
    private const string FujiRafHeader = "FUJIFILMCCD-RAW";

    public sealed class RawPhotoData
    {
        public byte[] previewBytes;
        public bool isSensorDecoded;
        public DateTime? captureTime;
        public string locationText;
        public string cameraText;
        public string apertureText;
    }

    private sealed class ExifMetadata
    {
        public DateTime? captureTime;
        public string make;
        public string model;
        public double? aperture;
        public string gpsLatitudeRef;
        public double[] gpsLatitude;
        public string gpsLongitudeRef;
        public double[] gpsLongitude;
    }

    public static bool IsRawExtension(string filePath)
    {
        var ext = (Path.GetExtension(filePath) ?? string.Empty).ToLowerInvariant();
        return ext == ".raw" ||
               ext == ".cr2" ||
               ext == ".cr3" ||
               ext == ".nef" ||
               ext == ".arw" ||
               ext == ".dng" ||
               ext == ".raf" ||
               ext == ".rw2" ||
               ext == ".orf" ||
               ext == ".srw" ||
               ext == ".pef";
    }

    public static bool TryParse(string filePath, out RawPhotoData result)
    {
        result = new RawPhotoData();
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return false;

        byte[] bytes;
        try
        {
            bytes = File.ReadAllBytes(filePath);
        }
        catch
        {
            return false;
        }

        if (bytes == null || bytes.Length < 4)
            return false;

        if (string.Equals(Path.GetExtension(filePath), ".dng", StringComparison.OrdinalIgnoreCase) &&
            TryDecodeUncompressedDng(bytes, out var decodedDngPreview))
        {
            result.previewBytes = decodedDngPreview;
            result.isSensorDecoded = true;
        }
        else if (TryExtractPreviewImage(bytes, out var previewBytes))
        {
            result.previewBytes = previewBytes;
            if (TryParseJpegExif(previewBytes, out var jpegExif))
                ApplyMetadata(result, jpegExif);
        }

        if (TryParseFirstTiffExif(bytes, out var fileExif))
            ApplyMissingMetadata(result, fileExif);

        return (result.previewBytes != null && result.previewBytes.Length > 0) ||
               result.captureTime.HasValue ||
               !string.IsNullOrWhiteSpace(result.locationText) ||
               !string.IsNullOrWhiteSpace(result.cameraText) ||
               !string.IsNullOrWhiteSpace(result.apertureText);
    }

    public static bool TryLoadDisplayBytes(string filePath, out byte[] imageBytes)
    {
        return TryLoadDisplayBytes(filePath, out imageBytes, out _);
    }

    public static bool TryLoadDisplayBytes(string filePath, out byte[] imageBytes, out RawPhotoData rawPhoto)
    {
        imageBytes = null;
        rawPhoto = null;

        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return false;

        if (IsRawExtension(filePath))
        {
            if (!TryParse(filePath, out rawPhoto) ||
                rawPhoto?.previewBytes == null ||
                rawPhoto.previewBytes.Length == 0)
            {
                return false;
            }

            imageBytes = rawPhoto.previewBytes;
            return true;
        }

        return StandardImageIO.TryLoadDisplayBytes(filePath, out imageBytes, out rawPhoto);
    }

    private struct TiffImageEntry
    {
        public ushort tag;
        public ushort type;
        public uint count;
        public uint valueOffset;
        public int entryOffset;
    }

    private sealed class UncompressedDng
    {
        public int width;
        public int height;
        public bool littleEndian;
        public ushort[] samples;
        public int[] cfaPattern = { 0, 1, 1, 2 };
        public float blackLevel;
        public float whiteLevel = 65535f;
    }

    private static bool TryDecodeUncompressedDng(byte[] bytes, out byte[] pngBytes)
    {
        pngBytes = null;
        if (!TryReadUncompressedDng(bytes, out var dng))
            return false;

        Texture2D texture = null;
        try
        {
            var pixels = new Color32[dng.width * dng.height];
            var whiteBalance = EstimateDngWhiteBalance(dng);
            for (var y = 0; y < dng.height; y++)
            {
                for (var x = 0; x < dng.width; x++)
                {
                    var rgb = new Vector3(
                        SampleDngChannel(dng, x, y, 0),
                        SampleDngChannel(dng, x, y, 1),
                        SampleDngChannel(dng, x, y, 2));
                    rgb = new Vector3(rgb.x * whiteBalance.x, rgb.y * whiteBalance.y, rgb.z * whiteBalance.z);
                    var max = Mathf.Max(rgb.x, Mathf.Max(rgb.y, rgb.z));
                    if (max > 1f)
                        rgb /= max;
                    rgb = new Vector3(
                        Mathf.Pow(Mathf.Clamp01(rgb.x), 1f / 2.2f),
                        Mathf.Pow(Mathf.Clamp01(rgb.y), 1f / 2.2f),
                        Mathf.Pow(Mathf.Clamp01(rgb.z), 1f / 2.2f));
                    pixels[y * dng.width + x] = new Color(rgb.x, rgb.y, rgb.z, 1f);
                }
            }

            texture = new Texture2D(dng.width, dng.height, TextureFormat.RGBA32, false, true);
            texture.SetPixels32(pixels);
            texture.Apply(false, false);
            pngBytes = texture.EncodeToPNG();
            return pngBytes != null && pngBytes.Length > 0;
        }
        catch
        {
            pngBytes = null;
            return false;
        }
        finally
        {
            if (texture != null)
                UnityEngine.Object.Destroy(texture);
        }
    }

    private static Vector3 EstimateDngWhiteBalance(UncompressedDng dng)
    {
        var sums = Vector3.zero;
        var counts = Vector3.zero;
        var sampleStride = Mathf.Max(1, Mathf.Min(dng.width, dng.height) / 256);
        for (var y = 0; y < dng.height; y += sampleStride)
        {
            for (var x = 0; x < dng.width; x += sampleStride)
            {
                var channel = GetDngCfaChannel(dng, x, y);
                var value = ReadDngSample(dng, x, y);
                if (channel == 0)
                {
                    sums.x += value;
                    counts.x++;
                }
                else if (channel == 1)
                {
                    sums.y += value;
                    counts.y++;
                }
                else if (channel == 2)
                {
                    sums.z += value;
                    counts.z++;
                }
            }
        }

        var averages = new Vector3(
            sums.x / Mathf.Max(1f, counts.x),
            sums.y / Mathf.Max(1f, counts.y),
            sums.z / Mathf.Max(1f, counts.z));
        var target = Mathf.Max(0.05f, averages.y);
        return new Vector3(
            Mathf.Clamp(target / Mathf.Max(0.02f, averages.x), 0.5f, 2.5f),
            1f,
            Mathf.Clamp(target / Mathf.Max(0.02f, averages.z), 0.5f, 2.5f));
    }

    private static float SampleDngChannel(UncompressedDng dng, int x, int y, int channel)
    {
        if (GetDngCfaChannel(dng, x, y) == channel)
            return ReadDngSample(dng, x, y);

        var sum = 0f;
        var weight = 0f;
        for (var dy = -2; dy <= 2; dy++)
        {
            for (var dx = -2; dx <= 2; dx++)
            {
                if (dx == 0 && dy == 0)
                    continue;
                var sampleX = Mathf.Clamp(x + dx, 0, dng.width - 1);
                var sampleY = Mathf.Clamp(y + dy, 0, dng.height - 1);
                if (GetDngCfaChannel(dng, sampleX, sampleY) != channel)
                    continue;
                var distanceSquared = dx * dx + dy * dy;
                var sampleWeight = 1f / Mathf.Max(1f, distanceSquared);
                sum += ReadDngSample(dng, sampleX, sampleY) * sampleWeight;
                weight += sampleWeight;
            }
        }

        return weight > 0f ? sum / weight : ReadDngSample(dng, x, y);
    }

    private static int GetDngCfaChannel(UncompressedDng dng, int x, int y)
    {
        return dng.cfaPattern[(y & 1) * 2 + (x & 1)];
    }

    private static float ReadDngSample(UncompressedDng dng, int x, int y)
    {
        var sample = dng.samples[y * dng.width + x];
        return Mathf.Clamp01((sample - dng.blackLevel) / Mathf.Max(1f, dng.whiteLevel - dng.blackLevel));
    }

    private static bool TryReadUncompressedDng(byte[] bytes, out UncompressedDng result)
    {
        result = null;
        if (!LooksLikeTiffHeader(bytes, 0))
            return false;

        var littleEndian = bytes[0] == (byte)'I';
        var firstIfd = ReadUInt32(bytes, 4, littleEndian);
        if (firstIfd == 0 || firstIfd > int.MaxValue)
            return false;

        var pendingIfdOffsets = new Queue<uint>();
        pendingIfdOffsets.Enqueue(firstIfd);
        var visitedIfdOffsets = new HashSet<uint>();
        while (pendingIfdOffsets.Count > 0)
        {
            var ifdOffset = pendingIfdOffsets.Dequeue();
            if (!visitedIfdOffsets.Add(ifdOffset) || ifdOffset > int.MaxValue)
                continue;
            if (!TryReadTiffImageEntries(bytes, (int)ifdOffset, littleEndian, out var entries, out var nextIfdOffset))
                continue;

            if (TryGetTiffEntry(entries, 330, out var subIfdEntry) &&
                TryGetTiffUnsignedValues(bytes, subIfdEntry, littleEndian, out var subIfdOffsets))
            {
                foreach (var subIfdOffset in subIfdOffsets)
                    pendingIfdOffsets.Enqueue(subIfdOffset);
            }
            if (nextIfdOffset != 0)
                pendingIfdOffsets.Enqueue(nextIfdOffset);

            if (TryBuildUncompressedDng(bytes, entries, littleEndian, out result))
                return true;
        }

        return false;
    }

    private static bool TryReadTiffImageEntries(byte[] bytes, int ifdOffset, bool littleEndian, out List<TiffImageEntry> entries, out uint nextIfdOffset)
    {
        entries = new List<TiffImageEntry>();
        nextIfdOffset = 0;
        if (ifdOffset < 0 || ifdOffset + 2 > bytes.Length)
            return false;

        var entryCount = ReadUInt16(bytes, ifdOffset, littleEndian);
        var entriesEnd = ifdOffset + 2 + entryCount * 12;
        if (entriesEnd + 4 > bytes.Length)
            return false;

        for (var i = 0; i < entryCount; i++)
        {
            var entryOffset = ifdOffset + 2 + i * 12;
            entries.Add(new TiffImageEntry
            {
                tag = ReadUInt16(bytes, entryOffset, littleEndian),
                type = ReadUInt16(bytes, entryOffset + 2, littleEndian),
                count = ReadUInt32(bytes, entryOffset + 4, littleEndian),
                valueOffset = ReadUInt32(bytes, entryOffset + 8, littleEndian),
                entryOffset = entryOffset
            });
        }

        nextIfdOffset = ReadUInt32(bytes, entriesEnd, littleEndian);
        return true;
    }

    private static bool TryBuildUncompressedDng(byte[] bytes, List<TiffImageEntry> entries, bool littleEndian, out UncompressedDng result)
    {
        result = null;
        if (!TryGetTiffUnsignedValue(bytes, entries, 256, littleEndian, out var width) ||
            !TryGetTiffUnsignedValue(bytes, entries, 257, littleEndian, out var height) ||
            !TryGetTiffUnsignedValue(bytes, entries, 259, littleEndian, out var compression) ||
            !TryGetTiffUnsignedValue(bytes, entries, 262, littleEndian, out var photometric) ||
            !TryGetTiffUnsignedValue(bytes, entries, 277, littleEndian, out var samplesPerPixel) ||
            !TryGetTiffUnsignedValue(bytes, entries, 278, littleEndian, out var rowsPerStrip) ||
            !TryGetTiffEntry(entries, 273, out var stripOffsetsEntry) ||
            !TryGetTiffEntry(entries, 279, out var stripByteCountsEntry) ||
            !TryGetTiffEntry(entries, 258, out var bitsPerSampleEntry) ||
            width == 0 || height == 0 || width > 16000 || height > 16000 ||
            compression != 1 || photometric != 32803 || samplesPerPixel != 1 || rowsPerStrip == 0)
        {
            return false;
        }

        if (!TryGetTiffUnsignedValues(bytes, bitsPerSampleEntry, littleEndian, out var bitsPerSample) ||
            bitsPerSample.Length != 1 || bitsPerSample[0] != 16 ||
            !TryGetTiffUnsignedValues(bytes, stripOffsetsEntry, littleEndian, out var stripOffsets) ||
            !TryGetTiffUnsignedValues(bytes, stripByteCountsEntry, littleEndian, out var stripByteCounts) ||
            stripOffsets.Length == 0 || stripOffsets.Length != stripByteCounts.Length)
        {
            return false;
        }

        if (TryGetTiffEntry(entries, 33421, out var cfaDimensionsEntry) &&
            (!TryGetTiffUnsignedValues(bytes, cfaDimensionsEntry, littleEndian, out var cfaDimensions) ||
             cfaDimensions.Length < 2 ||
             cfaDimensions[0] != 2 ||
             cfaDimensions[1] != 2))
        {
            return false;
        }

        var cfaPattern = new[] { 0, 1, 1, 2 };
        if (TryGetTiffEntry(entries, 33422, out var cfaPatternEntry) &&
            TryGetTiffUnsignedValues(bytes, cfaPatternEntry, littleEndian, out var sourceCfaPattern) &&
            sourceCfaPattern.Length >= 4)
        {
            for (var i = 0; i < 4; i++)
            {
                if (sourceCfaPattern[i] > 2)
                    return false;
                cfaPattern[i] = (int)sourceCfaPattern[i];
            }
        }

        var pixelCount = (long)width * height;
        if (pixelCount > 128L * 1024L * 1024L)
            return false;

        var samples = new ushort[(int)pixelCount];
        var currentRow = 0;
        for (var stripIndex = 0; stripIndex < stripOffsets.Length && currentRow < height; stripIndex++)
        {
            var stripRows = (int)Math.Min(rowsPerStrip, height - currentRow);
            var expectedByteCount = (long)stripRows * width * 2;
            if (stripOffsets[stripIndex] > int.MaxValue ||
                stripByteCounts[stripIndex] < expectedByteCount ||
                stripOffsets[stripIndex] + expectedByteCount > bytes.Length)
            {
                return false;
            }

            var dataOffset = (int)stripOffsets[stripIndex];
            for (var row = 0; row < stripRows; row++)
            {
                for (var column = 0; column < width; column++)
                    samples[(currentRow + row) * (int)width + column] = ReadUInt16(bytes, dataOffset + (row * (int)width + column) * 2, littleEndian);
            }
            currentRow += stripRows;
        }
        if (currentRow != height)
            return false;

        result = new UncompressedDng
        {
            width = (int)width,
            height = (int)height,
            littleEndian = littleEndian,
            samples = samples,
            cfaPattern = cfaPattern
        };
        if (TryGetTiffUnsignedValue(bytes, entries, 50714, littleEndian, out var blackLevel))
            result.blackLevel = blackLevel;
        if (TryGetTiffUnsignedValue(bytes, entries, 50717, littleEndian, out var whiteLevel))
            result.whiteLevel = whiteLevel;
        return result.whiteLevel > result.blackLevel;
    }

    private static bool TryGetTiffEntry(List<TiffImageEntry> entries, ushort tag, out TiffImageEntry entry)
    {
        for (var i = 0; i < entries.Count; i++)
        {
            if (entries[i].tag == tag)
            {
                entry = entries[i];
                return true;
            }
        }

        entry = default;
        return false;
    }

    private static bool TryGetTiffUnsignedValue(byte[] bytes, List<TiffImageEntry> entries, ushort tag, bool littleEndian, out uint value)
    {
        value = 0;
        if (!TryGetTiffEntry(entries, tag, out var entry) ||
            !TryGetTiffUnsignedValues(bytes, entry, littleEndian, out var values) ||
            values.Length == 0)
        {
            return false;
        }

        value = values[0];
        return true;
    }

    private static bool TryGetTiffUnsignedValues(byte[] bytes, TiffImageEntry entry, bool littleEndian, out uint[] values)
    {
        values = null;
        var elementSize = entry.type == 1 ? 1 : entry.type == 3 ? 2 : entry.type == 4 ? 4 : 0;
        if (elementSize == 0 || entry.count == 0 || entry.count > 1024)
            return false;

        var byteCount = (long)elementSize * entry.count;
        var dataOffset = byteCount <= 4 ? entry.entryOffset + 8 : (long)entry.valueOffset;
        if (dataOffset < 0 || dataOffset + byteCount > bytes.Length)
            return false;

        values = new uint[entry.count];
        for (var i = 0; i < values.Length; i++)
        {
            var offset = (int)dataOffset + i * elementSize;
            values[i] = elementSize == 1
                ? bytes[offset]
                : elementSize == 2
                    ? ReadUInt16(bytes, offset, littleEndian)
                    : ReadUInt32(bytes, offset, littleEndian);
        }
        return true;
    }

    public static bool TryReadMetadata(string filePath, out RawPhotoData result)
    {
        result = null;
        if (string.IsNullOrWhiteSpace(filePath) || !File.Exists(filePath))
            return false;

        if (IsRawExtension(filePath))
            return TryParse(filePath, out result);

        return StandardImageIO.TryReadMetadata(filePath, out result);
    }

    internal static bool TryReadJpegOrTiffMetadataFromBytes(byte[] imageBytes, out RawPhotoData result)
    {
        result = TryReadStandardPhotoMetadata(imageBytes);
        return HasMetadata(result);
    }

    internal static bool TryReadTiffMetadataFromBytes(byte[] exifTiffBytes, out RawPhotoData result)
    {
        result = null;
        if (exifTiffBytes == null || exifTiffBytes.Length < 8)
            return false;

        if (!TryParseTiffExif(exifTiffBytes, 0, out var metadata))
            return false;

        result = new RawPhotoData();
        ApplyMetadata(result, metadata);
        return HasMetadata(result);
    }

    private static void ApplyMetadata(RawPhotoData target, ExifMetadata source)
    {
        if (target == null || source == null)
            return;

        target.captureTime = source.captureTime;
        target.cameraText = BuildCameraText(source.make, source.model);
        target.apertureText = FormatAperture(source.aperture);
        target.locationText = BuildGpsText(source.gpsLatitudeRef, source.gpsLatitude, source.gpsLongitudeRef, source.gpsLongitude);
    }

    private static void ApplyMissingMetadata(RawPhotoData target, ExifMetadata source)
    {
        if (target == null || source == null)
            return;

        target.captureTime ??= source.captureTime;
        if (string.IsNullOrWhiteSpace(target.cameraText))
            target.cameraText = BuildCameraText(source.make, source.model);
        if (string.IsNullOrWhiteSpace(target.apertureText))
            target.apertureText = FormatAperture(source.aperture);
        if (string.IsNullOrWhiteSpace(target.locationText))
            target.locationText = BuildGpsText(source.gpsLatitudeRef, source.gpsLatitude, source.gpsLongitudeRef, source.gpsLongitude);
    }

    private static RawPhotoData TryReadStandardPhotoMetadata(byte[] imageBytes)
    {
        if (imageBytes == null || imageBytes.Length < 4)
            return null;

        ExifMetadata metadata = null;
        if (!TryParseJpegExif(imageBytes, out metadata) &&
            !TryParseFirstTiffExif(imageBytes, out metadata))
        {
            return null;
        }

        var result = new RawPhotoData();
        ApplyMetadata(result, metadata);
        return HasMetadata(result) ? result : null;
    }

    private static bool HasMetadata(RawPhotoData data)
    {
        return data != null &&
               (data.captureTime.HasValue ||
                !string.IsNullOrWhiteSpace(data.locationText) ||
                !string.IsNullOrWhiteSpace(data.cameraText) ||
                !string.IsNullOrWhiteSpace(data.apertureText));
    }

    private static bool TryExtractPreviewImage(byte[] bytes, out byte[] previewBytes)
    {
        previewBytes = null;
        if (bytes == null || bytes.Length < 4)
            return false;

        if (TryExtractFujiRafPreview(bytes, out previewBytes))
            return true;

        return TryExtractLargestEmbeddedJpeg(bytes, out previewBytes);
    }

    private static bool TryExtractFujiRafPreview(byte[] bytes, out byte[] jpegBytes)
    {
        jpegBytes = null;
        if (bytes == null || bytes.Length < 112)
            return false;

        if (!StartsWithAscii(bytes, FujiRafHeader))
            return false;

        var jpegOffset = (int)ReadUInt32BigEndian(bytes, 84);
        var jpegLength = (int)ReadUInt32BigEndian(bytes, 88);
        if (!TrySliceJpeg(bytes, jpegOffset, jpegLength, out jpegBytes))
            return false;

        return true;
    }

    private static bool TryExtractLargestEmbeddedJpeg(byte[] bytes, out byte[] jpegBytes)
    {
        jpegBytes = null;
        if (bytes == null || bytes.Length < 4)
            return false;

        var bestStart = -1;
        var bestLength = 0;
        for (var i = 0; i < bytes.Length - 2; i++)
        {
            if (bytes[i] != 0xFF || bytes[i + 1] != 0xD8 || bytes[i + 2] != 0xFF)
                continue;

            var end = FindJpegEnd(bytes, i + 2);
            if (end <= i)
                continue;

            var length = end - i;
            if (length > bestLength)
            {
                bestStart = i;
                bestLength = length;
            }

            i = Math.Max(i, end - 1);
        }

        if (bestStart < 0 || bestLength <= 0)
            return false;

        jpegBytes = new byte[bestLength];
        Buffer.BlockCopy(bytes, bestStart, jpegBytes, 0, bestLength);
        return true;
    }

    private static bool TrySliceJpeg(byte[] bytes, int offset, int length, out byte[] jpegBytes)
    {
        jpegBytes = null;
        if (bytes == null || offset < 0 || length <= 0 || offset + length > bytes.Length)
            return false;
        if (length < 4 || bytes[offset] != 0xFF || bytes[offset + 1] != 0xD8)
            return false;

        var end = offset + length;
        if (bytes[end - 2] != 0xFF || bytes[end - 1] != 0xD9)
        {
            var fallbackEnd = FindJpegEnd(bytes, offset + 2);
            if (fallbackEnd <= offset)
                return false;

            end = fallbackEnd;
            length = end - offset;
        }

        jpegBytes = new byte[length];
        Buffer.BlockCopy(bytes, offset, jpegBytes, 0, length);
        return true;
    }

    private static int FindJpegEnd(byte[] bytes, int start)
    {
        for (var i = Math.Max(0, start); i < bytes.Length - 1; i++)
        {
            if (bytes[i] == 0xFF && bytes[i + 1] == 0xD9)
                return i + 2;
        }

        return -1;
    }

    private static bool TryParseJpegExif(byte[] jpegBytes, out ExifMetadata metadata)
    {
        metadata = null;
        if (jpegBytes == null || jpegBytes.Length < 10 || jpegBytes[0] != 0xFF || jpegBytes[1] != 0xD8)
            return false;

        var offset = 2;
        while (offset + 4 < jpegBytes.Length)
        {
            if (jpegBytes[offset] != 0xFF)
                break;

            var marker = jpegBytes[offset + 1];
            if (marker == 0xD9 || marker == 0xDA)
                break;

            if (offset + 4 > jpegBytes.Length)
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
                var tiffOffset = offset + 10;
                return TryParseTiffExif(jpegBytes, tiffOffset, out metadata);
            }

            offset += 2 + segmentLength;
        }

        return false;
    }

    private static bool TryParseFirstTiffExif(byte[] bytes, out ExifMetadata metadata)
    {
        metadata = null;
        if (bytes == null || bytes.Length < 8)
            return false;

        for (var i = 0; i < bytes.Length - 7; i++)
        {
            if (!LooksLikeTiffHeader(bytes, i))
                continue;

            if (TryParseTiffExif(bytes, i, out metadata))
                return true;
        }

        return false;
    }

    private static bool LooksLikeTiffHeader(byte[] bytes, int offset)
    {
        return offset + 7 < bytes.Length &&
               ((bytes[offset] == 0x49 && bytes[offset + 1] == 0x49 && bytes[offset + 2] == 0x2A && bytes[offset + 3] == 0x00) ||
                (bytes[offset] == 0x4D && bytes[offset + 1] == 0x4D && bytes[offset + 2] == 0x00 && bytes[offset + 3] == 0x2A));
    }

    private static bool TryParseTiffExif(byte[] bytes, int tiffOffset, out ExifMetadata metadata)
    {
        metadata = new ExifMetadata();
        if (!LooksLikeTiffHeader(bytes, tiffOffset))
            return false;

        var littleEndian = bytes[tiffOffset] == 0x49;
        var ifd0Relative = ReadUInt32(bytes, tiffOffset + 4, littleEndian);
        if (ifd0Relative <= 0 || tiffOffset + ifd0Relative >= bytes.Length)
            return false;

        uint exifIfdRelative = 0;
        uint gpsIfdRelative = 0;
        if (!TryReadIfd(bytes, tiffOffset, tiffOffset + (int)ifd0Relative, littleEndian, metadata, out exifIfdRelative, out gpsIfdRelative))
            return false;

        if (exifIfdRelative > 0 && tiffOffset + exifIfdRelative < bytes.Length)
            TryReadIfd(bytes, tiffOffset, tiffOffset + (int)exifIfdRelative, littleEndian, metadata, out _, out _);

        if (gpsIfdRelative > 0 && tiffOffset + gpsIfdRelative < bytes.Length)
            TryReadGpsIfd(bytes, tiffOffset, tiffOffset + (int)gpsIfdRelative, littleEndian, metadata);

        return metadata.captureTime.HasValue ||
               !string.IsNullOrWhiteSpace(metadata.make) ||
               !string.IsNullOrWhiteSpace(metadata.model) ||
               metadata.aperture.HasValue ||
               (metadata.gpsLatitude != null && metadata.gpsLongitude != null);
    }

    private static bool TryReadIfd(
        byte[] bytes,
        int tiffOffset,
        int ifdOffset,
        bool littleEndian,
        ExifMetadata metadata,
        out uint exifIfdRelative,
        out uint gpsIfdRelative)
    {
        exifIfdRelative = 0;
        gpsIfdRelative = 0;
        if (ifdOffset < 0 || ifdOffset + 2 > bytes.Length)
            return false;

        var entryCount = ReadUInt16(bytes, ifdOffset, littleEndian);
        if (entryCount <= 0 || entryCount > 512)
            return false;

        var valid = false;
        for (var i = 0; i < entryCount; i++)
        {
            var entryOffset = ifdOffset + 2 + i * 12;
            if (entryOffset + 12 > bytes.Length)
                return false;

            var tag = ReadUInt16(bytes, entryOffset, littleEndian);
            var type = ReadUInt16(bytes, entryOffset + 2, littleEndian);
            var valueCount = ReadUInt32(bytes, entryOffset + 4, littleEndian);
            var valueOffset = ReadUInt32(bytes, entryOffset + 8, littleEndian);
            var valueByteCount = GetValueByteCount(type, valueCount);
            if (valueByteCount <= 0)
                continue;

            int dataOffset;
            if (valueByteCount <= 4)
            {
                dataOffset = entryOffset + 8;
            }
            else
            {
                dataOffset = tiffOffset + (int)valueOffset;
                if (dataOffset < 0 || dataOffset + valueByteCount > bytes.Length)
                    continue;
            }

            switch (tag)
            {
                case 0x010F:
                    metadata.make ??= ReadAscii(bytes, dataOffset, valueByteCount);
                    valid = true;
                    break;
                case 0x0110:
                    metadata.model ??= ReadAscii(bytes, dataOffset, valueByteCount);
                    valid = true;
                    break;
                case 0x0132:
                    metadata.captureTime ??= ParseExifDate(ReadAscii(bytes, dataOffset, valueByteCount));
                    valid = true;
                    break;
                case 0x829D:
                    metadata.aperture ??= ReadRationalValue(bytes, dataOffset, littleEndian, type);
                    valid = true;
                    break;
                case 0x8769:
                    exifIfdRelative = valueOffset;
                    valid = true;
                    break;
                case 0x8825:
                    gpsIfdRelative = valueOffset;
                    valid = true;
                    break;
                case 0x9003:
                    metadata.captureTime ??= ParseExifDate(ReadAscii(bytes, dataOffset, valueByteCount));
                    valid = true;
                    break;
            }
        }

        return valid;
    }

    private static void TryReadGpsIfd(byte[] bytes, int tiffOffset, int ifdOffset, bool littleEndian, ExifMetadata metadata)
    {
        if (ifdOffset < 0 || ifdOffset + 2 > bytes.Length)
            return;

        var entryCount = ReadUInt16(bytes, ifdOffset, littleEndian);
        if (entryCount <= 0 || entryCount > 256)
            return;

        for (var i = 0; i < entryCount; i++)
        {
            var entryOffset = ifdOffset + 2 + i * 12;
            if (entryOffset + 12 > bytes.Length)
                return;

            var tag = ReadUInt16(bytes, entryOffset, littleEndian);
            var type = ReadUInt16(bytes, entryOffset + 2, littleEndian);
            var valueCount = ReadUInt32(bytes, entryOffset + 4, littleEndian);
            var valueOffset = ReadUInt32(bytes, entryOffset + 8, littleEndian);
            var valueByteCount = GetValueByteCount(type, valueCount);
            if (valueByteCount <= 0)
                continue;

            int dataOffset;
            if (valueByteCount <= 4)
            {
                dataOffset = entryOffset + 8;
            }
            else
            {
                dataOffset = tiffOffset + (int)valueOffset;
                if (dataOffset < 0 || dataOffset + valueByteCount > bytes.Length)
                    continue;
            }

            switch (tag)
            {
                case 0x0001:
                    metadata.gpsLatitudeRef ??= ReadAscii(bytes, dataOffset, valueByteCount);
                    break;
                case 0x0002:
                    metadata.gpsLatitude ??= ReadRationalArray(bytes, dataOffset, littleEndian, valueCount, type);
                    break;
                case 0x0003:
                    metadata.gpsLongitudeRef ??= ReadAscii(bytes, dataOffset, valueByteCount);
                    break;
                case 0x0004:
                    metadata.gpsLongitude ??= ReadRationalArray(bytes, dataOffset, littleEndian, valueCount, type);
                    break;
            }
        }
    }

    private static int GetValueByteCount(ushort type, uint count)
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

    private static string ReadAscii(byte[] bytes, int offset, int byteCount)
    {
        if (bytes == null || byteCount <= 0 || offset < 0 || offset + byteCount > bytes.Length)
            return null;

        var text = System.Text.Encoding.ASCII.GetString(bytes, offset, byteCount);
        return string.IsNullOrWhiteSpace(text) ? null : text.Trim('\0', ' ', '\t', '\r', '\n');
    }

    private static double? ReadRationalValue(byte[] bytes, int offset, bool littleEndian, ushort type)
    {
        if (bytes == null || offset < 0 || offset + 8 > bytes.Length)
            return null;

        if (type == 5)
        {
            var numerator = ReadUInt32(bytes, offset, littleEndian);
            var denominator = ReadUInt32(bytes, offset + 4, littleEndian);
            if (denominator == 0)
                return null;
            return numerator / (double)denominator;
        }

        if (type == 10)
        {
            var numerator = ReadInt32(bytes, offset, littleEndian);
            var denominator = ReadInt32(bytes, offset + 4, littleEndian);
            if (denominator == 0)
                return null;
            return numerator / (double)denominator;
        }

        return null;
    }

    private static double[] ReadRationalArray(byte[] bytes, int offset, bool littleEndian, uint count, ushort type)
    {
        if (count == 0 || count > 32)
            return null;

        var values = new double[count];
        for (var i = 0; i < count; i++)
        {
            var value = ReadRationalValue(bytes, offset + i * 8, littleEndian, type);
            if (!value.HasValue)
                return null;
            values[i] = value.Value;
        }

        return values;
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

    private static ushort ReadUInt16(byte[] bytes, int offset, bool littleEndian)
    {
        return littleEndian
            ? (ushort)(bytes[offset] | (bytes[offset + 1] << 8))
            : (ushort)((bytes[offset] << 8) | bytes[offset + 1]);
    }

    private static uint ReadUInt32(byte[] bytes, int offset, bool littleEndian)
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

    private static int ReadInt32(byte[] bytes, int offset, bool littleEndian)
    {
        return unchecked((int)ReadUInt32(bytes, offset, littleEndian));
    }

    private static DateTime? ParseExifDate(string text)
    {
        if (string.IsNullOrWhiteSpace(text))
            return null;

        if (DateTime.TryParseExact(
                text.Trim(),
                new[] { "yyyy:MM:dd HH:mm:ss", "yyyy-MM-dd HH:mm:ss" },
                CultureInfo.InvariantCulture,
                DateTimeStyles.None,
                out var value))
        {
            return value;
        }

        return null;
    }

    private static string BuildCameraText(string make, string model)
    {
        make = NormalizeText(make);
        model = NormalizeText(model);
        if (string.IsNullOrWhiteSpace(model))
            return make;
        if (string.IsNullOrWhiteSpace(make))
            return model;
        if (model.StartsWith(make, StringComparison.OrdinalIgnoreCase))
            return model;
        return make + " " + model;
    }

    private static string FormatAperture(double? value)
    {
        if (!value.HasValue || value.Value <= 0.01d)
            return null;

        return "f/" + value.Value.ToString("0.0#", CultureInfo.InvariantCulture);
    }

    private static string BuildGpsText(string latRef, double[] lat, string lonRef, double[] lon)
    {
        if (lat == null || lon == null || lat.Length < 3 || lon.Length < 3)
            return null;

        var latitude = lat[0] + lat[1] / 60d + lat[2] / 3600d;
        var longitude = lon[0] + lon[1] / 60d + lon[2] / 3600d;

        if (string.Equals(NormalizeText(latRef), "S", StringComparison.OrdinalIgnoreCase))
            latitude = -latitude;
        if (string.Equals(NormalizeText(lonRef), "W", StringComparison.OrdinalIgnoreCase))
            longitude = -longitude;

        return $"GPS {latitude:0.0000}, {longitude:0.0000}";
    }

    private static string NormalizeText(string text)
    {
        return string.IsNullOrWhiteSpace(text) ? null : text.Trim();
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
}
