using System;
using System.Globalization;
using System.IO;

public static class RawPhotoParser
{
    private const string FujiRafHeader = "FUJIFILMCCD-RAW";

    public sealed class RawPhotoData
    {
        public byte[] previewBytes;
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

        if (TryExtractPreviewImage(bytes, out var previewBytes))
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
