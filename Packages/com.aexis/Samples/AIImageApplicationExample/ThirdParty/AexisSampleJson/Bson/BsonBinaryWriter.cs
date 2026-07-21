#define NETSTANDARD2_0
#define HAVE_ADO_NET
#define HAVE_APP_DOMAIN
#define HAVE_ASYNC
#define HAVE_BIG_INTEGER
#define HAVE_BINARY_FORMATTER
#define HAVE_BINARY_SERIALIZATION
#define HAVE_BINARY_EXCEPTION_SERIALIZATION
#define HAVE_CHAR_TO_LOWER_WITH_CULTURE
#define HAVE_CHAR_TO_STRING_WITH_CULTURE
#define HAVE_COM_ATTRIBUTES
#define HAVE_COMPONENT_MODEL
#define HAVE_CONCURRENT_COLLECTIONS
#define HAVE_COVARIANT_GENERICS
#define HAVE_DATA_CONTRACTS
#define HAVE_DATE_TIME_OFFSET
#define HAVE_DB_NULL_TYPE_CODE
#define HAVE_DYNAMIC
#define HAVE_EMPTY_TYPES
#define HAVE_ENTITY_FRAMEWORK
#define HAVE_EXPRESSIONS
#define HAVE_FAST_REVERSE
#define HAVE_FSHARP_TYPES
#define HAVE_FULL_REFLECTION
#define HAVE_GUID_TRY_PARSE
#define HAVE_HASH_SET
#define HAVE_ICLONEABLE
#define HAVE_ICONVERTIBLE
#define HAVE_IGNORE_DATA_MEMBER_ATTRIBUTE
#define HAVE_INOTIFY_COLLECTION_CHANGED
#define HAVE_INOTIFY_PROPERTY_CHANGING
#define HAVE_ISET
#define HAVE_LINQ
#define HAVE_MEMORY_BARRIER
#define HAVE_METHOD_IMPL_ATTRIBUTE
#define HAVE_NON_SERIALIZED_ATTRIBUTE
#define HAVE_READ_ONLY_COLLECTIONS
#define HAVE_SECURITY_SAFE_CRITICAL_ATTRIBUTE
#define HAVE_SERIALIZATION_BINDER_BIND_TO_NAME
#define HAVE_STREAM_READER_WRITER_CLOSE
#define HAVE_STRING_JOIN_WITH_ENUMERABLE
#define HAVE_TIME_SPAN_PARSE_WITH_CULTURE
#define HAVE_TIME_SPAN_TO_STRING_WITH_CULTURE
#define HAVE_TIME_ZONE_INFO
#define HAVE_TRACE_WRITER
#define HAVE_TYPE_DESCRIPTOR
#define HAVE_UNICODE_SURROGATE_DETECTION
#define HAVE_VARIANT_TYPE_PARAMETERS
#define HAVE_VERSION_TRY_PARSE
#define HAVE_XLINQ
#define HAVE_XML_DOCUMENT
#define HAVE_XML_DOCUMENT_TYPE
#define HAVE_CONCURRENT_DICTIONARY
#define HAVE_REGEX_TIMEOUTS
#region License
// Copyright (c) 2007 James Newton-King
//
// Permission is hereby granted, free of charge, to any person
// obtaining a copy of this software and associated documentation
// files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use,
// copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following
// conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
// OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
// HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
// WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
// FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
// OTHER DEALINGS IN THE SOFTWARE.
#endregion

using System;
using System.Globalization;
using System.IO;
using System.Text;
using Aexis.Samples.Json.Utilities;

#nullable disable

namespace Aexis.Samples.Json.Bson
{
    internal class BsonBinaryWriter
    {
        private static readonly Encoding Encoding = new UTF8Encoding(false);

        private readonly BinaryWriter _writer;

        private byte[] _largeByteBuffer;

        public DateTimeKind DateTimeKindHandling { get; set; }

        public BsonBinaryWriter(BinaryWriter writer)
        {
            DateTimeKindHandling = DateTimeKind.Utc;
            _writer = writer;
        }

        public void Flush()
        {
            _writer.Flush();
        }

        public void Close()
        {
#if HAVE_STREAM_READER_WRITER_CLOSE
            _writer.Close();
#else
            _writer.Dispose();
#endif
        }

        public void WriteToken(BsonToken t)
        {
            CalculateSize(t);
            WriteTokenInternal(t);
        }

        private void WriteTokenInternal(BsonToken t)
        {
            switch (t.Type)
            {
                case BsonType.Object:
                {
                    BsonObject value = (BsonObject)t;
                    _writer.Write(value.CalculatedSize);
                    foreach (BsonProperty property in value)
                    {
                        _writer.Write((sbyte)property.Value.Type);
                        WriteString((string)property.Name.Value, property.Name.ByteCount, null);
                        WriteTokenInternal(property.Value);
                    }
                    _writer.Write((byte)0);
                }
                    break;
                case BsonType.Array:
                {
                    BsonArray value = (BsonArray)t;
                    _writer.Write(value.CalculatedSize);
                    ulong index = 0;
                    foreach (BsonToken c in value)
                    {
                        _writer.Write((sbyte)c.Type);
                        WriteString(index.ToString(CultureInfo.InvariantCulture), MathUtils.IntLength(index), null);
                        WriteTokenInternal(c);
                        index++;
                    }
                    _writer.Write((byte)0);
                }
                    break;
                case BsonType.Integer:
                {
                    BsonValue value = (BsonValue)t;
                    _writer.Write(Convert.ToInt32(value.Value, CultureInfo.InvariantCulture));
                }
                    break;
                case BsonType.Long:
                {
                    BsonValue value = (BsonValue)t;
                    _writer.Write(Convert.ToInt64(value.Value, CultureInfo.InvariantCulture));
                }
                    break;
                case BsonType.Number:
                {
                    BsonValue value = (BsonValue)t;
                    _writer.Write(Convert.ToDouble(value.Value, CultureInfo.InvariantCulture));
                }
                    break;
                case BsonType.String:
                {
                    BsonString value = (BsonString)t;
                    WriteString((string)value.Value, value.ByteCount, value.CalculatedSize - 4);
                }
                    break;
                case BsonType.Boolean:
                    _writer.Write(t == BsonBoolean.True);
                    break;
                case BsonType.Null:
                case BsonType.Undefined:
                    break;
                case BsonType.Date:
                {
                    BsonValue value = (BsonValue)t;

                    long ticks = 0;

                    if (value.Value is DateTime dateTime)
                    {
                        if (DateTimeKindHandling == DateTimeKind.Utc)
                        {
                            dateTime = dateTime.ToUniversalTime();
                        }
                        else if (DateTimeKindHandling == DateTimeKind.Local)
                        {
                            dateTime = dateTime.ToLocalTime();
                        }

                        ticks = DateTimeUtils.ConvertDateTimeToJavaScriptTicks(dateTime, false);
                    }
#if HAVE_DATE_TIME_OFFSET
                    else
                    {
                        DateTimeOffset dateTimeOffset = (DateTimeOffset)value.Value;
                        ticks = DateTimeUtils.ConvertDateTimeToJavaScriptTicks(dateTimeOffset.UtcDateTime, dateTimeOffset.Offset);
                    }
#endif

                    _writer.Write(ticks);
                }
                    break;
                case BsonType.Binary:
                {
                    BsonBinary value = (BsonBinary)t;

                    byte[] data = (byte[])value.Value;
                    _writer.Write(data.Length);
                    _writer.Write((byte)value.BinaryType);
                    _writer.Write(data);
                }
                    break;
                case BsonType.Oid:
                {
                    BsonValue value = (BsonValue)t;

                    byte[] data = (byte[])value.Value;
                    _writer.Write(data);
                }
                    break;
                case BsonType.Regex:
                {
                    BsonRegex value = (BsonRegex)t;

                    WriteString((string)value.Pattern.Value, value.Pattern.ByteCount, null);
                    WriteString((string)value.Options.Value, value.Options.ByteCount, null);
                }
                    break;
                default:
                    throw new ArgumentOutOfRangeException(nameof(t), "Unexpected token when writing BSON: {0}".FormatWith(CultureInfo.InvariantCulture, t.Type));
            }
        }

        private void WriteString(string s, int byteCount, int? calculatedlengthPrefix)
        {
            if (calculatedlengthPrefix != null)
            {
                _writer.Write(calculatedlengthPrefix.GetValueOrDefault());
            }

            WriteUtf8Bytes(s, byteCount);

            _writer.Write((byte)0);
        }

        public void WriteUtf8Bytes(string s, int byteCount)
        {
            if (s != null)
            {
                if (byteCount <= 256)
                {
                    if (_largeByteBuffer == null)
                    {
                        _largeByteBuffer = new byte[256];
                    }

                    Encoding.GetBytes(s, 0, s.Length, _largeByteBuffer, 0);
                    _writer.Write(_largeByteBuffer, 0, byteCount);
                }
                else
                {
                    byte[] bytes = Encoding.GetBytes(s);
                    _writer.Write(bytes);
                }
            }
        }

        private int CalculateSize(int stringByteCount)
        {
            return stringByteCount + 1;
        }

        private int CalculateSizeWithLength(int stringByteCount, bool includeSize)
        {
            int baseSize = (includeSize)
                ? 5 // size bytes + terminator
                : 1; // terminator

            return baseSize + stringByteCount;
        }

        private int CalculateSize(BsonToken t)
        {
            switch (t.Type)
            {
                case BsonType.Object:
                {
                    BsonObject value = (BsonObject)t;

                    int bases = 4;
                    foreach (BsonProperty p in value)
                    {
                        int size = 1;
                        size += CalculateSize(p.Name);
                        size += CalculateSize(p.Value);

                        bases += size;
                    }
                    bases += 1;
                    value.CalculatedSize = bases;
                    return bases;
                }
                case BsonType.Array:
                {
                    BsonArray value = (BsonArray)t;

                    int size = 4;
                    ulong index = 0;
                    foreach (BsonToken c in value)
                    {
                        size += 1;
                        size += CalculateSize(MathUtils.IntLength(index));
                        size += CalculateSize(c);
                        index++;
                    }
                    size += 1;
                    value.CalculatedSize = size;

                    return value.CalculatedSize;
                }
                case BsonType.Integer:
                    return 4;
                case BsonType.Long:
                    return 8;
                case BsonType.Number:
                    return 8;
                case BsonType.String:
                {
                    BsonString value = (BsonString)t;
                    string s = (string)value.Value;
                    value.ByteCount = (s != null) ? Encoding.GetByteCount(s) : 0;
                    value.CalculatedSize = CalculateSizeWithLength(value.ByteCount, value.IncludeLength);

                    return value.CalculatedSize;
                }
                case BsonType.Boolean:
                    return 1;
                case BsonType.Null:
                case BsonType.Undefined:
                    return 0;
                case BsonType.Date:
                    return 8;
                case BsonType.Binary:
                {
                    BsonBinary value = (BsonBinary)t;

                    byte[] data = (byte[])value.Value;
                    value.CalculatedSize = 4 + 1 + data.Length;

                    return value.CalculatedSize;
                }
                case BsonType.Oid:
                    return 12;
                case BsonType.Regex:
                {
                    BsonRegex value = (BsonRegex)t;
                    int size = 0;
                    size += CalculateSize(value.Pattern);
                    size += CalculateSize(value.Options);
                    value.CalculatedSize = size;

                    return value.CalculatedSize;
                }
                default:
                    throw new ArgumentOutOfRangeException(nameof(t), "Unexpected token when writing BSON: {0}".FormatWith(CultureInfo.InvariantCulture, t.Type));
            }
        }
    }
}