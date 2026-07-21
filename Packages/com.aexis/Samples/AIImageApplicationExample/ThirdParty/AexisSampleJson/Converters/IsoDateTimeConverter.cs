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
using Aexis.Samples.Json.Utilities;

namespace Aexis.Samples.Json.Converters
{
    /// <summary>
    /// Converts a <see cref="DateTime"/> to and from the ISO 8601 date format (e.g. <c>"2008-04-12T12:53Z"</c>).
    /// </summary>
    public class IsoDateTimeConverter : DateTimeConverterBase
    {
        private const string DefaultDateTimeFormat = "yyyy'-'MM'-'dd'T'HH':'mm':'ss.FFFFFFFK";

        private DateTimeStyles _dateTimeStyles = DateTimeStyles.RoundtripKind;
        private string? _dateTimeFormat;
        private CultureInfo? _culture;

        /// <summary>
        /// Gets or sets the date time styles used when converting a date to and from JSON.
        /// </summary>
        /// <value>The date time styles used when converting a date to and from JSON.</value>
        public DateTimeStyles DateTimeStyles
        {
            get => _dateTimeStyles;
            set => _dateTimeStyles = value;
        }

        /// <summary>
        /// Gets or sets the date time format used when converting a date to and from JSON.
        /// </summary>
        /// <value>The date time format used when converting a date to and from JSON.</value>
        public string? DateTimeFormat
        {
            get => _dateTimeFormat ?? string.Empty;
            set => _dateTimeFormat = (StringUtils.IsNullOrEmpty(value)) ? null : value;
        }

        /// <summary>
        /// Gets or sets the culture used when converting a date to and from JSON.
        /// </summary>
        /// <value>The culture used when converting a date to and from JSON.</value>
        public CultureInfo Culture
        {
            get => _culture ?? CultureInfo.CurrentCulture;
            set => _culture = value;
        }

        /// <summary>
        /// Writes the JSON representation of the object.
        /// </summary>
        /// <param name="writer">The <see cref="JsonWriter"/> to write to.</param>
        /// <param name="value">The value.</param>
        /// <param name="serializer">The calling serializer.</param>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            string text;

            if (value is DateTime dateTime)
            {
                if ((_dateTimeStyles & DateTimeStyles.AdjustToUniversal) == DateTimeStyles.AdjustToUniversal
                    || (_dateTimeStyles & DateTimeStyles.AssumeUniversal) == DateTimeStyles.AssumeUniversal)
                {
                    dateTime = dateTime.ToUniversalTime();
                }

                text = dateTime.ToString(_dateTimeFormat ?? DefaultDateTimeFormat, Culture);
            }
#if HAVE_DATE_TIME_OFFSET
            else if (value is DateTimeOffset dateTimeOffset)
            {
                if ((_dateTimeStyles & DateTimeStyles.AdjustToUniversal) == DateTimeStyles.AdjustToUniversal
                    || (_dateTimeStyles & DateTimeStyles.AssumeUniversal) == DateTimeStyles.AssumeUniversal)
                {
                    dateTimeOffset = dateTimeOffset.ToUniversalTime();
                }

                text = dateTimeOffset.ToString(_dateTimeFormat ?? DefaultDateTimeFormat, Culture);
            }
#endif
            else
            {
                throw new JsonSerializationException("Unexpected value when converting date. Expected DateTime or DateTimeOffset, got {0}.".FormatWith(CultureInfo.InvariantCulture, ReflectionUtils.GetObjectType(value)!));
            }

            writer.WriteValue(text);
        }

        /// <summary>
        /// Reads the JSON representation of the object.
        /// </summary>
        /// <param name="reader">The <see cref="JsonReader"/> to read from.</param>
        /// <param name="objectType">Type of the object.</param>
        /// <param name="existingValue">The existing value of object being read.</param>
        /// <param name="serializer">The calling serializer.</param>
        /// <returns>The object value.</returns>
        public override object? ReadJson(JsonReader reader, Type objectType, object? existingValue, JsonSerializer serializer)
        {
            bool nullable = ReflectionUtils.IsNullableType(objectType);
            if (reader.TokenType == JsonToken.Null)
            {
                if (!nullable)
                {
                    throw JsonSerializationException.Create(reader, "Cannot convert null value to {0}.".FormatWith(CultureInfo.InvariantCulture, objectType));
                }

                return null;
            }

#if HAVE_DATE_TIME_OFFSET
            Type t = (nullable)
                ? Nullable.GetUnderlyingType(objectType)!
                : objectType;
#endif

            if (reader.TokenType == JsonToken.Date)
            {
#if HAVE_DATE_TIME_OFFSET
                if (t == typeof(DateTimeOffset))
                {
                    return (reader.Value is DateTimeOffset) ? reader.Value : new DateTimeOffset((DateTime)reader.Value!);
                }

                // converter is expected to return a DateTime
                if (reader.Value is DateTimeOffset offset)
                {
                    return offset.DateTime;
                }
#endif

                return reader.Value;
            }

            if (reader.TokenType != JsonToken.String)
            {
                throw JsonSerializationException.Create(reader, "Unexpected token parsing date. Expected String, got {0}.".FormatWith(CultureInfo.InvariantCulture, reader.TokenType));
            }

            string? dateText = reader.Value?.ToString();

            if (StringUtils.IsNullOrEmpty(dateText) && nullable)
            {
                return null;
            }

            MiscellaneousUtils.Assert(dateText != null);

#if HAVE_DATE_TIME_OFFSET
            if (t == typeof(DateTimeOffset))
            {
                if (!StringUtils.IsNullOrEmpty(_dateTimeFormat))
                {
                    return DateTimeOffset.ParseExact(dateText, _dateTimeFormat, Culture, _dateTimeStyles);
                }
                else
                {
                    return DateTimeOffset.Parse(dateText, Culture, _dateTimeStyles);
                }
            }
#endif

            if (!StringUtils.IsNullOrEmpty(_dateTimeFormat))
            {
                return DateTime.ParseExact(dateText, _dateTimeFormat, Culture, _dateTimeStyles);
            }
            else
            {
                return DateTime.Parse(dateText, Culture, _dateTimeStyles);
            }
        }
    }
}
