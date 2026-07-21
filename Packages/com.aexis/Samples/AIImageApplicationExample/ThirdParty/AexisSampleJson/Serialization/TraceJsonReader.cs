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
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text;

namespace Aexis.Samples.Json.Serialization
{
    internal class TraceJsonReader : JsonReader, IJsonLineInfo
    {
        private readonly JsonReader _innerReader;
        private readonly JsonTextWriter _textWriter;
        private readonly StringWriter _sw;

        public TraceJsonReader(JsonReader innerReader)
        {
            _innerReader = innerReader;

            _sw = new StringWriter(CultureInfo.InvariantCulture);
            // prefix the message in the stringwriter to avoid concat with a potentially large JSON string
            _sw.Write("Deserialized JSON: " + Environment.NewLine);

            _textWriter = new JsonTextWriter(_sw);
            _textWriter.Formatting = Formatting.Indented;
        }

        public string GetDeserializedJsonMessage()
        {
            return _sw.ToString();
        }

        public override bool Read()
        {
            bool value = _innerReader.Read();
            WriteCurrentToken();
            return value;
        }

        public override int? ReadAsInt32()
        {
            int? value = _innerReader.ReadAsInt32();
            WriteCurrentToken();
            return value;
        }

        public override string? ReadAsString()
        {
            string? value = _innerReader.ReadAsString();
            WriteCurrentToken();
            return value;
        }

        public override byte[]? ReadAsBytes()
        {
            byte[]? value = _innerReader.ReadAsBytes();
            WriteCurrentToken();
            return value;
        }

        public override decimal? ReadAsDecimal()
        {
            decimal? value = _innerReader.ReadAsDecimal();
            WriteCurrentToken();
            return value;
        }

        public override double? ReadAsDouble()
        {
            double? value = _innerReader.ReadAsDouble();
            WriteCurrentToken();
            return value;
        }

        public override bool? ReadAsBoolean()
        {
            bool? value = _innerReader.ReadAsBoolean();
            WriteCurrentToken();
            return value;
        }

        public override DateTime? ReadAsDateTime()
        {
            DateTime? value = _innerReader.ReadAsDateTime();
            WriteCurrentToken();
            return value;
        }

#if HAVE_DATE_TIME_OFFSET
        public override DateTimeOffset? ReadAsDateTimeOffset()
        {
            DateTimeOffset? value = _innerReader.ReadAsDateTimeOffset();
            WriteCurrentToken();
            return value;
        }
#endif

        public void WriteCurrentToken()
        {
            _textWriter.WriteToken(_innerReader, false, false, true);
        }

        public override int Depth => _innerReader.Depth;

        public override string Path => _innerReader.Path;

        public override char QuoteChar
        {
            get => _innerReader.QuoteChar;
            protected internal set => _innerReader.QuoteChar = value;
        }

        public override JsonToken TokenType => _innerReader.TokenType;

        public override object? Value => _innerReader.Value;

        public override Type? ValueType => _innerReader.ValueType;

        public override void Close()
        {
            _innerReader.Close();
        }

        bool IJsonLineInfo.HasLineInfo()
        {
            return _innerReader is IJsonLineInfo lineInfo && lineInfo.HasLineInfo();
        }

        int IJsonLineInfo.LineNumber => (_innerReader is IJsonLineInfo lineInfo) ? lineInfo.LineNumber : 0;

        int IJsonLineInfo.LinePosition => (_innerReader is IJsonLineInfo lineInfo) ? lineInfo.LinePosition : 0;
    }
}