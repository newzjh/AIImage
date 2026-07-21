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
using System.Collections;
using System.Globalization;
using System.Runtime.Serialization.Formatters;
using Aexis.Samples.Json.Utilities;
using System.Runtime.Serialization;

namespace Aexis.Samples.Json.Serialization
{
    internal class JsonSerializerProxy : JsonSerializer
    {
        private readonly JsonSerializerInternalReader? _serializerReader;
        private readonly JsonSerializerInternalWriter? _serializerWriter;
        internal readonly JsonSerializer _serializer;

        public override event EventHandler<ErrorEventArgs>? Error
        {
            add => _serializer.Error += value;
            remove => _serializer.Error -= value;
        }

        public override IReferenceResolver? ReferenceResolver
        {
            get => _serializer.ReferenceResolver;
            set => _serializer.ReferenceResolver = value;
        }

        public override ITraceWriter? TraceWriter
        {
            get => _serializer.TraceWriter;
            set => _serializer.TraceWriter = value;
        }

        public override IEqualityComparer? EqualityComparer
        {
            get => _serializer.EqualityComparer;
            set => _serializer.EqualityComparer = value;
        }

        public override JsonConverterCollection Converters => _serializer.Converters;

        public override DefaultValueHandling DefaultValueHandling
        {
            get => _serializer.DefaultValueHandling;
            set => _serializer.DefaultValueHandling = value;
        }

        public override IContractResolver ContractResolver
        {
            get => _serializer.ContractResolver;
            set => _serializer.ContractResolver = value;
        }

        public override MissingMemberHandling MissingMemberHandling
        {
            get => _serializer.MissingMemberHandling;
            set => _serializer.MissingMemberHandling = value;
        }

        public override NullValueHandling NullValueHandling
        {
            get => _serializer.NullValueHandling;
            set => _serializer.NullValueHandling = value;
        }

        public override ObjectCreationHandling ObjectCreationHandling
        {
            get => _serializer.ObjectCreationHandling;
            set => _serializer.ObjectCreationHandling = value;
        }

        public override ReferenceLoopHandling ReferenceLoopHandling
        {
            get => _serializer.ReferenceLoopHandling;
            set => _serializer.ReferenceLoopHandling = value;
        }

        public override PreserveReferencesHandling PreserveReferencesHandling
        {
            get => _serializer.PreserveReferencesHandling;
            set => _serializer.PreserveReferencesHandling = value;
        }

        public override TypeNameHandling TypeNameHandling
        {
            get => _serializer.TypeNameHandling;
            set => _serializer.TypeNameHandling = value;
        }

        public override MetadataPropertyHandling MetadataPropertyHandling
        {
            get => _serializer.MetadataPropertyHandling;
            set => _serializer.MetadataPropertyHandling = value;
        }

        [Obsolete("TypeNameAssemblyFormat is obsolete. Use TypeNameAssemblyFormatHandling instead.")]
        public override FormatterAssemblyStyle TypeNameAssemblyFormat
        {
            get => _serializer.TypeNameAssemblyFormat;
            set => _serializer.TypeNameAssemblyFormat = value;
        }

        public override TypeNameAssemblyFormatHandling TypeNameAssemblyFormatHandling
        {
            get => _serializer.TypeNameAssemblyFormatHandling;
            set => _serializer.TypeNameAssemblyFormatHandling = value;
        }

        public override ConstructorHandling ConstructorHandling
        {
            get => _serializer.ConstructorHandling;
            set => _serializer.ConstructorHandling = value;
        }

        [Obsolete("Binder is obsolete. Use SerializationBinder instead.")]
        public override SerializationBinder Binder
        {
            get => _serializer.Binder;
            set => _serializer.Binder = value;
        }

        public override ISerializationBinder SerializationBinder
        {
            get => _serializer.SerializationBinder;
            set => _serializer.SerializationBinder = value;
        }

        public override StreamingContext Context
        {
            get => _serializer.Context;
            set => _serializer.Context = value;
        }

        public override Formatting Formatting
        {
            get => _serializer.Formatting;
            set => _serializer.Formatting = value;
        }

        public override DateFormatHandling DateFormatHandling
        {
            get => _serializer.DateFormatHandling;
            set => _serializer.DateFormatHandling = value;
        }

        public override DateTimeZoneHandling DateTimeZoneHandling
        {
            get => _serializer.DateTimeZoneHandling;
            set => _serializer.DateTimeZoneHandling = value;
        }

        public override DateParseHandling DateParseHandling
        {
            get => _serializer.DateParseHandling;
            set => _serializer.DateParseHandling = value;
        }

        public override FloatFormatHandling FloatFormatHandling
        {
            get => _serializer.FloatFormatHandling;
            set => _serializer.FloatFormatHandling = value;
        }

        public override FloatParseHandling FloatParseHandling
        {
            get => _serializer.FloatParseHandling;
            set => _serializer.FloatParseHandling = value;
        }

        public override StringEscapeHandling StringEscapeHandling
        {
            get => _serializer.StringEscapeHandling;
            set => _serializer.StringEscapeHandling = value;
        }

        public override string DateFormatString
        {
            get => _serializer.DateFormatString;
            set => _serializer.DateFormatString = value;
        }

        public override CultureInfo Culture
        {
            get => _serializer.Culture;
            set => _serializer.Culture = value;
        }

        public override int? MaxDepth
        {
            get => _serializer.MaxDepth;
            set => _serializer.MaxDepth = value;
        }

        public override bool CheckAdditionalContent
        {
            get => _serializer.CheckAdditionalContent;
            set => _serializer.CheckAdditionalContent = value;
        }

        internal JsonSerializerInternalBase GetInternalSerializer()
        {
            if (_serializerReader != null)
            {
                return _serializerReader;
            }
            else
            {
                return _serializerWriter!;
            }
        }

        public JsonSerializerProxy(JsonSerializerInternalReader serializerReader)
        {
            ValidationUtils.ArgumentNotNull(serializerReader, nameof(serializerReader));

            _serializerReader = serializerReader;
            _serializer = serializerReader.Serializer;
        }

        public JsonSerializerProxy(JsonSerializerInternalWriter serializerWriter)
        {
            ValidationUtils.ArgumentNotNull(serializerWriter, nameof(serializerWriter));

            _serializerWriter = serializerWriter;
            _serializer = serializerWriter.Serializer;
        }

        internal override object? DeserializeInternal(JsonReader reader, Type? objectType)
        {
            if (_serializerReader != null)
            {
                return _serializerReader.Deserialize(reader, objectType, false);
            }
            else
            {
                return _serializer.Deserialize(reader, objectType);
            }
        }

        internal override void PopulateInternal(JsonReader reader, object target)
        {
            if (_serializerReader != null)
            {
                _serializerReader.Populate(reader, target);
            }
            else
            {
                _serializer.Populate(reader, target);
            }
        }

        internal override void SerializeInternal(JsonWriter jsonWriter, object? value, Type? rootType)
        {
            if (_serializerWriter != null)
            {
                _serializerWriter.Serialize(jsonWriter, value, rootType);
            }
            else
            {
                _serializer.Serialize(jsonWriter, value);
            }
        }
    }
}