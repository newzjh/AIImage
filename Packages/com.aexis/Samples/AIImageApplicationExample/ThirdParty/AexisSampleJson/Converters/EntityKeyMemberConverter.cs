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

#if HAVE_ENTITY_FRAMEWORK
using System;
using Aexis.Samples.Json.Serialization;
using System.Globalization;
using Aexis.Samples.Json.Utilities;
using System.Diagnostics;

namespace Aexis.Samples.Json.Converters
{
    /// <summary>
    /// Converts an Entity Framework <see cref="T:System.Data.EntityKeyMember"/> to and from JSON.
    /// </summary>
    public class EntityKeyMemberConverter : JsonConverter
    {
        private const string EntityKeyMemberFullTypeName = "System.Data.EntityKeyMember";

        private const string KeyPropertyName = "Key";
        private const string TypePropertyName = "Type";
        private const string ValuePropertyName = "Value";

        private static ReflectionObject? _reflectionObject;

        /// <summary>
        /// Writes the JSON representation of the object.
        /// </summary>
        /// <param name="writer">The <see cref="JsonWriter"/> to write to.</param>
        /// <param name="value">The value.</param>
        /// <param name="serializer">The calling serializer.</param>
        public override void WriteJson(JsonWriter writer, object? value, JsonSerializer serializer)
        {
            if (value == null)
            {
                writer.WriteNull();
                return;
            }

            EnsureReflectionObject(value.GetType());
            MiscellaneousUtils.Assert(_reflectionObject != null);

            DefaultContractResolver? resolver = serializer.ContractResolver as DefaultContractResolver;

            string keyName = (string)_reflectionObject.GetValue(value, KeyPropertyName)!;
            object? keyValue = _reflectionObject.GetValue(value, ValuePropertyName);

            Type? keyValueType = keyValue?.GetType();

            writer.WriteStartObject();
            writer.WritePropertyName((resolver != null) ? resolver.GetResolvedPropertyName(KeyPropertyName) : KeyPropertyName);
            writer.WriteValue(keyName);
            writer.WritePropertyName((resolver != null) ? resolver.GetResolvedPropertyName(TypePropertyName) : TypePropertyName);
            writer.WriteValue(keyValueType?.FullName);

            writer.WritePropertyName((resolver != null) ? resolver.GetResolvedPropertyName(ValuePropertyName) : ValuePropertyName);

            if (keyValueType != null)
            {
                if (JsonSerializerInternalWriter.TryConvertToString(keyValue!, keyValueType, out string? valueJson))
                {
                    writer.WriteValue(valueJson);
                }
                else
                {
                    writer.WriteValue(keyValue);
                }
            }
            else
            {
                writer.WriteNull();
            }

            writer.WriteEndObject();
        }

        private static void ReadAndAssertProperty(JsonReader reader, string propertyName)
        {
            reader.ReadAndAssert();

            if (reader.TokenType != JsonToken.PropertyName || !string.Equals(reader.Value?.ToString(), propertyName, StringComparison.OrdinalIgnoreCase))
            {
                throw new JsonSerializationException("Expected JSON property '{0}'.".FormatWith(CultureInfo.InvariantCulture, propertyName));
            }
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
            EnsureReflectionObject(objectType);
            MiscellaneousUtils.Assert(_reflectionObject != null);

            object entityKeyMember = _reflectionObject.Creator!();

            ReadAndAssertProperty(reader, KeyPropertyName);
            reader.ReadAndAssert();
            _reflectionObject.SetValue(entityKeyMember, KeyPropertyName, reader.Value?.ToString());

            ReadAndAssertProperty(reader, TypePropertyName);
            reader.ReadAndAssert();
            string? type = reader.Value?.ToString();

            Type t = Type.GetType(type!)!;

            ReadAndAssertProperty(reader, ValuePropertyName);
            reader.ReadAndAssert();
            _reflectionObject.SetValue(entityKeyMember, ValuePropertyName, serializer.Deserialize(reader, t));

            reader.ReadAndAssert();

            return entityKeyMember;
        }

        private static void EnsureReflectionObject(Type objectType)
        {
            if (_reflectionObject == null)
            {
                _reflectionObject = ReflectionObject.Create(objectType, KeyPropertyName, ValuePropertyName);
            }
        }

        /// <summary>
        /// Determines whether this instance can convert the specified object type.
        /// </summary>
        /// <param name="objectType">Type of the object.</param>
        /// <returns>
        /// 	<c>true</c> if this instance can convert the specified object type; otherwise, <c>false</c>.
        /// </returns>
        public override bool CanConvert(Type objectType)
        {
            return objectType.AssignableToTypeName(EntityKeyMemberFullTypeName, false);
        }
    }
}

#endif