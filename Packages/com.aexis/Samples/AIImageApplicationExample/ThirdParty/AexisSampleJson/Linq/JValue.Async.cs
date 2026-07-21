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

#if HAVE_ASYNC

using System;
using System.Globalization;
#if HAVE_BIG_INTEGER
using System.Numerics;
#endif
using System.Threading;
using System.Threading.Tasks;
using Aexis.Samples.Json.Utilities;

namespace Aexis.Samples.Json.Linq
{
    public partial class JValue
    {
        /// <summary>
        /// Writes this token to a <see cref="JsonWriter"/> asynchronously.
        /// </summary>
        /// <param name="writer">A <see cref="JsonWriter"/> into which this method will write.</param>
        /// <param name="cancellationToken">The token to monitor for cancellation requests.</param>
        /// <param name="converters">A collection of <see cref="JsonConverter"/> which will be used when writing the token.</param>
        /// <returns>A <see cref="Task"/> that represents the asynchronous write operation.</returns>
        public override Task WriteToAsync(JsonWriter writer, CancellationToken cancellationToken, params JsonConverter[] converters)
        {
            if (converters != null && converters.Length > 0 && _value != null)
            {
                JsonConverter? matchingConverter = JsonSerializer.GetMatchingConverter(converters, _value.GetType());
                if (matchingConverter != null && matchingConverter.CanWrite)
                {
                    // TODO: Call WriteJsonAsync when it exists.
                    matchingConverter.WriteJson(writer, _value, JsonSerializer.CreateDefault());
                    return AsyncUtils.CompletedTask;
                }
            }

            switch (_valueType)
            {
                case JTokenType.Comment:
                    return writer.WriteCommentAsync(_value?.ToString(), cancellationToken);
                case JTokenType.Raw:
                    return writer.WriteRawValueAsync(_value?.ToString(), cancellationToken);
                case JTokenType.Null:
                    return writer.WriteNullAsync(cancellationToken);
                case JTokenType.Undefined:
                    return writer.WriteUndefinedAsync(cancellationToken);
                case JTokenType.Integer:
                    if (_value is int i)
                    {
                        return writer.WriteValueAsync(i, cancellationToken);
                    }

                    if (_value is long l)
                    {
                        return writer.WriteValueAsync(l, cancellationToken);
                    }

                    if (_value is ulong ul)
                    {
                        return writer.WriteValueAsync(ul, cancellationToken);
                    }

#if HAVE_BIG_INTEGER
                    if (_value is BigInteger integer)
                    {
                        return writer.WriteValueAsync(integer, cancellationToken);
                    }
#endif

                    return writer.WriteValueAsync(Convert.ToInt64(_value, CultureInfo.InvariantCulture), cancellationToken);
                case JTokenType.Float:
                    if (_value is decimal dec)
                    {
                        return writer.WriteValueAsync(dec, cancellationToken);
                    }

                    if (_value is double d)
                    {
                        return writer.WriteValueAsync(d, cancellationToken);
                    }

                    if (_value is float f)
                    {
                        return writer.WriteValueAsync(f, cancellationToken);
                    }

                    return writer.WriteValueAsync(Convert.ToDouble(_value, CultureInfo.InvariantCulture), cancellationToken);
                case JTokenType.String:
                    return writer.WriteValueAsync(_value?.ToString(), cancellationToken);
                case JTokenType.Boolean:
                    return writer.WriteValueAsync(Convert.ToBoolean(_value, CultureInfo.InvariantCulture), cancellationToken);
                case JTokenType.Date:
                    if (_value is DateTimeOffset offset)
                    {
                        return writer.WriteValueAsync(offset, cancellationToken);
                    }

                    return writer.WriteValueAsync(Convert.ToDateTime(_value, CultureInfo.InvariantCulture), cancellationToken);
                case JTokenType.Bytes:
                    return writer.WriteValueAsync((byte[]?)_value, cancellationToken);
                case JTokenType.Guid:
                    return writer.WriteValueAsync(_value != null ? (Guid?)_value : null, cancellationToken);
                case JTokenType.TimeSpan:
                    return writer.WriteValueAsync(_value != null ? (TimeSpan?)_value : null, cancellationToken);
                case JTokenType.Uri:
                    return writer.WriteValueAsync((Uri?)_value, cancellationToken);
            }

            throw MiscellaneousUtils.CreateArgumentOutOfRangeException(nameof(Type), _valueType, "Unexpected token type.");
        }
    }
}

#endif
