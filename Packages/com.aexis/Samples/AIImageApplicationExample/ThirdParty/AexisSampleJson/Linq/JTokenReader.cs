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
using Aexis.Samples.Json.Utilities;

namespace Aexis.Samples.Json.Linq
{
    /// <summary>
    /// Represents a reader that provides fast, non-cached, forward-only access to serialized JSON data.
    /// </summary>
    public class JTokenReader : JsonReader, IJsonLineInfo
    {
        private readonly JToken _root;
        private string? _initialPath;
        private JToken? _parent;
        private JToken? _current;

        /// <summary>
        /// Gets the <see cref="JToken"/> at the reader's current position.
        /// </summary>
        public JToken? CurrentToken => _current;

        /// <summary>
        /// Initializes a new instance of the <see cref="JTokenReader"/> class.
        /// </summary>
        /// <param name="token">The token to read from.</param>
        public JTokenReader(JToken token)
        {
            ValidationUtils.ArgumentNotNull(token, nameof(token));

            _root = token;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="JTokenReader"/> class.
        /// </summary>
        /// <param name="token">The token to read from.</param>
        /// <param name="initialPath">The initial path of the token. It is prepended to the returned <see cref="Path"/>.</param>
        public JTokenReader(JToken token, string initialPath)
            : this(token)
        {
            _initialPath = initialPath;
        }

        /// <summary>
        /// Reads the next JSON token from the underlying <see cref="JToken"/>.
        /// </summary>
        /// <returns>
        /// <c>true</c> if the next token was read successfully; <c>false</c> if there are no more tokens to read.
        /// </returns>
        public override bool Read()
        {
            if (CurrentState != State.Start)
            {
                if (_current == null)
                {
                    return false;
                }

                if (_current is JContainer container && _parent != container)
                {
                    return ReadInto(container);
                }
                else
                {
                    return ReadOver(_current);
                }
            }

            // The current value could already be the root value if it is a comment
            if (_current == _root)
            {
                return false;
            }

            _current = _root;
            SetToken(_current);
            return true;
        }

        private bool ReadOver(JToken t)
        {
            if (t == _root)
            {
                return ReadToEnd();
            }

            JToken? next = t.Next;
            if ((next == null || next == t) || t == t.Parent!.Last)
            {
                if (t.Parent == null)
                {
                    return ReadToEnd();
                }

                return SetEnd(t.Parent);
            }
            else
            {
                _current = next;
                SetToken(_current);
                return true;
            }
        }

        private bool ReadToEnd()
        {
            _current = null;
            SetToken(JsonToken.None);
            return false;
        }

        private JsonToken? GetEndToken(JContainer c)
        {
            switch (c.Type)
            {
                case JTokenType.Object:
                    return JsonToken.EndObject;
                case JTokenType.Array:
                    return JsonToken.EndArray;
                case JTokenType.Constructor:
                    return JsonToken.EndConstructor;
                case JTokenType.Property:
                    return null;
                default:
                    throw MiscellaneousUtils.CreateArgumentOutOfRangeException(nameof(c.Type), c.Type, "Unexpected JContainer type.");
            }
        }

        private bool ReadInto(JContainer c)
        {
            JToken? firstChild = c.First;
            if (firstChild == null)
            {
                return SetEnd(c);
            }
            else
            {
                SetToken(firstChild);
                _current = firstChild;
                _parent = c;
                return true;
            }
        }

        private bool SetEnd(JContainer c)
        {
            JsonToken? endToken = GetEndToken(c);
            if (endToken != null)
            {
                SetToken(endToken.GetValueOrDefault());
                _current = c;
                _parent = c;
                return true;
            }
            else
            {
                return ReadOver(c);
            }
        }

        private void SetToken(JToken token)
        {
            switch (token.Type)
            {
                case JTokenType.Object:
                    SetToken(JsonToken.StartObject);
                    break;
                case JTokenType.Array:
                    SetToken(JsonToken.StartArray);
                    break;
                case JTokenType.Constructor:
                    SetToken(JsonToken.StartConstructor, ((JConstructor)token).Name);
                    break;
                case JTokenType.Property:
                    SetToken(JsonToken.PropertyName, ((JProperty)token).Name);
                    break;
                case JTokenType.Comment:
                    SetToken(JsonToken.Comment, ((JValue)token).Value);
                    break;
                case JTokenType.Integer:
                    SetToken(JsonToken.Integer, ((JValue)token).Value);
                    break;
                case JTokenType.Float:
                    SetToken(JsonToken.Float, ((JValue)token).Value);
                    break;
                case JTokenType.String:
                    SetToken(JsonToken.String, ((JValue)token).Value);
                    break;
                case JTokenType.Boolean:
                    SetToken(JsonToken.Boolean, ((JValue)token).Value);
                    break;
                case JTokenType.Null:
                    SetToken(JsonToken.Null, ((JValue)token).Value);
                    break;
                case JTokenType.Undefined:
                    SetToken(JsonToken.Undefined, ((JValue)token).Value);
                    break;
                case JTokenType.Date:
                    {
                        object? v = ((JValue)token).Value;
                        if (v is DateTime dt)
                        {
                            v = DateTimeUtils.EnsureDateTime(dt, DateTimeZoneHandling);
                        }

                        SetToken(JsonToken.Date, v);
                        break;
                    }
                case JTokenType.Raw:
                    SetToken(JsonToken.Raw, ((JValue)token).Value);
                    break;
                case JTokenType.Bytes:
                    SetToken(JsonToken.Bytes, ((JValue)token).Value);
                    break;
                case JTokenType.Guid:
                    SetToken(JsonToken.String, SafeToString(((JValue)token).Value));
                    break;
                case JTokenType.Uri:
                    {
                        object? v = ((JValue)token).Value;
                        SetToken(JsonToken.String, v is Uri uri ? uri.OriginalString : SafeToString(v));
                        break;
                    }
                case JTokenType.TimeSpan:
                    SetToken(JsonToken.String, SafeToString(((JValue)token).Value));
                    break;
                default:
                    throw MiscellaneousUtils.CreateArgumentOutOfRangeException(nameof(token.Type), token.Type, "Unexpected JTokenType.");
            }
        }

        private string? SafeToString(object? value)
        {
            return value?.ToString();
        }

        bool IJsonLineInfo.HasLineInfo()
        {
            if (CurrentState == State.Start)
            {
                return false;
            }

            IJsonLineInfo? info = _current;
            return (info != null && info.HasLineInfo());
        }

        int IJsonLineInfo.LineNumber
        {
            get
            {
                if (CurrentState == State.Start)
                {
                    return 0;
                }

                IJsonLineInfo? info = _current;
                if (info != null)
                {
                    return info.LineNumber;
                }

                return 0;
            }
        }

        int IJsonLineInfo.LinePosition
        {
            get
            {
                if (CurrentState == State.Start)
                {
                    return 0;
                }

                IJsonLineInfo? info = _current;
                if (info != null)
                {
                    return info.LinePosition;
                }

                return 0;
            }
        }

        /// <summary>
        /// Gets the path of the current JSON token. 
        /// </summary>
        public override string Path
        {
            get
            {
                string path = base.Path;

                if (_initialPath == null)
                {
                    _initialPath = _root.Path;
                }

                if (!StringUtils.IsNullOrEmpty(_initialPath))
                {
                    if (StringUtils.IsNullOrEmpty(path))
                    {
                        return _initialPath;
                    }

                    if (path.StartsWith('['))
                    {
                        path = _initialPath + path;
                    }
                    else
                    {
                        path = _initialPath + "." + path;
                    }
                }

                return path;
            }
        }
    }
}