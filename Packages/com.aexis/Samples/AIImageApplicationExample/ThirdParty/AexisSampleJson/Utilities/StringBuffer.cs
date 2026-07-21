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

namespace Aexis.Samples.Json.Utilities
{
    /// <summary>
    /// Builds a string. Unlike <see cref="System.Text.StringBuilder"/> this class lets you reuse its internal buffer.
    /// </summary>
    internal struct StringBuffer
    {
        private char[]? _buffer;
        private int _position;

        public int Position
        {
            get => _position;
            set => _position = value;
        }

        public bool IsEmpty => _buffer == null;

        public StringBuffer(IArrayPool<char>? bufferPool, int initalSize) : this(BufferUtils.RentBuffer(bufferPool, initalSize))
        {
        }

        private StringBuffer(char[] buffer)
        {
            _buffer = buffer;
            _position = 0;
        }

        public void Append(IArrayPool<char>? bufferPool, char value)
        {
            // test if the buffer array is large enough to take the value
            if (_position == _buffer!.Length)
            {
                EnsureSize(bufferPool, 1);
            }

            // set value and increment poisition
            _buffer![_position++] = value;
        }

        public void Append(IArrayPool<char>? bufferPool, char[] buffer, int startIndex, int count)
        {
            if (_position + count >= _buffer!.Length)
            {
                EnsureSize(bufferPool, count);
            }

            Array.Copy(buffer, startIndex, _buffer, _position, count);

            _position += count;
        }

        public void Clear(IArrayPool<char>? bufferPool)
        {
            if (_buffer != null)
            {
                BufferUtils.ReturnBuffer(bufferPool, _buffer);
                _buffer = null;
            }
            _position = 0;
        }

        private void EnsureSize(IArrayPool<char>? bufferPool, int appendLength)
        {
            char[] newBuffer = BufferUtils.RentBuffer(bufferPool, (_position + appendLength) * 2);

            if (_buffer != null)
            {
                Array.Copy(_buffer, newBuffer, _position);
                BufferUtils.ReturnBuffer(bufferPool, _buffer);
            }

            _buffer = newBuffer;
        }

        public override string ToString()
        {
            return ToString(0, _position);
        }

        public string ToString(int start, int length)
        {
            MiscellaneousUtils.Assert(_buffer != null);
            return new string(_buffer, start, length);
        }

        public char[]? InternalBuffer => _buffer;
    }
}