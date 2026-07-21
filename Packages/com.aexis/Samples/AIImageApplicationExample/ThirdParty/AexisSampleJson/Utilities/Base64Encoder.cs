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
using System.IO;
#if HAVE_ASYNC
using System.Threading;
using System.Threading.Tasks;
#endif

namespace Aexis.Samples.Json.Utilities
{
    internal class Base64Encoder
    {
        private const int Base64LineSize = 76;
        private const int LineSizeInBytes = 57;

        private readonly char[] _charsLine = new char[Base64LineSize];
        private readonly TextWriter _writer;

        private byte[]? _leftOverBytes;
        private int _leftOverBytesCount;

        public Base64Encoder(TextWriter writer)
        {
            ValidationUtils.ArgumentNotNull(writer, nameof(writer));
            _writer = writer;
        }

        private void ValidateEncode(byte[] buffer, int index, int count)
        {
            if (buffer == null)
            {
                throw new ArgumentNullException(nameof(buffer));
            }

            if (index < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(index));
            }

            if (count < 0)
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }

            if (count > (buffer.Length - index))
            {
                throw new ArgumentOutOfRangeException(nameof(count));
            }
        }

        public void Encode(byte[] buffer, int index, int count)
        {
            ValidateEncode(buffer, index, count);

            if (_leftOverBytesCount > 0)
            {
                if (FulfillFromLeftover(buffer, index, ref count))
                {
                    return;
                }

                int num2 = Convert.ToBase64CharArray(_leftOverBytes!, 0, 3, _charsLine, 0);
                WriteChars(_charsLine, 0, num2);
            }

            StoreLeftOverBytes(buffer, index, ref count);

            int num4 = index + count;
            int length = LineSizeInBytes;
            while (index < num4)
            {
                if ((index + length) > num4)
                {
                    length = num4 - index;
                }
                int num6 = Convert.ToBase64CharArray(buffer, index, length, _charsLine, 0);
                WriteChars(_charsLine, 0, num6);
                index += length;
            }
        }

        private void StoreLeftOverBytes(byte[] buffer, int index, ref int count)
        {
            int leftOverBytesCount = count % 3;
            if (leftOverBytesCount > 0)
            {
                count -= leftOverBytesCount;
                if (_leftOverBytes == null)
                {
                    _leftOverBytes = new byte[3];
                }

                for (int i = 0; i < leftOverBytesCount; i++)
                {
                    _leftOverBytes[i] = buffer[index + count + i];
                }
            }

            _leftOverBytesCount = leftOverBytesCount;
        }

        private bool FulfillFromLeftover(byte[] buffer, int index, ref int count)
        {
            int leftOverBytesCount = _leftOverBytesCount;
            while (leftOverBytesCount < 3 && count > 0)
            {
                _leftOverBytes![leftOverBytesCount++] = buffer[index++];
                count--;
            }

            if (count == 0 && leftOverBytesCount < 3)
            {
                _leftOverBytesCount = leftOverBytesCount;
                return true;
            }

            return false;
        }

        public void Flush()
        {
            if (_leftOverBytesCount > 0)
            {
                int count = Convert.ToBase64CharArray(_leftOverBytes!, 0, _leftOverBytesCount, _charsLine, 0);
                WriteChars(_charsLine, 0, count);
                _leftOverBytesCount = 0;
            }
        }

        private void WriteChars(char[] chars, int index, int count)
        {
            _writer.Write(chars, index, count);
        }

#if HAVE_ASYNC

        public async Task EncodeAsync(byte[] buffer, int index, int count, CancellationToken cancellationToken)
        {
            ValidateEncode(buffer, index, count);

            if (_leftOverBytesCount > 0)
            {
                if (FulfillFromLeftover(buffer, index, ref count))
                {
                    return;
                }

                int num2 = Convert.ToBase64CharArray(_leftOverBytes!, 0, 3, _charsLine, 0);
                await WriteCharsAsync(_charsLine, 0, num2, cancellationToken).ConfigureAwait(false);
            }

            StoreLeftOverBytes(buffer, index, ref count);

            int num4 = index + count;
            int length = LineSizeInBytes;
            while (index < num4)
            {
                if (index + length > num4)
                {
                    length = num4 - index;
                }
                int num6 = Convert.ToBase64CharArray(buffer, index, length, _charsLine, 0);
                await WriteCharsAsync(_charsLine, 0, num6, cancellationToken).ConfigureAwait(false);
                index += length;
            }
        }

        private Task WriteCharsAsync(char[] chars, int index, int count, CancellationToken cancellationToken)
        {
            return _writer.WriteAsync(chars, index, count, cancellationToken);
        }

        public Task FlushAsync(CancellationToken cancellationToken)
        {
            if (cancellationToken.IsCancellationRequested)
            {
                return cancellationToken.FromCanceled();
            }

            if (_leftOverBytesCount > 0)
            {
                int count = Convert.ToBase64CharArray(_leftOverBytes!, 0, _leftOverBytesCount, _charsLine, 0);
                _leftOverBytesCount = 0;
                return WriteCharsAsync(_charsLine, 0, count, cancellationToken);
            }

            return AsyncUtils.CompletedTask;
        }

#endif

    }
}