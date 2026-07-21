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
// Copyright (c) 2022 James Newton-King
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

namespace Aexis.Samples.Json.Utilities
{
    internal static class BoxedPrimitives
    {
        internal static object Get(bool value) => value ? BooleanTrue : BooleanFalse;

        internal static readonly object BooleanTrue = true;
        internal static readonly object BooleanFalse = false;

        internal static object Get(int value) => value switch
        {
            -1 => Int32_M1,
            0 => Int32_0,
            1 => Int32_1,
            2 => Int32_2,
            3 => Int32_3,
            4 => Int32_4,
            5 => Int32_5,
            6 => Int32_6,
            7 => Int32_7,
            8 => Int32_8,
            _ => value,
        };

        // integers tend to be weighted towards a handful of low numbers; we could argue
        // for days over the "correct" range to have special handling, but I'm arbitrarily
        // mirroring the same decision as the IL opcodes, which has M1 thru 8
        internal static readonly object Int32_M1 = -1;
        internal static readonly object Int32_0 = 0;
        internal static readonly object Int32_1 = 1;
        internal static readonly object Int32_2 = 2;
        internal static readonly object Int32_3 = 3;
        internal static readonly object Int32_4 = 4;
        internal static readonly object Int32_5 = 5;
        internal static readonly object Int32_6 = 6;
        internal static readonly object Int32_7 = 7;
        internal static readonly object Int32_8 = 8;

        internal static object Get(long value) => value switch
        {
            -1 => Int64_M1,
            0 => Int64_0,
            1 => Int64_1,
            2 => Int64_2,
            3 => Int64_3,
            4 => Int64_4,
            5 => Int64_5,
            6 => Int64_6,
            7 => Int64_7,
            8 => Int64_8,
            _ => value,
        };

        internal static readonly object Int64_M1 = -1L;
        internal static readonly object Int64_0 = 0L;
        internal static readonly object Int64_1 = 1L;
        internal static readonly object Int64_2 = 2L;
        internal static readonly object Int64_3 = 3L;
        internal static readonly object Int64_4 = 4L;
        internal static readonly object Int64_5 = 5L;
        internal static readonly object Int64_6 = 6L;
        internal static readonly object Int64_7 = 7L;
        internal static readonly object Int64_8 = 8L;

        internal static object Get(decimal value) => value == decimal.Zero ? DecimalZero : value;

        private static readonly object DecimalZero = decimal.Zero;

        internal static object Get(double value)
        {
            if (value == 0.0d)
            {
                return DoubleZero;
            }
            if (double.IsInfinity(value))
            {
                return double.IsPositiveInfinity(value) ? DoublePositiveInfinity : DoubleNegativeInfinity;
            }
            if (double.IsNaN(value))
            {
                return DoubleNaN;
            }
            return value;
        }

        internal static readonly object DoubleNaN = double.NaN;
        internal static readonly object DoublePositiveInfinity = double.PositiveInfinity;
        internal static readonly object DoubleNegativeInfinity = double.NegativeInfinity;
        internal static readonly object DoubleZero = (double)0;
    }
}
