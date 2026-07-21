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

using System.Collections;
using System.Collections.Generic;

#nullable disable

namespace Aexis.Samples.Json.Bson
{
    internal abstract class BsonToken
    {
        public abstract BsonType Type { get; }
        public BsonToken Parent { get; set; }
        public int CalculatedSize { get; set; }
    }

    internal class BsonObject : BsonToken, IEnumerable<BsonProperty>
    {
        private readonly List<BsonProperty> _children = new List<BsonProperty>();

        public void Add(string name, BsonToken token)
        {
            _children.Add(new BsonProperty { Name = new BsonString(name, false), Value = token });
            token.Parent = this;
        }

        public override BsonType Type => BsonType.Object;

        public IEnumerator<BsonProperty> GetEnumerator()
        {
            return _children.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    internal class BsonArray : BsonToken, IEnumerable<BsonToken>
    {
        private readonly List<BsonToken> _children = new List<BsonToken>();

        public void Add(BsonToken token)
        {
            _children.Add(token);
            token.Parent = this;
        }

        public override BsonType Type => BsonType.Array;

        public IEnumerator<BsonToken> GetEnumerator()
        {
            return _children.GetEnumerator();
        }

        IEnumerator IEnumerable.GetEnumerator()
        {
            return GetEnumerator();
        }
    }

    internal class BsonEmpty : BsonToken
    {
        public static readonly BsonToken Null = new BsonEmpty(BsonType.Null);
        public static readonly BsonToken Undefined = new BsonEmpty(BsonType.Undefined);

        private BsonEmpty(BsonType type)
        {
            Type = type;
        }

        public override BsonType Type { get; }
    }

    internal class BsonValue : BsonToken
    {
        private readonly object _value;
        private readonly BsonType _type;

        public BsonValue(object value, BsonType type)
        {
            _value = value;
            _type = type;
        }

        public object Value => _value;

        public override BsonType Type => _type;
    }

    internal class BsonBoolean : BsonValue
    {
        public static readonly BsonBoolean False = new BsonBoolean(false);
        public static readonly BsonBoolean True = new BsonBoolean(true);

        private BsonBoolean(bool value)
            : base(value, BsonType.Boolean)
        {
        }
    }

    internal class BsonString : BsonValue
    {
        public int ByteCount { get; set; }
        public bool IncludeLength { get; }

        public BsonString(object value, bool includeLength)
            : base(value, BsonType.String)
        {
            IncludeLength = includeLength;
        }
    }

    internal class BsonBinary : BsonValue
    {
        public BsonBinaryType BinaryType { get; set; }

        public BsonBinary(byte[] value, BsonBinaryType binaryType)
            : base(value, BsonType.Binary)
        {
            BinaryType = binaryType;
        }
    }

    internal class BsonRegex : BsonToken
    {
        public BsonString Pattern { get; set; }
        public BsonString Options { get; set; }

        public BsonRegex(string pattern, string options)
        {
            Pattern = new BsonString(pattern, false);
            Options = new BsonString(options, false);
        }

        public override BsonType Type => BsonType.Regex;
    }

    internal class BsonProperty
    {
        public BsonString Name { get; set; }
        public BsonToken Value { get; set; }
    }
}