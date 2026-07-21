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
using System.Diagnostics;
using System.Reflection;
using Aexis.Samples.Json.Utilities;

#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#endif

namespace Aexis.Samples.Json.Serialization
{
    /// <summary>
    /// Maps a JSON property to a .NET member or constructor parameter.
    /// </summary>
    public class JsonProperty
    {
        internal Required? _required;
        internal bool _hasExplicitDefaultValue;

        private object? _defaultValue;
        private bool _hasGeneratedDefaultValue;
        private string? _propertyName;
        internal bool _skipPropertyNameEscape;
        private Type? _propertyType;

        // use to cache contract during deserialization
        internal JsonContract? PropertyContract { get; set; }

        /// <summary>
        /// Gets or sets the name of the property.
        /// </summary>
        /// <value>The name of the property.</value>
        public string? PropertyName
        {
            get => _propertyName;
            set
            {
                _propertyName = value;
                _skipPropertyNameEscape = !JavaScriptUtils.ShouldEscapeJavaScriptString(_propertyName, JavaScriptUtils.HtmlCharEscapeFlags);
            }
        }

        /// <summary>
        /// Gets or sets the type that declared this property.
        /// </summary>
        /// <value>The type that declared this property.</value>
        public Type? DeclaringType { get; set; }

        /// <summary>
        /// Gets or sets the order of serialization of a member.
        /// </summary>
        /// <value>The numeric order of serialization.</value>
        public int? Order { get; set; }

        /// <summary>
        /// Gets or sets the name of the underlying member or parameter.
        /// </summary>
        /// <value>The name of the underlying member or parameter.</value>
        public string? UnderlyingName { get; set; }

        /// <summary>
        /// Gets the <see cref="IValueProvider"/> that will get and set the <see cref="JsonProperty"/> during serialization.
        /// </summary>
        /// <value>The <see cref="IValueProvider"/> that will get and set the <see cref="JsonProperty"/> during serialization.</value>
        public IValueProvider? ValueProvider { get; set; }

        /// <summary>
        /// Gets or sets the <see cref="IAttributeProvider"/> for this property.
        /// </summary>
        /// <value>The <see cref="IAttributeProvider"/> for this property.</value>
        public IAttributeProvider? AttributeProvider { get; set; }

        /// <summary>
        /// Gets or sets the type of the property.
        /// </summary>
        /// <value>The type of the property.</value>
        public Type? PropertyType
        {
            get => _propertyType;
            set
            {
                if (_propertyType != value)
                {
                    _propertyType = value;
                    _hasGeneratedDefaultValue = false;
                }
            }
        }

        /// <summary>
        /// Gets or sets the <see cref="JsonConverter" /> for the property.
        /// If set this converter takes precedence over the contract converter for the property type.
        /// </summary>
        /// <value>The converter.</value>
        public JsonConverter? Converter { get; set; }

        /// <summary>
        /// Gets or sets the member converter.
        /// </summary>
        /// <value>The member converter.</value>
        [Obsolete("MemberConverter is obsolete. Use Converter instead.")]
        public JsonConverter? MemberConverter
        {
            get => Converter;
            set => Converter = value;
        }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="JsonProperty"/> is ignored.
        /// </summary>
        /// <value><c>true</c> if ignored; otherwise, <c>false</c>.</value>
        public bool Ignored { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="JsonProperty"/> is readable.
        /// </summary>
        /// <value><c>true</c> if readable; otherwise, <c>false</c>.</value>
        public bool Readable { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="JsonProperty"/> is writable.
        /// </summary>
        /// <value><c>true</c> if writable; otherwise, <c>false</c>.</value>
        public bool Writable { get; set; }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="JsonProperty"/> has a member attribute.
        /// </summary>
        /// <value><c>true</c> if has a member attribute; otherwise, <c>false</c>.</value>
        public bool HasMemberAttribute { get; set; }

        /// <summary>
        /// Gets the default value.
        /// </summary>
        /// <value>The default value.</value>
        public object? DefaultValue
        {
            get
            {
                if (!_hasExplicitDefaultValue)
                {
                    return null;
                }

                return _defaultValue;
            }
            set
            {
                _hasExplicitDefaultValue = true;
                _defaultValue = value;
            }
        }

        internal object? GetResolvedDefaultValue()
        {
            if (_propertyType == null)
            {
                return null;
            }

            if (!_hasExplicitDefaultValue && !_hasGeneratedDefaultValue)
            {
                _defaultValue = ReflectionUtils.GetDefaultValue(_propertyType);
                _hasGeneratedDefaultValue = true;
            }

            return _defaultValue;
        }

        /// <summary>
        /// Gets or sets a value indicating whether this <see cref="JsonProperty"/> is required.
        /// </summary>
        /// <value>A value indicating whether this <see cref="JsonProperty"/> is required.</value>
        public Required Required
        {
            get => _required ?? Required.Default;
            set => _required = value;
        }

        /// <summary>
        /// Gets a value indicating whether <see cref="Required"/> has a value specified.
        /// </summary>
        public bool IsRequiredSpecified => _required != null;

        /// <summary>
        /// Gets or sets a value indicating whether this property preserves object references.
        /// </summary>
        /// <value>
        /// 	<c>true</c> if this instance is reference; otherwise, <c>false</c>.
        /// </value>
        public bool? IsReference { get; set; }

        /// <summary>
        /// Gets or sets the property null value handling.
        /// </summary>
        /// <value>The null value handling.</value>
        public NullValueHandling? NullValueHandling { get; set; }

        /// <summary>
        /// Gets or sets the property default value handling.
        /// </summary>
        /// <value>The default value handling.</value>
        public DefaultValueHandling? DefaultValueHandling { get; set; }

        /// <summary>
        /// Gets or sets the property reference loop handling.
        /// </summary>
        /// <value>The reference loop handling.</value>
        public ReferenceLoopHandling? ReferenceLoopHandling { get; set; }

        /// <summary>
        /// Gets or sets the property object creation handling.
        /// </summary>
        /// <value>The object creation handling.</value>
        public ObjectCreationHandling? ObjectCreationHandling { get; set; }

        /// <summary>
        /// Gets or sets or sets the type name handling.
        /// </summary>
        /// <value>The type name handling.</value>
        public TypeNameHandling? TypeNameHandling { get; set; }

        /// <summary>
        /// Gets or sets a predicate used to determine whether the property should be serialized.
        /// </summary>
        /// <value>A predicate used to determine whether the property should be serialized.</value>
        public Predicate<object>? ShouldSerialize { get; set; }

        /// <summary>
        /// Gets or sets a predicate used to determine whether the property should be deserialized.
        /// </summary>
        /// <value>A predicate used to determine whether the property should be deserialized.</value>
        public Predicate<object>? ShouldDeserialize { get; set; }

        /// <summary>
        /// Gets or sets a predicate used to determine whether the property should be serialized.
        /// </summary>
        /// <value>A predicate used to determine whether the property should be serialized.</value>
        public Predicate<object>? GetIsSpecified { get; set; }

        /// <summary>
        /// Gets or sets an action used to set whether the property has been deserialized.
        /// </summary>
        /// <value>An action used to set whether the property has been deserialized.</value>
        public Action<object, object?>? SetIsSpecified { get; set; }

        /// <summary>
        /// Returns a <see cref="String"/> that represents this instance.
        /// </summary>
        /// <returns>
        /// A <see cref="String"/> that represents this instance.
        /// </returns>
        public override string ToString()
        {
            return PropertyName ?? string.Empty;
        }

        /// <summary>
        /// Gets or sets the converter used when serializing the property's collection items.
        /// </summary>
        /// <value>The collection's items converter.</value>
        public JsonConverter? ItemConverter { get; set; }

        /// <summary>
        /// Gets or sets whether this property's collection items are serialized as a reference.
        /// </summary>
        /// <value>Whether this property's collection items are serialized as a reference.</value>
        public bool? ItemIsReference { get; set; }

        /// <summary>
        /// Gets or sets the type name handling used when serializing the property's collection items.
        /// </summary>
        /// <value>The collection's items type name handling.</value>
        public TypeNameHandling? ItemTypeNameHandling { get; set; }

        /// <summary>
        /// Gets or sets the reference loop handling used when serializing the property's collection items.
        /// </summary>
        /// <value>The collection's items reference loop handling.</value>
        public ReferenceLoopHandling? ItemReferenceLoopHandling { get; set; }

        internal void WritePropertyName(JsonWriter writer)
        {
            string? propertyName = PropertyName;
            MiscellaneousUtils.Assert(propertyName != null);

            if (_skipPropertyNameEscape)
            {
                writer.WritePropertyName(propertyName, false);
            }
            else
            {
                writer.WritePropertyName(propertyName);
            }
        }
    }
}