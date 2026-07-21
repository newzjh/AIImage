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

namespace Aexis.Samples.Json.Serialization
{
    /// <summary>
    /// A base class for resolving how property names and dictionary keys are serialized.
    /// </summary>
    public abstract class NamingStrategy
    {
        /// <summary>
        /// A flag indicating whether dictionary keys should be processed.
        /// Defaults to <c>false</c>.
        /// </summary>
        public bool ProcessDictionaryKeys { get; set; }

        /// <summary>
        /// A flag indicating whether extension data names should be processed.
        /// Defaults to <c>false</c>.
        /// </summary>
        public bool ProcessExtensionDataNames { get; set; }

        /// <summary>
        /// A flag indicating whether explicitly specified property names,
        /// e.g. a property name customized with a <see cref="JsonPropertyAttribute"/>, should be processed.
        /// Defaults to <c>false</c>.
        /// </summary>
        public bool OverrideSpecifiedNames { get; set; }

        /// <summary>
        /// Gets the serialized name for a given property name.
        /// </summary>
        /// <param name="name">The initial property name.</param>
        /// <param name="hasSpecifiedName">A flag indicating whether the property has had a name explicitly specified.</param>
        /// <returns>The serialized property name.</returns>
        public virtual string GetPropertyName(string name, bool hasSpecifiedName)
        {
            if (hasSpecifiedName && !OverrideSpecifiedNames)
            {
                return name;
            }

            return ResolvePropertyName(name);
        }

        /// <summary>
        /// Gets the serialized name for a given extension data name.
        /// </summary>
        /// <param name="name">The initial extension data name.</param>
        /// <returns>The serialized extension data name.</returns>
        public virtual string GetExtensionDataName(string name)
        {
            if (!ProcessExtensionDataNames)
            {
                return name;
            }

            return ResolvePropertyName(name);
        }

        /// <summary>
        /// Gets the serialized key for a given dictionary key.
        /// </summary>
        /// <param name="key">The initial dictionary key.</param>
        /// <returns>The serialized dictionary key.</returns>
        public virtual string GetDictionaryKey(string key)
        {
            if (!ProcessDictionaryKeys)
            {
                return key;
            }

            return ResolvePropertyName(key);
        }

        /// <summary>
        /// Resolves the specified property name.
        /// </summary>
        /// <param name="name">The property name to resolve.</param>
        /// <returns>The resolved property name.</returns>
        protected abstract string ResolvePropertyName(string name);

        /// <summary>
        /// Hash code calculation
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            unchecked
            {
                var hashCode = GetType().GetHashCode();     // make sure different types do not result in equal values
                hashCode = (hashCode * 397) ^ ProcessDictionaryKeys.GetHashCode();
                hashCode = (hashCode * 397) ^ ProcessExtensionDataNames.GetHashCode();
                hashCode = (hashCode * 397) ^ OverrideSpecifiedNames.GetHashCode();
                return hashCode;
            }
        }

        /// <summary>
        /// Object equality implementation
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(object? obj) => Equals(obj as NamingStrategy);

        /// <summary>
        /// Compare to another NamingStrategy
        /// </summary>
        /// <param name="other"></param>
        /// <returns></returns>
        protected bool Equals(NamingStrategy? other)
        {
            if (other == null)
            {
                return false;
            }

            return GetType() == other.GetType() &&
                ProcessDictionaryKeys == other.ProcessDictionaryKeys &&
                ProcessExtensionDataNames == other.ProcessExtensionDataNames &&
                OverrideSpecifiedNames == other.OverrideSpecifiedNames;
        }
    }
}