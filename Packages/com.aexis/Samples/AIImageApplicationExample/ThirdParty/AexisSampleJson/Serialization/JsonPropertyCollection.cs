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
using System.Collections.Generic;
using System.Text;
using System.Collections.ObjectModel;
using Aexis.Samples.Json.Utilities;
using System.Globalization;
using System.Runtime.CompilerServices;
using System.Diagnostics;
using System.Diagnostics.CodeAnalysis;

namespace Aexis.Samples.Json.Serialization
{
    /// <summary>
    /// A collection of <see cref="JsonProperty"/> objects.
    /// </summary>
    public class JsonPropertyCollection : KeyedCollection<string, JsonProperty>
    {
        private readonly Type _type;
        private readonly List<JsonProperty> _list;

        /// <summary>
        /// Initializes a new instance of the <see cref="JsonPropertyCollection"/> class.
        /// </summary>
        /// <param name="type">The type.</param>
        public JsonPropertyCollection(Type type)
            : base(StringComparer.Ordinal)
        {
            ValidationUtils.ArgumentNotNull(type, "type");
            _type = type;

            // foreach over List<T> to avoid boxing the Enumerator
            _list = (List<JsonProperty>)Items;
        }

        /// <summary>
        /// When implemented in a derived class, extracts the key from the specified element.
        /// </summary>
        /// <param name="item">The element from which to extract the key.</param>
        /// <returns>The key for the specified element.</returns>
        protected override string GetKeyForItem(JsonProperty item)
        {
            return item.PropertyName!;
        }

        /// <summary>
        /// Adds a <see cref="JsonProperty"/> object.
        /// </summary>
        /// <param name="property">The property to add to the collection.</param>
        public void AddProperty(JsonProperty property)
        {
            MiscellaneousUtils.Assert(property.PropertyName != null);

            if (Contains(property.PropertyName))
            {
                // don't overwrite existing property with ignored property
                if (property.Ignored)
                {
                    return;
                }

                JsonProperty existingProperty = this[property.PropertyName];
                bool duplicateProperty = true;

                if (existingProperty.Ignored)
                {
                    // remove ignored property so it can be replaced in collection
                    Remove(existingProperty);
                    duplicateProperty = false;
                }
                else
                {
                    if (property.DeclaringType != null && existingProperty.DeclaringType != null)
                    {
                        if (property.DeclaringType.IsSubclassOf(existingProperty.DeclaringType)
                            || (existingProperty.DeclaringType.IsInterface() && property.DeclaringType.ImplementInterface(existingProperty.DeclaringType)))
                        {
                            // current property is on a derived class and hides the existing
                            Remove(existingProperty);
                            duplicateProperty = false;
                        }
                        if (existingProperty.DeclaringType.IsSubclassOf(property.DeclaringType)
                            || (property.DeclaringType.IsInterface() && existingProperty.DeclaringType.ImplementInterface(property.DeclaringType)))
                        {
                            // current property is hidden by the existing so don't add it
                            return;
                        }
                        
                        if (_type.ImplementInterface(existingProperty.DeclaringType) && _type.ImplementInterface(property.DeclaringType))
                        {
                            // current property was already defined on another interface
                            return;
                        }
                    }
                }

                if (duplicateProperty)
                {
                    throw new JsonSerializationException("A member with the name '{0}' already exists on '{1}'. Use the JsonPropertyAttribute to specify another name.".FormatWith(CultureInfo.InvariantCulture, property.PropertyName, _type));
                }
            }

            Add(property);
        }

        /// <summary>
        /// Gets the closest matching <see cref="JsonProperty"/> object.
        /// First attempts to get an exact case match of <paramref name="propertyName"/> and then
        /// a case insensitive match.
        /// </summary>
        /// <param name="propertyName">Name of the property.</param>
        /// <returns>A matching property if found.</returns>
        public JsonProperty? GetClosestMatchProperty(string propertyName)
        {
            JsonProperty? property = GetProperty(propertyName, StringComparison.Ordinal);
            if (property == null)
            {
                property = GetProperty(propertyName, StringComparison.OrdinalIgnoreCase);
            }

            return property;
        }

        private bool TryGetProperty(string key, [NotNullWhen(true)]out JsonProperty? item)
        {
            if (Dictionary == null)
            {
                item = default;
                return false;
            }

            return Dictionary.TryGetValue(key, out item);
        }

        /// <summary>
        /// Gets a property by property name.
        /// </summary>
        /// <param name="propertyName">The name of the property to get.</param>
        /// <param name="comparisonType">Type property name string comparison.</param>
        /// <returns>A matching property if found.</returns>
        public JsonProperty? GetProperty(string propertyName, StringComparison comparisonType)
        {
            // KeyedCollection has an ordinal comparer
            if (comparisonType == StringComparison.Ordinal)
            {
                if (TryGetProperty(propertyName, out JsonProperty? property))
                {
                    return property;
                }

                return null;
            }

            for (int i = 0; i < _list.Count; i++)
            {
                JsonProperty property = _list[i];
                if (string.Equals(propertyName, property.PropertyName, comparisonType))
                {
                    return property;
                }
            }

            return null;
        }
    }
}
