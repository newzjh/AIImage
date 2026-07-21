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
using Aexis.Samples.Json.Utilities;
using System.Globalization;
using System.Diagnostics.CodeAnalysis;
using System.Runtime.CompilerServices;
#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#else
using System.Linq;

#endif

namespace Aexis.Samples.Json.Linq
{
    /// <summary>
    /// Contains the LINQ to JSON extension methods.
    /// </summary>
    public static class Extensions
    {
        /// <summary>
        /// Returns a collection of tokens that contains the ancestors of every token in the source collection.
        /// </summary>
        /// <typeparam name="T">The type of the objects in source, constrained to <see cref="JToken"/>.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the ancestors of every token in the source collection.</returns>
        public static IJEnumerable<JToken> Ancestors<T>(this IEnumerable<T> source) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(j => j.Ancestors()).AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of tokens that contains every token in the source collection, and the ancestors of every token in the source collection.
        /// </summary>
        /// <typeparam name="T">The type of the objects in source, constrained to <see cref="JToken"/>.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains every token in the source collection, the ancestors of every token in the source collection.</returns>
        public static IJEnumerable<JToken> AncestorsAndSelf<T>(this IEnumerable<T> source) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(j => j.AncestorsAndSelf()).AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of tokens that contains the descendants of every token in the source collection.
        /// </summary>
        /// <typeparam name="T">The type of the objects in source, constrained to <see cref="JContainer"/>.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the descendants of every token in the source collection.</returns>
        public static IJEnumerable<JToken> Descendants<T>(this IEnumerable<T> source) where T : JContainer
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(j => j.Descendants()).AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of tokens that contains every token in the source collection, and the descendants of every token in the source collection.
        /// </summary>
        /// <typeparam name="T">The type of the objects in source, constrained to <see cref="JContainer"/>.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains every token in the source collection, and the descendants of every token in the source collection.</returns>
        public static IJEnumerable<JToken> DescendantsAndSelf<T>(this IEnumerable<T> source) where T : JContainer
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(j => j.DescendantsAndSelf()).AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of child properties of every object in the source collection.
        /// </summary>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JObject"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JProperty"/> that contains the properties of every object in the source collection.</returns>
        public static IJEnumerable<JProperty> Properties(this IEnumerable<JObject> source)
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(d => d.Properties()).AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of child values of every object in the source collection with the given key.
        /// </summary>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <param name="key">The token key.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the values of every token in the source collection with the given key.</returns>
        public static IJEnumerable<JToken> Values(this IEnumerable<JToken> source, object? key)
        {
            return Values<JToken, JToken>(source, key)!.AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of child values of every object in the source collection.
        /// </summary>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the values of every token in the source collection.</returns>
        public static IJEnumerable<JToken> Values(this IEnumerable<JToken> source)
        {
            return source.Values(null);
        }

        /// <summary>
        /// Returns a collection of converted child values of every object in the source collection with the given key.
        /// </summary>
        /// <typeparam name="U">The type to convert the values to.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <param name="key">The token key.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> that contains the converted values of every token in the source collection with the given key.</returns>
        public static IEnumerable<U?> Values<U>(this IEnumerable<JToken> source, object key)
        {
            return Values<JToken, U>(source, key);
        }

        /// <summary>
        /// Returns a collection of converted child values of every object in the source collection.
        /// </summary>
        /// <typeparam name="U">The type to convert the values to.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> that contains the converted values of every token in the source collection.</returns>
        public static IEnumerable<U?> Values<U>(this IEnumerable<JToken> source)
        {
            return Values<JToken, U>(source, null);
        }

        /// <summary>
        /// Converts the value.
        /// </summary>
        /// <typeparam name="U">The type to convert the value to.</typeparam>
        /// <param name="value">A <see cref="JToken"/> cast as a <see cref="IEnumerable{T}"/> of <see cref="JToken"/>.</param>
        /// <returns>A converted value.</returns>
        public static U? Value<U>(this IEnumerable<JToken> value)
        {
            return value.Value<JToken, U>();
        }

        /// <summary>
        /// Converts the value.
        /// </summary>
        /// <typeparam name="T">The source collection type.</typeparam>
        /// <typeparam name="U">The type to convert the value to.</typeparam>
        /// <param name="value">A <see cref="JToken"/> cast as a <see cref="IEnumerable{T}"/> of <see cref="JToken"/>.</param>
        /// <returns>A converted value.</returns>
        public static U? Value<T, U>(this IEnumerable<T> value) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(value, nameof(value));

            if (!(value is JToken token))
            {
                throw new ArgumentException("Source value must be a JToken.");
            }

            return token.Convert<JToken, U>();
        }

        internal static IEnumerable<U?> Values<T, U>(this IEnumerable<T> source, object? key) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            if (key == null)
            {
                foreach (T token in source)
                {
                    if (token is JValue value)
                    {
                        yield return Convert<JValue, U>(value);
                    }
                    else
                    {
                        foreach (JToken t in token.Children())
                        {
                            yield return t.Convert<JToken, U>();
                        }
                    }
                }
            }
            else
            {
                foreach (T token in source)
                {
                    JToken? value = token[key];
                    if (value != null)
                    {
                        yield return value.Convert<JToken, U>();
                    }
                }
            }
        }

        //TODO
        //public static IEnumerable<T> InDocumentOrder<T>(this IEnumerable<T> source) where T : JObject;

        /// <summary>
        /// Returns a collection of child tokens of every array in the source collection.
        /// </summary>
        /// <typeparam name="T">The source collection type.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the values of every token in the source collection.</returns>
        public static IJEnumerable<JToken> Children<T>(this IEnumerable<T> source) where T : JToken
        {
            return Children<T, JToken>(source)!.AsJEnumerable();
        }

        /// <summary>
        /// Returns a collection of converted child tokens of every array in the source collection.
        /// </summary>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <typeparam name="U">The type to convert the values to.</typeparam>
        /// <typeparam name="T">The source collection type.</typeparam>
        /// <returns>An <see cref="IEnumerable{T}"/> that contains the converted values of every token in the source collection.</returns>
        public static IEnumerable<U?> Children<T, U>(this IEnumerable<T> source) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            return source.SelectMany(c => c.Children()).Convert<JToken, U>();
        }

        internal static IEnumerable<U?> Convert<T, U>(this IEnumerable<T> source) where T : JToken
        {
            ValidationUtils.ArgumentNotNull(source, nameof(source));

            foreach (T token in source)
            {
                yield return Convert<JToken, U>(token);
            }
        }

        internal static U? Convert<T, U>(this T token) where T : JToken?
        {
            if (token == null)
            {
#pragma warning disable CS8653 // A default expression introduces a null value for a type parameter.
                return default;
#pragma warning restore CS8653 // A default expression introduces a null value for a type parameter.
            }

            if (token is U castValue
                // don't want to cast JValue to its interfaces, want to get the internal value
                && typeof(U) != typeof(IComparable) && typeof(U) != typeof(IFormattable))
            {
                return castValue;
            }
            else
            {
                if (!(token is JValue value))
                {
                    throw new InvalidCastException("Cannot cast {0} to {1}.".FormatWith(CultureInfo.InvariantCulture, token.GetType(), typeof(T)));
                }

                if (value.Value is U u)
                {
                    return u;
                }

                Type targetType = typeof(U);

                if (ReflectionUtils.IsNullableType(targetType))
                {
                    if (value.Value == null)
                    {
#pragma warning disable CS8653 // A default expression introduces a null value for a type parameter.
                        return default;
#pragma warning restore CS8653 // A default expression introduces a null value for a type parameter.
                    }

                    targetType = Nullable.GetUnderlyingType(targetType)!;
                }

                return (U?)System.Convert.ChangeType(value.Value, targetType, CultureInfo.InvariantCulture);
            }
        }

        //TODO
        //public static void Remove<T>(this IEnumerable<T> source) where T : JContainer;

        /// <summary>
        /// Returns the input typed as <see cref="IJEnumerable{T}"/>.
        /// </summary>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>The input typed as <see cref="IJEnumerable{T}"/>.</returns>
        public static IJEnumerable<JToken> AsJEnumerable(this IEnumerable<JToken> source)
        {
            return source.AsJEnumerable<JToken>();
        }

        /// <summary>
        /// Returns the input typed as <see cref="IJEnumerable{T}"/>.
        /// </summary>
        /// <typeparam name="T">The source collection type.</typeparam>
        /// <param name="source">An <see cref="IEnumerable{T}"/> of <see cref="JToken"/> that contains the source collection.</param>
        /// <returns>The input typed as <see cref="IJEnumerable{T}"/>.</returns>
        public static IJEnumerable<T> AsJEnumerable<T>(this IEnumerable<T> source) where T : JToken
        {
            if (source == null)
            {
                return null!;
            }
            else if (source is IJEnumerable<T> customEnumerable)
            {
                return customEnumerable;
            }
            else
            {
                return new JEnumerable<T>(source);
            }
        }
    }
}