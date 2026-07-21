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
#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#else
using System.Linq;
#endif
using System.Reflection;

namespace Aexis.Samples.Json.Utilities
{
#if PORTABLE
    internal static class MethodBinder
    {
        
        /// <summary>
        /// List of primitive types which can be widened.
        /// </summary>
        private static readonly Type[] PrimitiveTypes = new Type[]
        {
            typeof(bool),  typeof(char),   typeof(sbyte), typeof(byte),
            typeof(short), typeof(ushort), typeof(int),   typeof(uint),
            typeof(long),  typeof(ulong),  typeof(float), typeof(double)
        };

        /// <summary>
        /// Widening masks for primitive types above.
        /// Index of the value in this array defines a type we're widening,
        /// while the bits in mask define types it can be widened to (including itself).
        /// 
        /// For example, value at index 0 defines a bool type, and it only has bit 0 set, 
        /// i.e. bool values can be assigned only to bool.
        /// </summary>
        private static readonly int[] WideningMasks = new int[]
        {
            0x0001,        0x0FE2,         0x0D54,        0x0FFA,
            0x0D50,        0x0FE2,         0x0D40,        0x0F80,
            0x0D00,        0x0E00,         0x0C00,        0x0800
        };

        /// <summary>
        /// Checks if value of primitive type <paramref name="from"/> can be  
        /// assigned to parameter of primitive type <paramref name="to"/>.
        /// </summary>
        /// <param name="from">Source primitive type.</param>
        /// <param name="to">Target primitive type.</param>
        /// <returns><c>true</c> if source type can be widened to target type, <c>false</c> otherwise.</returns>
        private static bool CanConvertPrimitive(Type from, Type to)
        {
            if (from == to)
            {
                // same type
                return true;
            }

            int fromMask = 0;
            int toMask = 0;

            for (int i = 0; i < PrimitiveTypes.Length; i++)
            {
                if (PrimitiveTypes[i] == from)
                {
                    fromMask = WideningMasks[i];
                }
                else if (PrimitiveTypes[i] == to)
                {
                    toMask = 1 << i;
                }

                if (fromMask != 0 && toMask != 0)
                {
                    break;
                }
            }

            return (fromMask & toMask) != 0;
        }

        /// <summary>
        /// Checks if a set of values with given <paramref name="types"/> can be used
        /// to invoke a method with specified <paramref name="parameters"/>. 
        /// </summary>
        /// <param name="parameters">Method parameters.</param>
        /// <param name="types">Argument types.</param>
        /// <param name="enableParamArray">Try to pack extra arguments into the last parameter when it is marked up with <see cref="ParamArrayAttribute"/>.</param>
        /// <returns><c>true</c> if method can be called with given arguments, <c>false</c> otherwise.</returns>
        private static bool FilterParameters(ParameterInfo[] parameters, IList<Type> types, bool enableParamArray)
        {
            ValidationUtils.ArgumentNotNull(parameters, nameof(parameters));
            ValidationUtils.ArgumentNotNull(types, nameof(types));

            if (parameters.Length == 0)
            {
                // fast check for parameterless methods
                return types.Count == 0;
            }
            if (parameters.Length > types.Count)
            {
                // not all declared parameters were specified (optional parameters are not supported)
                return false;
            }

            // check if the last parameter is ParamArray
            Type? paramArrayType = null;

            if (enableParamArray)
            {
                ParameterInfo lastParam = parameters[parameters.Length - 1];
                if (lastParam.ParameterType.IsArray && lastParam.IsDefined(typeof(ParamArrayAttribute)))
                {
                    paramArrayType = lastParam.ParameterType.GetElementType();
                }
            }

            if (paramArrayType == null && parameters.Length != types.Count)
            {
                // when there's no ParamArray, number of parameters should match
                return false;
            }

            for (int i = 0; i < types.Count; i++)
            {
                Type paramType = (paramArrayType != null && i >= parameters.Length - 1) ? paramArrayType : parameters[i].ParameterType;

                if (paramType == types[i])
                {
                    // exact match with provided type
                    continue;
                }

                if (paramType == typeof(object))
                {
                    // parameter of type object matches anything
                    continue;
                }

                if (paramType.IsPrimitive())
                {
                    if (!types[i].IsPrimitive() || !CanConvertPrimitive(types[i], paramType))
                    {
                        // primitive parameter can only be assigned from compatible primitive type
                        return false;
                    }
                }
                else
                {
                    if (!paramType.IsAssignableFrom(types[i]))
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        /// <summary>
        /// Compares two sets of parameters to determine 
        /// which one suits better for given argument types.
        /// </summary>
        private class ParametersMatchComparer : IComparer<ParameterInfo[]>
        {
            private readonly IList<Type> _types;
            private readonly bool _enableParamArray;

            public ParametersMatchComparer(IList<Type> types, bool enableParamArray)
            {
                ValidationUtils.ArgumentNotNull(types, nameof(types));

                _types = types;
                _enableParamArray = enableParamArray;
            }

            public int Compare(ParameterInfo[] parameters1, ParameterInfo[] parameters2)
            {
                ValidationUtils.ArgumentNotNull(parameters1, nameof(parameters1));
                ValidationUtils.ArgumentNotNull(parameters2, nameof(parameters2));

                // parameterless method wins
                if (parameters1.Length == 0)
                {
                    return -1;
                }
                if (parameters2.Length == 0)
                {
                    return 1;
                }

                Type? paramArrayType1 = null, paramArrayType2 = null;

                if (_enableParamArray)
                {
                    ParameterInfo lastParam1 = parameters1[parameters1.Length - 1];
                    if (lastParam1.ParameterType.IsArray && lastParam1.IsDefined(typeof(ParamArrayAttribute)))
                    {
                        paramArrayType1 = lastParam1.ParameterType.GetElementType();
                    }

                    ParameterInfo lastParam2 = parameters2[parameters2.Length - 1];
                    if (lastParam2.ParameterType.IsArray && lastParam2.IsDefined(typeof(ParamArrayAttribute)))
                    {
                        paramArrayType2 = lastParam2.ParameterType.GetElementType();
                    }

                    // A method using params always loses to one not using params
                    if (paramArrayType1 != null && paramArrayType2 == null)
                    {
                        return 1;
                    }
                    if (paramArrayType2 != null && paramArrayType1 == null)
                    {
                        return -1;
                    }
                }

                for (int i = 0; i < _types.Count; i++)
                {
                    Type type1 = (paramArrayType1 != null && i >= parameters1.Length - 1) ? paramArrayType1 : parameters1[i].ParameterType;
                    Type type2 = (paramArrayType2 != null && i >= parameters2.Length - 1) ? paramArrayType2 : parameters2[i].ParameterType;

                    if (type1 == type2)
                    {
                        // exact match between parameter types doesn't change score
                        continue;
                    }

                    // exact match with source type decides winner immediately
                    if (type1 == _types[i])
                    {
                        return -1;
                    }
                    if (type2 == _types[i])
                    {
                        return 1;
                    }

                    int r = ChooseMorePreciseType(type1, type2);
                    if (r != 0)
                    {
                        // winner decided 
                        return r;
                    }
                }

                return 0;
            }

            private static int ChooseMorePreciseType(Type type1, Type type2)
            {
                if (type1.IsByRef || type2.IsByRef)
                {
                    if (type1.IsByRef && type2.IsByRef)
                    {
                        type1 = type1.GetElementType();
                        type2 = type2.GetElementType();
                    }
                    else if (type1.IsByRef)
                    {
                        type1 = type1.GetElementType();
                        if (type1 == type2)
                        {
                            return 1;
                        }
                    }
                    else
                    {
                        type2 = type2.GetElementType();
                        if (type2 == type1)
                        {
                            return -1;
                        }
                    }
                }

                bool c1FromC2, c2FromC1;

                if (type1.IsPrimitive() && type2.IsPrimitive())
                {
                    c1FromC2 = CanConvertPrimitive(type2, type1);
                    c2FromC1 = CanConvertPrimitive(type1, type2);
                }
                else
                {
                    c1FromC2 = type1.IsAssignableFrom(type2);
                    c2FromC1 = type2.IsAssignableFrom(type1);
                }

                if (c1FromC2 == c2FromC1)
                {
                    return 0;
                }

                return c1FromC2 ? 1 : -1;
            }

        }

        /// <summary>
        /// Returns a best method overload for given argument <paramref name="types"/>.
        /// </summary>
        /// <param name="candidates">List of method candidates.</param>
        /// <param name="types">Argument types.</param>
        /// <returns>Best method overload, or <c>null</c> if none matched.</returns>
        public static TMethod SelectMethod<TMethod>(IEnumerable<TMethod> candidates, IList<Type> types) where TMethod : MethodBase
        {
            ValidationUtils.ArgumentNotNull(candidates, nameof(candidates));
            ValidationUtils.ArgumentNotNull(types, nameof(types));

            // ParamArrays are not supported by ReflectionDelegateFactory
            // They will be treated like ordinary array arguments
            const bool enableParamArray = false;

            return candidates
                .Where(m => FilterParameters(m.GetParameters(), types, enableParamArray))
                .OrderBy(m => m.GetParameters(), new ParametersMatchComparer(types, enableParamArray))
                .FirstOrDefault();
        }

    }
#endif
}