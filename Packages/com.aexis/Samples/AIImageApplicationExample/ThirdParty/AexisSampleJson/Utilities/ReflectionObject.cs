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

using Aexis.Samples.Json.Serialization;
using System;
using System.Collections.Generic;
using System.Reflection;
using System.Globalization;
#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#else
using System.Linq;

#endif

namespace Aexis.Samples.Json.Utilities
{
    internal class ReflectionMember
    {
        public Type? MemberType { get; set; }
        public Func<object, object?>? Getter { get; set; }
        public Action<object, object?>? Setter { get; set; }
    }

    internal class ReflectionObject
    {
        public ObjectConstructor<object>? Creator { get; }
        public IDictionary<string, ReflectionMember> Members { get; }

        private ReflectionObject(ObjectConstructor<object>? creator)
        {
            Members = new Dictionary<string, ReflectionMember>();
            Creator = creator;
        }

        public object? GetValue(object target, string member)
        {
            Func<object, object?> getter = Members[member].Getter!;
            return getter(target);
        }

        public void SetValue(object target, string member, object? value)
        {
            Action<object, object?> setter = Members[member].Setter!;
            setter(target, value);
        }

        public Type GetType(string member)
        {
            return Members[member].MemberType!;
        }

        public static ReflectionObject Create(Type t, params string[] memberNames)
        {
            return Create(t, null, memberNames);
        }

        public static ReflectionObject Create(Type t, MethodBase? creator, params string[] memberNames)
        {
            ReflectionDelegateFactory delegateFactory = JsonTypeReflector.ReflectionDelegateFactory;

            ObjectConstructor<object>? creatorConstructor = null;
            if (creator != null)
            {
                creatorConstructor = delegateFactory.CreateParameterizedConstructor(creator);
            }
            else
            {
                if (ReflectionUtils.HasDefaultConstructor(t, false))
                {
                    Func<object> ctor = delegateFactory.CreateDefaultConstructor<object>(t);

                    creatorConstructor = args => ctor();
                }
            }

            ReflectionObject d = new ReflectionObject(creatorConstructor);

            foreach (string memberName in memberNames)
            {
                MemberInfo[] members = t.GetMember(memberName, BindingFlags.Instance | BindingFlags.Public);
                if (members.Length != 1)
                {
                    throw new ArgumentException("Expected a single member with the name '{0}'.".FormatWith(CultureInfo.InvariantCulture, memberName));
                }

                MemberInfo member = members.Single();

                ReflectionMember reflectionMember = new ReflectionMember();

                switch (member.MemberType())
                {
                    case MemberTypes.Field:
                    case MemberTypes.Property:
                        if (ReflectionUtils.CanReadMemberValue(member, false))
                        {
                            reflectionMember.Getter = delegateFactory.CreateGet<object>(member);
                        }

                        if (ReflectionUtils.CanSetMemberValue(member, false, false))
                        {
                            reflectionMember.Setter = delegateFactory.CreateSet<object>(member);
                        }
                        break;
                    case MemberTypes.Method:
                        MethodInfo method = (MethodInfo)member;
                        if (method.IsPublic)
                        {
                            ParameterInfo[] parameters = method.GetParameters();
                            if (parameters.Length == 0 && method.ReturnType != typeof(void))
                            {
                                MethodCall<object, object?> call = delegateFactory.CreateMethodCall<object>(method);
                                reflectionMember.Getter = target => call(target);
                            }
                            else if (parameters.Length == 1 && method.ReturnType == typeof(void))
                            {
                                MethodCall<object, object?> call = delegateFactory.CreateMethodCall<object>(method);
                                reflectionMember.Setter = (target, arg) => call(target, arg);
                            }
                        }
                        break;
                    default:
                        throw new ArgumentException("Unexpected member type '{0}' for member '{1}'.".FormatWith(CultureInfo.InvariantCulture, member.MemberType(), member.Name));
                }

                reflectionMember.MemberType = ReflectionUtils.GetMemberUnderlyingType(member);

                d.Members[memberName] = reflectionMember;
            }

            return d;
        }
    }
}
