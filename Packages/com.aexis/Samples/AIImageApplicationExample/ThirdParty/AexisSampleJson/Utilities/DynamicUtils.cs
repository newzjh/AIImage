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

#if HAVE_DYNAMIC
using System;
using System.Collections.Generic;
using System.Dynamic;
using System.Linq;
using System.Linq.Expressions;
#if !HAVE_REFLECTION_BINDER
using System.Reflection;
#else
using Microsoft.CSharp.RuntimeBinder;
#endif
using System.Runtime.CompilerServices;
using System.Text;
using System.Globalization;
using Aexis.Samples.Json.Serialization;
using System.Diagnostics;

namespace Aexis.Samples.Json.Utilities
{
    internal static class DynamicUtils
    {
        internal static class BinderWrapper
        {
#if !HAVE_REFLECTION_BINDER
            public const string CSharpAssemblyName = "Microsoft.CSharp, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a";

            private const string BinderTypeName = "Microsoft.CSharp.RuntimeBinder.Binder, " + CSharpAssemblyName;
            private const string CSharpArgumentInfoTypeName = "Microsoft.CSharp.RuntimeBinder.CSharpArgumentInfo, " + CSharpAssemblyName;
            private const string CSharpArgumentInfoFlagsTypeName = "Microsoft.CSharp.RuntimeBinder.CSharpArgumentInfoFlags, " + CSharpAssemblyName;
            private const string CSharpBinderFlagsTypeName = "Microsoft.CSharp.RuntimeBinder.CSharpBinderFlags, " + CSharpAssemblyName;

            private static object? _getCSharpArgumentInfoArray;
            private static object? _setCSharpArgumentInfoArray;
            private static MethodCall<object?, object?>? _getMemberCall;
            private static MethodCall<object?, object?>? _setMemberCall;
            private static bool _init;

            private static void Init()
            {
                if (!_init)
                {
                    Type? binderType = Type.GetType(BinderTypeName, false);
                    if (binderType == null)
                    {
                        throw new InvalidOperationException("Could not resolve type '{0}'. You may need to add a reference to Microsoft.CSharp.dll to work with dynamic types.".FormatWith(CultureInfo.InvariantCulture, BinderTypeName));
                    }

                    // None
                    _getCSharpArgumentInfoArray = CreateSharpArgumentInfoArray(0);
                    // None, Constant | UseCompileTimeType
                    _setCSharpArgumentInfoArray = CreateSharpArgumentInfoArray(0, 3);
                    CreateMemberCalls();

                    _init = true;
                }
            }

            private static object CreateSharpArgumentInfoArray(params int[] values)
            {
                Type csharpArgumentInfoType = Type.GetType(CSharpArgumentInfoTypeName, true)!;
                Type csharpArgumentInfoFlags = Type.GetType(CSharpArgumentInfoFlagsTypeName, true)!;

                Array a = Array.CreateInstance(csharpArgumentInfoType, values.Length);

                for (int i = 0; i < values.Length; i++)
                {
                    MethodInfo createArgumentInfoMethod = csharpArgumentInfoType.GetMethod("Create", new[] { csharpArgumentInfoFlags, typeof(string) })!;
                    object arg = createArgumentInfoMethod.Invoke(null, new object?[] { 0, null })!;
                    a.SetValue(arg, i);
                }

                return a;
            }

            private static void CreateMemberCalls()
            {
                Type csharpArgumentInfoType = Type.GetType(CSharpArgumentInfoTypeName, true)!;
                Type csharpBinderFlagsType = Type.GetType(CSharpBinderFlagsTypeName, true)!;
                Type binderType = Type.GetType(BinderTypeName, true)!;

                Type csharpArgumentInfoTypeEnumerableType = typeof(IEnumerable<>).MakeGenericType(csharpArgumentInfoType);

                MethodInfo getMemberMethod = binderType.GetMethod("GetMember", new[] { csharpBinderFlagsType, typeof(string), typeof(Type), csharpArgumentInfoTypeEnumerableType })!;
                _getMemberCall = JsonTypeReflector.ReflectionDelegateFactory.CreateMethodCall<object?>(getMemberMethod);

                MethodInfo setMemberMethod = binderType.GetMethod("SetMember", new[] { csharpBinderFlagsType, typeof(string), typeof(Type), csharpArgumentInfoTypeEnumerableType })!;
                _setMemberCall = JsonTypeReflector.ReflectionDelegateFactory.CreateMethodCall<object?>(setMemberMethod);
            }
#endif

            public static CallSiteBinder GetMember(string name, Type context)
            {
#if !HAVE_REFLECTION_BINDER
                Init();
                MiscellaneousUtils.Assert(_getMemberCall != null);
                MiscellaneousUtils.Assert(_getCSharpArgumentInfoArray != null);
                return (CallSiteBinder)_getMemberCall(null, 0, name, context, _getCSharpArgumentInfoArray)!;
#else
                return Binder.GetMember(
                    CSharpBinderFlags.None, name, context, new[] {CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.None, null)});
#endif
            }

            public static CallSiteBinder SetMember(string name, Type context)
            {
#if !HAVE_REFLECTION_BINDER
                Init();
                MiscellaneousUtils.Assert(_setMemberCall != null);
                MiscellaneousUtils.Assert(_setCSharpArgumentInfoArray != null);
                return (CallSiteBinder)_setMemberCall(null, 0, name, context, _setCSharpArgumentInfoArray)!;
#else
                return Binder.SetMember(
                    CSharpBinderFlags.None, name, context, new[]
                                                               {
                                                                   CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.UseCompileTimeType, null),
                                                                   CSharpArgumentInfo.Create(CSharpArgumentInfoFlags.Constant, null)
                                                               });
#endif
            }
        }

        public static IEnumerable<string> GetDynamicMemberNames(this IDynamicMetaObjectProvider dynamicProvider)
        {
            DynamicMetaObject metaObject = dynamicProvider.GetMetaObject(Expression.Constant(dynamicProvider));
            return metaObject.GetDynamicMemberNames();
        }
    }

    internal class NoThrowGetBinderMember : GetMemberBinder
    {
        private readonly GetMemberBinder _innerBinder;

        public NoThrowGetBinderMember(GetMemberBinder innerBinder)
            : base(innerBinder.Name, innerBinder.IgnoreCase)
        {
            _innerBinder = innerBinder;
        }

        public override DynamicMetaObject FallbackGetMember(DynamicMetaObject target, DynamicMetaObject? errorSuggestion)
        {
            DynamicMetaObject retMetaObject = _innerBinder.Bind(target, CollectionUtils.ArrayEmpty<DynamicMetaObject>());

            NoThrowExpressionVisitor noThrowVisitor = new NoThrowExpressionVisitor();
            Expression resultExpression = noThrowVisitor.Visit(retMetaObject.Expression);

            DynamicMetaObject finalMetaObject = new DynamicMetaObject(resultExpression, retMetaObject.Restrictions);
            return finalMetaObject;
        }
    }

    internal class NoThrowSetBinderMember : SetMemberBinder
    {
        private readonly SetMemberBinder _innerBinder;

        public NoThrowSetBinderMember(SetMemberBinder innerBinder)
            : base(innerBinder.Name, innerBinder.IgnoreCase)
        {
            _innerBinder = innerBinder;
        }

        public override DynamicMetaObject FallbackSetMember(DynamicMetaObject target, DynamicMetaObject value, DynamicMetaObject? errorSuggestion)
        {
            DynamicMetaObject retMetaObject = _innerBinder.Bind(target, new DynamicMetaObject[] { value });

            NoThrowExpressionVisitor noThrowVisitor = new NoThrowExpressionVisitor();
            Expression resultExpression = noThrowVisitor.Visit(retMetaObject.Expression);

            DynamicMetaObject finalMetaObject = new DynamicMetaObject(resultExpression, retMetaObject.Restrictions);
            return finalMetaObject;
        }
    }

    internal class NoThrowExpressionVisitor : ExpressionVisitor
    {
        internal static readonly object ErrorResult = new object();

        protected override Expression VisitConditional(ConditionalExpression node)
        {
            // if the result of a test is to throw an error, rewrite to result an error result value
            if (node.IfFalse.NodeType == ExpressionType.Throw)
            {
                return Expression.Condition(node.Test, node.IfTrue, Expression.Constant(ErrorResult));
            }

            return base.VisitConditional(node);
        }
    }
}

#endif