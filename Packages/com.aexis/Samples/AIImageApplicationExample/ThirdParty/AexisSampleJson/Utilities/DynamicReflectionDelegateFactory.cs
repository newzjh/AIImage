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

#if HAVE_REFLECTION_EMIT
using System;
using System.Collections.Generic;
#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#endif
using System.Reflection;
using System.Reflection.Emit;
using Aexis.Samples.Json.Serialization;
using System.Globalization;

namespace Aexis.Samples.Json.Utilities
{
    internal class DynamicReflectionDelegateFactory : ReflectionDelegateFactory
    {
        internal static DynamicReflectionDelegateFactory Instance { get; } = new DynamicReflectionDelegateFactory();

        private static DynamicMethod CreateDynamicMethod(string name, Type? returnType, Type[] parameterTypes, Type owner)
        {
            DynamicMethod dynamicMethod = !owner.IsInterface()
                ? new DynamicMethod(name, returnType, parameterTypes, owner, true)
                : new DynamicMethod(name, returnType, parameterTypes, owner.Module, true);

            return dynamicMethod;
        }

        public override ObjectConstructor<object> CreateParameterizedConstructor(MethodBase method)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod(method.ToString()!, typeof(object), new[] { typeof(object[]) }, method.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateMethodCallIL(method, generator, 0);
            
            return (ObjectConstructor<object>)dynamicMethod.CreateDelegate(typeof(ObjectConstructor<object>));
        }

        public override MethodCall<T, object?> CreateMethodCall<T>(MethodBase method)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod(method.ToString()!, typeof(object), new[] { typeof(object), typeof(object[]) }, method.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateMethodCallIL(method, generator, 1);

            return (MethodCall<T, object?>)dynamicMethod.CreateDelegate(typeof(MethodCall<T, object?>));
        }

        private void GenerateCreateMethodCallIL(MethodBase method, ILGenerator generator, int argsIndex)
        {
            ParameterInfo[] args = method.GetParameters();

            Label argsOk = generator.DefineLabel();

            // throw an error if the number of argument values doesn't match method parameters
            generator.Emit(OpCodes.Ldarg, argsIndex);
            generator.Emit(OpCodes.Ldlen);
            generator.Emit(OpCodes.Ldc_I4, args.Length);
            generator.Emit(OpCodes.Beq, argsOk);
            generator.Emit(OpCodes.Newobj, typeof(TargetParameterCountException).GetConstructor(ReflectionUtils.EmptyTypes)!);
            generator.Emit(OpCodes.Throw);

            generator.MarkLabel(argsOk);

            if (!method.IsConstructor && !method.IsStatic)
            {
                generator.PushInstance(method.DeclaringType!);
            }

            LocalBuilder localConvertible = generator.DeclareLocal(typeof(IConvertible));
            LocalBuilder localObject = generator.DeclareLocal(typeof(object));

            OpCode variableAddressOpCode = args.Length < 256 ? OpCodes.Ldloca_S : OpCodes.Ldloca;
            OpCode variableLoadOpCode = args.Length < 256 ? OpCodes.Ldloc_S : OpCodes.Ldloc;

            for (int i = 0; i < args.Length; i++)
            {
                ParameterInfo parameter = args[i];
                Type parameterType = parameter.ParameterType;

                if (parameterType.IsByRef)
                {
                    parameterType = parameterType.GetElementType()!;

                    LocalBuilder localVariable = generator.DeclareLocal(parameterType);

                    // don't need to set variable for 'out' parameter
                    if (!parameter.IsOut)
                    {
                        generator.PushArrayInstance(argsIndex, i);

                        if (parameterType.IsValueType())
                        {
                            Label skipSettingDefault = generator.DefineLabel();
                            Label finishedProcessingParameter = generator.DefineLabel();

                            // check if parameter is not null
                            generator.Emit(OpCodes.Brtrue_S, skipSettingDefault);

                            // parameter has no value, initialize to default
                            generator.Emit(variableAddressOpCode, localVariable);
                            generator.Emit(OpCodes.Initobj, parameterType);
                            generator.Emit(OpCodes.Br_S, finishedProcessingParameter);

                            // parameter has value, get value from array again and unbox and set to variable
                            generator.MarkLabel(skipSettingDefault);
                            generator.PushArrayInstance(argsIndex, i);
                            generator.UnboxIfNeeded(parameterType);
                            generator.Emit(OpCodes.Stloc_S, localVariable);

                            // parameter finished, we out!
                            generator.MarkLabel(finishedProcessingParameter);
                        }
                        else
                        {
                            generator.UnboxIfNeeded(parameterType);
                            generator.Emit(OpCodes.Stloc_S, localVariable);
                        }
                    }

                    generator.Emit(variableAddressOpCode, localVariable);
                }
                else if (parameterType.IsValueType())
                {
                    generator.PushArrayInstance(argsIndex, i);
                    generator.Emit(OpCodes.Stloc_S, localObject);

                    // have to check that value type parameters aren't null
                    // otherwise they will error when unboxed
                    Label skipSettingDefault = generator.DefineLabel();
                    Label finishedProcessingParameter = generator.DefineLabel();

                    // check if parameter is not null
                    generator.Emit(OpCodes.Ldloc_S, localObject);
                    generator.Emit(OpCodes.Brtrue_S, skipSettingDefault);

                    // parameter has no value, initialize to default
                    LocalBuilder localVariable = generator.DeclareLocal(parameterType);
                    generator.Emit(variableAddressOpCode, localVariable);
                    generator.Emit(OpCodes.Initobj, parameterType);
                    generator.Emit(variableLoadOpCode, localVariable);
                    generator.Emit(OpCodes.Br_S, finishedProcessingParameter);

                    // argument has value, try to convert it to parameter type
                    generator.MarkLabel(skipSettingDefault);
                    
                    if (parameterType.IsPrimitive())
                    {
                        // for primitive types we need to handle type widening (e.g. short -> int)
                        MethodInfo? toParameterTypeMethod = typeof(IConvertible)
                            .GetMethod("To" + parameterType.Name, new[] { typeof(IFormatProvider) });
                        
                        if (toParameterTypeMethod != null)
                        {
                            Label skipConvertible = generator.DefineLabel();

                            // check if argument type is an exact match for parameter type
                            // in this case we may use cheap unboxing instead
                            generator.Emit(OpCodes.Ldloc_S, localObject);
                            generator.Emit(OpCodes.Isinst, parameterType);
                            generator.Emit(OpCodes.Brtrue_S, skipConvertible);

                            // types don't match, check if argument implements IConvertible
                            generator.Emit(OpCodes.Ldloc_S, localObject);
                            generator.Emit(OpCodes.Isinst, typeof(IConvertible));
                            generator.Emit(OpCodes.Stloc_S, localConvertible);
                            generator.Emit(OpCodes.Ldloc_S, localConvertible);
                            generator.Emit(OpCodes.Brfalse_S, skipConvertible);

                            // convert argument to parameter type
                            generator.Emit(OpCodes.Ldloc_S, localConvertible);
                            generator.Emit(OpCodes.Ldnull);
                            generator.Emit(OpCodes.Callvirt, toParameterTypeMethod);
                            generator.Emit(OpCodes.Br_S, finishedProcessingParameter);

                            generator.MarkLabel(skipConvertible);
                        }
                    }

                    // we got here because either argument type matches parameter (conversion will succeed),
                    // or argument type doesn't match parameter, but we're out of options (conversion will fail)
                    generator.Emit(OpCodes.Ldloc_S, localObject);

                    generator.UnboxIfNeeded(parameterType);
                    
                    // parameter finished, we out!
                    generator.MarkLabel(finishedProcessingParameter);
                }
                else
                {
                    generator.PushArrayInstance(argsIndex, i);

                    generator.UnboxIfNeeded(parameterType);
                }
            }

            if (method.IsConstructor)
            {
                generator.Emit(OpCodes.Newobj, (ConstructorInfo)method);
            }
            else
            {
                generator.CallMethod((MethodInfo)method);
            }

            Type returnType = method.IsConstructor
                ? method.DeclaringType!
                : ((MethodInfo)method).ReturnType;

            if (returnType != typeof(void))
            {
                generator.BoxIfNeeded(returnType);
            }
            else
            {
                generator.Emit(OpCodes.Ldnull);
            }

            generator.Return();
        }

        public override Func<T> CreateDefaultConstructor<T>(Type type)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod("Create" + type.FullName, typeof(T), ReflectionUtils.EmptyTypes, type);
            dynamicMethod.InitLocals = true;
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateDefaultConstructorIL(type, generator, typeof(T));

            return (Func<T>)dynamicMethod.CreateDelegate(typeof(Func<T>));
        }

        private void GenerateCreateDefaultConstructorIL(Type type, ILGenerator generator, Type delegateType)
        {
            if (type.IsValueType())
            {
                generator.DeclareLocal(type);
                generator.Emit(OpCodes.Ldloc_0);

                // only need to box if the delegate isn't returning the value type
                if (type != delegateType)
                {
                    generator.Emit(OpCodes.Box, type);
                }
            }
            else
            {
                ConstructorInfo? constructorInfo =
                    type.GetConstructor(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance, null, ReflectionUtils.EmptyTypes, null);

                if (constructorInfo == null)
                {
                    throw new ArgumentException("Could not get constructor for {0}.".FormatWith(CultureInfo.InvariantCulture, type));
                }

                generator.Emit(OpCodes.Newobj, constructorInfo);
            }

            generator.Return();
        }

        public override Func<T, object?> CreateGet<T>(PropertyInfo propertyInfo)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod("Get" + propertyInfo.Name, typeof(object), new[] { typeof(T) }, propertyInfo.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateGetPropertyIL(propertyInfo, generator);

            return (Func<T, object?>)dynamicMethod.CreateDelegate(typeof(Func<T, object?>));
        }

        private void GenerateCreateGetPropertyIL(PropertyInfo propertyInfo, ILGenerator generator)
        {
            MethodInfo? getMethod = propertyInfo.GetGetMethod(true);
            if (getMethod == null)
            {
                throw new ArgumentException("Property '{0}' does not have a getter.".FormatWith(CultureInfo.InvariantCulture, propertyInfo.Name));
            }

            if (!getMethod.IsStatic)
            {
                generator.PushInstance(propertyInfo.DeclaringType!);
            }

            generator.CallMethod(getMethod);
            generator.BoxIfNeeded(propertyInfo.PropertyType);
            generator.Return();
        }

        public override Func<T, object?> CreateGet<T>(FieldInfo fieldInfo)
        {
            if (fieldInfo.IsLiteral)
            {
                object constantValue = fieldInfo.GetValue(null)!;
                Func<T, object?> getter = o => constantValue;
                return getter;
            }

            DynamicMethod dynamicMethod = CreateDynamicMethod("Get" + fieldInfo.Name, typeof(T), new[] { typeof(object) }, fieldInfo.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateGetFieldIL(fieldInfo, generator);

            return (Func<T, object?>)dynamicMethod.CreateDelegate(typeof(Func<T, object?>));
        }

        private void GenerateCreateGetFieldIL(FieldInfo fieldInfo, ILGenerator generator)
        {
            if (!fieldInfo.IsStatic)
            {
                generator.PushInstance(fieldInfo.DeclaringType!);
                generator.Emit(OpCodes.Ldfld, fieldInfo);
            }
            else
            {
                generator.Emit(OpCodes.Ldsfld, fieldInfo);
            }

            generator.BoxIfNeeded(fieldInfo.FieldType);
            generator.Return();
        }

        public override Action<T, object?> CreateSet<T>(FieldInfo fieldInfo)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod("Set" + fieldInfo.Name, null, new[] { typeof(T), typeof(object) }, fieldInfo.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateSetFieldIL(fieldInfo, generator);

            return (Action<T, object?>)dynamicMethod.CreateDelegate(typeof(Action<T, object?>));
        }

        internal static void GenerateCreateSetFieldIL(FieldInfo fieldInfo, ILGenerator generator)
        {
            if (!fieldInfo.IsStatic)
            {
                generator.PushInstance(fieldInfo.DeclaringType!);
            }

            generator.Emit(OpCodes.Ldarg_1);
            generator.UnboxIfNeeded(fieldInfo.FieldType);

            if (!fieldInfo.IsStatic)
            {
                generator.Emit(OpCodes.Stfld, fieldInfo);
            }
            else
            {
                generator.Emit(OpCodes.Stsfld, fieldInfo);
            }

            generator.Return();
        }

        public override Action<T, object?> CreateSet<T>(PropertyInfo propertyInfo)
        {
            DynamicMethod dynamicMethod = CreateDynamicMethod("Set" + propertyInfo.Name, null, new[] { typeof(T), typeof(object) }, propertyInfo.DeclaringType!);
            ILGenerator generator = dynamicMethod.GetILGenerator();

            GenerateCreateSetPropertyIL(propertyInfo, generator);

            return (Action<T, object?>)dynamicMethod.CreateDelegate(typeof(Action<T, object>));
        }

        internal static void GenerateCreateSetPropertyIL(PropertyInfo propertyInfo, ILGenerator generator)
        {
            MethodInfo setMethod = propertyInfo.GetSetMethod(true)!;
            if (!setMethod.IsStatic)
            {
                generator.PushInstance(propertyInfo.DeclaringType!);
            }

            generator.Emit(OpCodes.Ldarg_1);
            generator.UnboxIfNeeded(propertyInfo.PropertyType);
            generator.CallMethod(setMethod);
            generator.Return();
        }
    }
}

#endif