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
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Text.RegularExpressions;
using System.Diagnostics;
#if !HAVE_LINQ
using Aexis.Samples.Json.Utilities.LinqBridge;
#else
using System.Linq;
#endif
using Aexis.Samples.Json.Utilities;

namespace Aexis.Samples.Json.Linq.JsonPath
{
    internal enum QueryOperator
    {
        None = 0,
        Equals = 1,
        NotEquals = 2,
        Exists = 3,
        LessThan = 4,
        LessThanOrEquals = 5,
        GreaterThan = 6,
        GreaterThanOrEquals = 7,
        And = 8,
        Or = 9,
        RegexEquals = 10,
        StrictEquals = 11,
        StrictNotEquals = 12
    }

    internal abstract class QueryExpression
    {
        internal QueryOperator Operator;

        public QueryExpression(QueryOperator @operator)
        {
            Operator = @operator;
        }

        // For unit tests
        public bool IsMatch(JToken root, JToken t)
        {
            return IsMatch(root, t, null);
        }

        public abstract bool IsMatch(JToken root, JToken t, JsonSelectSettings? settings);
    }

    internal class CompositeExpression : QueryExpression
    {
        public List<QueryExpression> Expressions { get; set; }

        public CompositeExpression(QueryOperator @operator) : base(@operator)
        {
            Expressions = new List<QueryExpression>();
        }

        public override bool IsMatch(JToken root, JToken t, JsonSelectSettings? settings)
        {
            switch (Operator)
            {
                case QueryOperator.And:
                    foreach (QueryExpression e in Expressions)
                    {
                        if (!e.IsMatch(root, t, settings))
                        {
                            return false;
                        }
                    }
                    return true;
                case QueryOperator.Or:
                    foreach (QueryExpression e in Expressions)
                    {
                        if (e.IsMatch(root, t, settings))
                        {
                            return true;
                        }
                    }
                    return false;
                default:
                    throw new ArgumentOutOfRangeException();
            }
        }
    }

    internal class BooleanQueryExpression : QueryExpression
    {
        public readonly object Left;
        public readonly object? Right;

        public BooleanQueryExpression(QueryOperator @operator, object left, object? right) : base(@operator)
        {
            Left = left;
            Right = right;
        }

        private IEnumerable<JToken> GetResult(JToken root, JToken t, object? o)
        {
            if (o is JToken resultToken)
            {
                return new[] { resultToken };
            }

            if (o is List<PathFilter> pathFilters)
            {
                return JPath.Evaluate(pathFilters, root, t, null);
            }

            return CollectionUtils.ArrayEmpty<JToken>();
        }

        public override bool IsMatch(JToken root, JToken t, JsonSelectSettings? settings)
        {
            if (Operator == QueryOperator.Exists)
            {
                return GetResult(root, t, Left).Any();
            }

            using (IEnumerator<JToken> leftResults = GetResult(root, t, Left).GetEnumerator())
            {
                if (leftResults.MoveNext())
                {
                    IEnumerable<JToken> rightResultsEn = GetResult(root, t, Right);
                    ICollection<JToken> rightResults = rightResultsEn as ICollection<JToken> ?? rightResultsEn.ToList();

                    do
                    {
                        JToken leftResult = leftResults.Current;
                        foreach (JToken rightResult in rightResults)
                        {
                            if (MatchTokens(leftResult, rightResult, settings))
                            {
                                return true;
                            }
                        }
                    } while (leftResults.MoveNext());
                }
            }

            return false;
        }

        private bool MatchTokens(JToken leftResult, JToken rightResult, JsonSelectSettings? settings)
        {
            if (leftResult is JValue leftValue && rightResult is JValue rightValue)
            {
                switch (Operator)
                {
                    case QueryOperator.RegexEquals:
                        if (RegexEquals(leftValue, rightValue, settings))
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.Equals:
                        if (EqualsWithStringCoercion(leftValue, rightValue))
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.StrictEquals:
                        if (EqualsWithStrictMatch(leftValue, rightValue))
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.NotEquals:
                        if (!EqualsWithStringCoercion(leftValue, rightValue))
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.StrictNotEquals:
                        if (!EqualsWithStrictMatch(leftValue, rightValue))
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.GreaterThan:
                        if (leftValue.CompareTo(rightValue) > 0)
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.GreaterThanOrEquals:
                        if (leftValue.CompareTo(rightValue) >= 0)
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.LessThan:
                        if (leftValue.CompareTo(rightValue) < 0)
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.LessThanOrEquals:
                        if (leftValue.CompareTo(rightValue) <= 0)
                        {
                            return true;
                        }
                        break;
                    case QueryOperator.Exists:
                        return true;
                }
            }
            else
            {
                switch (Operator)
                {
                    case QueryOperator.Exists:
                    // you can only specify primitive types in a comparison
                    // notequals will always be true
                    case QueryOperator.NotEquals:
                        return true;
                }
            }

            return false;
        }

        private static bool RegexEquals(JValue input, JValue pattern, JsonSelectSettings? settings)
        {
            if (input.Type != JTokenType.String || pattern.Type != JTokenType.String)
            {
                return false;
            }

            string regexText = (string)pattern.Value!;
            int patternOptionDelimiterIndex = regexText.LastIndexOf('/');

            string patternText = regexText.Substring(1, patternOptionDelimiterIndex - 1);
            string optionsText = regexText.Substring(patternOptionDelimiterIndex + 1);

#if HAVE_REGEX_TIMEOUTS
            TimeSpan timeout = settings?.RegexMatchTimeout ?? Regex.InfiniteMatchTimeout;
            return Regex.IsMatch((string)input.Value!, patternText, MiscellaneousUtils.GetRegexOptions(optionsText), timeout);
#else
            return Regex.IsMatch((string)input.Value!, patternText, MiscellaneousUtils.GetRegexOptions(optionsText));
#endif
        }

        internal static bool EqualsWithStringCoercion(JValue value, JValue queryValue)
        {
            if (value.Equals(queryValue))
            {
                return true;
            }

            // Handle comparing an integer with a float
            // e.g. Comparing 1 and 1.0
            if ((value.Type == JTokenType.Integer && queryValue.Type == JTokenType.Float)
                || (value.Type == JTokenType.Float && queryValue.Type == JTokenType.Integer))
            {
                return JValue.Compare(value.Type, value.Value, queryValue.Value) == 0;
            }

            if (queryValue.Type != JTokenType.String)
            {
                return false;
            }

            string queryValueString = (string)queryValue.Value!;

            string currentValueString;

            // potential performance issue with converting every value to string?
            switch (value.Type)
            {
                case JTokenType.Date:
                    using (StringWriter writer = StringUtils.CreateStringWriter(64))
                    {
#if HAVE_DATE_TIME_OFFSET
                        if (value.Value is DateTimeOffset offset)
                        {
                            DateTimeUtils.WriteDateTimeOffsetString(writer, offset, DateFormatHandling.IsoDateFormat, null, CultureInfo.InvariantCulture);
                        }
                        else
#endif
                        {
                            DateTimeUtils.WriteDateTimeString(writer, (DateTime)value.Value!, DateFormatHandling.IsoDateFormat, null, CultureInfo.InvariantCulture);
                        }

                        currentValueString = writer.ToString();
                    }
                    break;
                case JTokenType.Bytes:
                    currentValueString = Convert.ToBase64String((byte[])value.Value!);
                    break;
                case JTokenType.Guid:
                case JTokenType.TimeSpan:
                    currentValueString = value.Value!.ToString()!;
                    break;
                case JTokenType.Uri:
                    currentValueString = ((Uri)value.Value!).OriginalString;
                    break;
                default:
                    return false;
            }

            return string.Equals(currentValueString, queryValueString, StringComparison.Ordinal);
        }

        internal static bool EqualsWithStrictMatch(JValue value, JValue queryValue)
        {
            MiscellaneousUtils.Assert(value != null);
            MiscellaneousUtils.Assert(queryValue != null);

            // Handle comparing an integer with a float
            // e.g. Comparing 1 and 1.0
            if ((value.Type == JTokenType.Integer && queryValue.Type == JTokenType.Float)
                || (value.Type == JTokenType.Float && queryValue.Type == JTokenType.Integer))
            {
                return JValue.Compare(value.Type, value.Value, queryValue.Value) == 0;
            }

            // we handle floats and integers the exact same way, so they are pseudo equivalent
            if (value.Type != queryValue.Type)
            {
                return false;
            }

            return value.Equals(queryValue);
        }
    }
}