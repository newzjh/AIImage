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
using System.Diagnostics;
using System.Globalization;
using System.Text;

namespace Aexis.Samples.Json.Serialization
{
    /// <summary>
    /// Represents a trace writer that writes to memory. When the trace message limit is
    /// reached then old trace messages will be removed as new messages are added.
    /// </summary>
    public class MemoryTraceWriter : ITraceWriter
    {
        private readonly Queue<string> _traceMessages;
        private readonly object _lock;

        /// <summary>
        /// Gets the <see cref="TraceLevel"/> that will be used to filter the trace messages passed to the writer.
        /// For example a filter level of <see cref="TraceLevel.Info"/> will exclude <see cref="TraceLevel.Verbose"/> messages and include <see cref="TraceLevel.Info"/>,
        /// <see cref="TraceLevel.Warning"/> and <see cref="TraceLevel.Error"/> messages.
        /// </summary>
        /// <value>
        /// The <see cref="TraceLevel"/> that will be used to filter the trace messages passed to the writer.
        /// </value>
        public TraceLevel LevelFilter { get; set; }

        /// <summary>
        /// Initializes a new instance of the <see cref="MemoryTraceWriter"/> class.
        /// </summary>
        public MemoryTraceWriter()
        {
            LevelFilter = TraceLevel.Verbose;
            _traceMessages = new Queue<string>();
            _lock = new object();
        }

        /// <summary>
        /// Writes the specified trace level, message and optional exception.
        /// </summary>
        /// <param name="level">The <see cref="TraceLevel"/> at which to write this trace.</param>
        /// <param name="message">The trace message.</param>
        /// <param name="ex">The trace exception. This parameter is optional.</param>
        public void Trace(TraceLevel level, string message, Exception? ex)
        {
            StringBuilder sb = new StringBuilder();
            sb.Append(DateTime.Now.ToString("yyyy'-'MM'-'dd'T'HH':'mm':'ss'.'fff", CultureInfo.InvariantCulture));
            sb.Append(" ");
            sb.Append(level.ToString("g"));
            sb.Append(" ");
            sb.Append(message);

            string s = sb.ToString();

            lock (_lock)
            {
                if (_traceMessages.Count >= 1000)
                {
                    _traceMessages.Dequeue();
                }

                _traceMessages.Enqueue(s);
            }
        }

        /// <summary>
        /// Returns an enumeration of the most recent trace messages.
        /// </summary>
        /// <returns>An enumeration of the most recent trace messages.</returns>
        public IEnumerable<string> GetTraceMessages()
        {
            return _traceMessages;
        }

        /// <summary>
        /// Returns a <see cref="String"/> of the most recent trace messages.
        /// </summary>
        /// <returns>
        /// A <see cref="String"/> of the most recent trace messages.
        /// </returns>
        public override string ToString()
        {
            lock (_lock)
            {
                StringBuilder sb = new StringBuilder();
                foreach (string traceMessage in _traceMessages)
                {
                    if (sb.Length > 0)
                    {
                        sb.AppendLine();
                    }

                    sb.Append(traceMessage);
                }

                return sb.ToString();
            }
        }
    }
}