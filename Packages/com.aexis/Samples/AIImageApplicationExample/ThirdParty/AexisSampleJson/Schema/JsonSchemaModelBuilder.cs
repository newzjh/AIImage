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

#nullable disable

namespace Aexis.Samples.Json.Schema
{
    [Obsolete("JSON Schema validation has been moved to its own package. See https://www.newtonsoft.com/jsonschema for more details.")]
    internal class JsonSchemaModelBuilder
    {
        private JsonSchemaNodeCollection _nodes = new JsonSchemaNodeCollection();
        private Dictionary<JsonSchemaNode, JsonSchemaModel> _nodeModels = new Dictionary<JsonSchemaNode, JsonSchemaModel>();
        private JsonSchemaNode _node;

        public JsonSchemaModel Build(JsonSchema schema)
        {
            _nodes = new JsonSchemaNodeCollection();
            _node = AddSchema(null, schema);

            _nodeModels = new Dictionary<JsonSchemaNode, JsonSchemaModel>();
            JsonSchemaModel model = BuildNodeModel(_node);

            return model;
        }

        public JsonSchemaNode AddSchema(JsonSchemaNode existingNode, JsonSchema schema)
        {
            string newId;
            if (existingNode != null)
            {
                if (existingNode.Schemas.Contains(schema))
                {
                    return existingNode;
                }

                newId = JsonSchemaNode.GetId(existingNode.Schemas.Union(new[] { schema }));
            }
            else
            {
                newId = JsonSchemaNode.GetId(new[] { schema });
            }

            if (_nodes.Contains(newId))
            {
                return _nodes[newId];
            }

            JsonSchemaNode currentNode = (existingNode != null)
                ? existingNode.Combine(schema)
                : new JsonSchemaNode(schema);

            _nodes.Add(currentNode);

            AddProperties(schema.Properties, currentNode.Properties);

            AddProperties(schema.PatternProperties, currentNode.PatternProperties);

            if (schema.Items != null)
            {
                for (int i = 0; i < schema.Items.Count; i++)
                {
                    AddItem(currentNode, i, schema.Items[i]);
                }
            }

            if (schema.AdditionalItems != null)
            {
                AddAdditionalItems(currentNode, schema.AdditionalItems);
            }

            if (schema.AdditionalProperties != null)
            {
                AddAdditionalProperties(currentNode, schema.AdditionalProperties);
            }

            if (schema.Extends != null)
            {
                foreach (JsonSchema jsonSchema in schema.Extends)
                {
                    currentNode = AddSchema(currentNode, jsonSchema);
                }
            }

            return currentNode;
        }

        public void AddProperties(IDictionary<string, JsonSchema> source, IDictionary<string, JsonSchemaNode> target)
        {
            if (source != null)
            {
                foreach (KeyValuePair<string, JsonSchema> property in source)
                {
                    AddProperty(target, property.Key, property.Value);
                }
            }
        }

        public void AddProperty(IDictionary<string, JsonSchemaNode> target, string propertyName, JsonSchema schema)
        {
            target.TryGetValue(propertyName, out JsonSchemaNode propertyNode);

            target[propertyName] = AddSchema(propertyNode, schema);
        }

        public void AddItem(JsonSchemaNode parentNode, int index, JsonSchema schema)
        {
            JsonSchemaNode existingItemNode = (parentNode.Items.Count > index)
                ? parentNode.Items[index]
                : null;

            JsonSchemaNode newItemNode = AddSchema(existingItemNode, schema);

            if (!(parentNode.Items.Count > index))
            {
                parentNode.Items.Add(newItemNode);
            }
            else
            {
                parentNode.Items[index] = newItemNode;
            }
        }

        public void AddAdditionalProperties(JsonSchemaNode parentNode, JsonSchema schema)
        {
            parentNode.AdditionalProperties = AddSchema(parentNode.AdditionalProperties, schema);
        }

        public void AddAdditionalItems(JsonSchemaNode parentNode, JsonSchema schema)
        {
            parentNode.AdditionalItems = AddSchema(parentNode.AdditionalItems, schema);
        }

        private JsonSchemaModel BuildNodeModel(JsonSchemaNode node)
        {
            if (_nodeModels.TryGetValue(node, out JsonSchemaModel model))
            {
                return model;
            }

            model = JsonSchemaModel.Create(node.Schemas);
            _nodeModels[node] = model;

            foreach (KeyValuePair<string, JsonSchemaNode> property in node.Properties)
            {
                if (model.Properties == null)
                {
                    model.Properties = new Dictionary<string, JsonSchemaModel>();
                }

                model.Properties[property.Key] = BuildNodeModel(property.Value);
            }
            foreach (KeyValuePair<string, JsonSchemaNode> property in node.PatternProperties)
            {
                if (model.PatternProperties == null)
                {
                    model.PatternProperties = new Dictionary<string, JsonSchemaModel>();
                }

                model.PatternProperties[property.Key] = BuildNodeModel(property.Value);
            }
            foreach (JsonSchemaNode t in node.Items)
            {
                if (model.Items == null)
                {
                    model.Items = new List<JsonSchemaModel>();
                }

                model.Items.Add(BuildNodeModel(t));
            }
            if (node.AdditionalProperties != null)
            {
                model.AdditionalProperties = BuildNodeModel(node.AdditionalProperties);
            }
            if (node.AdditionalItems != null)
            {
                model.AdditionalItems = BuildNodeModel(node.AdditionalItems);
            }

            return model;
        }
    }
}