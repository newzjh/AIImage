using Aexis;
using Aexis.Onnx;
using NUnit.Framework;
using System.Collections.Generic;

namespace Aexis.Tests.Editor
{
    public sealed class OnnxExecutionShapePlannerTests
    {
        [Test]
        public void DynamicTopK_RequiresCapacityBoundedGpuShapeTensor()
        {
            var rejected = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = "dynamic_k",
                opType = "TopK",
                inputRank = 2,
                dynamicParameter = true
            });
            Assert.That(rejected.code, Is.EqualTo("missing-dynamic-capacity"));

            var accepted = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = "dynamic_k",
                opType = "TopK",
                inputRank = 2,
                dynamicParameter = true,
                outputShape = new GpuShapeTensorContract
                {
                    rank = 2,
                    capacity = 64,
                    lengthPolicy = GpuShapeLengthPolicy.CapacityBounded,
                    overflowPolicy = "reject"
                }
            });
            Assert.That(accepted, Is.Null);
        }

        [Test]
        public void Scatter_RejectsDuplicateOrNonInt32IndicesAtPlanTime()
        {
            var diagnostic = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = "scatter_updates",
                opType = "ScatterND",
                inputRank = 2,
                indexDataType = TensorDataType.Int32,
                uniqueIndices = false
            });
            Assert.That(diagnostic.code, Is.EqualTo("scatter-conflict-undefined"));
            Assert.That(diagnostic.recommendedAction, Does.Contain("Deduplicate"));
        }

        [Test]
        public void Compress_RequiresGpuCompactionCapacityContract()
        {
            var diagnostic = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
            {
                name = "filter",
                opType = "Compress",
                inputRank = 1
            });
            Assert.That(diagnostic.code, Is.EqualTo("missing-gpu-compaction-contract"));
        }

        [Test]
        public void Adapter_MapsDynamicTopKToCanonicalNodeWithoutCpuShape()
        {
            var ok = OnnxExecutionAdapter.TryAdapt(new OnnxExecutionImportedNode
            {
                name = "topk_from_gpu", opType = "TopK", inputRank = 1,
                parameterComesFromTensor = true, outputCapacity = 8,
                inputs = new[] { "values", "k_gpu" }, outputs = new[] { "top_values", "top_indices" }
            }, out var node, out var diagnostic);
            Assert.That(ok, Is.True, diagnostic == null ? string.Empty : diagnostic.message);
            Assert.That(node.op, Is.EqualTo(CanonicalShapeIndexOp.TopK));
            Assert.That(node.contract.outputShape.lengthPolicy, Is.EqualTo(GpuShapeLengthPolicy.CapacityBounded));
            Assert.That(node.contract.outputShape.lengthTensor, Is.EqualTo("top_values.shape"));
        }

        [Test]
        public void Property_NonZeroCapacityContract_CoversRandomLengths()
        {
            var random = new System.Random(20260713);
            for (var iteration = 0; iteration < 128; iteration++)
            {
                var inputLength = random.Next(1, 65);
                var capacity = random.Next(1, 65);
                var accepted = OnnxExecutionShapePlanner.Validate(new OnnxExecutionNodeContract
                {
                    name = "nonzero_" + iteration, opType = "NonZero", inputRank = 1,
                    outputCapacity = capacity,
                    outputShape = new GpuShapeTensorContract { rank = 1, capacity = capacity, lengthPolicy = GpuShapeLengthPolicy.CapacityBounded, overflowPolicy = "reject", lengthTensor = "count" }
                });
                Assert.That(accepted, Is.Null, "input=" + inputLength + " capacity=" + capacity);
            }
        }

        [Test]
        public void Importer_ReadsOnnxProtobufAndBuildsDynamicTopK()
        {
            var options = new OnnxD3ImportOptions();
            options.outputCapacities["top_values"] = 8;
            var result = OnnxD3Importer.Import(BuildTopKModel(), options);
            Assert.That(result.IsPlanEligible, Is.True);
            Assert.That(result.opset, Is.EqualTo(13));
            Assert.That(result.nodes, Has.Length.EqualTo(1));
            Assert.That(result.nodes[0].op, Is.EqualTo(CanonicalShapeIndexOp.TopK));
            Assert.That(result.nodes[0].contract.dynamicParameter, Is.True);
            Assert.That(result.nodes[0].contract.outputShape.capacity, Is.EqualTo(8));
        }

        [Test]
        public void ModelReader_PreservesGraphValuedControlFlowAttributes()
        {
            var model = OnnxModelReader.Read(BuildIfModelWithGraphAttribute());

            Assert.That(model.graph.nodes, Has.Count.EqualTo(1));
            Assert.That(model.graph.nodes[0].attributes.TryGetValue("then_branch", out var attribute), Is.True);
            Assert.That(attribute.type, Is.EqualTo(5));
            Assert.That(attribute.graph, Is.Not.Null);
            Assert.That(attribute.graph.name, Is.EqualTo("then_graph"));
        }

        [Test]
        public void ModelReader_DensifiesBoundedSparseInitializerForImmutableTextureUpload()
        {
            var model = OnnxModelReader.Read(BuildSparseInitializerModel());

            Assert.That(model.graph.sparseInitializers.ContainsKey("sparse_weight"), Is.True);
            Assert.That(model.graph.initializers.TryGetValue("sparse_weight", out var dense), Is.True);
            Assert.That(dense.dims, Is.EqualTo(new long[] { 2, 3 }));
            Assert.That(dense.rawData.Length, Is.EqualTo(6 * sizeof(float)));
            Assert.That(System.BitConverter.ToSingle(dense.rawData, 0), Is.EqualTo(0f));
            Assert.That(System.BitConverter.ToSingle(dense.rawData, sizeof(float)), Is.EqualTo(2f));
            Assert.That(System.BitConverter.ToSingle(dense.rawData, 5 * sizeof(float)), Is.EqualTo(-3f));

            model.graph.inputs.Add(new OnnxValueInfo { name = "x", dataType = TensorDataType.Float32, onnxDataType = 1, dims = new long[] { 2, 3 } });
            model.graph.outputs.Add(new OnnxValueInfo { name = "y", dataType = TensorDataType.Float32, onnxDataType = 1, dims = new long[] { 2, 3 } });
            var add = new OnnxNode { name = "add_sparse_weight", opType = "Add" };
            add.inputs.Add("x"); add.inputs.Add("sparse_weight"); add.outputs.Add("y");
            model.graph.nodes.Add(add);

            var lowered = Aexis.Execution.AexisOnnxGraphLowering.Lower(model);
            var compiled = Aexis.Execution.AexisOnnxGraphCompiler.Compile(lowered);
            Assert.That(lowered.IsEligible, Is.True);
            Assert.That(compiled.immutableWeights, Is.EqualTo(new[] { 0f, 2f, 0f, 0f, 0f, -3f }));
        }

        [Test]
        public void ModelReader_RejectsSparseInitializerWithDuplicateCoordinates()
        {
            Assert.Throws<System.IO.InvalidDataException>(() => OnnxModelReader.Read(BuildSparseInitializerModel(duplicateCoordinates: true)));
        }

        private static byte[] BuildTopKModel()
        {
            var node = Message(FieldString(1, "values"), FieldString(1, "k"), FieldString(2, "top_values"), FieldString(2, "top_indices"), FieldString(3, "dynamic_topk"), FieldString(4, "TopK"));
            var graph = Message(FieldBytes(1, node), FieldBytes(11, ValueInfo("values", 1, 1)), FieldBytes(11, ValueInfo("k", 6, 1)));
            var opset = Message(FieldVarint(2, 13));
            return Message(FieldBytes(7, graph), FieldBytes(8, opset));
        }

        private static byte[] BuildIfModelWithGraphAttribute()
        {
            var thenGraph = Message(FieldString(2, "then_graph"));
            var branchAttribute = Message(
                FieldString(1, "then_branch"),
                FieldBytes(6, thenGraph),
                FieldVarint(20, 5));
            var node = Message(
                FieldString(1, "condition"),
                FieldString(2, "output"),
                FieldString(3, "if_node"),
                FieldString(4, "If"),
                FieldBytes(5, branchAttribute));
            var graph = Message(FieldBytes(1, node));
            var opset = Message(FieldVarint(2, 13));
            return Message(FieldBytes(7, graph), FieldBytes(8, opset));
        }

        private static byte[] BuildSparseInitializerModel(bool duplicateCoordinates = false)
        {
            var values = Tensor("sparse_weight", 1, new long[] { 2 }, FloatBytes(2f, -3f));
            var indices = Tensor("sparse_weight_indices", 7, new long[] { 2, 2 }, Int64Bytes(
                0, 1,
                duplicateCoordinates ? 0 : 1, duplicateCoordinates ? 1 : 2));
            var sparse = Message(
                FieldBytes(1, values),
                FieldBytes(2, indices),
                FieldVarint(3, 2),
                FieldVarint(3, 3));
            var graph = Message(FieldBytes(15, sparse));
            var opset = Message(FieldVarint(2, 13));
            return Message(FieldBytes(7, graph), FieldBytes(8, opset));
        }

        private static byte[] Tensor(string name, int type, long[] dims, byte[] rawData)
        {
            var fields = new List<byte[]>();
            for (var index = 0; index < dims.Length; index++) fields.Add(FieldVarint(1, (ulong)dims[index]));
            fields.Add(FieldVarint(2, (ulong)type));
            fields.Add(FieldString(8, name));
            fields.Add(FieldBytes(9, rawData));
            return Message(fields.ToArray());
        }

        private static byte[] FloatBytes(params float[] values)
        {
            var bytes = new byte[values.Length * sizeof(float)];
            System.Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        private static byte[] Int64Bytes(params long[] values)
        {
            var bytes = new byte[values.Length * sizeof(long)];
            System.Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        private static byte[] ValueInfo(string name, int type, int rank)
        {
            var dimensions = new List<byte[]>();
            for (var i = 0; i < rank; i++) dimensions.Add(FieldBytes(1, Message(FieldVarint(1, 1))));
            var shape = Message(dimensions.ToArray());
            var tensor = Message(FieldVarint(1, (ulong)type), FieldBytes(2, shape));
            return Message(FieldString(1, name), FieldBytes(2, Message(FieldBytes(1, tensor))));
        }

        private static byte[] FieldString(int field, string value) => FieldBytes(field, System.Text.Encoding.UTF8.GetBytes(value));
        private static byte[] FieldBytes(int field, byte[] value) { var bytes = new List<byte>(); WriteVarint(bytes, (ulong)(field << 3 | 2)); WriteVarint(bytes, (ulong)value.Length); bytes.AddRange(value); return bytes.ToArray(); }
        private static byte[] FieldVarint(int field, ulong value) { var bytes = new List<byte>(); WriteVarint(bytes, (ulong)(field << 3)); WriteVarint(bytes, value); return bytes.ToArray(); }
        private static byte[] Message(params byte[][] fields) { var bytes = new List<byte>(); foreach (var field in fields) bytes.AddRange(field); return bytes.ToArray(); }
        private static void WriteVarint(List<byte> destination, ulong value) { while (value >= 128) { destination.Add((byte)(value | 128)); value >>= 7; } destination.Add((byte)value); }
    }
}
