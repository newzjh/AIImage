using AIImage.Inference.Core;
using NUnit.Framework;
using System.Collections.Generic;

namespace AIImage.Inference.Validation.Tests
{
    public sealed class OnnxSentisShapePlannerTests
    {
        [Test]
        public void DynamicTopK_RequiresCapacityBoundedGpuShapeTensor()
        {
            var rejected = OnnxSentisShapePlanner.Validate(new OnnxSentisNodeContract
            {
                name = "dynamic_k",
                opType = "TopK",
                inputRank = 2,
                dynamicParameter = true
            });
            Assert.That(rejected.code, Is.EqualTo("missing-dynamic-capacity"));

            var accepted = OnnxSentisShapePlanner.Validate(new OnnxSentisNodeContract
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
            var diagnostic = OnnxSentisShapePlanner.Validate(new OnnxSentisNodeContract
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
            var diagnostic = OnnxSentisShapePlanner.Validate(new OnnxSentisNodeContract
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
            var ok = OnnxSentisAdapter.TryAdapt(new OnnxSentisImportedNode
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
                var accepted = OnnxSentisShapePlanner.Validate(new OnnxSentisNodeContract
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

        private static byte[] BuildTopKModel()
        {
            var node = Message(FieldString(1, "values"), FieldString(1, "k"), FieldString(2, "top_values"), FieldString(2, "top_indices"), FieldString(3, "dynamic_topk"), FieldString(4, "TopK"));
            var graph = Message(FieldBytes(1, node), FieldBytes(11, ValueInfo("values", 1, 1)), FieldBytes(11, ValueInfo("k", 6, 1)));
            var opset = Message(FieldVarint(2, 13));
            return Message(FieldBytes(7, graph), FieldBytes(8, opset));
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
