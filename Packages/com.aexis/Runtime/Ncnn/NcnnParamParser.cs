using Aexis.Execution;
using System;
using System.IO;

namespace Aexis.Ncnn
{
    // Preserves the NCNN-facing parser API while the graph representation stays shared.
    public static class NcnnParamParser
    {
        public static AexisGraphModel Parse(string text)
        {
            return AexisGraphModelParser.Parse(text);
        }

        public static AexisGraphModel Parse(byte[] binaryParam)
        {
            return AexisNcnnBinaryParam.Deserialize(binaryParam);
        }

        public static AexisGraphModel Parse(Stream binaryParam)
        {
            return AexisNcnnBinaryParam.Read(binaryParam);
        }

        public static byte[] WriteBinary(AexisGraphModel graph)
        {
            return AexisNcnnBinaryParam.Serialize(graph);
        }

        public static int MergeStringParamsByLayerName(
            AexisGraphModel target,
            AexisGraphModel source,
            bool overwriteExisting = false)
        {
            return AexisGraphModelParser.MergeStringParamsByLayerName(target, source, overwriteExisting);
        }
    }
}
