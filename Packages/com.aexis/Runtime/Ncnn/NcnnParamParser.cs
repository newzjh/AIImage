using Aexis.Execution;

namespace Aexis.Ncnn
{
    // Preserves the NCNN-facing parser API while the graph representation stays shared.
    public static class NcnnParamParser
    {
        public static AexisGraphModel Parse(string text)
        {
            return AexisGraphModelParser.Parse(text);
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
