using UnityEngine;

namespace Aexis.Execution
{
    // Importers populate this durable Unity asset. Runtime code can use binaryParam
    // directly through AexisNcnnBinaryParam without reparsing text .param files.
    public sealed class AexisModelAsset : ScriptableObject
    {
        public int formatVersion = 1;
        public string modelId = string.Empty;
        public string sourceFormat = string.Empty;
        public string compilerVersion = string.Empty;
        public bool eligible;
        public byte[] binaryParam;
        public byte[] weights;
        public byte[] source;
        [TextArea(3, 20)] public string manifestJson = string.Empty;
        [TextArea(3, 30)] public string diagnosticJson = string.Empty;
    }
}
