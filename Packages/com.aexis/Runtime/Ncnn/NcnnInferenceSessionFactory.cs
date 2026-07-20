using System;
using Aexis;
using UnityEngine;

namespace Aexis.Ncnn
{
    // Application runners use this package-owned factory rather than owning backend construction.
    public static class NcnnInferenceSessionFactory
    {
        public static NcnnGraphSession Create(NcnnOps ops)
        {
            return Create(ops, NcnnModelManifestLoader.TryLoadFromEnvironment());
        }

        public static NcnnGraphSession Create(NcnnOps ops, ModelManifest manifest)
        {
            if (ops == null)
                throw new ArgumentNullException(nameof(ops));
            var session = new NcnnGraphSession(ops);
            if (manifest != null)
                session.ApplyModelManifest(manifest);
            return session;
        }

        public static NcnnGraphSession Create(NcnnOps ops, string modelId, NcnnPrecisionMode precisionMode)
        {
            var manifest = NcnnModelManifestLoader.ResolveRunnerManifest(modelId, precisionMode);
            var appliedMode = NcnnModelManifestLoader.ResolveAppliedPrecision(modelId, precisionMode, manifest);
            var session = Create(ops, manifest);
            session.SetAppliedPrecisionMode(appliedMode);
            Debug.Log("[NcnnPrecision] model=" + (modelId ?? string.Empty)
                + " | requested=" + precisionMode
                + " | applied=" + appliedMode
                + " | manifest=" + (manifest?.modelId ?? "legacy-fp32")
                + " | activation=" + session.ResolveActivationTextureFormat(4)
                + " | weights=" + (manifest?.precision?.weightDataType.ToString() ?? "Float32"));
            return session;
        }
    }
}
