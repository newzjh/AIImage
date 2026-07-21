using System;
using Aexis;
using UnityEngine;
using Aexis.Execution;

namespace Aexis.Ncnn
{
    // Application runners use this package-owned factory rather than owning backend construction.
    public static class NcnnInferenceSessionFactory
    {
        public static AexisGraphSession Create(AexisOps ops)
        {
            return Create(ops, AexisModelManifestLoader.TryLoadFromEnvironment());
        }

        public static AexisGraphSession Create(AexisOps ops, ModelManifest manifest)
        {
            if (ops == null)
                throw new ArgumentNullException(nameof(ops));
            var session = new AexisGraphSession(ops);
            if (manifest != null)
                session.ApplyModelManifest(manifest);
            return session;
        }

        public static AexisGraphSession Create(AexisOps ops, string modelId, AexisPrecisionMode precisionMode)
        {
            var manifest = AexisModelManifestLoader.ResolveRunnerManifest(modelId, precisionMode);
            var appliedMode = AexisModelManifestLoader.ResolveAppliedPrecision(modelId, precisionMode, manifest);
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
