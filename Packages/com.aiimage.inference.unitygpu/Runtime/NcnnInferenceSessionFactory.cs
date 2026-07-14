using System;
using AIImage.Inference.Core;

namespace NcnnCompute
{
    // Application runners use this package-owned factory rather than owning backend construction.
    public static class NcnnInferenceSessionFactory
    {
        public static NcnnRepro Create(NcnnOps ops)
        {
            return Create(ops, NcnnModelManifestLoader.TryLoadFromEnvironment());
        }

        public static NcnnRepro Create(NcnnOps ops, ModelManifest manifest)
        {
            if (ops == null)
                throw new ArgumentNullException(nameof(ops));
            var session = new NcnnRepro(ops);
            if (manifest != null)
                session.ApplyModelManifest(manifest);
            return session;
        }
    }
}
