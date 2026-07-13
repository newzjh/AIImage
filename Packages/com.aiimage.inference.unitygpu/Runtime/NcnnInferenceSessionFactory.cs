using System;

namespace NcnnCompute
{
    // Application runners use this package-owned factory rather than owning backend construction.
    public static class NcnnInferenceSessionFactory
    {
        public static NcnnRepro Create(NcnnOps ops)
        {
            if (ops == null)
                throw new ArgumentNullException(nameof(ops));
            return new NcnnRepro(ops);
        }
    }
}
