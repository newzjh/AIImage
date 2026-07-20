using UnityEngine;

namespace Aexis.Async
{
    /// <summary>Centralizes the engine's frame scheduling policy without third-party async dependencies.</summary>
    public static class AexisAsync
    {
        /// <summary>Completes after Unity advances to the next frame.</summary>
        public static Awaitable YieldFrame()
        {
            return Awaitable.NextFrameAsync();
        }
    }
}
