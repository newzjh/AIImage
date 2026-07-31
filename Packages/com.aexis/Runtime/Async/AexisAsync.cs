using System.Threading.Tasks;
using UnityEngine;

namespace Aexis.Async
{
    /// <summary>Centralizes the engine's frame scheduling policy without third-party async dependencies.</summary>
    public static class AexisAsync
    {
        /// <summary>Completes after Unity advances to the next frame.</summary>
        public static async Task YieldFrame()
        {
#if UNITY_6000_0_OR_NEWER
            // Task.Yield only schedules another synchronization-context work item.
            // Under load it can resume within the current player-loop iteration,
            // allowing texture compute layers to be submitted as one GPU burst.
            await Awaitable.NextFrameAsync();
#else
            await Task.Yield();
#endif
        }
    }
}
