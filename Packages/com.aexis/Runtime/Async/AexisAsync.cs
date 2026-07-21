using System.Threading.Tasks;

namespace Aexis.Async
{
    /// <summary>Centralizes the engine's frame scheduling policy without third-party async dependencies.</summary>
    public static class AexisAsync
    {
        /// <summary>Completes after Unity advances to the next frame.</summary>
        public static async Task YieldFrame()
        {
            await Task.Yield();
        }
    }
}
