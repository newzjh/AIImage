using Cysharp.Threading.Tasks;

namespace Aexis.Async
{
    /// <summary>Centralizes the engine's UniTask scheduling policy.</summary>
    public static class AexisAsync
    {
        public static UniTask YieldFrame()
        {
            return UniTask.NextFrame();
        }
    }
}
