using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Samples.Runners
{
    internal static class AexisSampleUniTask
    {
        public static async UniTask<T> FromResult<T>(T value)
        {
            return value;
        }

        public static async UniTask<AsyncGPUReadbackRequest> RequestReadbackAsync(
            RenderTexture texture,
            TextureFormat format,
            CancellationToken cancellationToken)
        {
            var request = AsyncGPUReadback.Request(texture, 0, format);
            while (!request.done)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.NextFrame();
            }
            return request;
        }
    }
}
