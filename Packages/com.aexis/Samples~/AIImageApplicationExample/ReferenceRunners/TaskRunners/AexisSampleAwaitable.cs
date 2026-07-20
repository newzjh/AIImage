using System.Threading;
using UnityEngine;
using UnityEngine.Rendering;

namespace Aexis.Samples.Runners
{
    internal static class AexisSampleAwaitable
    {
        public static async Awaitable<T> FromResult<T>(T value)
        {
            return value;
        }

        public static async Awaitable<AsyncGPUReadbackRequest> RequestReadbackAsync(
            RenderTexture texture,
            TextureFormat format,
            CancellationToken cancellationToken)
        {
            var request = AsyncGPUReadback.Request(texture, 0, format);
            while (!request.done)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await Awaitable.NextFrameAsync();
            }
            return request;
        }
    }
}
