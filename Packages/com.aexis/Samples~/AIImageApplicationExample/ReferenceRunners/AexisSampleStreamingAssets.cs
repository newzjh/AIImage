using System;
using System.IO;
using System.Threading;
using Aexis.Samples.Async;
using UnityEngine;
using UnityEngine.Networking;

namespace Aexis.Samples
{
    public static class AexisSampleStreamingAssets
    {
        public static async UniTask<byte[]> ReadBytesAsync(string relativePath, CancellationToken cancellationToken = default)
        {
            if (string.IsNullOrWhiteSpace(relativePath))
                throw new ArgumentException("A StreamingAssets-relative path is required.", nameof(relativePath));

            var root = Application.streamingAssetsPath.TrimEnd('/', '\\');
            var path = root + "/" + relativePath.TrimStart('/', '\\');
            using var request = UnityWebRequest.Get(path);
            var operation = request.SendWebRequest();
            while (!operation.isDone)
            {
                cancellationToken.ThrowIfCancellationRequested();
                await UniTask.NextFrame();
            }

            if (request.result != UnityWebRequest.Result.Success)
                throw new IOException("Unable to load StreamingAssets file '" + relativePath + "': " + request.error);
            return request.downloadHandler.data;
        }

        public static async UniTask<string> ReadTextAsync(string relativePath, CancellationToken cancellationToken = default)
        {
            var bytes = await ReadBytesAsync(relativePath, cancellationToken);
            return System.Text.Encoding.UTF8.GetString(bytes);
        }
    }
}
