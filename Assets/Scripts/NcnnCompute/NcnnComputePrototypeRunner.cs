using System;
using System.IO;
using Cysharp.Threading.Tasks;
using UnityEngine;

namespace NcnnCompute
{
    public sealed class NcnnComputePrototypeRunner : MonoBehaviour
    {
        public string paramRelativePath = "RealESRGAN/models/realesrgan-x4plus.param";
        public Texture inputTexture;
        public RenderTexture outputTexture;

        public async UniTask<NcnnParamModel> LoadParamAsync()
        {
            var path = Path.Combine(Application.streamingAssetsPath, paramRelativePath);
            string txt;
            try
            {
                txt = await File.ReadAllTextAsync(path);
            }
            catch (Exception e)
            {
                throw new InvalidOperationException("read param failed: " + path + " " + e.Message);
            }
            return NcnnParamParser.Parse(txt);
        }

        private async void Start()
        {
            try
            {
                await LoadParamAsync();
                if (inputTexture == null)
                    return;
                var backend = new NcnnComputeBackend();
                using var t = backend.Passthrough(inputTexture);
                outputTexture = t.rt;
            }
            catch
            {
            }
        }
    }
}
