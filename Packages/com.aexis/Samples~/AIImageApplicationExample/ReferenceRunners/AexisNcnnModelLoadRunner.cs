using System;
using System.IO;
using System.Threading;
using Aexis.Ncnn;
using Aexis.Samples.Async;
using UnityEngine;

namespace Aexis.Samples
{
    [Serializable]
    public sealed class AexisNcnnSampleModel
    {
        public string displayName;
        public string paramRelativePath;
        public string binRelativePath;
    }

    public sealed class AexisNcnnModelLoadRunner : MonoBehaviour
    {
        [SerializeField] private bool loadOnStart = true;
        [SerializeField] private string paramRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.param";
        [SerializeField] private string binRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.bin";

        public event Action<float, string> LoadProgressChanged;
        public event Action<NcnnGraphSession> ModelLoaded;
        public event Action<Exception> ModelLoadFailed;

        public NcnnGraphSession Session { get; private set; }
        public bool IsLoaded => Session != null && Session.Model != null;

        private async UniTask Start()
        {
            if (loadOnStart)
                await LoadAsync();
        }

        public async UniTask LoadAsync(CancellationToken cancellationToken = default)
        {
            Release();
            try
            {
                var paramText = await AexisSampleStreamingAssets.ReadTextAsync(paramRelativePath, cancellationToken);
                var binBytes = await AexisSampleStreamingAssets.ReadBytesAsync(binRelativePath, cancellationToken);
                Session = NcnnInferenceSessionFactory.Create(new NcnnOps());
                using var stream = new MemoryStream(binBytes, writable: false);
                using var weights = new NcnnBinReader(stream);
                await Session.LoadModelAsync(paramText, weights, ReportProgress, cancellationToken);
                ModelLoaded?.Invoke(Session);
            }
            catch (Exception exception)
            {
                Release();
                ModelLoadFailed?.Invoke(exception);
                Debug.LogException(exception, this);
            }
        }

        public void SelectModel(AexisNcnnSampleModel model)
        {
            if (model == null)
                throw new ArgumentNullException(nameof(model));
            paramRelativePath = model.paramRelativePath;
            binRelativePath = model.binRelativePath;
        }

        public void Release()
        {
            if (Session == null)
                return;
            Session.Release();
            Session = null;
        }

        private void OnDestroy()
        {
            Release();
        }

        private void ReportProgress(NcnnGraphSession.LoadProgress progress)
        {
            LoadProgressChanged?.Invoke(progress.progress01, progress.stage);
        }
    }
}
