using System;
using System.Threading;
using Aexis.Onnx;
using UnityEngine;

namespace Aexis.Samples
{
    public sealed class AexisOnnxInspectionRunner : MonoBehaviour
    {
        [SerializeField] private string onnxRelativePath = "DeepFileV2/deepfillv2_case1.source.onnx";

        public event Action<OnnxModel> ModelInspected;
        public event Action<Exception> InspectionFailed;

        public OnnxModel Model { get; private set; }

        public async Awaitable InspectAsync(CancellationToken cancellationToken = default)
        {
            try
            {
                var bytes = await AexisSampleStreamingAssets.ReadBytesAsync(onnxRelativePath, cancellationToken);
                Model = OnnxModelReader.Read(bytes);
                ModelInspected?.Invoke(Model);
                Debug.Log("Aexis ONNX model: " + Model.graph.name + ", nodes=" + Model.graph.nodes.Count, this);
            }
            catch (Exception exception)
            {
                Model = null;
                InspectionFailed?.Invoke(exception);
                Debug.LogException(exception, this);
            }
        }
    }
}
