namespace Aexis.Samples
{
    public static class AexisSampleModelCatalog
    {
        public static readonly AexisNcnnSampleModel ClipImageEncoder = new AexisNcnnSampleModel
        {
            displayName = "Clip MobileCLIP S0 image encoder",
            paramRelativePath = "Clip/mobileclip_s0_export/image_encoder.ncnn.param",
            binRelativePath = "Clip/mobileclip_s0_export/image_encoder.ncnn.bin"
        };

        public static readonly AexisNcnnSampleModel CodeFormerEncoder = new AexisNcnnSampleModel
        {
            displayName = "CodeFormer encoder",
            paramRelativePath = "CodeFormer/models/encoder.param",
            binRelativePath = "CodeFormer/models/encoder.bin"
        };

        public static readonly AexisNcnnSampleModel CodeFormerGenerator = new AexisNcnnSampleModel
        {
            displayName = "CodeFormer generator",
            paramRelativePath = "CodeFormer/models/generator.param",
            binRelativePath = "CodeFormer/models/generator.bin"
        };

        public static readonly AexisNcnnSampleModel DeepFillV2 = new AexisNcnnSampleModel
        {
            displayName = "DeepFillV2 case 1",
            paramRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.param",
            binRelativePath = "DeepFileV2/deepfillv2_case1.ncnn.bin"
        };

        public static readonly AexisNcnnSampleModel Matting = new AexisNcnnSampleModel
        {
            displayName = "Matting",
            paramRelativePath = "Matting/matting.param",
            binRelativePath = "Matting/matting.bin"
        };

        public static readonly AexisNcnnSampleModel RealEsrgan = new AexisNcnnSampleModel
        {
            displayName = "RealESRGAN x4plus anime",
            paramRelativePath = "RealESRGAN/models/realesrgan-x4plus-anime.param",
            binRelativePath = "RealESRGAN/models/realesrgan-x4plus-anime.bin"
        };

        public static readonly AexisNcnnSampleModel YoloV8Seg = new AexisNcnnSampleModel
        {
            displayName = "YOLOv8n segmentation",
            paramRelativePath = "Yolo/yolov8n_seg.ncnn.param",
            binRelativePath = "Yolo/yolov8n_seg.ncnn.bin"
        };
    }
}
