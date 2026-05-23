#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#include "ncnn/layer.h"
#include "ncnn/mat.h"
#include "ncnn/net.h"

#if defined(_WIN32)
#define REALESRGAN_EXPORT __declspec(dllexport)
#else
#define REALESRGAN_EXPORT __attribute__((visibility("default")))
#endif

struct StyleConvWeights
{
    int data_size;
    int inc;
    int hid_dim;
    int num_output;
    std::vector<float> style_convs_modulated_conv_weight;
    std::vector<float> style_convs_modulated_conv_modulation_weight;
    std::vector<float> style_convs_modulated_conv_modulation_bias;
    std::vector<float> style_convs_weight;
    std::vector<float> style_convs_bias;
};

struct ToRgbConvWeights
{
    int data_size;
    int inc;
    int hid_dim;
    int num_output;
    std::vector<float> to_rgbs_modulated_conv_weight;
    std::vector<float> to_rgbs_modulated_conv_modulation_weight;
    std::vector<float> to_rgbs_modulated_conv_modulation_bias;
    std::vector<float> to_rgbs_bias;
};

static const int style_conv_sizes[][5] = {
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512},
    {256 * 512 * 3 * 3, 512 * 512, 512, 1, 256},
    {256 * 256 * 3 * 3, 256 * 512, 256, 1, 256},
    {128 * 256 * 3 * 3, 256 * 512, 256, 1, 128},
    {128 * 128 * 3 * 3, 128 * 512, 128, 1, 128},
    {64 * 128 * 3 * 3, 128 * 512, 128, 1, 64},
    {64 * 64 * 3 * 3, 64 * 512, 64, 1, 64},
    {512 * 512 * 3 * 3, 512 * 512, 512, 1, 512}};

static const int to_rgb_sizes[][4] = {
    {3 * 512 * 1 * 1, 512 * 512, 512, 3},
    {3 * 512 * 1 * 1, 512 * 512, 512, 3},
    {3 * 512 * 1 * 1, 512 * 512, 512, 3},
    {3 * 512 * 1 * 1, 512 * 512, 512, 3},
    {3 * 256 * 1 * 1, 256 * 512, 256, 3},
    {3 * 128 * 1 * 1, 128 * 512, 128, 3},
    {3 * 64 * 1 * 1, 64 * 512, 64, 3},
    {3 * 512 * 1 * 1, 512 * 512, 512, 3}};

static const int style_conv_channels[][3] = {
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 512},
    {512, 512, 256},
    {512, 256, 256},
    {512, 256, 128},
    {512, 128, 128},
    {512, 128, 64},
    {512, 64, 64},
    {512, 512, 512}};

static const int to_rgb_channels[][3] = {
    {512, 512, 3},
    {512, 512, 3},
    {512, 512, 3},
    {512, 512, 3},
    {512, 256, 3},
    {512, 128, 3},
    {512, 64, 3},
    {512, 512, 3}};

struct gfpgan_ctx
{
    ncnn::Net net;
    std::vector<StyleConvWeights> style_conv_weights;
    std::vector<ToRgbConvWeights> to_rgbs_conv_weights;
    ncnn::Mat const_input;
    std::string last_error;
};

static const char* set_error_gfp(gfpgan_ctx* ctx, const char* msg)
{
    if (!ctx) return msg;
    ctx->last_error = msg ? msg : "";
    return ctx->last_error.c_str();
}

static ncnn::Mat generate_noise(int c, int h, int w, const float* weight)
{
    unsigned seed = (unsigned)std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine gen(seed);
    std::normal_distribution<double> dis(0, 1);

    ncnn::Mat noise(w, h, c);
    for (size_t i = 0; i < noise.total(); i++)
        noise[i] = (float)dis(gen) * weight[0];
    return noise;
}

static void relu(ncnn::Mat& in, float slope)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("ReLU");
    ncnn::ParamDict pd;
    pd.set(0, slope);
    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward_inplace(in, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void binary_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");
    ncnn::ParamDict pd;
    pd.set(0, 0);
    op->load_param(pd);
    op->create_pipeline(opt);
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;
    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);
    c = tops[0];
    op->destroy_pipeline(opt);
    delete op;
}

static void binary_mul(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");
    ncnn::ParamDict pd;
    pd.set(0, 2);
    op->load_param(pd);
    op->create_pipeline(opt);
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;
    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);
    c = tops[0];
    op->destroy_pipeline(opt);
    delete op;
}

static void innerproduct(const ncnn::Mat& in, const float* weight, const float* bias, int inc, int num_output, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    opt.use_vulkan_compute = false;

    ncnn::Layer* op = ncnn::create_layer("InnerProduct");
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, 1);
    pd.set(2, inc * num_output);
    op->load_param(pd);

    ncnn::Mat weights[2];
    weights[0].create(inc * num_output);
    weights[1].create(num_output);
    for (int i = 0; i < num_output; i++)
    {
        for (int j = 0; j < inc; j++)
            weights[0][i * inc + j] = weight[i * inc + j];
    }
    for (int i = 0; i < num_output; i++)
        weights[1][i] = bias[i];
    op->load_model(ncnn::ModelBinFromMatArray(weights));
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void concat(const ncnn::Mat& a, const ncnn::Mat& b, int axis, ncnn::Mat& c)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Concat");
    ncnn::ParamDict pd;
    pd.set(0, axis);
    op->load_param(pd);
    op->create_pipeline(opt);
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;
    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);
    c = tops[0];
    op->destroy_pipeline(opt);
    delete op;
}

static void convolution(const ncnn::Mat& in, const float* weight, int inc, int num_output, int kernel_size, int padding, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;
    opt.use_vulkan_compute = false;

    ncnn::Layer* op = ncnn::create_layer("Convolution");
    ncnn::ParamDict pd;
    pd.set(0, num_output);
    pd.set(1, kernel_size);
    pd.set(5, 0);
    pd.set(6, inc * num_output * kernel_size * kernel_size);
    pd.set(7, 1);
    pd.set(4, padding);
    pd.set(14, padding);
    pd.set(15, padding);
    pd.set(16, padding);
    op->load_param(pd);

    ncnn::Mat weights[1];
    weights[0].create(inc * num_output * kernel_size * kernel_size);
    for (int i = 0; i < inc * num_output * kernel_size * kernel_size; i++)
        weights[0][i] = weight[i];
    op->load_model(ncnn::ModelBinFromMatArray(weights));
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void scale_op(const ncnn::Mat& in, float scale, int scale_data_size, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Scale");
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);
    pd.set(1, 0);
    op->load_param(pd);

    ncnn::Mat scales[1];
    scales[0].create(scale_data_size);
    for (int i = 0; i < scale_data_size; i++)
        scales[0][i] = scale;
    op->load_model(ncnn::ModelBinFromMatArray(scales));
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void upsample(const ncnn::Mat& in, float scale, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");
    ncnn::ParamDict pd;
    pd.set(0, 2);
    pd.set(1, scale);
    pd.set(2, scale);
    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward(in, out, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static void clip(ncnn::Mat& in, float minv, float maxv)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Clip");
    ncnn::ParamDict pd;
    pd.set(0, minv);
    pd.set(1, maxv);
    op->load_param(pd);
    op->create_pipeline(opt);
    op->forward_inplace(in, opt);
    op->destroy_pipeline(opt);
    delete op;
}

static int modulated_conv(ncnn::Mat& x, ncnn::Mat& style, const float* self_weight, const float* weights, const float* bias, int sample_mode, int demodulate, int inc, int num_output, int kernel_size, int hid_dim, ncnn::Mat& out)
{
    ncnn::Mat style_out;
    innerproduct(style, weights, bias, inc, hid_dim, style_out);
    ncnn::Mat weight(kernel_size, kernel_size, hid_dim, num_output);
    for (int i = 0; i < weight.c; i++)
    {
        ncnn::Mat channel = weight.channel(i);
        for (int j = 0; j < weight.d; j++)
        {
            ncnn::Mat d = channel.channel(j);
            for (int k = 0; k < d.h; k++)
            {
                for (int l = 0; l < d.w; l++)
                {
                    weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l] =
                        style_out.channel(0)[j] * self_weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l];
                }
            }
        }
    }

    if (demodulate == 1)
    {
        ncnn::Mat demod(num_output, 1, 1, 1);
        for (int i = 0; i < weight.c; i++)
        {
            ncnn::Mat channel = weight.channel(i);
            float sum = 0.f;
            for (int j = 0; j < weight.d; j++)
            {
                ncnn::Mat d = channel.channel(j);
                for (int k = 0; k < d.h; k++)
                {
                    for (int l = 0; l < d.w; l++)
                    {
                        float v = weight[i * weight.d * weight.h * weight.w + j * weight.h * weight.w + k * weight.w + l];
                        sum += v * v;
                    }
                }
            }
            demod[i] = 1.f / std::sqrt(sum + 0.00000001f);
        }
        for (int i = 0; i < weight.c; i++)
        {
            float* weight_data = weight.channel(i);
            for (int l = 0; l < weight.d; l++)
            {
                for (int j = 0; j < weight.h; j++)
                {
                    for (int k = 0; k < weight.w; k++)
                        weight_data[l * weight.h * weight.w + j * weight.w + k] = weight_data[l * weight.h * weight.w + j * weight.w + k] * demod[i];
                }
            }
        }
    }

    if (sample_mode == 1)
        upsample(x, 2.f, x);

    int padding = (int)std::floor(kernel_size / 2.f);
    convolution(x, (const float*)weight.data, hid_dim, num_output, kernel_size, padding, out);
    return 0;
}

static int to_rgbs(ncnn::Mat& out, ncnn::Mat& latent, ncnn::Mat& skip, int inc, int hid_dim, int num_output, const float* to_rgbs_modulated_conv_weight, const float* to_rgbs_modulated_conv_modulation_weight, const float* to_rgbs_modulated_conv_modulation_bias, const float* to_rgbs_bias)
{
    ncnn::Mat style;
    modulated_conv(out, latent, (const float*)to_rgbs_modulated_conv_weight, (const float*)to_rgbs_modulated_conv_modulation_weight, to_rgbs_modulated_conv_modulation_bias, 0, 0, inc, num_output, 1, hid_dim, style);
    ncnn::Mat bias(num_output, (void*)to_rgbs_bias);
    bias = bias.reshape(1, 1, num_output);
    if (skip.empty())
        binary_add(style, bias, skip);
    else
    {
        binary_add(style, bias, style);
        upsample(skip, 2.f, skip);
        binary_add(style, skip, skip);
    }
    return 0;
}

static int style_convs_modulated_conv(ncnn::Mat& x, ncnn::Mat style, int sample_mode, int demodulate, ncnn::Mat& out, int inc, int hid_dim, int num_output, const float* style_convs_modulated_conv_weight, const float* style_convs_modulated_conv_modulation_weight, const float* style_convs_modulated_conv_modulation_bias, const float* style_convs_weight, const float* style_convs_bias)
{
    ncnn::Mat conv_out;
    modulated_conv(x, style, (const float*)style_convs_modulated_conv_weight, (const float*)style_convs_modulated_conv_modulation_weight, style_convs_modulated_conv_modulation_bias, sample_mode, demodulate, inc, num_output, 3, hid_dim, conv_out);
    scale_op(conv_out, 1.4142135381698608f, num_output, conv_out);
    ncnn::Mat noise = generate_noise(1, conv_out.h, conv_out.w, style_convs_weight);
    binary_add(conv_out, noise, conv_out);
    ncnn::Mat bias(num_output, (void*)style_convs_bias);
    bias = bias.reshape(1, 1, num_output);
    binary_add(conv_out, bias, out);
    relu(out, 0.2f);
    return 0;
}

static int load_weights(const char* model_path, std::vector<StyleConvWeights>& style_conv_weights, std::vector<ToRgbConvWeights>& to_rgbs_conv_weights, ncnn::Mat& const_input)
{
    std::ifstream ifs(model_path, std::ios::binary | std::ios::in);
    if (!ifs.is_open())
        return -1;

    style_conv_weights.clear();
    to_rgbs_conv_weights.clear();

    for (int i = 0; i < 15; i++)
    {
        int data_size1 = style_conv_sizes[i][0];
        int data_size2 = style_conv_sizes[i][1];
        int data_size3 = style_conv_sizes[i][2];
        int data_size4 = style_conv_sizes[i][3];
        int data_size5 = style_conv_sizes[i][4];
        int data_size = data_size1 + data_size2 + data_size3 + data_size4 + data_size5;

        StyleConvWeights weights;
        weights.data_size = data_size;
        weights.inc = style_conv_channels[i][0];
        weights.hid_dim = style_conv_channels[i][1];
        weights.num_output = style_conv_channels[i][2];
        weights.style_convs_modulated_conv_weight.resize(data_size1);
        weights.style_convs_modulated_conv_modulation_weight.resize(data_size2);
        weights.style_convs_modulated_conv_modulation_bias.resize(data_size3);
        weights.style_convs_weight.resize(data_size4);
        weights.style_convs_bias.resize(data_size5);

        ifs.read((char*)weights.style_convs_modulated_conv_weight.data(), sizeof(float) * data_size1);
        ifs.read((char*)weights.style_convs_modulated_conv_modulation_weight.data(), sizeof(float) * data_size2);
        ifs.read((char*)weights.style_convs_modulated_conv_modulation_bias.data(), sizeof(float) * data_size3);
        ifs.read((char*)weights.style_convs_weight.data(), sizeof(float) * data_size4);
        ifs.read((char*)weights.style_convs_bias.data(), sizeof(float) * data_size5);
        style_conv_weights.push_back(std::move(weights));
    }

    for (int i = 0; i < 8; i++)
    {
        int data_size1 = to_rgb_sizes[i][0];
        int data_size2 = to_rgb_sizes[i][1];
        int data_size3 = to_rgb_sizes[i][2];
        int data_size4 = to_rgb_sizes[i][3];
        int data_size = data_size1 + data_size2 + data_size3 + data_size4;

        ToRgbConvWeights weights;
        weights.data_size = data_size;
        weights.inc = to_rgb_channels[i][0];
        weights.hid_dim = to_rgb_channels[i][1];
        weights.num_output = to_rgb_channels[i][2];
        weights.to_rgbs_modulated_conv_weight.resize(data_size1);
        weights.to_rgbs_modulated_conv_modulation_weight.resize(data_size2);
        weights.to_rgbs_modulated_conv_modulation_bias.resize(data_size3);
        weights.to_rgbs_bias.resize(data_size4);

        ifs.read((char*)weights.to_rgbs_modulated_conv_weight.data(), sizeof(float) * data_size1);
        ifs.read((char*)weights.to_rgbs_modulated_conv_modulation_weight.data(), sizeof(float) * data_size2);
        ifs.read((char*)weights.to_rgbs_modulated_conv_modulation_bias.data(), sizeof(float) * data_size3);
        ifs.read((char*)weights.to_rgbs_bias.data(), sizeof(float) * data_size4);
        to_rgbs_conv_weights.push_back(std::move(weights));
    }

    int const_input_size = 4 * 4 * 512;
    std::vector<float> const_input_data;
    const_input_data.resize(const_input_size);
    ifs.read((char*)const_input_data.data(), sizeof(float) * const_input_size);
    const_input = ncnn::Mat(512 * 4 * 4, (void*)const_input_data.data()).reshape(4, 4, 512).clone();
    ifs.close();
    return 0;
}

extern "C" REALESRGAN_EXPORT const char* Gfpgan_Create(const char* modelDir, const char* modelName, int gpuId, void** outCtx)
{
    (void)modelName;
    (void)gpuId;

    if (!outCtx) return "outCtx is null";
    *outCtx = nullptr;
    if (!modelDir) return "modelDir is null";

    auto* ctx = new gfpgan_ctx();

    std::string parampath = std::string(modelDir) + "/encoder.param";
    std::string modelpath = std::string(modelDir) + "/encoder.bin";
    std::string stylepath = std::string(modelDir) + "/style.bin";

#if defined(_WIN32)
    for (auto& ch : parampath) if (ch == '/') ch = '\\';
    for (auto& ch : modelpath) if (ch == '/') ch = '\\';
    for (auto& ch : stylepath) if (ch == '/') ch = '\\';
#endif

    ctx->net.opt.use_vulkan_compute = false;
    ctx->net.opt.num_threads = 4;

    int retp = ctx->net.load_param(parampath.c_str());
    if (retp < 0)
    {
        auto* err = set_error_gfp(ctx, "failed to load encoder param");
        delete ctx;
        return err;
    }

    int retm = ctx->net.load_model(modelpath.c_str());
    if (retm < 0)
    {
        auto* err = set_error_gfp(ctx, "failed to load encoder model");
        delete ctx;
        return err;
    }

    int retw = load_weights(stylepath.c_str(), ctx->style_conv_weights, ctx->to_rgbs_conv_weights, ctx->const_input);
    if (retw < 0)
    {
        auto* err = set_error_gfp(ctx, "failed to load style weights");
        delete ctx;
        return err;
    }

    *outCtx = ctx;
    return nullptr;
}

extern "C" REALESRGAN_EXPORT void Gfpgan_Destroy(void* ctxPtr)
{
    auto* ctx = (gfpgan_ctx*)ctxPtr;
    delete ctx;
}

extern "C" REALESRGAN_EXPORT const char* Gfpgan_ProcessRgba(void* ctxPtr, const unsigned char* rgba, int w, int h, unsigned char* outRgba, int outW, int outH)
{
    auto* ctx = (gfpgan_ctx*)ctxPtr;
    if (!ctx) return "ctx is null";
    if (!rgba || !outRgba) return set_error_gfp(ctx, "rgba/outRgba is null");
    if (w <= 0 || h <= 0) return set_error_gfp(ctx, "invalid input size");
    if (outW != w || outH != h) return set_error_gfp(ctx, "output size must match input");

    ncnn::Mat ncnn_in = ncnn::Mat::from_pixels_resize(rgba, ncnn::Mat::PIXEL_RGBA2RGB, w, h, 512, 512);
    const float mean_vals[3] = { 127.5f, 127.5f, 127.5f };
    const float norm_vals[3] = { 1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f };
    ncnn_in.substract_mean_normalize(mean_vals, norm_vals);

    ncnn::Extractor ex = ctx->net.create_extractor();
    if (ex.input("input.1", ncnn_in) != 0)
        return set_error_gfp(ctx, "encoder input failed");

    ncnn::Mat styles;
    if (ex.extract("420", styles) != 0)
        return set_error_gfp(ctx, "encoder extract styles failed");

    std::vector<ncnn::Mat> conditions;
    conditions.reserve(14);
    ncnn::Mat c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13;
    if (ex.extract("440", c0) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("443", c1) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("463", c2) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("466", c3) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("486", c4) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("489", c5) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("509", c6) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("512", c7) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("532", c8) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("535", c9) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("555", c10) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("558", c11) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("578", c12) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    if (ex.extract("581", c13) != 0) return set_error_gfp(ctx, "encoder extract cond failed");
    conditions.push_back(c0);
    conditions.push_back(c1);
    conditions.push_back(c2);
    conditions.push_back(c3);
    conditions.push_back(c4);
    conditions.push_back(c5);
    conditions.push_back(c6);
    conditions.push_back(c7);
    conditions.push_back(c8);
    conditions.push_back(c9);
    conditions.push_back(c10);
    conditions.push_back(c11);
    conditions.push_back(c12);
    conditions.push_back(c13);

    if (ctx->style_conv_weights.size() < 15 || ctx->to_rgbs_conv_weights.size() < 8 || ctx->const_input.empty())
        return set_error_gfp(ctx, "weights not loaded");

    ncnn::Mat latent_0 = styles.channel(0).row_range(0, 1);
    ncnn::Mat out;
    style_convs_modulated_conv(ctx->const_input, latent_0, 0, 1, out,
        ctx->style_conv_weights[14].inc, ctx->style_conv_weights[14].hid_dim, ctx->style_conv_weights[14].num_output,
        ctx->style_conv_weights[14].style_convs_modulated_conv_weight.data(), ctx->style_conv_weights[14].style_convs_modulated_conv_modulation_weight.data(), ctx->style_conv_weights[14].style_convs_modulated_conv_modulation_bias.data(),
        ctx->style_conv_weights[14].style_convs_weight.data(), ctx->style_conv_weights[14].style_convs_bias.data());

    ncnn::Mat latent_1 = styles.channel(0).row_range(1, 1);
    ncnn::Mat skip;
    to_rgbs(out, latent_1, skip,
        ctx->to_rgbs_conv_weights[7].inc, ctx->to_rgbs_conv_weights[7].hid_dim, ctx->to_rgbs_conv_weights[7].num_output,
        ctx->to_rgbs_conv_weights[7].to_rgbs_modulated_conv_weight.data(), ctx->to_rgbs_conv_weights[7].to_rgbs_modulated_conv_modulation_weight.data(), ctx->to_rgbs_conv_weights[7].to_rgbs_modulated_conv_modulation_bias.data(),
        ctx->to_rgbs_conv_weights[7].to_rgbs_bias.data());

    int j = 0;
    for (int i = 1; i < 14;)
    {
        ncnn::Mat latent = styles.channel(0).row_range(i, 1);
        style_convs_modulated_conv(out, latent, 1, 1, out,
            ctx->style_conv_weights[i - 1].inc, ctx->style_conv_weights[i - 1].hid_dim, ctx->style_conv_weights[i - 1].num_output,
            ctx->style_conv_weights[i - 1].style_convs_modulated_conv_weight.data(), ctx->style_conv_weights[i - 1].style_convs_modulated_conv_modulation_weight.data(), ctx->style_conv_weights[i - 1].style_convs_modulated_conv_modulation_bias.data(),
            ctx->style_conv_weights[i - 1].style_convs_weight.data(), ctx->style_conv_weights[i - 1].style_convs_bias.data());

        ncnn::Mat out_same = out.channel_range(0, out.c / 2);
        ncnn::Mat out_sft = out.channel_range(out.c / 2, out.c / 2);
        binary_mul(out_sft, conditions[i - 1], out_sft);
        binary_add(out_sft, conditions[i], out_sft);
        concat(out_same, out_sft, 0, out);

        latent = styles.channel(0).row_range(i + 1, 1);
        style_convs_modulated_conv(out, latent, 0, 1, out,
            ctx->style_conv_weights[i].inc, ctx->style_conv_weights[i].hid_dim, ctx->style_conv_weights[i].num_output,
            ctx->style_conv_weights[i].style_convs_modulated_conv_weight.data(), ctx->style_conv_weights[i].style_convs_modulated_conv_modulation_weight.data(), ctx->style_conv_weights[i].style_convs_modulated_conv_modulation_bias.data(),
            ctx->style_conv_weights[i].style_convs_weight.data(), ctx->style_conv_weights[i].style_convs_bias.data());

        latent = styles.channel(0).row_range(i + 2, 1);
        to_rgbs(out, latent, skip,
            ctx->to_rgbs_conv_weights[j].inc, ctx->to_rgbs_conv_weights[j].hid_dim, ctx->to_rgbs_conv_weights[j].num_output,
            ctx->to_rgbs_conv_weights[j].to_rgbs_modulated_conv_weight.data(), ctx->to_rgbs_conv_weights[j].to_rgbs_modulated_conv_modulation_weight.data(), ctx->to_rgbs_conv_weights[j].to_rgbs_modulated_conv_modulation_bias.data(),
            ctx->to_rgbs_conv_weights[j].to_rgbs_bias.data());

        i += 2;
        j += 1;
    }

    clip(skip, -1.f, 1.f);
    if (skip.w != 512 || skip.h != 512 || skip.c != 3)
        return set_error_gfp(ctx, "unexpected output shape");

    for (int i = 0; i < outW * outH; i++)
        outRgba[i * 4 + 3] = 255;

    for (int y = 0; y < outH; y++)
    {
        int sy = (int)std::floor((y + 0.5f) * 512.f / (float)outH);
        if (sy < 0) sy = 0;
        if (sy > 511) sy = 511;
        const float* pr = skip.channel(0).row(sy);
        const float* pg = skip.channel(1).row(sy);
        const float* pb = skip.channel(2).row(sy);
        for (int x = 0; x < outW; x++)
        {
            int sx = (int)std::floor((x + 0.5f) * 512.f / (float)outW);
            if (sx < 0) sx = 0;
            if (sx > 511) sx = 511;
            float r = pr[sx] * 0.5f + 0.5f;
            float g = pg[sx] * 0.5f + 0.5f;
            float b = pb[sx] * 0.5f + 0.5f;
            r = r < 0.f ? 0.f : (r > 1.f ? 1.f : r);
            g = g < 0.f ? 0.f : (g > 1.f ? 1.f : g);
            b = b < 0.f ? 0.f : (b > 1.f ? 1.f : b);
            size_t di = ((size_t)y * (size_t)outW + (size_t)x) * 4;
            outRgba[di + 0] = (unsigned char)(r * 255.f + 0.5f);
            outRgba[di + 1] = (unsigned char)(g * 255.f + 0.5f);
            outRgba[di + 2] = (unsigned char)(b * 255.f + 0.5f);
        }
    }

    return nullptr;
}

