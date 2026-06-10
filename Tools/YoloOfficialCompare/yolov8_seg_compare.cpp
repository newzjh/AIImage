#ifndef NOMINMAX
#define NOMINMAX
#endif

#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <direct.h>
#include <fstream>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "layer.h"
#include "net.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#define STBI_ONLY_BMP
#define STBI_ONLY_PNM
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

struct RectF
{
    float x;
    float y;
    float width;
    float height;
};

struct Object
{
    RectF rect;
    int label;
    float prob;
    int gindex;
    int mask_width;
    int mask_height;
    std::vector<unsigned char> mask;
};

static inline float intersection_area(const Object& a, const Object& b)
{
    const float x0 = std::max(a.rect.x, b.rect.x);
    const float y0 = std::max(a.rect.y, b.rect.y);
    const float x1 = std::min(a.rect.x + a.rect.width, b.rect.x + b.rect.width);
    const float y1 = std::min(a.rect.y + a.rect.height, b.rect.y + b.rect.height);
    if (x1 <= x0 || y1 <= y0)
        return 0.f;
    return (x1 - x0) * (y1 - y0);
}

static void qsort_descent_inplace(std::vector<Object>& objects, int left, int right)
{
    int i = left;
    int j = right;
    float p = objects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (objects[i].prob > p)
            i++;

        while (objects[j].prob < p)
            j--;

        if (i <= j)
        {
            std::swap(objects[i], objects[j]);
            i++;
            j--;
        }
    }

    if (left < j) qsort_descent_inplace(objects, left, j);
    if (i < right) qsort_descent_inplace(objects, i, right);
}

static void qsort_descent_inplace(std::vector<Object>& objects)
{
    if (objects.empty())
        return;
    qsort_descent_inplace(objects, 0, (int)objects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& objects, std::vector<int>& picked, float nms_threshold, bool agnostic = false)
{
    picked.clear();
    const int n = (int)objects.size();
    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = objects[i].rect.width * objects[i].rect.height;
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = objects[i];
        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = objects[picked[j]];
            if (!agnostic && a.label != b.label)
                continue;

            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            if (union_area > 0.f && inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

static inline float sigmoid(float x)
{
    return 1.f / (1.f + std::exp(-x));
}

static void generate_proposals(const ncnn::Mat& pred, int stride, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    const int w = in_pad.w;
    const int h = in_pad.h;
    const int num_grid_x = w / stride;
    const int num_grid_y = h / stride;
    const int reg_max_1 = 16;
    const int num_class = pred.w - reg_max_1 * 4;

    for (int y = 0; y < num_grid_y; y++)
    {
        for (int x = 0; x < num_grid_x; x++)
        {
            const ncnn::Mat pred_grid = pred.row_range(y * num_grid_x + x, 1);

            int label = -1;
            float score = -FLT_MAX;
            {
                const ncnn::Mat pred_score = pred_grid.range(reg_max_1 * 4, num_class);
                for (int k = 0; k < num_class; k++)
                {
                    const float s = pred_score[k];
                    if (s > score)
                    {
                        label = k;
                        score = s;
                    }
                }
                score = sigmoid(score);
            }

            if (score < prob_threshold)
                continue;

            ncnn::Mat pred_bbox = pred_grid.range(0, reg_max_1 * 4).reshape(reg_max_1, 4).clone();
            {
                ncnn::Layer* softmax = ncnn::create_layer("Softmax");
                ncnn::ParamDict pd;
                pd.set(0, 1);
                pd.set(1, 1);
                softmax->load_param(pd);

                ncnn::Option opt;
                opt.num_threads = 1;
                opt.use_packing_layout = false;

                softmax->create_pipeline(opt);
                softmax->forward_inplace(pred_bbox, opt);
                softmax->destroy_pipeline(opt);
                delete softmax;
            }

            float pred_ltrb[4];
            for (int k = 0; k < 4; k++)
            {
                float dis = 0.f;
                const float* dis_after_sm = pred_bbox.row(k);
                for (int l = 0; l < reg_max_1; l++)
                {
                    dis += l * dis_after_sm[l];
                }
                pred_ltrb[k] = dis * stride;
            }

            const float pb_cx = (x + 0.5f) * stride;
            const float pb_cy = (y + 0.5f) * stride;

            Object obj;
            obj.rect.x = pb_cx - pred_ltrb[0];
            obj.rect.y = pb_cy - pred_ltrb[1];
            obj.rect.width = pb_cx + pred_ltrb[2] - obj.rect.x;
            obj.rect.height = pb_cy + pred_ltrb[3] - obj.rect.y;
            obj.label = label;
            obj.prob = score;
            obj.gindex = y * num_grid_x + x;
            obj.mask_width = 0;
            obj.mask_height = 0;
            objects.push_back(obj);
        }
    }
}

static void generate_proposals(const ncnn::Mat& pred, const std::vector<int>& strides, const ncnn::Mat& in_pad, float prob_threshold, std::vector<Object>& objects)
{
    int pred_row_offset = 0;
    for (size_t i = 0; i < strides.size(); i++)
    {
        const int stride = strides[i];
        const int num_grid_x = in_pad.w / stride;
        const int num_grid_y = in_pad.h / stride;
        const int num_grid = num_grid_x * num_grid_y;

        std::vector<Object> objects_stride;
        generate_proposals(pred.row_range(pred_row_offset, num_grid), stride, in_pad, prob_threshold, objects_stride);
        for (size_t j = 0; j < objects_stride.size(); j++)
        {
            Object obj = objects_stride[j];
            obj.gindex += pred_row_offset;
            objects.push_back(obj);
        }

        pred_row_offset += num_grid;
    }
}

static bool write_gray_png(const std::string& path, int width, int height, const std::vector<unsigned char>& data)
{
    if ((int)data.size() != width * height)
        return false;
    return stbi_write_png(path.c_str(), width, height, 1, data.data(), width) != 0;
}

static bool write_rgba_png(const std::string& path, int width, int height, const std::vector<unsigned char>& data)
{
    if ((int)data.size() != width * height * 4)
        return false;
    return stbi_write_png(path.c_str(), width, height, 4, data.data(), width * 4) != 0;
}

static void dump_blob_f32(const std::string& path, const ncnn::Mat& m)
{
    std::ofstream os(path.c_str(), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!os.is_open())
        return;

    if (m.dims == 2)
    {
        os.write((const char*)m, (std::streamsize)(m.w * m.h * (int)sizeof(float)));
        return;
    }

    if (m.dims == 3)
    {
        for (int q = 0; q < m.c; q++)
        {
            const float* ptr = m.channel(q);
            os.write((const char*)ptr, (std::streamsize)(m.w * m.h * (int)sizeof(float)));
        }
    }
}

static void dump_blob_summary_2d(const std::string& path, const ncnn::Mat& m)
{
    std::ofstream os(path.c_str(), std::ios::out | std::ios::trunc);
    if (!os.is_open())
        return;

    const int count = m.w * m.h;
    const float* ptr = (const float*)m;
    int finite = 0;
    int nan_count = 0;
    int inf_count = 0;
    double sum = 0.0;
    float minv = FLT_MAX;
    float maxv = -FLT_MAX;
    for (int i = 0; i < count; i++)
    {
        const float v = ptr[i];
        if (std::isnan(v))
        {
            nan_count++;
            continue;
        }
        if (!std::isfinite(v))
        {
            inf_count++;
            continue;
        }
        finite++;
        sum += v;
        minv = std::min(minv, v);
        maxv = std::max(maxv, v);
    }

    os << "shape=2d " << m.w << "x" << m.h << "x1x1\n";
    os << "count=" << count << "\n";
    os << "finite=" << finite << "\n";
    os << "nan=" << nan_count << "\n";
    os << "inf=" << inf_count << "\n";
    os << "min=" << (finite > 0 ? minv : NAN) << "\n";
    os << "max=" << (finite > 0 ? maxv : NAN) << "\n";
    os << "mean=" << (finite > 0 ? (sum / finite) : NAN) << "\n";

    const int rows_to_dump[4] = {0, std::min(1, m.h - 1), m.h / 2, m.h - 1};
    int dumped[4] = {-1, -1, -1, -1};
    int dumped_count = 0;
    for (int r = 0; r < 4; r++)
    {
        const int row = rows_to_dump[r];
        bool seen = false;
        for (int k = 0; k < dumped_count; k++)
            if (dumped[k] == row) seen = true;
        if (seen || row < 0 || row >= m.h)
            continue;
        dumped[dumped_count++] = row;

        os << "row[" << row << "]=";
        const float* row_ptr = m.row(row);
        const int cols = std::min(m.w, 24);
        for (int x = 0; x < cols; x++)
        {
            if (x) os << ", ";
            os << row_ptr[x];
        }
        os << "\n";
    }
}

static void dump_blob_summary_3d(const std::string& path, const ncnn::Mat& m)
{
    std::ofstream os(path.c_str(), std::ios::out | std::ios::trunc);
    if (!os.is_open())
        return;

    const int count = m.w * m.h * m.c;
    int finite = 0;
    int nan_count = 0;
    int inf_count = 0;
    double sum = 0.0;
    float minv = FLT_MAX;
    float maxv = -FLT_MAX;
    for (int q = 0; q < m.c; q++)
    {
        const float* ptr = m.channel(q);
        for (int i = 0; i < m.w * m.h; i++)
        {
            const float v = ptr[i];
            if (std::isnan(v))
            {
                nan_count++;
                continue;
            }
            if (!std::isfinite(v))
            {
                inf_count++;
                continue;
            }
            finite++;
            sum += v;
            minv = std::min(minv, v);
            maxv = std::max(maxv, v);
        }
    }

    os << "shape=3d " << m.w << "x" << m.h << "x1x" << m.c << "\n";
    os << "count=" << count << "\n";
    os << "finite=" << finite << "\n";
    os << "nan=" << nan_count << "\n";
    os << "inf=" << inf_count << "\n";
    os << "min=" << (finite > 0 ? minv : NAN) << "\n";
    os << "max=" << (finite > 0 ? maxv : NAN) << "\n";
    os << "mean=" << (finite > 0 ? (sum / finite) : NAN) << "\n";
}

static bool try_extract_and_dump_summary(
    ncnn::Net& net,
    const ncnn::Mat& in_pad,
    const char* blob_name,
    const std::string& out_dir,
    const std::string& stem,
    std::string* error_log = 0)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.set_light_mode(false);
    ex.input("in0", in_pad);

    ncnn::Mat m;
    const int ret = ex.extract(blob_name, m);
    if (ret != 0 || m.empty())
    {
        if (error_log)
        {
            std::ostringstream oss;
            oss << "blob=" << blob_name << "\tret=" << ret << "\tdims=" << m.dims << "\tw=" << m.w << "\th=" << m.h << "\tc=" << m.c;
            if (ret != 0)
            {
                const char* reason = stbi_failure_reason();
                if (reason && reason[0])
                    oss << "\tstbi=" << reason;
            }
            error_log->append(oss.str());
            error_log->append("\n");
        }
        return false;
    }

    std::string path = out_dir + "\\" + stem + "_official_blob_" + blob_name + "_summary.txt";
    if (m.dims == 2)
        dump_blob_summary_2d(path, m);
    else if (m.dims == 3)
        dump_blob_summary_3d(path, m);

    dump_blob_f32(out_dir + "\\" + stem + "_official_blob_" + blob_name + "_f32.bin", m);
    return true;
}

static std::string basename_noext(const std::string& path)
{
    size_t slash = path.find_last_of("/\\");
    size_t start = slash == std::string::npos ? 0 : slash + 1;
    size_t dot = path.find_last_of('.');
    if (dot == std::string::npos || dot < start)
        dot = path.size();
    return path.substr(start, dot - start);
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::fprintf(stderr, "usage: yolov8_seg_compare <image> <model_dir> <out_dir>\n");
        return 2;
    }

    const std::string image_path = argv[1];
    const std::string model_dir = argv[2];
    const std::string out_dir = argv[3];
    const std::string param_path = model_dir + "\\yolov8n_seg.ncnn.param";
    const std::string bin_path = model_dir + "\\yolov8n_seg.ncnn.bin";

    int img_w = 0;
    int img_h = 0;
    int img_c = 0;
    unsigned char* rgb = stbi_load(image_path.c_str(), &img_w, &img_h, &img_c, 3);
    if (!rgb)
    {
        std::fprintf(stderr, "failed to load image: %s\n", image_path.c_str());
        return 3;
    }

    _mkdir(out_dir.c_str());

    ncnn::Net yolov8;
    yolov8.opt.use_vulkan_compute = false;
    yolov8.opt.num_threads = 1;
    if (yolov8.load_param(param_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load param: %s\n", param_path.c_str());
        stbi_image_free(rgb);
        return 4;
    }
    if (yolov8.load_model(bin_path.c_str()) != 0)
    {
        std::fprintf(stderr, "failed to load model: %s\n", bin_path.c_str());
        stbi_image_free(rgb);
        return 5;
    }

    const int target_size = 640;
    const float prob_threshold = 0.25f;
    const float nms_threshold = 0.45f;
    const float mask_threshold = 0.5f;
    const int max_stride = 32;
    std::vector<int> strides = {8, 16, 32};

    int w = img_w;
    int h = img_h;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = (int)(h * scale);
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = (int)(w * scale);
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb, ncnn::Mat::PIXEL_RGB, img_w, img_h, w, h);
    stbi_image_free(rgb);

    int wpad = (w + max_stride - 1) / max_stride * max_stride - w;
    int hpad = (h + max_stride - 1) / max_stride * max_stride - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 114.f);

    const float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in_pad.substract_mean_normalize(0, norm_vals);

    ncnn::Extractor ex = yolov8.create_extractor();
    ex.set_light_mode(false);
    ex.input("in0", in_pad);

    ncnn::Mat out;
    ncnn::Mat mask_feat;
    ncnn::Mat mask_protos;
    ex.extract("out0", out);
    ex.extract("out1", mask_feat);
    ex.extract("out2", mask_protos);

    dump_blob_summary_3d(out_dir + "\\" + basename_noext(image_path) + "_official_in0_summary.txt", in_pad);
    dump_blob_f32(out_dir + "\\" + basename_noext(image_path) + "_official_in0_f32.bin", in_pad);
    dump_blob_summary_2d(out_dir + "\\" + basename_noext(image_path) + "_official_out0_summary.txt", out);
    dump_blob_summary_2d(out_dir + "\\" + basename_noext(image_path) + "_official_out1_summary.txt", mask_feat);
    dump_blob_summary_3d(out_dir + "\\" + basename_noext(image_path) + "_official_out2_summary.txt", mask_protos);
    dump_blob_f32(out_dir + "\\" + basename_noext(image_path) + "_official_out0_f32.bin", out);
    dump_blob_f32(out_dir + "\\" + basename_noext(image_path) + "_official_out1_f32.bin", mask_feat);
    dump_blob_f32(out_dir + "\\" + basename_noext(image_path) + "_official_out2_f32.bin", mask_protos);

    const char* debug_blobs[] = {
        "139", "140", "141",
        "160", "161", "162",
        "180", "181", "182",
        "194", "195", "196",
        "201", "202", "203",
        "208", "209", "210",
        "222", "233", "244",
        "245", "246", "247", "248", "249", "250"
    };
    const std::string stem = basename_noext(image_path);
    std::string debug_extract_log;
    for (size_t i = 0; i < sizeof(debug_blobs) / sizeof(debug_blobs[0]); i++)
        try_extract_and_dump_summary(yolov8, in_pad, debug_blobs[i], out_dir, stem, &debug_extract_log);

    if (!debug_extract_log.empty())
    {
        std::ofstream dbg(out_dir + "\\" + stem + "_official_debug_extract_log.txt", std::ios::out | std::ios::trunc);
        dbg << debug_extract_log;
    }

    std::vector<Object> proposals;
    generate_proposals(out, strides, in_pad, prob_threshold, proposals);
    qsort_descent_inplace(proposals);

    std::vector<int> picked;
    nms_sorted_bboxes(proposals, picked, nms_threshold);

    const int count = (int)picked.size();
    ncnn::Mat objects_mask_feat(mask_feat.w, 1, count);
    std::vector<Object> objects(count);

    for (int i = 0; i < count; i++)
    {
        objects[i] = proposals[picked[i]];
        float x0 = (objects[i].rect.x - (wpad / 2)) / scale;
        float y0 = (objects[i].rect.y - (hpad / 2)) / scale;
        float x1 = (objects[i].rect.x + objects[i].rect.width - (wpad / 2)) / scale;
        float y1 = (objects[i].rect.y + objects[i].rect.height - (hpad / 2)) / scale;

        x0 = std::max(std::min(x0, (float)(img_w - 1)), 0.f);
        y0 = std::max(std::min(y0, (float)(img_h - 1)), 0.f);
        x1 = std::max(std::min(x1, (float)(img_w - 1)), 0.f);
        y1 = std::max(std::min(y1, (float)(img_h - 1)), 0.f);

        objects[i].rect.x = x0;
        objects[i].rect.y = y0;
        objects[i].rect.width = x1 - x0;
        objects[i].rect.height = y1 - y0;

        std::memcpy(objects_mask_feat.channel(i), mask_feat.row(objects[i].gindex), mask_feat.w * sizeof(float));
    }

    ncnn::Mat objects_mask;
    if (count > 0)
    {
        ncnn::Layer* gemm = ncnn::create_layer("Gemm");
        ncnn::ParamDict pd;
        pd.set(6, 1);
        pd.set(7, count);
        pd.set(8, mask_protos.w * mask_protos.h);
        pd.set(9, mask_feat.w);
        pd.set(10, -1);
        pd.set(11, 1);
        gemm->load_param(pd);

        ncnn::Option opt;
        opt.num_threads = 1;
        opt.use_packing_layout = false;
        gemm->create_pipeline(opt);

        std::vector<ncnn::Mat> gemm_inputs(2);
        gemm_inputs[0] = objects_mask_feat;
        gemm_inputs[1] = mask_protos.reshape(mask_protos.w * mask_protos.h, 1, mask_protos.c);
        std::vector<ncnn::Mat> gemm_outputs(1);
        gemm->forward(gemm_inputs, gemm_outputs, opt);
        objects_mask = gemm_outputs[0].reshape(mask_protos.w, mask_protos.h, count);

        gemm->destroy_pipeline(opt);
        delete gemm;

        ncnn::Layer* sigmoid_layer = ncnn::create_layer("Sigmoid");
        sigmoid_layer->create_pipeline(opt);
        sigmoid_layer->forward_inplace(objects_mask, opt);
        sigmoid_layer->destroy_pipeline(opt);
        delete sigmoid_layer;

        ncnn::Mat objects_mask_resized;
        ncnn::resize_bilinear(objects_mask, objects_mask_resized, (int)(in_pad.w / scale), (int)(in_pad.h / scale));
        objects_mask = objects_mask_resized;
    }

    std::vector<unsigned char> union_mask(img_w * img_h, 0);
    for (int i = 0; i < count; i++)
    {
        Object& obj = objects[i];
        const ncnn::Mat mm = objects_mask.channel(i);
        const int rect_w = std::max(0, (int)obj.rect.width);
        const int rect_h = std::max(0, (int)obj.rect.height);
        obj.mask_width = rect_w;
        obj.mask_height = rect_h;
        obj.mask.assign(rect_w * rect_h, 0);

        for (int y = 0; y < rect_h; y++)
        {
            const float* pmm = mm.row((int)(hpad / 2 / scale + obj.rect.y + y)) + (int)(wpad / 2 / scale + obj.rect.x);
            for (int x = 0; x < rect_w; x++)
            {
                const unsigned char keep = pmm[x] > mask_threshold ? 255 : 0;
                obj.mask[y * rect_w + x] = keep;
                if (keep)
                {
                    const int gx = (int)obj.rect.x + x;
                    const int gy = (int)obj.rect.y + y;
                    if (gx >= 0 && gx < img_w && gy >= 0 && gy < img_h)
                        union_mask[gy * img_w + gx] = 255;
                }
            }
        }
    }

    std::vector<unsigned char> overlay;
    {
        int srcw = 0;
        int srch = 0;
        int srcc = 0;
        unsigned char* src = stbi_load(image_path.c_str(), &srcw, &srch, &srcc, 3);
        overlay.resize(srcw * srch * 4, 255);
        if (src)
        {
            for (int i = 0; i < srcw * srch; i++)
            {
                const bool masked = union_mask[i] > 0;
                const float alpha = masked ? 0.45f : 0.f;
                overlay[i * 4 + 0] = (unsigned char)std::round(src[i * 3 + 0] * (1.f - alpha) + 255.f * alpha);
                overlay[i * 4 + 1] = (unsigned char)std::round(src[i * 3 + 1] * (1.f - alpha) + 90.f * alpha);
                overlay[i * 4 + 2] = (unsigned char)std::round(src[i * 3 + 2] * (1.f - alpha) + 90.f * alpha);
                overlay[i * 4 + 3] = 255;
            }
            stbi_image_free(src);
        }
    }

    write_gray_png(out_dir + "\\" + stem + "_official_union_mask.png", img_w, img_h, union_mask);
    write_rgba_png(out_dir + "\\" + stem + "_official_overlay.png", img_w, img_h, overlay);
    for (int i = 0; i < count; i++)
    {
        std::ostringstream oss;
        oss << out_dir << "\\" << stem << "_official_obj_" << i << ".png";
        write_gray_png(oss.str(), objects[i].mask_width, objects[i].mask_height, objects[i].mask);
    }

    std::ofstream summary(out_dir + "\\" + stem + "_official_summary.txt", std::ios::out | std::ios::trunc);
    summary << "source=" << img_w << "x" << img_h << "\n";
    summary << "letterbox_resized=" << w << "x" << h << "\n";
    summary << "letterbox_input=" << in_pad.w << "x" << in_pad.h << "\n";
    summary << "pad=" << (wpad / 2) << "," << (hpad / 2) << "," << (wpad - wpad / 2) << "," << (hpad - hpad / 2) << "\n";
    summary << "scale=" << scale << "\n";
    summary << "detections=" << count << "\n";
    for (int i = 0; i < count; i++)
    {
        const Object& obj = objects[i];
        int mask_pixels = 0;
        for (size_t k = 0; k < obj.mask.size(); k++)
            if (obj.mask[k]) mask_pixels++;
        summary << i
                << "\tlabel=" << obj.label
                << "\tprob=" << obj.prob
                << "\tgrid=" << obj.gindex
                << "\trect=" << obj.rect.x << "," << obj.rect.y << "," << obj.rect.width << "," << obj.rect.height
                << "\tmask_pixels=" << mask_pixels
                << "\n";
    }
    summary.close();

    std::printf("official detections=%d\n", count);
    for (int i = 0; i < count; i++)
    {
        const Object& obj = objects[i];
        std::printf("%d label=%d prob=%.6f rect=%.2f,%.2f,%.2f,%.2f\n", i, obj.label, obj.prob, obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);
    }

    return 0;
}
