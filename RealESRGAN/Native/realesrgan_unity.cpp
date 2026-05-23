#include <algorithm>
#include <string>
#include <vector>

#if defined(_WIN32) && !defined(NOMINMAX)
#define NOMINMAX
#endif

#if defined(_WIN32)
#include <windows.h>
#endif

#include "ncnn/cpu.h"
#include "ncnn/gpu.h"
#include "ncnn/mat.h"
#include "ncnn/net.h"

#if defined(_WIN32)
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif

#if defined(_WIN32)
#define REALESRGAN_EXPORT __declspec(dllexport)
#else
#define REALESRGAN_EXPORT __attribute__((visibility("default")))
#endif

extern "C"
{
    typedef void(*realesrgan_progress_cb)(void* user, float progress01, const char* utf8Message);

    struct realesrgan_ctx
    {
        ncnn::Net net;
        int gpuid = 0;
        bool use_vulkan = true;
        int model_factor = 4;
        int prepadding = 10;
        bool tta_mode = false;
        realesrgan_progress_cb progress = nullptr;
        void* user = nullptr;
        std::string last_error;
    };

    static thread_local std::string g_last_error;
    static const char* set_global_error(const char* msg)
    {
        g_last_error = msg ? msg : "";
        return g_last_error.c_str();
    }

#if defined(_WIN32)
    static int seh_filter_set_error(const char* where, EXCEPTION_POINTERS* ep)
    {
        unsigned int code = ep && ep->ExceptionRecord ? (unsigned int)ep->ExceptionRecord->ExceptionCode : 0;
        void* addr = ep && ep->ExceptionRecord ? ep->ExceptionRecord->ExceptionAddress : nullptr;
        const char* modname = "";
        void* base = nullptr;
        unsigned long long offset = 0;
        if (addr)
        {
            MEMORY_BASIC_INFORMATION mbi;
            if (VirtualQuery(addr, &mbi, sizeof(mbi)) == sizeof(mbi))
            {
                base = mbi.AllocationBase;
                offset = (unsigned long long)((uintptr_t)addr - (uintptr_t)base);
                static char modpath[MAX_PATH] = { 0 };
                modpath[0] = 0;
                if (base)
                {
                    DWORD len = GetModuleFileNameA((HMODULE)base, modpath, MAX_PATH);
                    if (len > 0 && len < MAX_PATH)
                        modname = modpath;
                }
            }
        }
        char buf[512];
        snprintf(buf, sizeof(buf), "native exception in %s: 0x%08X addr=%p base=%p off=0x%llX mod=%s",
            where ? where : "unknown", code, addr, base, offset, modname);
        set_global_error(buf);
        return EXCEPTION_EXECUTE_HANDLER;
    }
#endif
    static int g_gpu_refcount = 0;

#if defined(_WIN32)
    static bool has_vulkan_loader()
    {
        HMODULE h = LoadLibraryA("vulkan-1.dll");
        if (!h) return false;
        FreeLibrary(h);
        return true;
    }

    static bool get_system32_vulkan_path(char* outPath, size_t outPathSize)
    {
        if (!outPath || outPathSize == 0) return false;
        outPath[0] = 0;
        char sysdir[MAX_PATH] = { 0 };
        UINT n = GetSystemDirectoryA(sysdir, MAX_PATH);
        if (n == 0 || n >= MAX_PATH) return false;
        std::string p = std::string(sysdir) + "\\vulkan-1.dll";
        if (p.size() + 1 > outPathSize) return false;
        memcpy(outPath, p.c_str(), p.size() + 1);
        return true;
    }
#endif

    struct vk_alloc_scope
    {
        ncnn::VulkanDevice* dev = nullptr;
        ncnn::VkAllocator* blob = nullptr;
        ncnn::VkAllocator* staging = nullptr;
        bool active = false;
        ~vk_alloc_scope()
        {
            if (!active || !dev) return;
            if (blob) dev->reclaim_blob_allocator(blob);
            if (staging) dev->reclaim_staging_allocator(staging);
        }
    };

    static int infer_model_factor_from_name(const std::string& name)
    {
        int v = 0;
        for (size_t i = 0; i + 1 < name.size(); i++)
        {
            if (name[i] == 'x' && name[i + 1] >= '0' && name[i + 1] <= '9')
            {
                v = name[i + 1] - '0';
                break;
            }
            if (name[i] >= '0' && name[i] <= '9' && name[i + 1] == 'x')
            {
                v = name[i] - '0';
                break;
            }
        }
        if (v < 2) v = 2;
        if (v > 4) v = 4;
        return v;
    }

    static const char* set_error(realesrgan_ctx* ctx, const char* msg)
    {
        if (!ctx) return msg;
        ctx->last_error = msg ? msg : "";
        return ctx->last_error.c_str();
    }

    static void report(realesrgan_ctx* ctx, float p, const char* msg)
    {
        if (!ctx || !ctx->progress) return;
        ctx->progress(ctx->user, p, msg);
    }

    static inline int clampi(int v, int lo, int hi)
    {
        if (v < lo) return lo;
        if (v > hi) return hi;
        return v;
    }

    static void rgba_to_rgb_tile_clamp(
        const unsigned char* rgba,
        int w,
        int h,
        int x0,
        int y0,
        int tw,
        int th,
        std::vector<unsigned char>& rgb_out)
    {
        rgb_out.resize((size_t)tw * (size_t)th * 3);
        for (int ty = 0; ty < th; ty++)
        {
            int sy = clampi(y0 + ty, 0, h - 1);
            for (int tx = 0; tx < tw; tx++)
            {
                int sx = clampi(x0 + tx, 0, w - 1);
                size_t si = ((size_t)sy * (size_t)w + (size_t)sx) * 4;
                size_t di = ((size_t)ty * (size_t)tw + (size_t)tx) * 3;
                rgb_out[di + 0] = rgba[si + 0];
                rgb_out[di + 1] = rgba[si + 1];
                rgb_out[di + 2] = rgba[si + 2];
            }
        }
    }

    static void upscale_alpha_bilinear(
        const unsigned char* rgba,
        int w,
        int h,
        unsigned char* outRgba,
        int outW,
        int outH,
        int scale)
    {
        if (!rgba || !outRgba || w <= 0 || h <= 0 || outW <= 0 || outH <= 0) return;
        if (scale <= 0) scale = 1;

        for (int y = 0; y < outH; y++)
        {
            float sy = (y + 0.5f) / (float)scale - 0.5f;
            int y0 = (int)floorf(sy);
            float fy = sy - y0;
            int y1 = y0 + 1;
            y0 = clampi(y0, 0, h - 1);
            y1 = clampi(y1, 0, h - 1);

            for (int x = 0; x < outW; x++)
            {
                float sx = (x + 0.5f) / (float)scale - 0.5f;
                int x0 = (int)floorf(sx);
                float fx = sx - x0;
                int x1 = x0 + 1;
                x0 = clampi(x0, 0, w - 1);
                x1 = clampi(x1, 0, w - 1);

                float a00 = rgba[((size_t)y0 * (size_t)w + (size_t)x0) * 4 + 3] / 255.f;
                float a10 = rgba[((size_t)y0 * (size_t)w + (size_t)x1) * 4 + 3] / 255.f;
                float a01 = rgba[((size_t)y1 * (size_t)w + (size_t)x0) * 4 + 3] / 255.f;
                float a11 = rgba[((size_t)y1 * (size_t)w + (size_t)x1) * 4 + 3] / 255.f;

                float a0 = a00 + (a10 - a00) * fx;
                float a1 = a01 + (a11 - a01) * fx;
                float a = a0 + (a1 - a0) * fy;
                if (a < 0.f) a = 0.f;
                if (a > 1.f) a = 1.f;

                outRgba[((size_t)y * (size_t)outW + (size_t)x) * 4 + 3] = (unsigned char)(a * 255.f + 0.5f);
            }
        }
    }

    static const char* Realesrgan_Create_Impl(
        const char* modelDir,
        const char* modelName,
        int modelFactor,
        int gpuid,
        int prepadding,
        int ttaMode,
        void* user,
        realesrgan_progress_cb progress,
        void** outCtx)
    {
        if (!outCtx) return "outCtx is null";
        *outCtx = nullptr;
        if (!modelDir || !modelName) return "modelDir/modelName is null";

        const bool request_cpu = (gpuid == -2);

#if defined(_WIN32)
        if (!request_cpu && !has_vulkan_loader())
            return set_global_error("vulkan-1.dll not found. Install GPU driver with Vulkan runtime.");
#endif

        const bool use_vulkan = !request_cpu;
        if (use_vulkan)
        {
            const bool need_create_gpu = (g_gpu_refcount == 0);
            if (need_create_gpu)
            {
#if defined(_WIN32)
                char vkpath[MAX_PATH] = { 0 };
                if (get_system32_vulkan_path(vkpath, MAX_PATH))
                    ncnn::create_gpu_instance(vkpath);
                else
                    ncnn::create_gpu_instance();
#else
                ncnn::create_gpu_instance();
#endif
            }
            if (ncnn::get_gpu_count() <= 0)
            {
                if (need_create_gpu)
                    ncnn::destroy_gpu_instance();
                return set_global_error("ncnn get_gpu_count() == 0. Vulkan device not available.");
            }
            g_gpu_refcount++;
        }

        auto* ctx = new realesrgan_ctx();
        ctx->user = user;
        ctx->progress = progress;
        ctx->use_vulkan = use_vulkan;
        ctx->gpuid = gpuid < 0 ? 0 : gpuid;
        ctx->tta_mode = ttaMode != 0;
        ctx->prepadding = prepadding > 0 ? prepadding : 10;
        ctx->model_factor = modelFactor > 0 ? modelFactor : infer_model_factor_from_name(modelName);

        std::string base(modelName);
        std::string parampath = std::string(modelDir) + "/" + base + ".param";
        std::string modelpath = std::string(modelDir) + "/" + base + ".bin";

#if defined(_WIN32)
        for (auto& ch : parampath) if (ch == '/') ch = '\\';
        for (auto& ch : modelpath) if (ch == '/') ch = '\\';
#endif

        ctx->net.opt.use_vulkan_compute = use_vulkan;
        // Force fp32 path first to avoid pack/layout expectations without the official preproc/postproc pipelines.
        // This sacrifices performance, but should make the basic extractor path stable.
        ctx->net.opt.use_fp16_packed = false;
        ctx->net.opt.use_fp16_storage = false;
        ctx->net.opt.use_fp16_arithmetic = false;
        ctx->net.opt.use_int8_storage = false;
        ctx->net.opt.use_int8_arithmetic = false;
        if (use_vulkan)
            ctx->net.set_vulkan_device(ctx->gpuid);

        report(ctx, 0.01f, "loading model");

        int retp = ctx->net.load_param(parampath.c_str());
        if (retp != 0)
        {
            auto* err = set_error(ctx, "failed to load param");
            delete ctx;
            if (use_vulkan)
            {
                g_gpu_refcount--;
                if (g_gpu_refcount == 0) ncnn::destroy_gpu_instance();
            }
            return err;
        }
        int retm = ctx->net.load_model(modelpath.c_str());
        if (retm != 0)
        {
            auto* err = set_error(ctx, "failed to load model");
            delete ctx;
            if (use_vulkan)
            {
                g_gpu_refcount--;
                if (g_gpu_refcount == 0) ncnn::destroy_gpu_instance();
            }
            return err;
        }

        *outCtx = ctx;
        report(ctx, 0.02f, "model loaded");
        return nullptr;
    }

    REALESRGAN_EXPORT const char* Realesrgan_Create(
        const char* modelDir,
        const char* modelName,
        int modelFactor,
        int gpuid,
        int prepadding,
        int ttaMode,
        void* user,
        realesrgan_progress_cb progress,
        void** outCtx)
    {
#if defined(_WIN32)
        __try
        {
#endif
            return Realesrgan_Create_Impl(modelDir, modelName, modelFactor, gpuid, prepadding, ttaMode, user, progress, outCtx);
#if defined(_WIN32)
        }
        __except (seh_filter_set_error("Realesrgan_Create", GetExceptionInformation()))
        {
            return set_global_error(g_last_error.c_str());
        }
#endif
    }

    REALESRGAN_EXPORT void Realesrgan_Destroy(void* ctxPtr)
    {
        auto* ctx = (realesrgan_ctx*)ctxPtr;
        const bool use_vulkan = ctx ? ctx->use_vulkan : false;
        delete ctx;

        if (!use_vulkan)
            return;

        g_gpu_refcount--;
        if (g_gpu_refcount <= 0)
        {
            g_gpu_refcount = 0;
            ncnn::destroy_gpu_instance();
        }
    }

    static const char* Realesrgan_ProcessRgba_Impl(
        void* ctxPtr,
        const unsigned char* rgba,
        int w,
        int h,
        unsigned char* outRgba,
        int outW,
        int outH,
        int tileSize,
        int scale)
    {
        auto* ctx = (realesrgan_ctx*)ctxPtr;
        if (!ctx) return "ctx is null";
        if (!rgba || !outRgba) return set_error(ctx, "rgba/outRgba is null");
        if (w <= 0 || h <= 0) return set_error(ctx, "invalid input size");

        const int factor = ctx->model_factor;
        if (scale != factor) return set_error(ctx, "scale does not match model factor");
        if (outW != w * factor || outH != h * factor) return set_error(ctx, "output size does not match scale");

        const int pad = ctx->prepadding;
        int tile = tileSize;
        if (tile <= 0) tile = 256;
        if (tile < 32) tile = 32;
        if (tile > 512) tile = 512;

        const int xtiles = (w + tile - 1) / tile;
        const int ytiles = (h + tile - 1) / tile;
        const int totalTiles = std::max(1, xtiles * ytiles);
        int doneTiles = 0;

        report(ctx, 0.05f, "preprocess");

        const bool use_vulkan = ctx->use_vulkan && ctx->net.opt.use_vulkan_compute;

        const ncnn::VulkanDevice* vkdev_const = nullptr;
        ncnn::VulkanDevice* vkdev = nullptr;
        vk_alloc_scope alloc_scope;
        ncnn::Option opt = ctx->net.opt;
        if (use_vulkan)
        {
            vkdev_const = ctx->net.vulkan_device();
            if (!vkdev_const)
                return set_error(ctx, "vulkan device is null");
            vkdev = const_cast<ncnn::VulkanDevice*>(vkdev_const);

            alloc_scope.dev = vkdev;
            alloc_scope.blob = vkdev->acquire_blob_allocator();
            alloc_scope.staging = vkdev->acquire_staging_allocator();
            alloc_scope.active = true;

            opt.blob_vkallocator = alloc_scope.blob;
            opt.workspace_vkallocator = alloc_scope.blob;
            opt.staging_vkallocator = alloc_scope.staging;
        }

        // Fill alpha first (will overwrite later if no alpha)
        for (int i = 0; i < outW * outH; i++)
            outRgba[i * 4 + 3] = 255;

        std::vector<unsigned char> rgb_tile;

        for (int yi = 0; yi < ytiles; yi++)
        {
            for (int xi = 0; xi < xtiles; xi++)
            {
                const int x0 = xi * tile;
                const int y0 = yi * tile;
                const int x1 = std::min(x0 + tile, w);
                const int y1 = std::min(y0 + tile, h);
                const int tw = x1 - x0;
                const int th = y1 - y0;

                const int in_x0 = x0 - pad;
                const int in_y0 = y0 - pad;
                const int in_tw = tw + pad * 2;
                const int in_th = th + pad * 2;

                rgba_to_rgb_tile_clamp(rgba, w, h, in_x0, in_y0, in_tw, in_th, rgb_tile);
                ncnn::Mat in = ncnn::Mat::from_pixels(rgb_tile.data(), ncnn::Mat::PIXEL_RGB, in_tw, in_th);

                const float norm[3] = { 1.f / 255.f, 1.f / 255.f, 1.f / 255.f };
                in.substract_mean_normalize(nullptr, norm);

                ncnn::Extractor ex = ctx->net.create_extractor();
                ex.set_light_mode(true);
                if (use_vulkan)
                {
                    ex.set_blob_vkallocator(alloc_scope.blob);
                    ex.set_workspace_vkallocator(alloc_scope.blob);
                    ex.set_staging_vkallocator(alloc_scope.staging);
                }

                int ri = 0;
                if (use_vulkan)
                {
                    ncnn::VkCompute cmd(vkdev);
                    ncnn::VkMat in_gpu;
                    cmd.record_clone(in, in_gpu, opt);
                    cmd.submit_and_wait();
                    cmd.reset();
                    ri = ex.input("data", in_gpu);
                }
                else
                {
                    ri = ex.input("data", in);
                }
                if (ri != 0)
                {
                    char buf[256];
                    snprintf(buf, sizeof(buf), "input failed (ri=%d) tile=(%d,%d) in=%dx%d pad=%d", ri, xi, yi, in_tw, in_th, pad);
                    return set_error(ctx, buf);
                }

                ncnn::Mat out;
                if (use_vulkan)
                {
                    ncnn::VkCompute cmd(vkdev);
                    ncnn::VkMat out_gpu;
                    int ro = ex.extract("output", out_gpu, cmd);
                    if (ro != 0)
                    {
                        char buf[320];
                        snprintf(buf, sizeof(buf), "extract output failed (ro=%d) tile=(%d,%d) in=%dx%d pad=%d scale=%d", ro, xi, yi, in_tw, in_th, pad, factor);
                        return set_error(ctx, buf);
                    }
                    cmd.submit_and_wait();
                    cmd.reset();

                    cmd.record_clone(out_gpu, out, opt);
                    cmd.submit_and_wait();
                    cmd.reset();
                }
                else
                {
                    int ro = ex.extract("output", out);
                    if (ro != 0)
                    {
                        char buf[320];
                        snprintf(buf, sizeof(buf), "extract output failed (ro=%d) tile=(%d,%d) in=%dx%d pad=%d scale=%d", ro, xi, yi, in_tw, in_th, pad, factor);
                        return set_error(ctx, buf);
                    }
                }

                const int expectedW = in_tw * factor;
                const int expectedH = in_th * factor;
                if (out.w != expectedW || out.h != expectedH || out.c != 3)
                    return set_error(ctx, "unexpected output shape");

                // crop center (remove padding), then paste into outRgba
                const int crop_x0 = pad * factor;
                const int crop_y0 = pad * factor;
                const int crop_w = tw * factor;
                const int crop_h = th * factor;

                for (int oy = 0; oy < crop_h; oy++)
                {
                    int sy = crop_y0 + oy;
                    int dy = y0 * factor + oy;
                    const float* pr = out.channel(0).row(sy);
                    const float* pg = out.channel(1).row(sy);
                    const float* pb = out.channel(2).row(sy);

                    for (int ox = 0; ox < crop_w; ox++)
                    {
                        int sx = crop_x0 + ox;
                        int dx = x0 * factor + ox;
                        float r = pr[sx];
                        float g = pg[sx];
                        float b = pb[sx];
                        if (r < 0.f) r = 0.f; if (r > 1.f) r = 1.f;
                        if (g < 0.f) g = 0.f; if (g > 1.f) g = 1.f;
                        if (b < 0.f) b = 0.f; if (b > 1.f) b = 1.f;
                        size_t di = ((size_t)dy * (size_t)outW + (size_t)dx) * 4;
                        outRgba[di + 0] = (unsigned char)(r * 255.f + 0.5f);
                        outRgba[di + 1] = (unsigned char)(g * 255.f + 0.5f);
                        outRgba[di + 2] = (unsigned char)(b * 255.f + 0.5f);
                    }
                }

                doneTiles++;
                float p = 0.10f + 0.80f * (doneTiles / (float)totalTiles);
                report(ctx, p, "inference");
            }
        }

        // Upscale alpha if there is any transparency (fast scan)
        bool has_alpha = false;
        for (int i = 0; i < w * h; i++)
        {
            if (rgba[i * 4 + 3] != 255)
            {
                has_alpha = true;
                break;
            }
        }
        if (has_alpha)
        {
            report(ctx, 0.93f, "alpha");
            upscale_alpha_bilinear(rgba, w, h, outRgba, outW, outH, factor);
        }

        report(ctx, 0.98f, "done");
        return nullptr;
    }

    REALESRGAN_EXPORT const char* Realesrgan_ProcessRgba(
        void* ctxPtr,
        const unsigned char* rgba,
        int w,
        int h,
        unsigned char* outRgba,
        int outW,
        int outH,
        int tileSize,
        int scale)
    {
#if defined(_WIN32)
        __try
        {
#endif
            return Realesrgan_ProcessRgba_Impl(ctxPtr, rgba, w, h, outRgba, outW, outH, tileSize, scale);
#if defined(_WIN32)
        }
        __except (seh_filter_set_error("Realesrgan_ProcessRgba", GetExceptionInformation()))
        {
            return set_global_error(g_last_error.c_str());
        }
#endif
    }
}
