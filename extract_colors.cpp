#include <getopt.h>
#include <csignal>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <deque>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <chrono>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <sys/stat.h>
#include <map>
#include <cstdlib>
#include <climits>
#include <optional>
#include <memory>
#include <cstdint>
#include <new>

extern "C" {
#include <libavformat/avformat.h>
#include <libavcodec/avcodec.h>
#include <libavutil/avutil.h>
#include <libavutil/pixdesc.h>
#include <libavutil/imgutils.h>
#include <libavutil/dict.h>
#include <libswscale/swscale.h>
}

#include <nlohmann/json.hpp>
using json = nlohmann::ordered_json;

#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#endif

namespace constants {
    constexpr int PROGRESS_BAR_WIDTH = 20;
    constexpr int SEEK_THRESHOLD = 10;
}

struct AVFormatContextDeleter {
    void operator()(AVFormatContext* ctx) const { if (ctx) avformat_close_input(&ctx); }
};
using AVFormatContextPtr = std::unique_ptr<AVFormatContext, AVFormatContextDeleter>;

struct AVCodecContextDeleter {
    void operator()(AVCodecContext* ctx) const { if (ctx) avcodec_free_context(&ctx); }
};
using AVCodecContextPtr = std::unique_ptr<AVCodecContext, AVCodecContextDeleter>;

struct AVFrameDeleter {
    void operator()(AVFrame* frame) const { if (frame) av_frame_free(&frame); }
};
using AVFramePtr = std::unique_ptr<AVFrame, AVFrameDeleter>;

struct AVPacketDeleter {
    void operator()(AVPacket* pkt) const { if (pkt) av_packet_free(&pkt); }
};
using AVPacketPtr = std::unique_ptr<AVPacket, AVPacketDeleter>;

struct SwsContextDeleter {
    void operator()(SwsContext* ctx) const { if (ctx) sws_freeContext(ctx); }
};
using SwsContextPtr = std::unique_ptr<SwsContext, SwsContextDeleter>;

enum class ScaleAlgo { BILINEAR, BICUBIC, LANCZOS, POINT, FAST_BILINEAR };

struct Options {
    std::string input_file;
    std::string output_file;
    bool force_overwrite = false;
    int target_bitdepth = 0;
    std::optional<std::vector<int>> crop;
    std::string crop_arg;
    std::optional<std::vector<std::pair<int,int>>> trim_segments;
    int step = 1;
    float scale_factor = 1.0f;
    bool enable_rgb1d = true;
    bool enable_rgb3d = true;
    bool enable_rgbmoments = true;
    int rgb1d_bins = 0;
    int rgb3d_bins = 0;
    int batch_frames = 1;
    int prefetch = 2;
    int workers = 0;
    bool float_binning = false;
    ScaleAlgo scale_algo = ScaleAlgo::BILINEAR;
    bool error = false;
    std::string error_msg;

    void print_help(const char* progname) const;
    void print_version() const;
};

struct VideoInfo {
    std::string file_name;
    int width = 0, height = 0;
    int crop_left = 0, crop_top = 0, crop_width = 0, crop_height = 0;
    std::string crop_input;
    int scaled_width = 0, scaled_height = 0;
    int bitdepth = 8;
    int64_t total_frames = 0;
    AVRational framerate = {0,1};
    AVPixelFormat pix_fmt = AV_PIX_FMT_NONE;
    std::string pix_fmt_name;
    std::string color_primaries, color_trc, color_space;
    int color_range = 0, chroma_location = 0, field_order = 0;
    std::string codec_name, codec_long_name;
    int profile = 0, level = 0;
    int64_t bitrate = 0;
    std::string container_format, container_format_long;
    int64_t file_size = 0;
    double duration_sec = 0.0;
    int64_t start_time = AV_NOPTS_VALUE;
    int64_t overall_bitrate = 0;
    std::map<std::string, std::string> metadata;
};

struct WorkerParams {
    int bitdepth;
    int max_val;
    bool enable_rgb1d;
    bool enable_rgb3d;
    bool enable_rgbmoments;
    size_t rgb1d_bins;
    int rgb3d_bins;
    int width, height;
    bool float_binning;
};

struct PartialResults {
    std::vector<uint64_t> rgb1d;
    std::vector<uint64_t> rgb3d;
    double sum[3] = {0,0,0};
    double sum2[3] = {0,0,0};
    double sum3[3] = {0,0,0};
    size_t pixel_count = 0;

    void merge(const PartialResults& other, const WorkerParams& p);
};

template<typename T, size_t Alignment = 32>
class AlignedAllocator {
public:
    using value_type = T;
    using pointer = T*;
    using const_pointer = const T*;
    using reference = T&;
    using const_reference = const T&;
    using size_type = size_t;
    using difference_type = ptrdiff_t;

    template<typename U>
    struct rebind {
        using other = AlignedAllocator<U, Alignment>;
    };

    AlignedAllocator() = default;
    template<typename U>
    AlignedAllocator(const AlignedAllocator<U, Alignment>&) {}

    T* allocate(size_t n) {
        if (n == 0) return nullptr;
        if (n > size_t(-1) / sizeof(T)) throw std::bad_alloc();
        void* ptr = ::operator new(n * sizeof(T), std::align_val_t(Alignment));
        if (!ptr) throw std::bad_alloc();
        return static_cast<T*>(ptr);
    }

    void deallocate(T* p, size_t) noexcept {
        ::operator delete(p, std::align_val_t(Alignment));
    }

    template<typename U, size_t A>
    bool operator==(const AlignedAllocator<U, A>&) const { return true; }
    template<typename U, size_t A>
    bool operator!=(const AlignedAllocator<U, A>&) const { return false; }
};

using AlignedBuffer = std::vector<uint8_t, AlignedAllocator<uint8_t>>;

struct Batch {
    AlignedBuffer data;
    size_t num_frames;

    Batch(AlignedBuffer&& d, size_t n) : data(std::move(d)), num_frames(n) {}
};

template<typename T>
class ConcurrentQueue {
    std::deque<T> queue_;
    mutable std::mutex mutex_;
    std::condition_variable not_empty_;
    std::condition_variable not_full_;
    size_t capacity_;
    bool stopped_ = false;
public:
    explicit ConcurrentQueue(size_t capacity) : capacity_(capacity) {}

    void push(T&& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_full_.wait(lock, [this] { return queue_.size() < capacity_ || stopped_; });
        if (stopped_) return;
        queue_.push_back(std::move(item));
        not_empty_.notify_one();
    }

    bool pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        not_empty_.wait(lock, [this] { return !queue_.empty() || stopped_; });
        if (queue_.empty() && stopped_) return false;
        item = std::move(queue_.front());
        queue_.pop_front();
        not_full_.notify_one();
        return true;
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        not_empty_.notify_all();
        not_full_.notify_all();
    }

    size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

class BufferPool {
    std::deque<AlignedBuffer> pool_;
    mutable std::mutex mutex_;
    std::condition_variable cv_;
    bool stopped_ = false;
    size_t buffer_size_;
public:
    explicit BufferPool(size_t buffer_size, size_t initial_count = 0)
        : buffer_size_(buffer_size) {
        for (size_t i = 0; i < initial_count; ++i) {
            AlignedBuffer buf;
            buf.reserve(buffer_size_);
            pool_.push_back(std::move(buf));
        }
    }

    AlignedBuffer borrow() {
        std::unique_lock<std::mutex> lock(mutex_);
        cv_.wait(lock, [this] { return !pool_.empty() || stopped_; });
        if (stopped_ && pool_.empty()) return {};
        AlignedBuffer buf = std::move(pool_.front());
        pool_.pop_front();
        if (buf.capacity() < buffer_size_) {
            buf.reserve(buffer_size_);
        }
        return buf;
    }

    void give_back(AlignedBuffer buf) {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            buf.clear();
            pool_.push_back(std::move(buf));
        }
        cv_.notify_one();
    }

    void stop() {
        {
            std::lock_guard<std::mutex> lock(mutex_);
            stopped_ = true;
        }
        cv_.notify_all();
    }
};

namespace util {
    std::string format_file_size(uint64_t bytes) {
        const char* units[] = {"B", "KiB", "MiB", "GiB", "TiB"};
        int unit = 0;
        double size = static_cast<double>(bytes);
        while (size >= 1024.0 && unit < 4) { size /= 1024.0; ++unit; }
        std::ostringstream oss;
        oss << std::fixed << std::setprecision(2) << size << " " << units[unit];
        return oss.str();
    }

    std::map<std::string, std::string> dict_to_map(AVDictionary* dict) {
        std::map<std::string, std::string> result;
        AVDictionaryEntry* entry = nullptr;
        while ((entry = av_dict_get(dict, "", entry, AV_DICT_IGNORE_SUFFIX)))
            result[entry->key] = entry->value;
        return result;
    }

    const char* color_range_name(int range) {
        switch (range) {
            case AVCOL_RANGE_UNSPECIFIED: return "unspecified";
            case AVCOL_RANGE_MPEG: return "limited (MPEG)";
            case AVCOL_RANGE_JPEG: return "full (JPEG)";
            default: return "unknown";
        }
    }

    const char* chroma_location_name(int loc) {
        switch (loc) {
            case AVCHROMA_LOC_UNSPECIFIED: return "unspecified";
            case AVCHROMA_LOC_LEFT: return "left";
            case AVCHROMA_LOC_CENTER: return "center";
            case AVCHROMA_LOC_TOPLEFT: return "topleft";
            case AVCHROMA_LOC_TOP: return "top";
            case AVCHROMA_LOC_BOTTOMLEFT: return "bottomleft";
            case AVCHROMA_LOC_BOTTOM: return "bottom";
            default: return "unknown";
        }
    }

    const char* field_order_name(int order) {
        switch (order) {
            case AV_FIELD_UNKNOWN: return "unknown";
            case AV_FIELD_PROGRESSIVE: return "progressive";
            case AV_FIELD_TT: return "interlaced (top first)";
            case AV_FIELD_BB: return "interlaced (bottom first)";
            case AV_FIELD_TB: return "interlaced (top/bottom)";
            case AV_FIELD_BT: return "interlaced (bottom/top)";
            default: return "unknown";
        }
    }

    const char* scale_algo_name(ScaleAlgo a) {
        switch (a) {
            case ScaleAlgo::BILINEAR: return "bilinear";
            case ScaleAlgo::BICUBIC: return "bicubic";
            case ScaleAlgo::LANCZOS: return "lanczos";
            case ScaleAlgo::POINT: return "point";
            case ScaleAlgo::FAST_BILINEAR: return "fast-bilinear";
            default: return "unknown";
        }
    }
}

namespace process {
    template<bool rgb1d, bool rgb3d, bool moments, bool float_bin, typename T>
    static PartialResults rgb_batch_impl_tmpl(const Batch& batch, const WorkerParams& p) {
        PartialResults res;
        size_t total_pixels = batch.num_frames * static_cast<size_t>(p.width) * p.height;
        res.pixel_count = total_pixels;

        const int shift = (sizeof(T) == 2) ? (16 - p.bitdepth) : 0;
        const int max_val = p.max_val;
        const size_t rgb1d_bins = p.rgb1d_bins;
        const int rgb3d_bins = p.rgb3d_bins;

        const double rgb1d_scale = (rgb1d_bins - 1) / static_cast<double>(max_val);
        const double rgb3d_scale = (rgb3d_bins - 1) / static_cast<double>(max_val);
        const int64_t rgb3d_factor = ((static_cast<int64_t>(rgb3d_bins - 1) << 16) + max_val/2) / max_val;

        if constexpr (rgb1d) res.rgb1d.assign(3 * rgb1d_bins, 0);
        if constexpr (rgb3d) res.rgb3d.assign(static_cast<size_t>(rgb3d_bins) * rgb3d_bins * rgb3d_bins, 0);

        uint64_t* __restrict rgb1d_ptr = rgb1d ? res.rgb1d.data() : nullptr;
        uint64_t* __restrict rgb3d_ptr = rgb3d ? res.rgb3d.data() : nullptr;

        const size_t offset_g = rgb1d_bins;
        const size_t offset_b = 2 * rgb1d_bins;

        double sum_r = 0.0, sum_g = 0.0, sum_b = 0.0;
        double sum2_r = 0.0, sum2_g = 0.0, sum2_b = 0.0;
        double sum3_r = 0.0, sum3_g = 0.0, sum3_b = 0.0;

        const T* __restrict data = reinterpret_cast<const T*>(batch.data.data());

        for (size_t i = 0; i < total_pixels; ++i) {
            uint16_t r = static_cast<uint16_t>(data[0] >> shift);
            uint16_t g = static_cast<uint16_t>(data[1] >> shift);
            uint16_t b = static_cast<uint16_t>(data[2] >> shift);
            data += 3;

            if constexpr (rgb1d) {
                if constexpr (float_bin) {
                    int ri = static_cast<int>(r * rgb1d_scale + 0.5f);
                    int gi = static_cast<int>(g * rgb1d_scale + 0.5f);
                    int bi = static_cast<int>(b * rgb1d_scale + 0.5f);
                    ri = (ri < 0) ? 0 : (ri >= static_cast<int>(rgb1d_bins) ? static_cast<int>(rgb1d_bins)-1 : ri);
                    gi = (gi < 0) ? 0 : (gi >= static_cast<int>(rgb1d_bins) ? static_cast<int>(rgb1d_bins)-1 : gi);
                    bi = (bi < 0) ? 0 : (bi >= static_cast<int>(rgb1d_bins) ? static_cast<int>(rgb1d_bins)-1 : bi);
                    rgb1d_ptr[ri]++;
                    rgb1d_ptr[offset_g + gi]++;
                    rgb1d_ptr[offset_b + bi]++;
                } else {
                    rgb1d_ptr[r]++;
                    rgb1d_ptr[offset_g + g]++;
                    rgb1d_ptr[offset_b + b]++;
                }
            }

            if constexpr (rgb3d) {
                if constexpr (float_bin) {
                    int rq = static_cast<int>(r * rgb3d_scale + 0.5f);
                    int gq = static_cast<int>(g * rgb3d_scale + 0.5f);
                    int bq = static_cast<int>(b * rgb3d_scale + 0.5f);
                    rq = (rq < 0) ? 0 : (rq >= rgb3d_bins ? rgb3d_bins-1 : rq);
                    gq = (gq < 0) ? 0 : (gq >= rgb3d_bins ? rgb3d_bins-1 : gq);
                    bq = (bq < 0) ? 0 : (bq >= rgb3d_bins ? rgb3d_bins-1 : bq);
                    size_t idx = (static_cast<size_t>(rq) * rgb3d_bins + gq) * rgb3d_bins + bq;
                    rgb3d_ptr[idx]++;
                } else {
                    uint16_t rq = static_cast<uint16_t>((r * rgb3d_factor) >> 16);
                    uint16_t gq = static_cast<uint16_t>((g * rgb3d_factor) >> 16);
                    uint16_t bq = static_cast<uint16_t>((b * rgb3d_factor) >> 16);
                    size_t idx = (static_cast<size_t>(rq) * rgb3d_bins + gq) * rgb3d_bins + bq;
                    rgb3d_ptr[idx]++;
                }
            }

            if constexpr (moments) {
                double rf = r, gf = g, bf = b;
                sum_r += rf;   sum_g += gf;   sum_b += bf;
                sum2_r += rf*rf; sum2_g += gf*gf; sum2_b += bf*bf;
                sum3_r += rf*rf*rf; sum3_g += gf*gf*gf; sum3_b += bf*bf*bf;
            }
        }

        if constexpr (moments) {
            res.sum[0] = sum_r;   res.sum[1] = sum_g;   res.sum[2] = sum_b;
            res.sum2[0] = sum2_r; res.sum2[1] = sum2_g; res.sum2[2] = sum2_b;
            res.sum3[0] = sum3_r; res.sum3[1] = sum3_g; res.sum3[2] = sum3_b;
        }

        return res;
    }

    static PartialResults process_batch_rgb(const Batch& batch, const WorkerParams& p) {
        if (p.enable_rgb1d && p.enable_rgb3d && p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, true, true, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, true, true, true, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && p.enable_rgb3d && p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, true, true, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, true, true, false, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && p.enable_rgb3d && !p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, true, false, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, true, false, true, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && p.enable_rgb3d && !p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, true, false, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, true, false, false, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && !p.enable_rgb3d && p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, false, true, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, false, true, true, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && !p.enable_rgb3d && p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, false, true, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, false, true, false, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && !p.enable_rgb3d && !p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, false, false, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, false, false, true, uint16_t>(batch, p);
        }
        if (p.enable_rgb1d && !p.enable_rgb3d && !p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<true, false, false, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<true, false, false, false, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && p.enable_rgb3d && p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, true, true, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, true, true, true, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && p.enable_rgb3d && p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, true, true, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, true, true, false, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && p.enable_rgb3d && !p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, true, false, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, true, false, true, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && p.enable_rgb3d && !p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, true, false, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, true, false, false, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && !p.enable_rgb3d && p.enable_rgbmoments && p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, false, true, true, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, false, true, true, uint16_t>(batch, p);
        }
        if (!p.enable_rgb1d && !p.enable_rgb3d && p.enable_rgbmoments && !p.float_binning) {
            if (p.bitdepth <= 8)
                return rgb_batch_impl_tmpl<false, false, true, false, uint8_t>(batch, p);
            else
                return rgb_batch_impl_tmpl<false, false, true, false, uint16_t>(batch, p);
        }
        return PartialResults{};
    }
}

void PartialResults::merge(const PartialResults& other, const WorkerParams& p) {
    if (p.enable_rgb1d) {
        if (rgb1d.empty()) rgb1d.assign(3 * p.rgb1d_bins, 0);
        for (size_t i = 0; i < rgb1d.size(); ++i) rgb1d[i] += other.rgb1d[i];
    }
    if (p.enable_rgb3d) {
        if (rgb3d.empty()) rgb3d.assign(static_cast<size_t>(p.rgb3d_bins) * p.rgb3d_bins * p.rgb3d_bins, 0);
        for (size_t i = 0; i < rgb3d.size(); ++i) rgb3d[i] += other.rgb3d[i];
    }
    if (p.enable_rgbmoments) {
        for (int c = 0; c < 3; ++c) {
            sum[c] += other.sum[c];
            sum2[c] += other.sum2[c];
            sum3[c] += other.sum3[c];
        }
        pixel_count += other.pixel_count;
    }
}

std::atomic<bool> global_stop_flag(false);
void signal_handler(int) { global_stop_flag = true; }

static void print_progress_bar(int64_t processed, int64_t total, double elapsed,
                               double fps,
                               double recent_input_ms,
                               double recent_decode_ms,
                               double recent_process_ms,
                               int active_workers, int total_workers) {
    double progress = (total > 0) ? static_cast<double>(processed) / total : 0.0;
    int pos = static_cast<int>(constants::PROGRESS_BAR_WIDTH * progress);

    std::cout << "\r[";
    for (int i = 0; i < constants::PROGRESS_BAR_WIDTH; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }
    std::cout << "] ";

    if (total > 0) {
        std::cout << std::fixed << std::setprecision(2) << (progress * 100.0) << "% ";
        std::cout << "(" << processed << "/" << total << ") ";
    } else {
        std::cout << processed << " frames ";
    }

    if (total > 0 && processed > 0 && elapsed > 0.0) {
        //double avg_fps = processed / elapsed;
        int64_t remaining = total - processed;
        //double eta_sec = remaining / avg_fps;
        double eta_sec = remaining / fps;
        int eta_h = static_cast<int>(eta_sec) / 3600;
        int eta_m = (static_cast<int>(eta_sec) % 3600) / 60;
        int eta_s = static_cast<int>(eta_sec) % 60;

        std::cout << "In: " << recent_input_ms << "ms"
                  << " | Dec: " << recent_decode_ms << "ms"
                  << " | Proc: " << recent_process_ms << "ms"
                  << " | Workers: " << active_workers << "/" << total_workers
                  << " | FPS: " << fps
                  << " | ETA: " << std::right << std::setfill('0')
                  << std::setw(2) << eta_h << ":"
                  << std::setw(2) << eta_m << ":"
                  << std::setw(2) << eta_s;
    }

    std::cout << "        ";
    std::cout.flush();
}

static void print_video_information(const VideoInfo& info, const Options& opts,
                                    int target_bitdepth,
                                    int rgb1d_bins, int rgb3d_bins,
                                    int num_threads) {

    const int label_width = 23;
    auto print_pair = [&](const std::string& label, const std::string& value) {
        std::cout << std::left << std::setw(label_width) << label << value << "\n";
    };
    auto print_sub = [&](const std::string& sub_label, const std::string& value) {
        std::string full_label = "- " + sub_label;
        std::cout << std::left << std::setw(label_width) << full_label << value << "\n";
    };

    std::cout << "\n---- Input ----\n";
    std::cout << "\n";
    print_pair("File", info.file_name);
    print_pair("Resolution", std::to_string(info.width) + "x" + std::to_string(info.height));
    if (info.bitrate > 0)
        print_pair("Bitrate", std::to_string(info.bitrate / 1000) + " kb/s");
    if (info.total_frames > 0)
        print_pair("Total frames", std::to_string(info.total_frames));
    print_pair("Frame rate", std::to_string(av_q2d(info.framerate)) + " fps");
    print_pair("Bit depth", std::to_string(info.bitdepth));
    print_pair("Codec", info.codec_long_name + " (" + info.codec_name + ")");
    print_pair("Field order", util::field_order_name(info.field_order));
    print_pair("Pixel format", info.pix_fmt_name);
    print_pair("Color space", info.color_space);
    print_pair("Color primaries", info.color_primaries);
    print_pair("Color transfer", info.color_trc);
    print_pair("Color range", util::color_range_name(info.color_range));
    print_pair("Chroma location", util::chroma_location_name(info.chroma_location));
    if (info.profile != AV_PROFILE_UNKNOWN)
        print_pair("Profile", std::to_string(info.profile));
    if (info.level > 0)
        print_pair("Level", std::to_string(info.level));

    std::cout << "\n---- Parameters ----\n";

    if (!info.crop_input.empty()) {
        std::cout << "\n";
        print_pair("Crop input", info.crop_input);
        print_pair("Crop", "");
        print_sub("Left",   std::to_string(info.crop_left));
        print_sub("Top",    std::to_string(info.crop_top));
        print_sub("Width",  std::to_string(info.crop_width));
        print_sub("Height", std::to_string(info.crop_height));
    }

    if (opts.trim_segments) {
        std::cout << "\n";
        print_pair("Trim", "");
        for (size_t i = 0; i < opts.trim_segments->size(); ++i) {
            const auto& seg = (*opts.trim_segments)[i];
            print_sub("Segment " + std::to_string(i+1),
                      std::to_string(seg.first) + "," + std::to_string(seg.second));
        }
    }

    if (opts.scale_factor != 1.0f) {
        std::cout << "\n";
        print_pair("Scaling algorithm", util::scale_algo_name(opts.scale_algo));
        print_pair("Scale", std::to_string(opts.scale_factor));
        print_pair("Scaled resolution", std::to_string(info.scaled_width) + "x" + std::to_string(info.scaled_height));
    }

    std::cout << "\nExtraction\n";
    if (opts.enable_rgb1d)      { print_sub("RGB 1D", std::to_string(rgb1d_bins) + " bins"); }
    if (opts.enable_rgb3d)      { print_sub("RGB 3D", std::to_string(rgb3d_bins) + " bins"); }
    if (opts.enable_rgbmoments) { print_sub("RGB Moments", ""); }

    if (opts.float_binning) {
        std::cout << "\nAccuracy\n";
        std::cout << "- Floating-point binning for histograms\n";
    }

    std::cout << "\nPerformance\n";
    print_sub("Step",     std::to_string(opts.step));
    print_sub("Batch",    std::to_string(opts.batch_frames));
    print_sub("Workers",  std::to_string(num_threads));
    int queue_capacity = opts.prefetch * num_threads;
    print_sub("Prefetch", std::to_string(num_threads) + " * " + std::to_string(opts.prefetch) + " = " + std::to_string(queue_capacity));

    std::cout << "\n---- Output ----\n";
    std::cout << "\n";
    print_pair("File", opts.output_file);

    std::cout << std::endl;
}

static void producer_thread_function(AVFormatContext* fmt_ctx_raw, AVCodecContext* dec_ctx_raw,
                                      SwsContext* sws_ctx_raw, int video_stream_idx,
                                      const Options& opts, int target_bitdepth,
                                      int crop_left, int crop_top, int crop_width, int crop_height,
                                      int out_width, int out_height, AVPixelFormat out_pix_fmt,
                                      ConcurrentQueue<Batch>& queue, BufferPool& buffer_pool,
                                      std::atomic<bool>& stop_flag,
                                      int64_t& total_frames_read,
                                      std::atomic<uint64_t>& total_decode_time_us,
                                      std::atomic<uint64_t>& total_input_wait_us) {

    AVPacketPtr packet(av_packet_alloc());
    AVFramePtr frame(av_frame_alloc());
    if (!packet || !frame) {
        std::cerr << "Failed to allocate packet or frame\n";
        stop_flag = true;
        return;
    }

    AVStream* stream = fmt_ctx_raw->streams[video_stream_idx];
    AVRational frame_rate = av_guess_frame_rate(fmt_ctx_raw, stream, nullptr);
    if (frame_rate.num == 0 || frame_rate.den == 0) {
        frame_rate = {25, 1};
    }
    double fps = av_q2d(frame_rate);
    double time_base_sec = av_q2d(stream->time_base);

    auto pts_to_frame = [&](int64_t pts) -> int64_t {
        if (pts == AV_NOPTS_VALUE) return -1;
        int64_t frame_index = av_rescale_q(pts, stream->time_base, av_inv_q(frame_rate));
        return frame_index + 1;
    };

    auto seek_to_frame = [&](int64_t abs_frame) {
        double target_sec = (abs_frame - 1) / fps;
        int64_t target_pts = static_cast<int64_t>(target_sec / time_base_sec + 0.5);
        auto seek_start = std::chrono::high_resolution_clock::now();
        av_seek_frame(fmt_ctx_raw, video_stream_idx, target_pts, AVSEEK_FLAG_BACKWARD);
        avcodec_flush_buffers(dec_ctx_raw);
        auto seek_end = std::chrono::high_resolution_clock::now();
        total_input_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(seek_end - seek_start).count();
    };

    std::vector<std::pair<int64_t, int64_t>> segments;
    if (opts.trim_segments) {
        for (const auto& p : *opts.trim_segments) {
            segments.emplace_back(p.first, p.second);
        }
    } else {
        segments.emplace_back(1, INT64_MAX);
    }

    size_t current_seg = 0;
    int64_t target_abs_frame = -1;

    if (segments.empty()) {
        return;
    }
    current_seg = 0;
    target_abs_frame = segments[0].first;
    seek_to_frame(target_abs_frame);

    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(out_pix_fmt);
    if (!desc) {
        std::cerr << "Invalid target pixel format\n";
        stop_flag = true;
        return;
    }
    int bytes_per_pixel = (target_bitdepth <= 8) ? 1 : 2;
    size_t frame_bytes = 3 * static_cast<size_t>(out_width) * out_height * bytes_per_pixel;

    int frames_in_batch = 0;
    AlignedBuffer current_batch = buffer_pool.borrow();
    current_batch.clear();
    current_batch.reserve(static_cast<size_t>(opts.batch_frames) * frame_bytes);

    bool need_crop = (crop_left != 0 || crop_top != 0 ||
                      crop_width != dec_ctx_raw->width ||
                      crop_height != dec_ctx_raw->height);
    bool need_scale = (out_width != crop_width || out_height != crop_height);
    bool can_copy_direct = !need_crop && !need_scale && dec_ctx_raw->pix_fmt == out_pix_fmt;

    int64_t fallback_counter = 0;

    bool all_done = false;

    while (!stop_flag.load() && !global_stop_flag.load() && !all_done) {
        auto read_start = std::chrono::high_resolution_clock::now();
        if (av_read_frame(fmt_ctx_raw, packet.get()) < 0) {
            break;
        }
        auto read_end = std::chrono::high_resolution_clock::now();
        total_input_wait_us += std::chrono::duration_cast<std::chrono::microseconds>(read_end - read_start).count();

        if (packet->stream_index != video_stream_idx) {
            continue;
        }

        int ret = avcodec_send_packet(dec_ctx_raw, packet.get());
        if (ret < 0) break;

        while (true) {
            ret = avcodec_receive_frame(dec_ctx_raw, frame.get());
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) {
                stop_flag = true;
                break;
            }

            int64_t abs_frame_num = -1;
            int64_t pts = frame->pts;
            if (pts == AV_NOPTS_VALUE) pts = frame->best_effort_timestamp;
            if (pts != AV_NOPTS_VALUE && time_base_sec > 0.0 && fps > 0.0) {
                abs_frame_num = pts_to_frame(pts);
            }
            if (abs_frame_num < 0) {
                static bool warned = false;
                if (!warned) {
                    std::cerr << "Warning: PTS missing, stepping may be inaccurate.\n";
                    warned = true;
                }
                abs_frame_num = ++fallback_counter;
            }

            if (abs_frame_num < segments[current_seg].first) {
                continue;
            }

            while (current_seg < segments.size() && abs_frame_num > segments[current_seg].second) {
                current_seg++;
                if (current_seg >= segments.size()) {
                    if (frames_in_batch > 0) {
                        queue.push(Batch(std::move(current_batch), frames_in_batch));
                        total_frames_read += frames_in_batch;
                    }
                    all_done = true;
                    break;
                }
                target_abs_frame = segments[current_seg].first;
                seek_to_frame(target_abs_frame);
                break;
            }
            if (current_seg >= segments.size()) break;

            if (abs_frame_num == target_abs_frame) {
                auto frame_start = std::chrono::high_resolution_clock::now();

                size_t offset = current_batch.size();
                current_batch.resize(offset + frame_bytes);

                if (can_copy_direct && frame->format == out_pix_fmt) {
                    int ret = av_image_copy_to_buffer(current_batch.data() + offset, static_cast<int>(frame_bytes),
                                                      frame->data, frame->linesize,
                                                      out_pix_fmt, out_width, out_height, 1);
                    if (ret < 0) {
                        std::cerr << "Error copying frame directly\n";
                        stop_flag = true;
                        break;
                    }
                } else {
                    uint8_t* dst[4] = {nullptr};
                    int dst_linesize[4] = {0};
                    size_t plane_size = static_cast<size_t>(out_width) * out_height * bytes_per_pixel;
                    dst[0] = current_batch.data() + offset;
                    dst[1] = dst[0] + plane_size;
                    dst[2] = dst[1] + plane_size;
                    dst_linesize[0] = out_width * bytes_per_pixel;
                    dst_linesize[1] = out_width * bytes_per_pixel;
                    dst_linesize[2] = out_width * bytes_per_pixel;

                    sws_scale(sws_ctx_raw,
                              frame->data, frame->linesize,
                              crop_top, crop_height,
                              dst, dst_linesize);
                }

                auto frame_end = std::chrono::high_resolution_clock::now();
                total_decode_time_us += std::chrono::duration_cast<std::chrono::microseconds>(frame_end - frame_start).count();

                frames_in_batch++;
                if (frames_in_batch >= opts.batch_frames) {
                    queue.push(Batch(std::move(current_batch), frames_in_batch));
                    total_frames_read += frames_in_batch;
                    current_batch = buffer_pool.borrow();
                    current_batch.clear();
                    current_batch.reserve(static_cast<size_t>(opts.batch_frames) * frame_bytes);
                    frames_in_batch = 0;
                }

                target_abs_frame += opts.step;

                if (target_abs_frame <= segments[current_seg].second) {
                    if (opts.step > 1 && opts.step > constants::SEEK_THRESHOLD) {
                        seek_to_frame(target_abs_frame);
                        break;
                    }
                } else {
                    current_seg++;
                    if (current_seg < segments.size()) {
                        target_abs_frame = segments[current_seg].first;
                        seek_to_frame(target_abs_frame);
                        break;
                    } else {
                        all_done = true;
                        break;
                    }
                }
            }
        }
        if (all_done) break;
        av_packet_unref(packet.get());
    }

    if (frames_in_batch > 0) {
        queue.push(Batch(std::move(current_batch), frames_in_batch));
        total_frames_read += frames_in_batch;
    } else {
        buffer_pool.give_back(std::move(current_batch));
    }
}

static int extract_color_information(const Options& opts) {
    AVFormatContext* raw_fmt_ctx = nullptr;
    if (avformat_open_input(&raw_fmt_ctx, opts.input_file.c_str(), nullptr, nullptr) < 0) {
        std::cerr << "Could not open input file: " << opts.input_file << "\n";
        return 1;
    }
    AVFormatContextPtr fmt_ctx(raw_fmt_ctx);

#ifdef __linux__
    int fd = fileno(fmt_ctx->pb ? fmt_ctx->pb->logfile : 0);
    if (fd > 0) {
        posix_fadvise(fd, 0, 0, POSIX_FADV_SEQUENTIAL);
    }
#endif

    if (avformat_find_stream_info(fmt_ctx.get(), nullptr) < 0) {
        std::cerr << "Could not find stream information\n";
        return 1;
    }

    int video_stream_idx = av_find_best_stream(fmt_ctx.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
    if (video_stream_idx < 0) {
        std::cerr << "No video stream found\n";
        return 1;
    }

    AVStream* stream = fmt_ctx->streams[video_stream_idx];
    AVCodecParameters* codecpar = stream->codecpar;
    const AVCodec* codec = avcodec_find_decoder(codecpar->codec_id);
    if (!codec) {
        std::cerr << "Unsupported codec\n";
        return 1;
    }

    AVCodecContext* raw_dec_ctx = avcodec_alloc_context3(codec);
    if (!raw_dec_ctx) {
        std::cerr << "Failed to allocate codec context\n";
        return 1;
    }
    AVCodecContextPtr dec_ctx(raw_dec_ctx);
    avcodec_parameters_to_context(dec_ctx.get(), codecpar);

    dec_ctx->thread_count = std::thread::hardware_concurrency();
    dec_ctx->thread_type = FF_THREAD_FRAME;

    if (avcodec_open2(dec_ctx.get(), codec, nullptr) < 0) {
        std::cerr << "Could not open codec\n";
        return 1;
    }

    VideoInfo info;
    info.file_name = opts.input_file;
    info.width = dec_ctx->width;
    info.height = dec_ctx->height;
    info.pix_fmt = dec_ctx->pix_fmt;
    info.pix_fmt_name = av_get_pix_fmt_name(dec_ctx->pix_fmt) ?: "unknown";
    info.framerate = av_guess_frame_rate(fmt_ctx.get(), stream, nullptr);
    info.total_frames = stream->nb_frames;
    if (info.total_frames <= 0) {
        if (fmt_ctx->duration != AV_NOPTS_VALUE) {
            double duration_sec = fmt_ctx->duration / static_cast<double>(AV_TIME_BASE);
            double fps = av_q2d(info.framerate);
            info.total_frames = static_cast<int64_t>(duration_sec * fps);
        }
    }

    const AVPixFmtDescriptor* desc = av_pix_fmt_desc_get(dec_ctx->pix_fmt);
    info.bitdepth = desc ? desc->comp[0].depth : 8;

    info.color_primaries = av_color_primaries_name(dec_ctx->color_primaries) ?: "unknown";
    info.color_trc = av_color_transfer_name(dec_ctx->color_trc) ?: "unknown";
    info.color_space = av_color_space_name(dec_ctx->colorspace) ?: "unknown";
    info.color_range = dec_ctx->color_range;
    info.chroma_location = dec_ctx->chroma_sample_location;
    info.field_order = dec_ctx->field_order;

    info.codec_name = avcodec_get_name(codecpar->codec_id);
    info.codec_long_name = codec->long_name ?: "unknown";
    info.profile = codecpar->profile;
    info.level = codecpar->level;
    info.bitrate = codecpar->bit_rate;

    info.container_format = fmt_ctx->iformat->name;
    info.container_format_long = fmt_ctx->iformat->long_name;

    struct stat st;
    if (stat(opts.input_file.c_str(), &st) == 0) info.file_size = st.st_size;

    if (fmt_ctx->duration != AV_NOPTS_VALUE) info.duration_sec = fmt_ctx->duration / static_cast<double>(AV_TIME_BASE);
    info.start_time = fmt_ctx->start_time;
    info.overall_bitrate = fmt_ctx->bit_rate;

    info.metadata = util::dict_to_map(fmt_ctx->metadata);

    int crop_left = 0, crop_top = 0, crop_width = info.width, crop_height = info.height;
    if (opts.crop) {
        const auto& c = *opts.crop;
        int a = c[0], b = c[1], c1 = c[2], d = c[3];
        if (c1 >= 0 && d >= 0) {
            crop_left = a; crop_top = b; crop_width = c1; crop_height = d;
        } else {
            crop_left = a; crop_top = b;
            crop_width = info.width - a - (-c1);
            crop_height = info.height - b - (-d);
        }
        if (crop_left < 0 || crop_top < 0 || crop_width <= 0 || crop_height <= 0 ||
            crop_left + crop_width > info.width || crop_top + crop_height > info.height) {
            std::cerr << "Invalid crop region.\n";
            return 1;
        }
        info.crop_left = crop_left; info.crop_top = crop_top;
        info.crop_width = crop_width; info.crop_height = crop_height;
        info.crop_input = opts.crop_arg;
    } else {
        info.crop_left = 0; info.crop_top = 0;
        info.crop_width = info.width; info.crop_height = info.height;
    }

    int out_width = static_cast<int>(info.crop_width * opts.scale_factor);
    int out_height = static_cast<int>(info.crop_height * opts.scale_factor);
    if (out_width < 1) out_width = 1;
    if (out_height < 1) out_height = 1;
    info.scaled_width = out_width;
    info.scaled_height = out_height;

    if (opts.trim_segments) {
        const auto& segs = *opts.trim_segments;
        for (const auto& seg : segs) {
            if (seg.first > info.total_frames || seg.second > info.total_frames) {
                std::cerr << "Warning: Trim segment [" << seg.first << "," << seg.second << "] exceeds total frames.\n";
            }
        }
    }

    int target_bitdepth = opts.target_bitdepth;
    if (target_bitdepth == 0) target_bitdepth = info.bitdepth;
    
    int rgb1d_bins   = opts.rgb1d_bins   != 0 ? opts.rgb1d_bins   : std::min(1024, (1 << target_bitdepth));
    int rgb3d_bins   = opts.rgb3d_bins   != 0 ? opts.rgb3d_bins   : 65;

    bool need_rgb = opts.enable_rgb1d || opts.enable_rgb3d || opts.enable_rgbmoments;
    if (!need_rgb) {
        std::cerr << "No extraction enabled, nothing to do.\n";
        return 1;
    }

    AVPixelFormat out_pix_fmt = (target_bitdepth <= 8) ? AV_PIX_FMT_RGB24 : AV_PIX_FMT_RGB48LE;

    int sws_flags = SWS_BILINEAR;
    switch (opts.scale_algo) {
        case ScaleAlgo::BICUBIC: sws_flags = SWS_BICUBIC; break;
        case ScaleAlgo::LANCZOS: sws_flags = SWS_LANCZOS; break;
        case ScaleAlgo::POINT:   sws_flags = SWS_POINT;   break;
        case ScaleAlgo::FAST_BILINEAR: sws_flags = SWS_FAST_BILINEAR; break;
        default: break;
    }
    SwsContext* raw_sws_ctx = sws_getContext(info.width, info.height, info.pix_fmt,
                                              out_width, out_height, out_pix_fmt,
                                              sws_flags, nullptr, nullptr, nullptr);
    if (!raw_sws_ctx) {
        std::cerr << "Could not initialize scaling context\n";
        return 1;
    }
    SwsContextPtr sws_ctx(raw_sws_ctx);

    int max_val = (1 << target_bitdepth) - 1;

    WorkerParams wparams;
    wparams.bitdepth = target_bitdepth;
    wparams.enable_rgb1d = opts.enable_rgb1d;
    wparams.enable_rgb3d = opts.enable_rgb3d;
    wparams.enable_rgbmoments = opts.enable_rgbmoments;
    wparams.max_val = max_val;
    wparams.rgb1d_bins = rgb1d_bins;
    wparams.rgb3d_bins = rgb3d_bins;
    wparams.width = out_width;
    wparams.height = out_height;
    wparams.float_binning = opts.float_binning;

    int num_threads = (opts.workers > 0) ? opts.workers : static_cast<int>(std::thread::hardware_concurrency());
    if (num_threads < 1) num_threads = 1;

    print_video_information(info, opts, target_bitdepth,
                            rgb1d_bins, rgb3d_bins,
                            num_threads);

    int queue_capacity = opts.prefetch * num_threads;
    if (queue_capacity < 1) queue_capacity = 1;
    ConcurrentQueue<Batch> queue(queue_capacity);

    size_t frame_bytes = 3 * static_cast<size_t>(out_width) * out_height *
                         ((target_bitdepth <= 8) ? 1 : 2);
    size_t buffer_size = static_cast<size_t>(opts.batch_frames) * frame_bytes;
    BufferPool buffer_pool(buffer_size, num_threads + 1);

    std::atomic<bool> stop_flag(false);
    int64_t total_frames_read = 0;

    std::atomic<uint64_t> total_decode_time_us(0);
    std::atomic<uint64_t> total_process_time_us(0);
    std::atomic<uint64_t> total_input_wait_us(0);
    std::atomic<int> active_workers(0);

    std::thread producer(producer_thread_function, fmt_ctx.get(), dec_ctx.get(), sws_ctx.get(),
                         video_stream_idx, std::cref(opts), target_bitdepth,
                         crop_left, crop_top, crop_width, crop_height,
                         out_width, out_height, out_pix_fmt,
                         std::ref(queue), std::ref(buffer_pool),
                         std::ref(stop_flag), std::ref(total_frames_read),
                         std::ref(total_decode_time_us), std::ref(total_input_wait_us));

    std::vector<PartialResults> thread_results(num_threads);
    std::vector<std::thread> workers;
    std::atomic<int64_t> processed_frames(0);

    auto worker_function = [&](int tid) {
        auto& local = thread_results[tid];
        
        while (!stop_flag.load() && !global_stop_flag.load()) {
            Batch batch({{}}, 0);
            if (!queue.pop(batch)) break;

            active_workers++;

            auto proc_start = std::chrono::high_resolution_clock::now();

            PartialResults partial = process::process_batch_rgb(batch, wparams);

            auto proc_end = std::chrono::high_resolution_clock::now();
            uint64_t proc_time = std::chrono::duration_cast<std::chrono::microseconds>(proc_end - proc_start).count();
            total_process_time_us.fetch_add(proc_time, std::memory_order_relaxed);

            local.merge(partial, wparams);
            processed_frames.fetch_add(batch.num_frames, std::memory_order_relaxed);

            buffer_pool.give_back(std::move(batch.data));

            active_workers--;
        }
    };

    for (int i = 0; i < num_threads; ++i)
        workers.emplace_back(worker_function, i);

    int64_t total_effective_frames = 0;
    if (opts.trim_segments) {
        for (const auto& seg : *opts.trim_segments) {
            int64_t len = seg.second - seg.first + 1;
            total_effective_frames += (len + opts.step - 1) / opts.step;
        }
    } else if (info.total_frames > 0) {
        total_effective_frames = (info.total_frames + opts.step - 1) / opts.step;
    } else {
        total_effective_frames = 0;
    }

    std::atomic<bool> progress_done(false);
    std::thread progress_thread([&]() {
        auto start_time = std::chrono::steady_clock::now();
        int64_t last_processed = 0;
        auto last_time = start_time;
    
        uint64_t last_input_us  = 0;
        uint64_t last_decode_us = 0;
        uint64_t last_process_us = 0;
    
        while (!progress_done.load() && !global_stop_flag.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
    
            auto now = std::chrono::steady_clock::now();
            double elapsed = std::chrono::duration<double>(now - start_time).count();
    
            int64_t current = processed_frames.load();
            int64_t total = total_effective_frames;
    
            uint64_t cur_input_us  = total_input_wait_us.load();
            uint64_t cur_decode_us = total_decode_time_us.load();
            uint64_t cur_process_us = total_process_time_us.load();
    
            double dt = std::chrono::duration<double>(now - last_time).count();
            int64_t delta_frames = current - last_processed;
    
            double fps = (dt > 0.0) ? delta_frames / dt : 0.0;

            double recent_input_ms = 0.0, recent_decode_ms = 0.0, recent_process_ms = 0.0;
            if (delta_frames > 0 && dt > 0.0) {
                uint64_t delta_input_us  = cur_input_us - last_input_us;
                uint64_t delta_decode_us = cur_decode_us - last_decode_us;
                uint64_t delta_process_us = cur_process_us - last_process_us;
    
                recent_input_ms  = (delta_input_us  / 1000.0) / delta_frames;
                recent_decode_ms = (delta_decode_us / 1000.0) / delta_frames;
                recent_process_ms = (delta_process_us / 1000.0) / delta_frames;
            }
    
            print_progress_bar(current, total, elapsed, fps,
                               recent_input_ms, recent_decode_ms, recent_process_ms,
                               active_workers.load(), num_threads);
    
            last_processed = current;
            last_time = now;
            last_input_us  = cur_input_us;
            last_decode_us = cur_decode_us;
            last_process_us = cur_process_us;
        }
        std::cout << std::endl;
    });

    producer.join();
    stop_flag = true;
    queue.stop();
    buffer_pool.stop();
    for (auto& t : workers) t.join();

    progress_done = true;
    progress_thread.join();

    PartialResults final_results;
    for (auto& tr : thread_results)
        final_results.merge(tr, wparams);

    json output;

    json file_metadata;
    file_metadata["name"] = opts.input_file;
    file_metadata["size_bytes"] = info.file_size;
    file_metadata["size_human"] = util::format_file_size(info.file_size);
    file_metadata["container"] = info.container_format;
    file_metadata["container_long"] = info.container_format_long;
    file_metadata["duration_sec"] = info.duration_sec;
    file_metadata["start_time"] = info.start_time;
    file_metadata["bitrate"] = info.overall_bitrate;
    if (!info.metadata.empty()) file_metadata["tags"] = info.metadata;
    output["metadata"]["file"] = file_metadata;

    json video_metadata;
    video_metadata["frames_total"] = info.total_frames;
    video_metadata["frame_rate"] = av_q2d(info.framerate);
    video_metadata["frame_rate_num"] = info.framerate.num;
    video_metadata["frame_rate_den"] = info.framerate.den;
    video_metadata["original_width"] = info.width;
    video_metadata["original_height"] = info.height;
    video_metadata["original_bitdepth"] = info.bitdepth;
    video_metadata["pixel_format"] = info.pix_fmt_name;
    video_metadata["color_primaries"] = info.color_primaries;
    video_metadata["color_trc"] = info.color_trc;
    video_metadata["color_space"] = info.color_space;
    video_metadata["color_range"] = util::color_range_name(info.color_range);
    video_metadata["chroma_location"] = util::chroma_location_name(info.chroma_location);
    video_metadata["field_order"] = util::field_order_name(info.field_order);
    video_metadata["codec_name"] = info.codec_name;
    video_metadata["codec_long_name"] = info.codec_long_name;
    video_metadata["profile"] = info.profile;
    video_metadata["level"] = info.level;
    video_metadata["bitrate"] = info.bitrate;
    output["metadata"]["video"] = video_metadata;

    json extract_metadata;
    extract_metadata["frames_processed"] = total_frames_read;
    extract_metadata["frames_expected"] = total_effective_frames;
    extract_metadata["target_bitdepth"] = target_bitdepth;
    extract_metadata["crop_left"] = info.crop_left;
    extract_metadata["crop_top"] = info.crop_top;
    extract_metadata["crop_width"] = info.crop_width;
    extract_metadata["crop_height"] = info.crop_height;
    extract_metadata["scale_factor"] = opts.scale_factor;
    extract_metadata["scaled_width"] = info.scaled_width;
    extract_metadata["scaled_height"] = info.scaled_height;
    extract_metadata["scale_algo"]    = util::scale_algo_name(opts.scale_algo);
    extract_metadata["float_binning"] = opts.float_binning;
    
    if (opts.trim_segments) {
        json trim_array = json::array();
        for (const auto& seg : *opts.trim_segments) {
            trim_array.push_back({{"start", seg.first}, {"end", seg.second}});
        }
        extract_metadata["trim"] = trim_array;
    }
    
    json parameters;

    if (opts.enable_rgb1d || opts.enable_rgb3d || (opts.enable_rgbmoments && final_results.pixel_count > 0)) {
        json rgb_params;
        if (opts.enable_rgb1d) {
            rgb_params["1d"] = opts.enable_rgb1d;
            rgb_params["1d_bins"] = rgb1d_bins;
         }
        if (opts.enable_rgb3d) {
            rgb_params["3d"] = opts.enable_rgb3d;
            rgb_params["3d_bins"] = rgb3d_bins;
        }
        if (opts.enable_rgbmoments && final_results.pixel_count > 0) {
            rgb_params["moments"] = opts.enable_rgbmoments;
        }
        parameters["rgb"] = rgb_params;
    }
    
    extract_metadata["parameters"] = parameters;
    
    json performance;
    performance["step"] = opts.step;
    performance["batch_frames"] = opts.batch_frames;
    performance["prefetch"] = opts.prefetch;
    performance["workers"] = num_threads;
    extract_metadata["performance"] = performance;

    output["metadata"]["extraction"] = extract_metadata;

    json data_obj;

    if (opts.enable_rgb1d || opts.enable_rgb3d || (opts.enable_rgbmoments && final_results.pixel_count > 0)) {
        json rgb_obj;
        if (opts.enable_rgb1d) {
            size_t n = rgb1d_bins;
            std::vector<uint64_t> R(final_results.rgb1d.begin(), final_results.rgb1d.begin() + n);
            std::vector<uint64_t> G(final_results.rgb1d.begin() + n, final_results.rgb1d.begin() + 2*n);
            std::vector<uint64_t> B(final_results.rgb1d.begin() + 2*n, final_results.rgb1d.end());
            rgb_obj["r"] = std::move(R);
            rgb_obj["g"] = std::move(G);
            rgb_obj["b"] = std::move(B);
        }
        if (opts.enable_rgb3d) {
            rgb_obj["3d"] = final_results.rgb3d;
        }
        if (opts.enable_rgbmoments && final_results.pixel_count > 0) {
            json moments;
            moments["pixel_count"] = final_results.pixel_count;
            moments["sum"]["R"] = final_results.sum[0];
            moments["sum"]["G"] = final_results.sum[1];
            moments["sum"]["B"] = final_results.sum[2];
            moments["sum2"]["R"] = final_results.sum2[0];
            moments["sum2"]["G"] = final_results.sum2[1];
            moments["sum2"]["B"] = final_results.sum2[2];
            moments["sum3"]["R"] = final_results.sum3[0];
            moments["sum3"]["G"] = final_results.sum3[1];
            moments["sum3"]["B"] = final_results.sum3[2];
            rgb_obj["moments"] = std::move(moments);
        }
        data_obj["rgb"] = std::move(rgb_obj);
    }

    if (!data_obj.empty()) {
        output["data"] = std::move(data_obj);
    }

    std::ofstream out(opts.output_file);
    if (!out) {
        std::cerr << "Could not open output file for writing: " << opts.output_file << "\n";
        return 1;
    }
    out << output.dump(2);
    out.close();

    std::cout << "Color information saved to " << opts.output_file << "\n";
    return 0;
}

static int parse_integer(const char* arg, const std::string& name, int min_val) {
    char* endptr;
    long val = strtol(arg, &endptr, 10);
    if (*endptr != '\0' || val < min_val || val > INT_MAX) {
        throw std::invalid_argument("option --" + name + " requires an integer >= " + std::to_string(min_val));
    }
    return static_cast<int>(val);
}

static float parse_floating_point(const char* arg, const std::string& name, float min_val, float max_val) {
    char* endptr;
    float val = strtof(arg, &endptr);
    if (*endptr != '\0' || val < min_val || val > max_val) {
        throw std::invalid_argument("option --" + name + " requires a float between " +
                                    std::to_string(min_val) + " and " + std::to_string(max_val));
    }
    return val;
}

static std::vector<int> parse_comma_separated_integers(const char* arg, const std::string& name, size_t expected_count, int min_val) {
    std::string s(arg);
    std::vector<int> result;
    size_t pos = 0;
    while (pos < s.size()) {
        size_t comma = s.find(',', pos);
        std::string token = s.substr(pos, comma - pos);
        char* endptr;
        long val = strtol(token.c_str(), &endptr, 10);
        if (*endptr != '\0' || val < min_val || val > INT_MAX) {
            throw std::invalid_argument("option --" + name + " requires integers >= " + std::to_string(min_val) +
                                        " separated by commas.");
        }
        result.push_back(static_cast<int>(val));
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    if (result.size() != expected_count) {
        throw std::invalid_argument("option --" + name + " expects exactly " + std::to_string(expected_count) +
                                    " values (comma-separated).");
    }
    return result;
}

static std::vector<std::pair<int,int>> parse_trim_specification(const char* arg) {
    std::string s(arg);
    std::vector<std::pair<int,int>> segments;

    if (s.find('-') != std::string::npos) {
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string token = s.substr(pos, comma - pos);
            size_t dash = token.find('-');
            if (dash == std::string::npos) {
                throw std::invalid_argument("trim range must contain '-' when using dash syntax");
            }
            std::string start_str = token.substr(0, dash);
            std::string end_str = token.substr(dash + 1);
            char* endptr;
            long start = strtol(start_str.c_str(), &endptr, 10);
            if (*endptr != '\0' || start < 1) throw std::invalid_argument("invalid trim start");
            long end = strtol(end_str.c_str(), &endptr, 10);
            if (*endptr != '\0' || end < 1) throw std::invalid_argument("invalid trim end");
            segments.emplace_back(static_cast<int>(start), static_cast<int>(end));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
    } else {
        std::vector<int> nums;
        size_t pos = 0;
        while (pos < s.size()) {
            size_t comma = s.find(',', pos);
            std::string token = s.substr(pos, comma - pos);
            char* endptr;
            long val = strtol(token.c_str(), &endptr, 10);
            if (*endptr != '\0' || val < 1) throw std::invalid_argument("invalid trim integer");
            nums.push_back(static_cast<int>(val));
            if (comma == std::string::npos) break;
            pos = comma + 1;
        }
        if (nums.size() % 2 != 0) {
            throw std::invalid_argument("trim requires an even number of integers (pairs)");
        }
        for (size_t i = 0; i < nums.size(); i += 2) {
            segments.emplace_back(nums[i], nums[i+1]);
        }
    }

    for (const auto& p : segments) {
        if (p.first > p.second) {
            throw std::invalid_argument("trim start must be <= end in each segment");
        }
    }

    std::sort(segments.begin(), segments.end());

    for (size_t i = 1; i < segments.size(); ++i) {
        if (segments[i].first <= segments[i-1].second) {
            throw std::invalid_argument("trim segments must not overlap and must be in increasing order");
        }
    }

    return segments;
}

static Options parse_options(int argc, char** argv) {
    Options opts;

    static struct option long_options[] = {
        {"input",                required_argument, 0, 'i'},
        {"output",               required_argument, 0, 'o'},
        {"bitdepth",             required_argument, 0, 'b'},
        {"force",                no_argument,       0, 'f'},
        {"help",                 no_argument,       0, 'h'},
        {"version",              no_argument,       0, 'v'},
        {"crop",                 required_argument, 0, 1},
        {"trim",                 required_argument, 0, 2},
        {"methods",              required_argument, 0, 300},
        {"rgb1d-bins",           required_argument, 0, 9},
        {"rgb3d-bins",           required_argument, 0, 10},
        {"step",                 required_argument, 0, 's'},
        {"scale",                required_argument, 0, 16},
        {"batch",                required_argument, 0, 17},
        {"prefetch",             required_argument, 0, 18},
        {"workers",              required_argument, 0, 19},
        {"float-binning",        no_argument,       0, 20},
        {"scaling",              required_argument, 0, 26},
        {0, 0, 0, 0}
    };

    int opt;
    int option_index = 0;
    while ((opt = getopt_long(argc, argv, "i:o:s:b:fhv", long_options, &option_index)) != -1) {
        try {
            switch (opt) {
                case 'i': opts.input_file = optarg; break;
                case 'o': opts.output_file = optarg; break;
                case 'b': opts.target_bitdepth = parse_integer(optarg, "bitdepth", 0); break;
                case 'f': opts.force_overwrite = true; break;
                case 'h': opts.print_help(argv[0]); exit(0);
                case 'v': opts.print_version(); exit(0);
                case 1: 
                    opts.crop = parse_comma_separated_integers(optarg, "crop", 4, -10000);
                    opts.crop_arg = optarg;
                    break;
                case 2: {
                    auto segs = parse_trim_specification(optarg);
                    opts.trim_segments = segs;
                    break;
                }
                case 300: {
                    std::string arg(optarg);
                    std::vector<std::string> methods;
                    size_t pos = 0;
                    while (pos < arg.size()) {
                        size_t comma = arg.find(',', pos);
                        std::string token = arg.substr(pos, comma - pos);
                        if (!token.empty())
                            methods.push_back(token);
                        if (comma == std::string::npos) break;
                        pos = comma + 1;
                    }
                    bool all_present = std::find(methods.begin(), methods.end(), "all") != methods.end();
                    if (all_present) {
                        opts.enable_rgb1d = true;
                        opts.enable_rgb3d = true;
                        opts.enable_rgbmoments = true;
                    } else {
                        opts.enable_rgb1d = false;
                        opts.enable_rgb3d = false;
                        opts.enable_rgbmoments = false;
                        for (const auto& m : methods) {
                            if (m == "rgb1d") opts.enable_rgb1d = true;
                            else if (m == "rgb3d") opts.enable_rgb3d = true;
                            else if (m == "rgbmoments") opts.enable_rgbmoments = true;
                            else throw std::invalid_argument("Invalid method in --methods: " + m);
                        }
                    }
                    break;
                }
                case 9: opts.rgb1d_bins = parse_integer(optarg, "rgb1d-bins", 2); break;
                case 10: opts.rgb3d_bins = parse_integer(optarg, "rgb3d-bins", 2); break;
                case 's': opts.step = parse_integer(optarg, "step", 1); break;
                case 16: opts.scale_factor = parse_floating_point(optarg, "scale", 0.0f, 1.0f); break;
                case 17: opts.batch_frames = parse_integer(optarg, "batch", 1); break;
                case 18: opts.prefetch = parse_integer(optarg, "prefetch", 1); break;
                case 19: opts.workers = parse_integer(optarg, "workers", 0); break;
                case 20: opts.float_binning = true; break;
                case 26: {
                    std::string arg(optarg);
                    if (arg == "bilinear") opts.scale_algo = ScaleAlgo::BILINEAR;
                    else if (arg == "bicubic") opts.scale_algo = ScaleAlgo::BICUBIC;
                    else if (arg == "lanczos") opts.scale_algo = ScaleAlgo::LANCZOS;
                    else if (arg == "point") opts.scale_algo = ScaleAlgo::POINT;
                    else if (arg == "fast-bilinear") opts.scale_algo = ScaleAlgo::FAST_BILINEAR;
                    else throw std::invalid_argument("Invalid scaling (must be bilinear, bicubic, lanczos, point, fast-bilinear)");
                    break;
                }
                default: opts.error = true; opts.error_msg = "Unknown option"; return opts;
            }
        } catch (const std::exception& e) {
            opts.error = true;
            opts.error_msg = e.what();
            return opts;
        }
    }

    if (opts.input_file.empty()) {
        opts.error = true;
        opts.error_msg = "No input file specified. Use -i <file>.";
    }
    if (opts.target_bitdepth != 0 && opts.target_bitdepth != 8 && opts.target_bitdepth != 10 &&
        opts.target_bitdepth != 12 && opts.target_bitdepth != 16) {
        opts.error = true;
        opts.error_msg = "Target bit depth must be 0 (auto), 8, 10, 12, or 16.";
    }
    return opts;
}

void Options::print_help(const char* progname) const {
    std::cout << "Usage: " << progname << " -i <input_file> [options]\n\n"
              << "Extracts color information from a video file and writes a JSON report.\n\n"
              << "Required options:\n"
              << "  -i, --input <file>             Input video file.\n\n"
              << "General options:\n"
              << "  -o, --output <file>            Output JSON file (default: input name with '_colors.json').\n"
              << "  -b, --bitdepth <depth>         Target bit depth: 8, 10, 12, 16, or 0 for original (default: 0).\n"
              << "  -f, --force                    Overwrite output file if it exists.\n"
              << "  -h, --help                     Show this help message.\n"
              << "  -v, --version                  Show version information.\n\n"
              << "Crop and trim:\n"
              << "  --crop <l,t,-r,-b>             left, top, -right, -bottom.\n"
              << "  --crop <l,t,w,h>               left, top, width, height.\n"
              << "  --trim <start,end>...          Process frames in given segments (1-based, inclusive).\n"
              << "                                  Syntax: comma-separated pairs (e.g., 1000,2000,3000,4000)\n"
              << "                                  or dash-separated ranges (e.g., 1000-2000,3000-4000).\n"
              << "                                  Segments must be sorted and non-overlapping.\n\n"
              << "Method selection (all enabled by default):\n"
              << "  --methods <list>               Comma-separated list of methods to enable:\n"
              << "                                  rgb1d, rgb3d, rgbmoments\n"
              << "                                  Use 'all' to enable everything (default).\n\n"
              << "Histogram resolution options (if not set, sensible defaults are chosen based on --bitdepth):\n"
              << "  --rgb1d-bins <N>               Number of bins per channel for RGB 1D histograms.\n"
              << "  --rgb3d-bins <N>               Number of bins per channel for RGB 3D histogram.\n\n"
              << "Accuracy control (default: off = fastest):\n"
              << "  --float-binning                Use floating-point binning for histograms.\n\n"
              << "Performance tuning:\n"
              << "  --scaling <algo>               Pixel scaling: bilinear, bicubic, lanczos, point, fast-bilinear (default: bilinear).\n"
              << "  --scale <F>                    Scale down image after crop (0.0 - 1.0, default: 1.0).\n"
              << "  --step <N>                     Process every Nth frame within each segment (default: 1).\n"
              << "  --batch <N>                    Frames per batch (default: 1).\n"
              << "  --prefetch <N>                 Batches queued per worker (default: 2).\n"
              << "  --workers <N>                  Number of worker threads (0 = auto, default: 0).\n\n"
              << "Examples:\n"
              << "  " << progname << " -i video.mkv -o colors.json\n"
              << "  " << progname << " -i video.mkv --step 5 --crop 10,10,100,200 --trim 100,1000\n"
              << "  " << progname << " -i video.mkv --methods rgb1d,rgb3d --float-binning\n";
}

void Options::print_version() const {
    std::cout << "extract_colors version 2.6 (RGB only, YUV/LAB removed)\n"
              << "Uses FFmpeg " << av_version_info() << " and nlohmann/json.\n";
}

int main(int argc, char** argv) {
    signal(SIGINT, signal_handler);

    auto opts = parse_options(argc, argv);
    if (opts.error) {
        std::cerr << "Error: " << opts.error_msg << "\n";
        return 1;
    }

    if (opts.output_file.empty()) {
        size_t dot = opts.input_file.find_last_of('.');
        if (dot == std::string::npos) {
            opts.output_file = opts.input_file + "_colors.json";
        } else {
            opts.output_file = opts.input_file.substr(0, dot) + "_colors.json";
        }
    }

    if (!opts.force_overwrite) {
        std::ifstream f(opts.output_file.c_str());
        if (f.good()) {
            std::cerr << "Output file " << opts.output_file << " exists. Use --force to overwrite.\n";
            return 1;
        }
    }

    avformat_network_init();

    int ret = extract_color_information(opts);

    avformat_network_deinit();
    return ret;
}