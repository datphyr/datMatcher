#include <getopt.h>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <tuple>
#include <functional>
#include <random>
#include <numeric>
#include <cstring>
#include <omp.h>
#include <nlohmann/json.hpp>

using json = nlohmann::ordered_json;
namespace fs = std::filesystem;

class Histogram1D {
public:
    int bins;
    std::vector<double> pdf;
    std::vector<double> cdf_edges;

    Histogram1D() = default;
    Histogram1D(const std::vector<uint64_t>& counts, int bins_, double smoothing = 0.0) {
        bins = bins_;
        double total_raw = 0.0;
        for (auto c : counts) total_raw += c;
        double total = total_raw + smoothing * bins;
        pdf.resize(bins);
        if (total == 0.0) {
            for (int i = 0; i < bins; ++i) pdf[i] = 1.0 / bins;
        } else {
            for (int i = 0; i < bins; ++i) pdf[i] = (counts[i] + smoothing) / total;
        }
        cdf_edges.resize(bins + 1);
        cdf_edges[0] = 0.0;
        for (int i = 0; i < bins; ++i) cdf_edges[i+1] = cdf_edges[i] + pdf[i];
        double last = cdf_edges.back();
        if (last > 0.0) {
            for (int i = 1; i <= bins; ++i) cdf_edges[i] /= last;
        }
    }

    Histogram1D(const std::vector<double>& prob, int bins_, double total = 0.0) {
        bins = bins_;
        pdf = prob;
        if (total == 0.0) {
            total = 0.0;
            for (double v : pdf) total += v;
        }
        if (total > 0.0) {
            for (double& v : pdf) v /= total;
        }
        cdf_edges.resize(bins + 1);
        cdf_edges[0] = 0.0;
        for (int i = 0; i < bins; ++i) {
            cdf_edges[i+1] = cdf_edges[i] + pdf[i];
        }
        double last = cdf_edges.back();
        if (last > 0.0) {
            for (int i = 1; i <= bins; ++i) cdf_edges[i] /= last;
        }
    }

    double getCDF(double x) const {
        if (x <= 0.0) return 0.0;
        if (x >= 1.0) return 1.0;
        double pos = x * bins;
        int idx = static_cast<int>(pos);
        if (idx >= bins) idx = bins - 1;
        double frac = pos - idx;
        return cdf_edges[idx] + frac * (cdf_edges[idx+1] - cdf_edges[idx]);
    }

    double invertCDF(double p) const {
        if (p <= 0.0) return 0.0;
        if (p >= 1.0) return 1.0;
        auto it = std::upper_bound(cdf_edges.begin(), cdf_edges.end(), p);
        int idx = it - cdf_edges.begin() - 1;
        if (idx < 0) idx = 0;
        double cdf_low = cdf_edges[idx];
        double cdf_high = cdf_edges[idx+1];
        double frac = (cdf_high > cdf_low) ? (p - cdf_low) / (cdf_high - cdf_low) : 0.0;
        return (idx + frac) / bins;
    }
};

class Histogram3D {
public:
    int bins_r, bins_g, bins_b;
    std::vector<double> pdf;

    Histogram3D() = default;
    Histogram3D(const std::vector<uint64_t>& counts, int br, int bg, int bb, double smoothing = 0.0) {
        bins_r = br; bins_g = bg; bins_b = bb;
        size_t sz = static_cast<size_t>(br) * bg * bb;
        double total_raw = 0.0;
        for (auto c : counts) total_raw += c;
        double total = total_raw + smoothing * sz;
        pdf.resize(sz);
        if (total > 0.0) {
            for (size_t i = 0; i < sz; ++i) pdf[i] = (counts[i] + smoothing) / total;
        } else {
            std::fill(pdf.begin(), pdf.end(), 1.0 / sz);
        }
    }

    Histogram3D resample(int newR, int newG, int newB) const {
        Histogram3D res;
        res.bins_r = newR; res.bins_g = newG; res.bins_b = newB;
        size_t sz = static_cast<size_t>(newR) * newG * newB;
        res.pdf.assign(sz, 0.0);
        const int br = bins_r, bg = bins_g, bb = bins_b;
        const double invNewR = 1.0 / newR;
        const double invNewG = 1.0 / newG;
        const double invNewB = 1.0 / newB;
        const double* pdf_ptr = pdf.data();

        #pragma omp parallel for collapse(3) schedule(static)
        for (int bi = 0; bi < newB; ++bi) {
            for (int gi = 0; gi < newG; ++gi) {
                for (int ri = 0; ri < newR; ++ri) {
                    double b = (bi + 0.5) * invNewB;
                    double g = (gi + 0.5) * invNewG;
                    double r = (ri + 0.5) * invNewR;
                    double r_pos = r * br - 0.5;
                    double g_pos = g * bg - 0.5;
                    double b_pos = b * bb - 0.5;
                    int r0 = static_cast<int>(std::floor(r_pos));
                    int g0 = static_cast<int>(std::floor(g_pos));
                    int b0 = static_cast<int>(std::floor(b_pos));
                    double fr = r_pos - r0;
                    double fg = g_pos - g0;
                    double fb = b_pos - b0;
                    auto get = [&](int rr, int gg, int bb_) -> double {
                        if (rr < 0 || rr >= br || gg < 0 || gg >= bg || bb_ < 0 || bb_ >= bb) return 0.0;
                        return pdf_ptr[(bb_ * bg + gg) * br + rr];
                    };
                    double val = 0.0;
                    val += (1-fr)*(1-fg)*(1-fb) * get(r0,   g0,   b0);
                    val +=    fr *(1-fg)*(1-fb) * get(r0+1, g0,   b0);
                    val += (1-fr)*   fg *(1-fb) * get(r0,   g0+1, b0);
                    val +=    fr *   fg *(1-fb) * get(r0+1, g0+1, b0);
                    val += (1-fr)*(1-fg)*   fb  * get(r0,   g0,   b0+1);
                    val +=    fr *(1-fg)*   fb  * get(r0+1, g0,   b0+1);
                    val += (1-fr)*   fg *   fb  * get(r0,   g0+1, b0+1);
                    val +=    fr *   fg *   fb  * get(r0+1, g0+1, b0+1);
                    res.pdf[(bi * newG + gi) * newR + ri] = val;
                }
            }
        }
        double sum = 0.0;
        for (double v : res.pdf) sum += v;
        if (sum > 0.0) {
            for (double& v : res.pdf) v /= sum;
        }
        return res;
    }

    void match3D_IDT(const Histogram3D& target, std::vector<double>& r_out,
                     std::vector<double>& g_out, std::vector<double>& b_out, int num_iter) const {
        int N = bins_r;
        int total = N * N * N;
        double total_src = 0.0, total_tgt = 0.0;
        for (double p : pdf) total_src += p;
        for (double p : target.pdf) total_tgt += p;

        auto marginal = [&](const std::vector<double>& hist, int N) {
            std::vector<double> marg_r(N,0), marg_g(N,0), marg_b(N,0);
            for (int b = 0; b < N; ++b)
                for (int g = 0; g < N; ++g)
                    for (int r = 0; r < N; ++r) {
                        double p = hist[(b*N + g)*N + r];
                        marg_r[r] += p;
                        marg_g[g] += p;
                        marg_b[b] += p;
                    }
            return std::tuple{marg_r, marg_g, marg_b};
        };

        auto [src_marg_r, src_marg_g, src_marg_b] = marginal(pdf, N);
        auto [tgt_marg_r, tgt_marg_g, tgt_marg_b] = marginal(target.pdf, N);

        Histogram1D tgt_cdf_r(tgt_marg_r, N, total_tgt);
        Histogram1D tgt_cdf_g(tgt_marg_g, N, total_tgt);
        Histogram1D tgt_cdf_b(tgt_marg_b, N, total_tgt);

        std::vector<double> mapping_r(total), mapping_g(total), mapping_b(total);
        for (int b = 0; b < N; ++b) {
            double bnorm = (b + 0.5) / N;
            for (int g = 0; g < N; ++g) {
                double gnorm = (g + 0.5) / N;
                for (int r = 0; r < N; ++r) {
                    double rnorm = (r + 0.5) / N;
                    int idx = (b * N + g) * N + r;
                    mapping_r[idx] = rnorm;
                    mapping_g[idx] = gnorm;
                    mapping_b[idx] = bnorm;
                }
            }
        }

        std::vector<std::tuple<double, double, int>> items(total);
        std::vector<double> new_coord(total);
        for (int iter = 0; iter < num_iter; ++iter) {
            for (int axis = 0; axis < 3; ++axis) {
                for (int idx = 0; idx < total; ++idx) {
                    double val;
                    if (axis == 0) val = mapping_r[idx];
                    else if (axis == 1) val = mapping_g[idx];
                    else val = mapping_b[idx];
                    items[idx] = {val, pdf[idx], idx};
                }
                std::sort(items.begin(), items.end(),
                    [](const auto& a, const auto& b) { return std::get<0>(a) < std::get<0>(b); });
                double cum = 0.0;
                for (size_t i = 0; i < items.size(); ++i) {
                    cum += std::get<1>(items[i]);
                    double percentile = cum / total_src;
                    double new_c;
                    if (axis == 0) new_c = tgt_cdf_r.invertCDF(percentile);
                    else if (axis == 1) new_c = tgt_cdf_g.invertCDF(percentile);
                    else new_c = tgt_cdf_b.invertCDF(percentile);
                    new_coord[std::get<2>(items[i])] = new_c;
                }
                for (int idx = 0; idx < total; ++idx) {
                    if (axis == 0) mapping_r[idx] = new_coord[idx];
                    else if (axis == 1) mapping_g[idx] = new_coord[idx];
                    else mapping_b[idx] = new_coord[idx];
                }
            }
        }
        r_out.swap(mapping_r);
        g_out.swap(mapping_g);
        b_out.swap(mapping_b);
    }

    void match3D_JointCDF(const Histogram3D& target, std::vector<double>& r_out,
                          std::vector<double>& g_out, std::vector<double>& b_out) const {
        if (bins_r != target.bins_r || bins_g != target.bins_g || bins_b != target.bins_b)
            throw std::runtime_error("Histogram3D::match3D_JointCDF: grid size mismatch");
        int total = bins_r * bins_g * bins_b;
        std::vector<double> src_cdf(total + 1, 0.0);
        std::vector<double> tgt_cdf(total + 1, 0.0);
        for (int i = 0; i < total; ++i) {
            src_cdf[i+1] = src_cdf[i] + pdf[i];
            tgt_cdf[i+1] = tgt_cdf[i] + target.pdf[i];
        }
        double src_sum = src_cdf.back();
        double tgt_sum = tgt_cdf.back();
        if (src_sum > 0.0)
            for (int i = 1; i <= total; ++i) src_cdf[i] /= src_sum;
        if (tgt_sum > 0.0)
            for (int i = 1; i <= total; ++i) tgt_cdf[i] /= tgt_sum;
        src_cdf[0] = 0.0; tgt_cdf[0] = 0.0;

        std::vector<int> mapping(total);
        for (int i = 0; i < total; ++i) {
            double p = src_cdf[i];
            auto it = std::upper_bound(tgt_cdf.begin(), tgt_cdf.end(), p);
            int j = it - tgt_cdf.begin() - 1;
            j = std::clamp(j, 0, total - 1);
            mapping[i] = j;
        }

        r_out.resize(total);
        g_out.resize(total);
        b_out.resize(total);
        for (int idx = 0; idx < total; ++idx) {
            int j = mapping[idx];
            int r = j % bins_r;
            int g = (j / bins_r) % bins_g;
            int b = j / (bins_r * bins_g);
            r_out[idx] = (r + 0.5) / bins_r;
            g_out[idx] = (g + 0.5) / bins_g;
            b_out[idx] = (b + 0.5) / bins_b;
        }
    }
};

void resample_lut(int old_size,
                  const std::vector<double>& r_map,
                  const std::vector<double>& g_map,
                  const std::vector<double>& b_map,
                  int new_size,
                  std::vector<double>& r_out,
                  std::vector<double>& g_out,
                  std::vector<double>& b_out) {
    r_out.assign(new_size * new_size * new_size, 0.0);
    g_out.assign(new_size * new_size * new_size, 0.0);
    b_out.assign(new_size * new_size * new_size, 0.0);

    const double inv_new = 1.0 / new_size;
    #pragma omp parallel for collapse(3) schedule(static)
    for (int bi = 0; bi < new_size; ++bi) {
        for (int gi = 0; gi < new_size; ++gi) {
            for (int ri = 0; ri < new_size; ++ri) {
                double b = (bi + 0.5) * inv_new;
                double g = (gi + 0.5) * inv_new;
                double r = (ri + 0.5) * inv_new;

                double r_pos = r * old_size - 0.5;
                double g_pos = g * old_size - 0.5;
                double b_pos = b * old_size - 0.5;

                int r0 = static_cast<int>(std::floor(r_pos));
                int g0 = static_cast<int>(std::floor(g_pos));
                int b0 = static_cast<int>(std::floor(b_pos));
                double fr = r_pos - r0;
                double fg = g_pos - g0;
                double fb = b_pos - b0;

                auto get = [&](int rr, int gg, int bb) -> std::tuple<double,double,double> {
                    rr = std::clamp(rr, 0, old_size-1);
                    gg = std::clamp(gg, 0, old_size-1);
                    bb = std::clamp(bb, 0, old_size-1);
                    size_t idx = (bb * old_size + gg) * old_size + rr;
                    return {r_map[idx], g_map[idx], b_map[idx]};
                };

                auto [v000_r, v000_g, v000_b] = get(r0,   g0,   b0);
                auto [v100_r, v100_g, v100_b] = get(r0+1, g0,   b0);
                auto [v010_r, v010_g, v010_b] = get(r0,   g0+1, b0);
                auto [v110_r, v110_g, v110_b] = get(r0+1, g0+1, b0);
                auto [v001_r, v001_g, v001_b] = get(r0,   g0,   b0+1);
                auto [v101_r, v101_g, v101_b] = get(r0+1, g0,   b0+1);
                auto [v011_r, v011_g, v011_b] = get(r0,   g0+1, b0+1);
                auto [v111_r, v111_g, v111_b] = get(r0+1, g0+1, b0+1);

                double r_val = 
                    (1-fr)*(1-fg)*(1-fb) * v000_r +
                       fr *(1-fg)*(1-fb) * v100_r +
                    (1-fr)*   fg *(1-fb) * v010_r +
                       fr *   fg *(1-fb) * v110_r +
                    (1-fr)*(1-fg)*   fb  * v001_r +
                       fr *(1-fg)*   fb  * v101_r +
                    (1-fr)*   fg *   fb  * v011_r +
                       fr *   fg *   fb  * v111_r;

                double g_val = 
                    (1-fr)*(1-fg)*(1-fb) * v000_g +
                       fr *(1-fg)*(1-fb) * v100_g +
                    (1-fr)*   fg *(1-fb) * v010_g +
                       fr *   fg *(1-fb) * v110_g +
                    (1-fr)*(1-fg)*   fb  * v001_g +
                       fr *(1-fg)*   fb  * v101_g +
                    (1-fr)*   fg *   fb  * v011_g +
                       fr *   fg *   fb  * v111_g;

                double b_val = 
                    (1-fr)*(1-fg)*(1-fb) * v000_b +
                       fr *(1-fg)*(1-fb) * v100_b +
                    (1-fr)*   fg *(1-fb) * v010_b +
                       fr *   fg *(1-fb) * v110_b +
                    (1-fr)*(1-fg)*   fb  * v001_b +
                       fr *(1-fg)*   fb  * v101_b +
                    (1-fr)*   fg *   fb  * v011_b +
                       fr *   fg *   fb  * v111_b;

                size_t idx_out = (bi * new_size + gi) * new_size + ri;
                r_out[idx_out] = std::clamp(r_val, 0.0, 1.0);
                g_out[idx_out] = std::clamp(g_val, 0.0, 1.0);
                b_out[idx_out] = std::clamp(b_val, 0.0, 1.0);
            }
        }
    }
}

struct VideoData {
    int bitdepth = 8;
    int max_val = 255;
    bool has_rgb_1d = false;
    bool has_rgb_3d = false;
    bool has_rgb_moments = false;
    int rgb_1d_bins = 0;
    std::vector<uint64_t> rgb_1d_r, rgb_1d_g, rgb_1d_b;
    int rgb_3d_bins = 0;
    std::vector<uint64_t> rgb_3d;
    uint64_t rgb_moments_pixel_count = 0;
    double rgb_moments_sum[3] = {0,0,0};
    double rgb_moments_sum2[3] = {0,0,0};
    double rgb_moments_sum3[3] = {0,0,0};

    double mean_r() const { return rgb_moments_pixel_count ? rgb_moments_sum[0] / rgb_moments_pixel_count : 0.0; }
    double mean_g() const { return rgb_moments_pixel_count ? rgb_moments_sum[1] / rgb_moments_pixel_count : 0.0; }
    double mean_b() const { return rgb_moments_pixel_count ? rgb_moments_sum[2] / rgb_moments_pixel_count : 0.0; }
    double var_r() const {
        if (rgb_moments_pixel_count < 2) return 1.0;
        double mean = mean_r();
        return (rgb_moments_sum2[0] - 2*mean*rgb_moments_sum[0] + rgb_moments_pixel_count*mean*mean) / (rgb_moments_pixel_count - 1);
    }
    double var_g() const { double mean = mean_g(); return (rgb_moments_sum2[1] - 2*mean*rgb_moments_sum[1] + rgb_moments_pixel_count*mean*mean) / (rgb_moments_pixel_count - 1); }
    double var_b() const { double mean = mean_b(); return (rgb_moments_sum2[2] - 2*mean*rgb_moments_sum[2] + rgb_moments_pixel_count*mean*mean) / (rgb_moments_pixel_count - 1); }
    double std_r() const { return std::sqrt(var_r()); }
    double std_g() const { return std::sqrt(var_g()); }
    double std_b() const { return std::sqrt(var_b()); }
};

VideoData read_json(const std::string& filename) {
    VideoData data;
    std::ifstream f(filename);
    if (!f.is_open()) throw std::runtime_error("Cannot open " + filename);
    json j;
    f >> j;

    if (j.contains("metadata") && j["metadata"].contains("video") &&
        j["metadata"]["video"].contains("original_bitdepth")) {
        data.bitdepth = j["metadata"]["video"]["original_bitdepth"];
        data.max_val = (1 << data.bitdepth) - 1;
    }

    if (j.contains("metadata") && j["metadata"].contains("extraction") &&
        j["metadata"]["extraction"].contains("parameters")) {
        auto& params = j["metadata"]["extraction"]["parameters"];
        if (params.contains("rgb")) {
            auto& rgb = params["rgb"];
            if (rgb.contains("1d_bins")) data.rgb_1d_bins = rgb["1d_bins"];
            if (rgb.contains("3d_bins")) data.rgb_3d_bins = rgb["3d_bins"];
        }
    }

    if (j.contains("data")) {
        auto& data_obj = j["data"];
        if (data_obj.contains("rgb")) {
            auto& rgb = data_obj["rgb"];
            if (rgb.contains("r")) {
                data.rgb_1d_r = rgb["r"].get<std::vector<uint64_t>>();
                data.has_rgb_1d = true;
                if (data.rgb_1d_bins == 0 && !data.rgb_1d_r.empty())
                    data.rgb_1d_bins = data.rgb_1d_r.size();
            }
            if (rgb.contains("g")) {
                data.rgb_1d_g = rgb["g"].get<std::vector<uint64_t>>();
                if (!data.has_rgb_1d && !data.rgb_1d_g.empty()) data.has_rgb_1d = true;
            }
            if (rgb.contains("b")) {
                data.rgb_1d_b = rgb["b"].get<std::vector<uint64_t>>();
                if (!data.has_rgb_1d && !data.rgb_1d_b.empty()) data.has_rgb_1d = true;
            }
            if (rgb.contains("3d")) {
                data.rgb_3d = rgb["3d"].get<std::vector<uint64_t>>();
                data.has_rgb_3d = true;
                if (data.rgb_3d_bins == 0) {
                    int cube_root = static_cast<int>(std::round(std::pow(data.rgb_3d.size(), 1.0/3.0)));
                    data.rgb_3d_bins = cube_root;
                }
                if (data.rgb_3d_bins > 0) {
                    int bins = data.rgb_3d_bins;
                    std::vector<uint64_t> reordered(data.rgb_3d.size());
                    for (int r = 0; r < bins; ++r) {
                        for (int g = 0; g < bins; ++g) {
                            for (int b = 0; b < bins; ++b) {
                                int src_idx = r * bins * bins + g * bins + b;
                                int dst_idx = b * bins * bins + g * bins + r;
                                reordered[dst_idx] = data.rgb_3d[src_idx];
                            }
                        }
                    }
                    data.rgb_3d = std::move(reordered);
                }
            }
            if (rgb.contains("moments")) {
                auto& mom = rgb["moments"];
                data.rgb_moments_pixel_count = mom.value("pixel_count", 0ULL);
                if (mom.contains("sum")) {
                    data.rgb_moments_sum[0] = mom["sum"].value("R", 0.0);
                    data.rgb_moments_sum[1] = mom["sum"].value("G", 0.0);
                    data.rgb_moments_sum[2] = mom["sum"].value("B", 0.0);
                }
                if (mom.contains("sum2")) {
                    data.rgb_moments_sum2[0] = mom["sum2"].value("R", 0.0);
                    data.rgb_moments_sum2[1] = mom["sum2"].value("G", 0.0);
                    data.rgb_moments_sum2[2] = mom["sum2"].value("B", 0.0);
                }
                if (mom.contains("sum3")) {
                    data.rgb_moments_sum3[0] = mom["sum3"].value("R", 0.0);
                    data.rgb_moments_sum3[1] = mom["sum3"].value("G", 0.0);
                    data.rgb_moments_sum3[2] = mom["sum3"].value("B", 0.0);
                }
                data.has_rgb_moments = true;
            }
        }
    }
    return data;
}

void write_cube_3d_separate(const std::string& filename, int size,
                            const std::vector<double>& r,
                            const std::vector<double>& g,
                            const std::vector<double>& b,
                            const std::string& title = "") {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("Cannot write " + filename);
    out << "TITLE \"" << title << "\"\n";
    out << "LUT_3D_SIZE " << size << "\n";
    out << "DOMAIN_MIN 0.0 0.0 0.0\n";
    out << "DOMAIN_MAX 1.0 1.0 1.0\n";
    out << "#\n";
    out << std::fixed << std::setprecision(6);
    for (size_t i = 0; i < r.size(); ++i) {
        out << r[i] << " " << g[i] << " " << b[i] << "\n";
    }
}

void smooth_lut_separate(std::vector<double>& r, std::vector<double>& g,
                         std::vector<double>& b, int size, double sigma) {
    if (sigma <= 0.0 || size < 3) return;
    int radius = static_cast<int>(std::ceil(2.0 * sigma));
    int kernel_size = 2 * radius + 1;
    std::vector<double> kernel(kernel_size);
    double sum = 0.0;
    for (int i = -radius; i <= radius; ++i) {
        double x = i;
        double val = std::exp(-(x*x) / (2.0 * sigma * sigma));
        kernel[i + radius] = val;
        sum += val;
    }
    for (double& v : kernel) v /= sum;

    auto smooth_channel = [&](std::vector<double>& ch) {
        std::vector<double> tmp = ch;
        #pragma omp parallel for collapse(3) schedule(static)
        for (int b = 0; b < size; ++b) {
            for (int g = 0; g < size; ++g) {
                for (int r = 0; r < size; ++r) {
                    int idx = (b * size + g) * size + r;
                    double val = 0.0;
                    double wsum = 0.0;
                    for (int k = -radius; k <= radius; ++k) {
                        int rr = r + k;
                        if (rr < 0 || rr >= size) continue;
                        int nidx = (b * size + g) * size + rr;
                        double kw = kernel[k + radius];
                        val += kw * tmp[nidx];
                        wsum += kw;
                    }
                    if (wsum > 0.0) ch[idx] = val / wsum;
                }
            }
        }
        tmp = ch;
        #pragma omp parallel for collapse(3) schedule(static)
        for (int b = 0; b < size; ++b) {
            for (int r = 0; r < size; ++r) {
                for (int g = 0; g < size; ++g) {
                    int idx = (b * size + g) * size + r;
                    double val = 0.0;
                    double wsum = 0.0;
                    for (int k = -radius; k <= radius; ++k) {
                        int gg = g + k;
                        if (gg < 0 || gg >= size) continue;
                        int nidx = (b * size + gg) * size + r;
                        double kw = kernel[k + radius];
                        val += kw * tmp[nidx];
                        wsum += kw;
                    }
                    if (wsum > 0.0) ch[idx] = val / wsum;
                }
            }
        }
        tmp = ch;
        #pragma omp parallel for collapse(3) schedule(static)
        for (int g = 0; g < size; ++g) {
            for (int r = 0; r < size; ++r) {
                for (int b = 0; b < size; ++b) {
                    int idx = (b * size + g) * size + r;
                    double val = 0.0;
                    double wsum = 0.0;
                    for (int k = -radius; k <= radius; ++k) {
                        int bb = b + k;
                        if (bb < 0 || bb >= size) continue;
                        int nidx = (bb * size + g) * size + r;
                        double kw = kernel[k + radius];
                        val += kw * tmp[nidx];
                        wsum += kw;
                    }
                    if (wsum > 0.0) ch[idx] = val / wsum;
                }
            }
        }
    };

    smooth_channel(r);
    smooth_channel(g);
    smooth_channel(b);
}

struct Options {
    std::string source_file;
    std::string target_file;
    std::string output_dir = ".";
    int size = 65;
    bool enable_rgb_1d = true;
    bool enable_rgb_3d_idt = true;
    bool enable_rgb_3d_joint = true;
    bool enable_rgb_3d_emd = false;
    bool enable_rgb_3d_sinkhorn = false;
    bool enable_rgb_moments = true;
    bool help = false;
    bool force = false;
    double input_smoothing_value = 0.000001;
    bool input_smoothing_is_percent = false;
    int idt_iterations = 10;
    double sinkhorn_epsilon = 0.01;
    int sinkhorn_iterations = 50;
    int emd_projections = 128;
    double output_smoothing_sigma = 0.0;
};

void print_help(const char* prog) {
    std::cout << "Usage: " << prog << " --source <src.json> --target <tgt.json> [options]\n"
              << "Generate color matching LUTs from source and target JSON files.\n\n"
              << "Required:\n"
              << "  --source <file>             Source JSON file.\n"
              << "  --target <file>             Target (reference) JSON file.\n\n"
              << "Options:\n"
              << "  --output <dir>              Output directory (default: .).\n"
              << "  --size <N>                  LUT size per dimension (default: 65).\n"
              << "  --methods <list>            Comma-separated list of methods to enable.\n"
              << "                              Options: rgb-1d, rgb-moments, rgb-3d-idt,\n"
              << "                                       rgb-3d-joint, rgb-3d-emd, rgb-3d-sinkhorn.\n"
              << "                              Use 'all' to enable everything (default).\n"
              << "  --input-smoothing <value>   Smoothing added to each histogram bin.\n"
              << "                              If value ends with '%', it is treated as a percentage\n"
              << "                              of the average bin count (e.g., 100% means add one\n"
              << "                              average count per bin). Otherwise it is an absolute\n"
              << "                              count added directly to every bin.\n"
              << "                              Default: 0.000001 (absolute).\n"
              << "  --output-smoothing <sigma>  Apply Gaussian smoothing with given sigma to output LUT.\n"
              << "                              (default: 0.0 = no smoothing).\n"
              << "  --idt-iterations <N>        Number of iterations for RGB-3D-IDT (default: 10).\n"
              << "  --emd-projections <N>       Number of random projections for RGB-3D-EMD (default: 128).\n"
              << "  --sinkhorn-epsilon <value>  Entropic regularization strength (default: 0.01).\n"
              << "  --sinkhorn-iterations <N>   Number of Sinkhorn iterations (default: 50).\n"
              << "  --force, -f                 Overwrite existing LUT files.\n"
              << "  --help                      Show this help.\n";
}

Options parse_options(int argc, char** argv) {
    Options opts;
    opts.enable_rgb_1d = false;
    opts.enable_rgb_3d_idt = false;
    opts.enable_rgb_3d_joint = false;
    opts.enable_rgb_3d_emd = false;
    opts.enable_rgb_3d_sinkhorn = false;
    opts.enable_rgb_moments = false;

    static struct option long_options[] = {
        {"source",   required_argument, 0, 1},
        {"target",   required_argument, 0, 2},
        {"output",   required_argument, 0, 3},
        {"size",     required_argument, 0, 4},
        {"methods",  required_argument, 0, 'm'},
        {"input-smoothing", required_argument, 0, 5},
        {"output-smoothing", required_argument, 0, 6},
        {"idt-iterations", required_argument, 0, 7},
        {"emd-projections", required_argument, 0, 8},
        {"sinkhorn-epsilon", required_argument, 0, 9},
        {"sinkhorn-iterations", required_argument, 0, 10},
        {"force",    no_argument,       0, 'f'},
        {"help",     no_argument,       0, 'h'},
        {0,0,0,0}
    };

    int opt;
    int option_index = 0;
    std::string methods_str;
    bool methods_provided = false;

    while ((opt = getopt_long(argc, argv, "hfm:", long_options, &option_index)) != -1) {
        switch (opt) {
            case 1: opts.source_file = optarg; break;
            case 2: opts.target_file = optarg; break;
            case 3: opts.output_dir = optarg; break;
            case 4: opts.size = std::stoi(optarg); if (opts.size < 2) opts.size = 2; break;
            case 'm': methods_str = optarg; methods_provided = true; break;
            case 5: {
                std::string arg = optarg;
                if (!arg.empty() && arg.back() == '%') {
                    opts.input_smoothing_is_percent = true;
                    arg.pop_back();
                }
                opts.input_smoothing_value = std::stod(arg);
                break;
            }
            case 6: opts.output_smoothing_sigma = std::stod(optarg); break;
            case 7: opts.idt_iterations = std::stoi(optarg); if (opts.idt_iterations < 1) opts.idt_iterations = 1; break;
            case 8: opts.emd_projections = std::stoi(optarg); if (opts.emd_projections < 1) opts.emd_projections = 1; break;
            case 9: opts.sinkhorn_epsilon = std::stod(optarg); break;
            case 10: opts.sinkhorn_iterations = std::stoi(optarg); if (opts.sinkhorn_iterations < 1) opts.sinkhorn_iterations = 1; break;
            case 'f': opts.force = true; break;
            case 'h': opts.help = true; break;
            default: break;
        }
    }

    if (opts.help || opts.source_file.empty() || opts.target_file.empty()) {
        print_help(argv[0]);
        exit(opts.help ? 0 : 1);
    }

    if (!methods_provided) {
        opts.enable_rgb_1d = true;
        opts.enable_rgb_3d_idt = true;
        opts.enable_rgb_3d_joint = true;
        opts.enable_rgb_3d_emd = true;
        opts.enable_rgb_3d_sinkhorn = true;
        opts.enable_rgb_moments = true;
    } else {
        std::stringstream ss(methods_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            token.erase(0, token.find_first_not_of(" \t\r\n"));
            token.erase(token.find_last_not_of(" \t\r\n") + 1);
            if (token == "all") {
                opts.enable_rgb_1d = true;
                opts.enable_rgb_3d_idt = true;
                opts.enable_rgb_3d_joint = true;
                opts.enable_rgb_3d_emd = true;
                opts.enable_rgb_3d_sinkhorn = true;
                opts.enable_rgb_moments = true;
            } else if (token == "rgb-1d") {
                opts.enable_rgb_1d = true;
            } else if (token == "rgb-3d-idt") {
                opts.enable_rgb_3d_idt = true;
            } else if (token == "rgb-3d-joint") {
                opts.enable_rgb_3d_joint = true;
            } else if (token == "rgb-3d-emd") {
                opts.enable_rgb_3d_emd = true;
            } else if (token == "rgb-3d-sinkhorn") {
                opts.enable_rgb_3d_sinkhorn = true;
            } else if (token == "rgb-moments") {
                opts.enable_rgb_moments = true;
            } else {
                std::cerr << "Warning: unknown method '" << token << "' ignored.\n";
            }
        }
    }
    return opts;
}

void print_info_table(const VideoData& src, const VideoData& tgt, const Options& opts) {
    const int label_width = 24;
    const int col_width   = 20;
    auto print_row = [&](const std::string& label,
                         const std::string& src_val,
                         const std::string& tgt_val) {
        std::cout << std::left << std::setw(label_width) << label
                  << std::setw(col_width) << src_val
                  << std::setw(col_width) << tgt_val << "\n";
    };
    auto print_pair = [&](const std::string& label, const std::string& value) {
        std::cout << std::left << std::setw(label_width) << label << value << "\n";
    };
    auto print_sub = [&](const std::string& sub_label,
                         const std::string& src_val,
                         const std::string& tgt_val) {
        print_row("- " + sub_label, src_val, tgt_val);
    };

    auto feat_str = [](bool present, int bins = 0) -> std::string {
        if (!present) return "no";
        if (bins > 0) return "yes (" + std::to_string(bins) + " bins)";
        return "yes";
    };

    print_pair("Source", opts.source_file);
    print_pair("Target", opts.target_file);
    
    std::cout << "\n";
    print_row("", "Source", "Target");
    print_row("", "------", "------");

    print_row("Bit depth", std::to_string(src.bitdepth), std::to_string(tgt.bitdepth));
    print_row("Max value", std::to_string(src.max_val), std::to_string(tgt.max_val));

    std::cout << "\nData\n";
    print_sub("RGB 1D",
              feat_str(src.has_rgb_1d, src.rgb_1d_bins),
              feat_str(tgt.has_rgb_1d, tgt.rgb_1d_bins));
    print_sub("RGB 3D",
              feat_str(src.has_rgb_3d, src.rgb_3d_bins),
              feat_str(tgt.has_rgb_3d, tgt.rgb_3d_bins));
    print_sub("RGB Moments",
              feat_str(src.has_rgb_moments),
              feat_str(tgt.has_rgb_moments));

    std::cout << "\nEnabled methods\n";
    auto method_str = [](bool enabled, bool available) -> std::string {
        if (!enabled) return "disabled";
        return available ? "enabled (data present)" : "enabled (data missing)";
    };
    auto print_param = [&](const std::string& label, const std::string& value) {
        std::cout << std::left << std::setw(label_width) << label << value << "\n";
    };
    print_param("- RGB 1D",       method_str(opts.enable_rgb_1d,      src.has_rgb_1d && tgt.has_rgb_1d));
    print_param("- RGB 3D Joint", method_str(opts.enable_rgb_3d_joint, src.has_rgb_3d && tgt.has_rgb_3d));
    print_param("- RGB 3D IDT",   method_str(opts.enable_rgb_3d_idt,      src.has_rgb_3d && tgt.has_rgb_3d));
    print_param("- RGB 3D EMD",   method_str(opts.enable_rgb_3d_emd, src.has_rgb_3d && tgt.has_rgb_3d));
    print_param("- RGB 3D Sinkhorn", method_str(opts.enable_rgb_3d_sinkhorn, src.has_rgb_3d && tgt.has_rgb_3d));
    print_param("- RGB Moments",  method_str(opts.enable_rgb_moments, src.has_rgb_moments && tgt.has_rgb_moments));

    std::cout << "\n---- Parameters ----\n\n";
    std::string smoothing_str = opts.input_smoothing_is_percent ? 
                                std::to_string(opts.input_smoothing_value) + "%" : 
                                std::to_string(opts.input_smoothing_value);
    print_param("Input smoothing", smoothing_str);
    print_param("Output smoothing", std::to_string(opts.output_smoothing_sigma));
    print_param("LUT size", std::to_string(opts.size) + "^3");
    print_param("IDT iterations", std::to_string(opts.idt_iterations));
    print_param("EMD projections", std::to_string(opts.emd_projections));
    print_param("Sinkhorn epsilon", std::to_string(opts.sinkhorn_epsilon));
    print_param("Sinkhorn iterations", std::to_string(opts.sinkhorn_iterations));

    std::cout << "\n---- Output ----\n\n";
    print_param("Output directory", opts.output_dir);
    std::cout << std::endl;
}

std::string basename_no_ext(const std::string& path) {
    fs::path p(path);
    return p.stem().string();
}

int main(int argc, char** argv) {
    try {
        Options opts = parse_options(argc, argv);
        fs::create_directories(opts.output_dir);
        VideoData src = read_json(opts.source_file);
        VideoData tgt = read_json(opts.target_file);
        print_info_table(src, tgt, opts);

        auto filename = [&](const std::string& method) -> std::string {
            fs::path out(opts.output_dir);
            out /= basename_no_ext(opts.source_file) + "_" +
                   basename_no_ext(opts.target_file) + "_" + method + ".cube";
            return out.string();
        };

        struct GenerationTask {
            std::string name;
            std::string filename;
            std::function<void()> generate;
        };
        std::vector<GenerationTask> tasks;

        auto compute_added_per_bin = [&](uint64_t total_pixels, int num_bins) -> double {
            if (num_bins == 0) return 0.0;
            if (opts.input_smoothing_is_percent) {
                double factor = opts.input_smoothing_value / 100.0;
                double avg = static_cast<double>(total_pixels) / num_bins;
                return factor * avg;
            } else {
                return opts.input_smoothing_value;
            }
        };

        if (opts.enable_rgb_1d && src.has_rgb_1d && tgt.has_rgb_1d) {
            tasks.push_back({"RGB 1D", filename("rgb-1d"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB 1D LUT...\n";
                auto total_r_src = std::accumulate(src.rgb_1d_r.begin(), src.rgb_1d_r.end(), 0ULL);
                auto total_g_src = std::accumulate(src.rgb_1d_g.begin(), src.rgb_1d_g.end(), 0ULL);
                auto total_b_src = std::accumulate(src.rgb_1d_b.begin(), src.rgb_1d_b.end(), 0ULL);
                auto total_r_tgt = std::accumulate(tgt.rgb_1d_r.begin(), tgt.rgb_1d_r.end(), 0ULL);
                auto total_g_tgt = std::accumulate(tgt.rgb_1d_g.begin(), tgt.rgb_1d_g.end(), 0ULL);
                auto total_b_tgt = std::accumulate(tgt.rgb_1d_b.begin(), tgt.rgb_1d_b.end(), 0ULL);

                double add_r_src = compute_added_per_bin(total_r_src, src.rgb_1d_bins);
                double add_g_src = compute_added_per_bin(total_g_src, src.rgb_1d_bins);
                double add_b_src = compute_added_per_bin(total_b_src, src.rgb_1d_bins);
                double add_r_tgt = compute_added_per_bin(total_r_tgt, tgt.rgb_1d_bins);
                double add_g_tgt = compute_added_per_bin(total_g_tgt, tgt.rgb_1d_bins);
                double add_b_tgt = compute_added_per_bin(total_b_tgt, tgt.rgb_1d_bins);

                Histogram1D histR_src(src.rgb_1d_r, src.rgb_1d_bins, add_r_src);
                Histogram1D histG_src(src.rgb_1d_g, src.rgb_1d_bins, add_g_src);
                Histogram1D histB_src(src.rgb_1d_b, src.rgb_1d_bins, add_b_src);
                Histogram1D histR_tgt(tgt.rgb_1d_r, tgt.rgb_1d_bins, add_r_tgt);
                Histogram1D histG_tgt(tgt.rgb_1d_g, tgt.rgb_1d_bins, add_g_tgt);
                Histogram1D histB_tgt(tgt.rgb_1d_b, tgt.rgb_1d_bins, add_b_tgt);

                std::vector<double> lutR(opts.size), lutG(opts.size), lutB(opts.size);
                for (int i = 0; i < opts.size; ++i) {
                    double x = i / double(opts.size - 1);
                    double p = histR_src.getCDF(x);
                    lutR[i] = histR_tgt.invertCDF(p);
                    p = histG_src.getCDF(x);
                    lutG[i] = histG_tgt.invertCDF(p);
                    p = histB_src.getCDF(x);
                    lutB[i] = histB_tgt.invertCDF(p);
                }

                int N = opts.size;
                size_t N3 = N * N * N;
                std::vector<double> r(N3), g(N3), b(N3);
                #pragma omp parallel for collapse(3) schedule(static)
                for (int bi = 0; bi < N; ++bi) {
                    for (int gi = 0; gi < N; ++gi) {
                        for (int ri = 0; ri < N; ++ri) {
                            size_t idx = (bi * N + gi) * N + ri;
                            r[idx] = lutR[ri];
                            g[idx] = lutG[gi];
                            b[idx] = lutB[bi];
                        }
                    }
                }
                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, N, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-1d"), N, r, g, b, "RGB 1D separable LUT");
            }});
        }

        if (opts.enable_rgb_3d_idt && src.has_rgb_3d && tgt.has_rgb_3d) {
            tasks.push_back({"RGB 3D IDT", filename("rgb-3d-idt"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB 3D LUT (IDT)...\n";

                int sb = src.rgb_3d_bins;
                int tb = tgt.rgb_3d_bins;
                int common_bins = std::max(sb, tb);

                auto total_src = std::accumulate(src.rgb_3d.begin(), src.rgb_3d.end(), 0ULL);
                auto total_tgt = std::accumulate(tgt.rgb_3d.begin(), tgt.rgb_3d.end(), 0ULL);
                double add_src = compute_added_per_bin(total_src, sb * sb * sb);
                double add_tgt = compute_added_per_bin(total_tgt, tb * tb * tb);

                Histogram3D h3_src_orig(src.rgb_3d, sb, sb, sb, add_src);
                Histogram3D h3_tgt_orig(tgt.rgb_3d, tb, tb, tb, add_tgt);

                auto src_res = h3_src_orig.resample(common_bins, common_bins, common_bins);
                auto tgt_res = h3_tgt_orig.resample(common_bins, common_bins, common_bins);

                std::vector<double> r_map, g_map, b_map;
                src_res.match3D_IDT(tgt_res, r_map, g_map, b_map, opts.idt_iterations);

                std::vector<double> r, g, b;
                resample_lut(common_bins, r_map, g_map, b_map, opts.size, r, g, b);

                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, opts.size, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-3d-idt"), opts.size, r, g, b, "RGB 3D IDT LUT");
            }});
        }

        if (opts.enable_rgb_3d_joint && src.has_rgb_3d && tgt.has_rgb_3d) {
            tasks.push_back({"RGB 3D Joint", filename("rgb-3d-joint"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB 3D LUT (Joint CDF)...\n";

                int sb = src.rgb_3d_bins;
                int tb = tgt.rgb_3d_bins;
                int common_bins = std::max(sb, tb);

                auto total_src = std::accumulate(src.rgb_3d.begin(), src.rgb_3d.end(), 0ULL);
                auto total_tgt = std::accumulate(tgt.rgb_3d.begin(), tgt.rgb_3d.end(), 0ULL);
                double add_src = compute_added_per_bin(total_src, sb * sb * sb);
                double add_tgt = compute_added_per_bin(total_tgt, tb * tb * tb);

                Histogram3D h3_src_orig(src.rgb_3d, sb, sb, sb, add_src);
                Histogram3D h3_tgt_orig(tgt.rgb_3d, tb, tb, tb, add_tgt);

                auto src_res = h3_src_orig.resample(common_bins, common_bins, common_bins);
                auto tgt_res = h3_tgt_orig.resample(common_bins, common_bins, common_bins);

                std::vector<double> r_map, g_map, b_map;
                src_res.match3D_JointCDF(tgt_res, r_map, g_map, b_map);

                std::vector<double> r, g, b;
                resample_lut(common_bins, r_map, g_map, b_map, opts.size, r, g, b);

                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, opts.size, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-3d-joint"), opts.size, r, g, b, "RGB 3D Joint CDF LUT");
            }});
        }

        if (opts.enable_rgb_3d_emd && src.has_rgb_3d && tgt.has_rgb_3d) {
            tasks.push_back({"RGB 3D EMD", filename("rgb-3d-emd"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB 3D LUT (EMD, Sliced Wasserstein)...\n";
                const int num_directions = opts.emd_projections;
                const int proj_bins = 1024;

                std::vector<std::array<double,3>> src_centers, tgt_centers;
                std::vector<double> src_weights, tgt_weights;
                int sb = src.rgb_3d_bins;
                for (int b = 0; b < sb; ++b) {
                    double b_norm = (b + 0.5) / sb;
                    for (int g = 0; g < sb; ++g) {
                        double g_norm = (g + 0.5) / sb;
                        for (int r = 0; r < sb; ++r) {
                            double r_norm = (r + 0.5) / sb;
                            int idx = (b * sb + g) * sb + r;
                            uint64_t count = src.rgb_3d[idx];
                            if (count > 0) {
                                src_centers.push_back({r_norm, g_norm, b_norm});
                                src_weights.push_back(static_cast<double>(count));
                            }
                        }
                    }
                }
                int tb = tgt.rgb_3d_bins;
                for (int b = 0; b < tb; ++b) {
                    double b_norm = (b + 0.5) / tb;
                    for (int g = 0; g < tb; ++g) {
                        double g_norm = (g + 0.5) / tb;
                        for (int r = 0; r < tb; ++r) {
                            double r_norm = (r + 0.5) / tb;
                            int idx = (b * tb + g) * tb + r;
                            uint64_t count = tgt.rgb_3d[idx];
                            if (count > 0) {
                                tgt_centers.push_back({r_norm, g_norm, b_norm});
                                tgt_weights.push_back(static_cast<double>(count));
                            }
                        }
                    }
                }

                std::random_device rd;
                std::vector<std::array<double,3>> directions(num_directions);
                #pragma omp parallel
                {
                    std::mt19937 gen(rd() + omp_get_thread_num());
                    std::normal_distribution<double> nd(0.0, 1.0);
                    #pragma omp for schedule(static)
                    for (int k = 0; k < num_directions; ++k) {
                        double x = nd(gen), y = nd(gen), z = nd(gen);
                        double norm = std::sqrt(x*x + y*y + z*z);
                        directions[k] = {x/norm, y/norm, z/norm};
                    }
                }

                struct ProjMap {
                    double min_val, max_val;
                    Histogram1D src_hist, tgt_hist;
                };
                std::vector<ProjMap> proj_maps(num_directions);

                #pragma omp parallel for schedule(static)
                for (int k = 0; k < num_directions; ++k) {
                    auto& d = directions[k];
                    std::vector<double> src_proj(src_centers.size());
                    double min_src = 1e100, max_src = -1e100;
                    for (size_t i = 0; i < src_centers.size(); ++i) {
                        double p = d[0]*src_centers[i][0] + d[1]*src_centers[i][1] + d[2]*src_centers[i][2];
                        src_proj[i] = p;
                        if (p < min_src) min_src = p;
                        if (p > max_src) max_src = p;
                    }
                    std::vector<double> tgt_proj(tgt_centers.size());
                    double min_tgt = 1e100, max_tgt = -1e100;
                    for (size_t i = 0; i < tgt_centers.size(); ++i) {
                        double p = d[0]*tgt_centers[i][0] + d[1]*tgt_centers[i][1] + d[2]*tgt_centers[i][2];
                        tgt_proj[i] = p;
                        if (p < min_tgt) min_tgt = p;
                        if (p > max_tgt) max_tgt = p;
                    }
                    double min_all = std::min(min_src, min_tgt);
                    double max_all = std::max(max_src, max_tgt);
                    double range = max_all - min_all;
                    if (range < 1e-12) range = 1.0;
                    double inv_range = 1.0 / range;
                    double scale = proj_bins * inv_range;

                    std::vector<uint64_t> src_counts(proj_bins, 0), tgt_counts(proj_bins, 0);
                    for (size_t i = 0; i < src_proj.size(); ++i) {
                        int bin = static_cast<int>((src_proj[i] - min_all) * scale);
                        bin = std::clamp(bin, 0, proj_bins - 1);
                        src_counts[bin] += static_cast<uint64_t>(src_weights[i]);
                    }
                    for (size_t i = 0; i < tgt_proj.size(); ++i) {
                        int bin = static_cast<int>((tgt_proj[i] - min_all) * scale);
                        bin = std::clamp(bin, 0, proj_bins - 1);
                        tgt_counts[bin] += static_cast<uint64_t>(tgt_weights[i]);
                    }

                    double total_src_w = std::accumulate(src_weights.begin(), src_weights.end(), 0.0);
                    double total_tgt_w = std::accumulate(tgt_weights.begin(), tgt_weights.end(), 0.0);
                    double add_src = compute_added_per_bin(static_cast<uint64_t>(total_src_w), proj_bins);
                    double add_tgt = compute_added_per_bin(static_cast<uint64_t>(total_tgt_w), proj_bins);

                    proj_maps[k].min_val = min_all;
                    proj_maps[k].max_val = max_all;
                    proj_maps[k].src_hist = Histogram1D(src_counts, proj_bins, add_src);
                    proj_maps[k].tgt_hist = Histogram1D(tgt_counts, proj_bins, add_tgt);
                }

                double A[3][3] = {{0,0,0},{0,0,0},{0,0,0}};
                for (int k = 0; k < num_directions; ++k) {
                    for (int i = 0; i < 3; ++i)
                        for (int j = 0; j < 3; ++j)
                            A[i][j] += directions[k][i] * directions[k][j];
                }
                double det = A[0][0]*(A[1][1]*A[2][2] - A[1][2]*A[2][1])
                           - A[0][1]*(A[1][0]*A[2][2] - A[1][2]*A[2][0])
                           + A[0][2]*(A[1][0]*A[2][1] - A[1][1]*A[2][0]);
                if (std::abs(det) < 1e-12) det = 1.0;
                double invA[3][3];
                invA[0][0] = (A[1][1]*A[2][2] - A[1][2]*A[2][1]) / det;
                invA[0][1] = (A[0][2]*A[2][1] - A[0][1]*A[2][2]) / det;
                invA[0][2] = (A[0][1]*A[1][2] - A[0][2]*A[1][1]) / det;
                invA[1][0] = (A[1][2]*A[2][0] - A[1][0]*A[2][2]) / det;
                invA[1][1] = (A[0][0]*A[2][2] - A[0][2]*A[2][0]) / det;
                invA[1][2] = (A[0][2]*A[1][0] - A[0][0]*A[1][2]) / det;
                invA[2][0] = (A[1][0]*A[2][1] - A[1][1]*A[2][0]) / det;
                invA[2][1] = (A[0][1]*A[2][0] - A[0][0]*A[2][1]) / det;
                invA[2][2] = (A[0][0]*A[1][1] - A[0][1]*A[1][0]) / det;

                std::array<std::vector<double>, 3> P;
                for (int i = 0; i < 3; ++i) {
                    P[i].resize(num_directions, 0.0);
                    for (int k = 0; k < num_directions; ++k) {
                        double sum = 0;
                        for (int j = 0; j < 3; ++j) sum += invA[i][j] * directions[k][j];
                        P[i][k] = sum;
                    }
                }

                int N = opts.size;
                size_t N3 = N * N * N;
                std::vector<double> r(N3), g(N3), b(N3);
                #pragma omp parallel for collapse(3) schedule(static)
                for (int bi = 0; bi < N; ++bi) {
                    double b_in = bi / double(N - 1);
                    for (int gi = 0; gi < N; ++gi) {
                        double g_in = gi / double(N - 1);
                        for (int ri = 0; ri < N; ++ri) {
                            double r_in = ri / double(N - 1);
                            std::array<double,3> x = {r_in, g_in, b_in};
                            std::vector<double> q(num_directions);
                            for (int k = 0; k < num_directions; ++k) {
                                double p = directions[k][0]*x[0] + directions[k][1]*x[1] + directions[k][2]*x[2];
                                double u = (p - proj_maps[k].min_val) / (proj_maps[k].max_val - proj_maps[k].min_val);
                                double v = proj_maps[k].tgt_hist.invertCDF(proj_maps[k].src_hist.getCDF(u));
                                q[k] = proj_maps[k].min_val + v * (proj_maps[k].max_val - proj_maps[k].min_val);
                            }
                            double r_out = 0, g_out = 0, b_out = 0;
                            for (int k = 0; k < num_directions; ++k) {
                                r_out += P[0][k] * q[k];
                                g_out += P[1][k] * q[k];
                                b_out += P[2][k] * q[k];
                            }
                            size_t idx = (bi * N + gi) * N + ri;
                            r[idx] = std::clamp(r_out, 0.0, 1.0);
                            g[idx] = std::clamp(g_out, 0.0, 1.0);
                            b[idx] = std::clamp(b_out, 0.0, 1.0);
                        }
                    }
                }
                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, N, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-3d-emd"), N, r, g, b, "RGB 3D EMD LUT");
            }});
        }

        if (opts.enable_rgb_3d_sinkhorn && src.has_rgb_3d && tgt.has_rgb_3d) {
            tasks.push_back({"RGB 3D Sinkhorn", filename("rgb-3d-sinkhorn"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB 3D LUT (Sinkhorn)...\n";

                int sb = src.rgb_3d_bins;
                int tb = tgt.rgb_3d_bins;
                int common_bins = std::max(sb, tb);

                auto total_src = std::accumulate(src.rgb_3d.begin(), src.rgb_3d.end(), 0ULL);
                auto total_tgt = std::accumulate(tgt.rgb_3d.begin(), tgt.rgb_3d.end(), 0ULL);
                double add_src = compute_added_per_bin(total_src, sb * sb * sb);
                double add_tgt = compute_added_per_bin(total_tgt, tb * tb * tb);

                Histogram3D h3_src_orig(src.rgb_3d, sb, sb, sb, add_src);
                Histogram3D h3_tgt_orig(tgt.rgb_3d, tb, tb, tb, add_tgt);

                auto src_res = h3_src_orig.resample(common_bins, common_bins, common_bins);
                auto tgt_res = h3_tgt_orig.resample(common_bins, common_bins, common_bins);

                int N = common_bins;
                size_t N3 = static_cast<size_t>(N) * N * N;

                const std::vector<double>& a = src_res.pdf;
                const std::vector<double>& b = tgt_res.pdf;

                std::vector<double> kernel(N * N, 0.0);
                double inv_eps = 1.0 / opts.sinkhorn_epsilon;
                for (int i = 0; i < N; ++i) {
                    double xi = (i + 0.5) / N;
                    for (int j = 0; j < N; ++j) {
                        double xj = (j + 0.5) / N;
                        double diff = xi - xj;
                        kernel[i * N + j] = std::exp(-diff * diff * inv_eps);
                    }
                }

                std::vector<double> temp(N3), temp2(N3), Kv(N3), Ku(N3);
                std::vector<double> u(N3, 1.0), v(N3, 1.0);

                auto apply_kernel = [&](const std::vector<double>& src, int dim,
                                         std::vector<double>& dst,
                                         std::vector<double>& tmp1,
                                         std::vector<double>& tmp2) {
                    if (dim == 0) {
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int b = 0; b < N; ++b) {
                            for (int g = 0; g < N; ++g) {
                                for (int r_out = 0; r_out < N; ++r_out) {
                                    double sum = 0.0;
                                    for (int r_in = 0; r_in < N; ++r_in) {
                                        size_t idx_in = (b * N + g) * N + r_in;
                                        sum += kernel[r_out * N + r_in] * src[idx_in];
                                    }
                                    size_t idx_out = (b * N + g) * N + r_out;
                                    dst[idx_out] = sum;
                                }
                            }
                        }
                    } else if (dim == 1) {
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int b = 0; b < N; ++b) {
                            for (int r = 0; r < N; ++r) {
                                for (int g_out = 0; g_out < N; ++g_out) {
                                    double sum = 0.0;
                                    for (int g_in = 0; g_in < N; ++g_in) {
                                        size_t idx_in = (b * N + g_in) * N + r;
                                        sum += kernel[g_out * N + g_in] * src[idx_in];
                                    }
                                    size_t idx_out = (b * N + g_out) * N + r;
                                    dst[idx_out] = sum;
                                }
                            }
                        }
                    } else {
                        #pragma omp parallel for collapse(2) schedule(static)
                        for (int g = 0; g < N; ++g) {
                            for (int r = 0; r < N; ++r) {
                                for (int b_out = 0; b_out < N; ++b_out) {
                                    double sum = 0.0;
                                    for (int b_in = 0; b_in < N; ++b_in) {
                                        size_t idx_in = (b_in * N + g) * N + r;
                                        sum += kernel[b_out * N + b_in] * src[idx_in];
                                    }
                                    size_t idx_out = (b_out * N + g) * N + r;
                                    dst[idx_out] = sum;
                                }
                            }
                        }
                    }
                };

                for (int iter = 0; iter < opts.sinkhorn_iterations; ++iter) {
                    apply_kernel(v, 0, temp, temp, temp2);
                    apply_kernel(temp, 1, temp2, temp, temp2);
                    apply_kernel(temp2, 2, Kv, temp, temp2);
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < N3; ++i) {
                        if (a[i] == 0.0) u[i] = 0.0;
                        else u[i] = a[i] / Kv[i];
                    }
                    apply_kernel(u, 0, temp, temp, temp2);
                    apply_kernel(temp, 1, temp2, temp, temp2);
                    apply_kernel(temp2, 2, Ku, temp, temp2);
                    #pragma omp parallel for schedule(static)
                    for (size_t i = 0; i < N3; ++i) {
                        if (b[i] == 0.0) v[i] = 0.0;
                        else v[i] = b[i] / Ku[i];
                    }
                }

                apply_kernel(v, 0, temp, temp, temp2);
                apply_kernel(temp, 1, temp2, temp, temp2);
                apply_kernel(temp2, 2, Kv, temp, temp2);

                std::vector<double> v_r(N3, 0.0), v_g(N3, 0.0), v_b(N3, 0.0);
                #pragma omp parallel for collapse(3) schedule(static)
                for (int b_idx = 0; b_idx < N; ++b_idx) {
                    double b_coord = (b_idx + 0.5) / N;
                    for (int g_idx = 0; g_idx < N; ++g_idx) {
                        double g_coord = (g_idx + 0.5) / N;
                        for (int r_idx = 0; r_idx < N; ++r_idx) {
                            double r_coord = (r_idx + 0.5) / N;
                            size_t idx = (b_idx * N + g_idx) * N + r_idx;
                            v_r[idx] = v[idx] * r_coord;
                            v_g[idx] = v[idx] * g_coord;
                            v_b[idx] = v[idx] * b_coord;
                        }
                    }
                }

                std::vector<double> Kv_r(N3), Kv_g(N3), Kv_b(N3);
                apply_kernel(v_r, 0, temp, temp, temp2);
                apply_kernel(temp, 1, temp2, temp, temp2);
                apply_kernel(temp2, 2, Kv_r, temp, temp2);

                apply_kernel(v_g, 0, temp, temp, temp2);
                apply_kernel(temp, 1, temp2, temp, temp2);
                apply_kernel(temp2, 2, Kv_g, temp, temp2);

                apply_kernel(v_b, 0, temp, temp, temp2);
                apply_kernel(temp, 1, temp2, temp, temp2);
                apply_kernel(temp2, 2, Kv_b, temp, temp2);

                std::vector<double> r_map(N3), g_map(N3), b_map(N3);
                #pragma omp parallel for collapse(3) schedule(static)
                for (int b_idx = 0; b_idx < N; ++b_idx) {
                    for (int g_idx = 0; g_idx < N; ++g_idx) {
                        for (int r_idx = 0; r_idx < N; ++r_idx) {
                            size_t idx = (b_idx * N + g_idx) * N + r_idx;
                            double denom = u[idx] * Kv[idx];
                            if (denom > 0.0) {
                                r_map[idx] = u[idx] * Kv_r[idx] / denom;
                                g_map[idx] = u[idx] * Kv_g[idx] / denom;
                                b_map[idx] = u[idx] * Kv_b[idx] / denom;
                            } else {
                                r_map[idx] = (r_idx + 0.5) / N;
                                g_map[idx] = (g_idx + 0.5) / N;
                                b_map[idx] = (b_idx + 0.5) / N;
                            }
                        }
                    }
                }

                std::vector<double> r, g, b;
                resample_lut(N, r_map, g_map, b_map, opts.size, r, g, b);

                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, opts.size, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-3d-sinkhorn"), opts.size, r, g, b, "RGB 3D Sinkhorn LUT");
            }});
        }

        if (opts.enable_rgb_moments && src.has_rgb_moments && tgt.has_rgb_moments) {
            tasks.push_back({"RGB Moments", filename("rgb-moments"), [&]() {
                #pragma omp critical(output)
                std::cout << "Generating RGB Moments LUT...\n";
                double mean_src_r = src.mean_r() / src.max_val;
                double mean_src_g = src.mean_g() / src.max_val;
                double mean_src_b = src.mean_b() / src.max_val;
                double std_src_r  = src.std_r() / src.max_val;
                double std_src_g  = src.std_g() / src.max_val;
                double std_src_b  = src.std_b() / src.max_val;
                double mean_tgt_r = tgt.mean_r() / tgt.max_val;
                double mean_tgt_g = tgt.mean_g() / tgt.max_val;
                double mean_tgt_b = tgt.mean_b() / tgt.max_val;
                double std_tgt_r  = tgt.std_r() / tgt.max_val;
                double std_tgt_g  = tgt.std_g() / tgt.max_val;
                double std_tgt_b  = tgt.std_b() / tgt.max_val;

                std::vector<double> lutR(opts.size), lutG(opts.size), lutB(opts.size);
                for (int i = 0; i < opts.size; ++i) {
                    double x = i / double(opts.size - 1);
                    lutR[i] = std::clamp((x - mean_src_r) / std_src_r * std_tgt_r + mean_tgt_r, 0.0, 1.0);
                    lutG[i] = std::clamp((x - mean_src_g) / std_src_g * std_tgt_g + mean_tgt_g, 0.0, 1.0);
                    lutB[i] = std::clamp((x - mean_src_b) / std_src_b * std_tgt_b + mean_tgt_b, 0.0, 1.0);
                }

                int N = opts.size;
                size_t N3 = N * N * N;
                std::vector<double> r(N3), g(N3), b(N3);
                #pragma omp parallel for collapse(3) schedule(static)
                for (int bi = 0; bi < N; ++bi) {
                    for (int gi = 0; gi < N; ++gi) {
                        for (int ri = 0; ri < N; ++ri) {
                            size_t idx = (bi * N + gi) * N + ri;
                            r[idx] = lutR[ri];
                            g[idx] = lutG[gi];
                            b[idx] = lutB[bi];
                        }
                    }
                }
                if (opts.output_smoothing_sigma > 0.0) {
                    smooth_lut_separate(r, g, b, N, opts.output_smoothing_sigma);
                }
                write_cube_3d_separate(filename("rgb-moments"), N, r, g, b, "RGB Moments linear LUT");
            }});
        }

        #pragma omp parallel
        #pragma omp single
        for (const auto& task : tasks) {
            #pragma omp task
            {
                if (fs::exists(task.filename) && !opts.force) {
                    #pragma omp critical
                    std::cout << "Skipping " << task.name << " LUT (file exists: " << task.filename << ")\n";
                } else {
                    if (fs::exists(task.filename) && opts.force) {
                        #pragma omp critical
                        std::cout << "Warning: overwriting existing file: " << task.filename << "\n";
                    }
                    task.generate();
                }
            }
        }

        std::cout << "All requested LUTs processed.\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}