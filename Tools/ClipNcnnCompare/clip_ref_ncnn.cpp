#define NOMINMAX
#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <regex>
#include <set>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <ncnn/net.h>
#include <ncnn/mat.h>

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

using std::string;
using std::vector;

static std::vector<std::string> load_vocab_lines(const std::string& path)
{
    std::ifstream ifs(path);
    if (!ifs.is_open())
        throw std::runtime_error("failed to open vocab: " + path);

    std::vector<std::string> lines;
    std::string line;
    while (std::getline(ifs, line))
    {
        if (!line.empty())
            lines.push_back(line);
    }
    return lines;
}

static std::set<std::pair<std::string, std::string>> get_pairs(const std::vector<std::string>& word)
{
    std::set<std::pair<std::string, std::string>> pairs;
    if (word.size() < 2)
        return pairs;
    for (size_t i = 0; i + 1 < word.size(); i++)
        pairs.insert({word[i], word[i + 1]});
    return pairs;
}

class SimpleTokenizerRef
{
public:
    SimpleTokenizerRef(const std::string& vocab_path, const std::string& bpe_path, int context_length = 77)
        : context_length_(context_length)
    {
        byte_encoder_ = load_vocab_lines(vocab_path);

        std::ifstream merges_file(bpe_path);
        if (!merges_file.is_open())
            throw std::runtime_error("failed to open merges: " + bpe_path);

        std::vector<std::pair<std::string, std::string>> merges;
        std::string line;
        std::getline(merges_file, line);
        while (std::getline(merges_file, line))
        {
            std::stringstream ss(line);
            std::string first;
            std::string second;
            ss >> first >> second;
            if (!first.empty() && !second.empty())
                merges.push_back({first, second});
        }

        std::vector<std::string> vocab;
        for (const auto& token : byte_encoder_)
            vocab.push_back(token);
        for (const auto& token : byte_encoder_)
            vocab.push_back(token + "</w>");
        for (const auto& merge : merges)
            vocab.push_back(merge.first + merge.second);
        vocab.push_back("<start_of_text>");
        vocab.push_back("<end_of_text>");

        for (size_t i = 0; i < vocab.size(); i++)
            encoder_[vocab[i]] = static_cast<int>(i);
        for (size_t i = 0; i < merges.size(); i++)
            bpe_ranks_[merges[i]] = static_cast<int>(i);

        sot_token_id_ = encoder_.at("<start_of_text>");
        eot_token_id_ = encoder_.at("<end_of_text>");
        pattern_ = std::regex(R"(<start_of_text>|<end_of_text>|'s|'t|'re|'ve|'m|'ll|'d|[a-zA-Z]+|[0-9]+|[^\s\w\d]+)", std::regex::icase);
    }

    std::vector<int> tokenize(const std::string& text) const
    {
        auto cleaned = clean(text);
        std::vector<int> tokens;
        tokens.push_back(sot_token_id_);

        auto it = std::sregex_iterator(cleaned.begin(), cleaned.end(), pattern_);
        auto end = std::sregex_iterator();
        for (; it != end; ++it)
        {
            auto token_str = it->str();
            std::string encoded_token;
            for (unsigned char b : token_str)
            {
            int index = static_cast<int>(b) - 33;
            if (index < 0 || index >= static_cast<int>(byte_encoder_.size()))
                throw std::runtime_error("prompt contains unsupported byte: " + std::to_string((int)b));
                encoded_token += byte_encoder_[index];
            }

            auto bpe_result = bpe(encoded_token);
            std::stringstream ss(bpe_result);
            std::string sub_token;
            while (ss >> sub_token)
            {
                auto found = encoder_.find(sub_token);
                if (found != encoder_.end())
                    tokens.push_back(found->second);
            }
        }

        tokens.push_back(eot_token_id_);
        if ((int)tokens.size() > context_length_)
        {
            tokens.resize(context_length_);
            tokens.back() = eot_token_id_;
        }
        tokens.resize(context_length_, 0);
        return tokens;
    }

    int eot_token_id() const
    {
        return eot_token_id_;
    }

private:
    static std::string clean(const std::string& input)
    {
        std::string out = input;
        out = std::regex_replace(out, std::regex("&lt;"), "<");
        out = std::regex_replace(out, std::regex("&gt;"), ">");
        out = std::regex_replace(out, std::regex("&quot;"), "\"");
        out = std::regex_replace(out, std::regex("&#39;"), "'");
        out = std::regex_replace(out, std::regex("&amp;"), "&");
        out = std::regex_replace(out, std::regex("^\\s+|\\s+$"), "");
        out = std::regex_replace(out, std::regex("\\s+"), " ");
        std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return (char)std::tolower(c); });
        return out;
    }

    std::string bpe(const std::string& token) const
    {
        auto cache_it = cache_.find(token);
        if (cache_it != cache_.end())
            return cache_it->second;

        std::vector<std::string> word;
        for (char c : token)
            word.push_back(std::string(1, c));
        if (word.empty())
            return token;

        word.back() += "</w>";
        auto pairs = get_pairs(word);
        while (!pairs.empty())
        {
            std::pair<std::string, std::string> best_pair;
            int best_rank = (std::numeric_limits<int>::max)();
            bool found = false;

            for (const auto& pair : pairs)
            {
                auto rank_it = bpe_ranks_.find(pair);
                if (rank_it != bpe_ranks_.end() && rank_it->second < best_rank)
                {
                    best_pair = pair;
                    best_rank = rank_it->second;
                    found = true;
                }
            }
            if (!found)
                break;

            std::vector<std::string> merged;
            for (size_t i = 0; i < word.size();)
            {
                if (i + 1 < word.size() && word[i] == best_pair.first && word[i + 1] == best_pair.second)
                {
                    merged.push_back(best_pair.first + best_pair.second);
                    i += 2;
                }
                else
                {
                    merged.push_back(word[i]);
                    i += 1;
                }
            }
            word = merged;
            if (word.size() == 1)
                break;
            pairs = get_pairs(word);
        }

        std::string result;
        for (size_t i = 0; i < word.size(); i++)
        {
            result += word[i];
            if (i + 1 < word.size())
                result += " ";
        }
        cache_[token] = result;
        return result;
    }

    int context_length_;
    std::vector<std::string> byte_encoder_;
    std::unordered_map<std::string, int> encoder_;
    std::map<std::pair<std::string, std::string>, int> bpe_ranks_;
    mutable std::unordered_map<std::string, std::string> cache_;
    std::regex pattern_;
    int sot_token_id_ = 49406;
    int eot_token_id_ = 49407;
};

static void normalize_inplace(std::vector<float>& values)
{
    double sum = 0.0;
    for (float v : values)
        sum += (double)v * (double)v;
    double norm = std::sqrt((std::max)(sum, 1e-24));
    for (float& v : values)
        v = (float)(v / norm);
}

static std::vector<float> encode_image(const std::string& model_root, const std::string& image_path)
{
    ncnn::Net image_encoder;
    image_encoder.load_param((model_root + "/image_encoder.ncnn.param").c_str());
    image_encoder.load_model((model_root + "/image_encoder.ncnn.bin").c_str());

    int w = 0;
    int h = 0;
    int c = 0;
    stbi_uc* pixels = stbi_load(image_path.c_str(), &w, &h, &c, 3);
    if (!pixels)
        throw std::runtime_error("failed to read image: " + image_path);

    const int target_size = 256;
    ncnn::Mat in = ncnn::Mat::from_pixels_resize(pixels, ncnn::Mat::PIXEL_RGB, w, h, target_size, target_size);
    stbi_image_free(pixels);
    float mean_vals[3] = {0.f, 0.f, 0.f};
    float norm_vals[3] = {1.f / 255.f, 1.f / 255.f, 1.f / 255.f};
    in.substract_mean_normalize(mean_vals, norm_vals);

    auto ex = image_encoder.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    std::vector<float> result(out.w * (std::max)(1, out.h));
    memcpy(result.data(), out.data, result.size() * sizeof(float));
    normalize_inplace(result);
    return result;
}

static std::vector<float> encode_text(const std::string& model_root, const std::vector<int>& tokens, int eot_token_id)
{
    ncnn::Net text_encoder;
    ncnn::Net projection;
    text_encoder.load_param((model_root + "/text_encoder.ncnn.param").c_str());
    text_encoder.load_model((model_root + "/text_encoder.ncnn.bin").c_str());
    projection.load_param((model_root + "/projection_layer.ncnn.param").c_str());
    projection.load_model((model_root + "/projection_layer.ncnn.bin").c_str());

    ncnn::Mat in((int)tokens.size());
    for (int i = 0; i < (int)tokens.size(); i++)
        in.row<int>(0)[i] = tokens[i];

    auto ex = text_encoder.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    int eot_index = 0;
    for (int i = 0; i < (int)tokens.size(); i++)
    {
        if (tokens[i] == eot_token_id)
        {
            eot_index = i;
            break;
        }
    }

    ncnn::Mat text_embed = out.row_range(eot_index, 1);
    auto ex2 = projection.create_extractor();
    ex2.input("in0", text_embed);
    ncnn::Mat out2;
    ex2.extract("out0", out2);

    std::vector<float> result(out2.w);
    memcpy(result.data(), out2.data, result.size() * sizeof(float));
    normalize_inplace(result);
    return result;
}

static double dot_product(const std::vector<float>& a, const std::vector<float>& b)
{
    double total = 0.0;
    for (size_t i = 0; i < a.size() && i < b.size(); i++)
        total += (double)a[i] * (double)b[i];
    return total;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        std::cerr << "Usage: clip_ref_ncnn <clip_root> <model_name> <image_path>\n";
        return 1;
    }

    const std::string clip_root = argv[1];
    const std::string model_name = argv[2];
    const std::string image_path = argv[3];
    const std::string model_root = clip_root + "/" + model_name;

    const std::vector<std::pair<std::string, std::string>> labels = {
        {"Portrait", "a portrait photo"},
        {"Landscape", "a landscape photo"},
        {"Night", "a night photo"},
        {"Food", "a photo of food"},
        {"Pet", "a photo of a pet"},
        {"Architecture", "an architecture photo"},
        {"Document", "a photo of a document"},
        {"Group", "a group photo"},
        {"Photo", "a photo"},
    };

    try
    {
        SimpleTokenizerRef tokenizer(clip_root + "/vocab.txt", clip_root + "/bpe_simple_vocab_16e6.txt");
        auto image_features = encode_image(model_root, image_path);

        std::vector<double> logits(labels.size());
        for (size_t i = 0; i < labels.size(); i++)
        {
            auto tokens = tokenizer.tokenize(labels[i].second);
            auto text_features = encode_text(model_root, tokens, tokenizer.eot_token_id());
            logits[i] = dot_product(image_features, text_features) * 100.0;
        }

        double max_logit = *std::max_element(logits.begin(), logits.end());
        double sum = 0.0;
        std::vector<double> probs(labels.size());
        for (size_t i = 0; i < logits.size(); i++)
        {
            probs[i] = std::exp(logits[i] - max_logit);
            sum += probs[i];
        }

        for (size_t i = 0; i < labels.size(); i++)
        {
            probs[i] /= sum;
            std::cout << labels[i].first << "\t" << logits[i] << "\t" << probs[i] << "\t" << labels[i].second << "\n";
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << "ERROR: " << e.what() << "\n";
        return 2;
    }

    return 0;
}
