#pragma once

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

struct ModelConfig {
    int d_model;
    int max_seq_len;
    int vocab_size;
    int n_heads;
    int ffn_hidden;
    int n_layers;
    int tie_word_embeddings;
    std::string token_embedding_path;
    std::string positional_embedding_path;
    std::string final_norm_weight_path;
    std::string final_norm_bias_path;
    std::string lm_head_path;
};

struct TransformerLayerWeights {
    std::vector<float> ln1_weight;
    std::vector<float> ln1_bias;
    std::vector<float> q_proj_weight;
    std::vector<float> q_proj_bias;
    std::vector<float> k_proj_weight;
    std::vector<float> k_proj_bias;
    std::vector<float> v_proj_weight;
    std::vector<float> v_proj_bias;
    std::vector<float> o_proj_weight;
    std::vector<float> o_proj_bias;
    std::vector<float> ln2_weight;
    std::vector<float> ln2_bias;
    std::vector<float> ffn_up_weight;
    std::vector<float> ffn_up_bias;
    std::vector<float> ffn_down_weight;
    std::vector<float> ffn_down_bias;
};

struct GPT2Weights {
    ModelConfig config;
    std::vector<float> token_embedding_table;
    std::vector<float> positional_embedding_table;
    std::vector<float> final_norm_weight;
    std::vector<float> final_norm_bias;
    std::vector<float> lm_head_weight;
    std::vector<TransformerLayerWeights> layers;
};

inline std::string read_text_file(const char* path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << '\n';
        std::exit(1);
    }

    return std::string((std::istreambuf_iterator<char>(file)),
                       std::istreambuf_iterator<char>());
}

inline std::string extract_json_string(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        std::cerr << "Missing key in config: " << key << '\n';
        std::exit(1);
    }

    const std::size_t colon_pos = text.find(':', key_pos);
    const std::size_t first_quote = text.find('"', colon_pos + 1);
    const std::size_t second_quote = text.find('"', first_quote + 1);
    if (colon_pos == std::string::npos || first_quote == std::string::npos ||
        second_quote == std::string::npos) {
        std::cerr << "Invalid string value for key: " << key << '\n';
        std::exit(1);
    }

    return text.substr(first_quote + 1, second_quote - first_quote - 1);
}

inline int extract_json_int(const std::string& text, const std::string& key) {
    const std::string needle = "\"" + key + "\"";
    const std::size_t key_pos = text.find(needle);
    if (key_pos == std::string::npos) {
        std::cerr << "Missing key in config: " << key << '\n';
        std::exit(1);
    }

    const std::size_t colon_pos = text.find(':', key_pos);
    const std::size_t value_start = text.find_first_of("-0123456789", colon_pos + 1);
    const std::size_t value_end = text.find_first_not_of("0123456789", value_start);
    if (colon_pos == std::string::npos || value_start == std::string::npos) {
        std::cerr << "Invalid integer value for key: " << key << '\n';
        std::exit(1);
    }

    return std::stoi(text.substr(value_start, value_end - value_start));
}

inline std::vector<int> load_ints(const char* path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << '\n';
        std::exit(1);
    }

    std::vector<int> values;
    int value = 0;
    while (file >> value) {
        values.push_back(value);
    }

    if (values.empty()) {
        std::cerr << "No integer values found in " << path << '\n';
        std::exit(1);
    }

    return values;
}

inline std::vector<float> load_floats(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file) {
        std::cerr << "Failed to open " << path << '\n';
        std::exit(1);
    }

    std::vector<float> values;
    float value = 0.0f;
    while (file >> value) {
        values.push_back(value);
    }

    if (values.empty()) {
        std::cerr << "No float values found in " << path << '\n';
        std::exit(1);
    }

    return values;
}

inline ModelConfig load_model_config(const char* path) {
    const std::string text = read_text_file(path);

    ModelConfig config{
        extract_json_int(text, "d_model"),
        extract_json_int(text, "max_seq_len"),
        extract_json_int(text, "vocab_size"),
        extract_json_int(text, "n_heads"),
        extract_json_int(text, "ffn_hidden"),
        extract_json_int(text, "n_layers"),
        extract_json_int(text, "tie_word_embeddings"),
        extract_json_string(text, "token_embedding_path"),
        extract_json_string(text, "positional_embedding_path"),
        extract_json_string(text, "final_norm_weight_path"),
        extract_json_string(text, "final_norm_bias_path"),
        extract_json_string(text, "lm_head_path"),
    };

    if (config.d_model <= 0 || config.max_seq_len <= 0 || config.vocab_size <= 0 ||
        config.n_heads <= 0 || config.ffn_hidden <= 0 || config.n_layers <= 0) {
        std::cerr << "Config dimensions must be positive\n";
        std::exit(1);
    }

    if (config.d_model % config.n_heads != 0) {
        std::cerr << "d_model must be divisible by n_heads\n";
        std::exit(1);
    }

    return config;
}

inline void expect_size(const std::vector<float>& values, std::size_t expected, const char* label) {
    if (values.size() != expected) {
        std::cerr << label << " size mismatch\n";
        std::exit(1);
    }
}

inline GPT2Weights load_gpt2_weights(const char* config_path) {
    const ModelConfig config = load_model_config(config_path);
    const std::filesystem::path base_dir = std::filesystem::path(config_path).parent_path();

    GPT2Weights weights{
        config,
        load_floats(base_dir / config.token_embedding_path),
        load_floats(base_dir / config.positional_embedding_path),
        load_floats(base_dir / config.final_norm_weight_path),
        load_floats(base_dir / config.final_norm_bias_path),
        {},
        {},
    };

    expect_size(weights.token_embedding_table,
                static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(config.d_model),
                "token embedding table");
    expect_size(weights.positional_embedding_table,
                static_cast<std::size_t>(config.max_seq_len) * static_cast<std::size_t>(config.d_model),
                "positional embedding table");
    expect_size(weights.final_norm_weight, static_cast<std::size_t>(config.d_model), "final norm weight");
    expect_size(weights.final_norm_bias, static_cast<std::size_t>(config.d_model), "final norm bias");

    if (config.tie_word_embeddings == 1) {
        weights.lm_head_weight.resize(
            static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.vocab_size));
        for (int token_id = 0; token_id < config.vocab_size; ++token_id) {
            for (int feature_idx = 0; feature_idx < config.d_model; ++feature_idx) {
                weights.lm_head_weight[feature_idx * config.vocab_size + token_id] =
                    weights.token_embedding_table[token_id * config.d_model + feature_idx];
            }
        }
    } else {
        weights.lm_head_weight = load_floats(base_dir / config.lm_head_path);
    }
    expect_size(weights.lm_head_weight,
                static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.vocab_size),
                "lm head weight");

    weights.layers.reserve(config.n_layers);
    for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
        const std::filesystem::path layer_dir = base_dir / ("layer_" + std::to_string(layer_idx));

        TransformerLayerWeights layer{
            load_floats(layer_dir / "ln1_weight.txt"),
            load_floats(layer_dir / "ln1_bias.txt"),
            load_floats(layer_dir / "q_proj_weight.txt"),
            load_floats(layer_dir / "q_proj_bias.txt"),
            load_floats(layer_dir / "k_proj_weight.txt"),
            load_floats(layer_dir / "k_proj_bias.txt"),
            load_floats(layer_dir / "v_proj_weight.txt"),
            load_floats(layer_dir / "v_proj_bias.txt"),
            load_floats(layer_dir / "o_proj_weight.txt"),
            load_floats(layer_dir / "o_proj_bias.txt"),
            load_floats(layer_dir / "ln2_weight.txt"),
            load_floats(layer_dir / "ln2_bias.txt"),
            load_floats(layer_dir / "ffn_up_weight.txt"),
            load_floats(layer_dir / "ffn_up_bias.txt"),
            load_floats(layer_dir / "ffn_down_weight.txt"),
            load_floats(layer_dir / "ffn_down_bias.txt"),
        };

        expect_size(layer.ln1_weight, static_cast<std::size_t>(config.d_model), "ln1_weight");
        expect_size(layer.ln1_bias, static_cast<std::size_t>(config.d_model), "ln1_bias");
        expect_size(layer.q_proj_weight,
                    static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.d_model),
                    "q_proj_weight");
        expect_size(layer.q_proj_bias, static_cast<std::size_t>(config.d_model), "q_proj_bias");
        expect_size(layer.k_proj_weight,
                    static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.d_model),
                    "k_proj_weight");
        expect_size(layer.k_proj_bias, static_cast<std::size_t>(config.d_model), "k_proj_bias");
        expect_size(layer.v_proj_weight,
                    static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.d_model),
                    "v_proj_weight");
        expect_size(layer.v_proj_bias, static_cast<std::size_t>(config.d_model), "v_proj_bias");
        expect_size(layer.o_proj_weight,
                    static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.d_model),
                    "o_proj_weight");
        expect_size(layer.o_proj_bias, static_cast<std::size_t>(config.d_model), "o_proj_bias");
        expect_size(layer.ln2_weight, static_cast<std::size_t>(config.d_model), "ln2_weight");
        expect_size(layer.ln2_bias, static_cast<std::size_t>(config.d_model), "ln2_bias");
        expect_size(layer.ffn_up_weight,
                    static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.ffn_hidden),
                    "ffn_up_weight");
        expect_size(layer.ffn_up_bias, static_cast<std::size_t>(config.ffn_hidden), "ffn_up_bias");
        expect_size(layer.ffn_down_weight,
                    static_cast<std::size_t>(config.ffn_hidden) * static_cast<std::size_t>(config.d_model),
                    "ffn_down_weight");
        expect_size(layer.ffn_down_bias, static_cast<std::size_t>(config.d_model), "ffn_down_bias");

        weights.layers.push_back(std::move(layer));
    }

    return weights;
}
