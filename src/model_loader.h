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
    std::string token_embedding_path;
    std::string positional_embedding_path;
    std::string q_proj_path;
    std::string k_proj_path;
    std::string v_proj_path;
    std::string o_proj_path;
    std::string ffn_up_path;
    std::string ffn_down_path;
};

struct EmbeddingWeights {
    ModelConfig config;
    std::vector<float> token_embedding_table;
    std::vector<float> positional_embedding_table;
};

struct AttentionWeights {
    ModelConfig config;
    std::vector<float> q_proj_weight;
    std::vector<float> k_proj_weight;
    std::vector<float> v_proj_weight;
    std::vector<float> o_proj_weight;
};

struct FFNWeights {
    ModelConfig config;
    std::vector<float> ffn_up_weight;
    std::vector<float> ffn_down_weight;
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
        extract_json_string(text, "token_embedding_path"),
        extract_json_string(text, "positional_embedding_path"),
        extract_json_string(text, "q_proj_path"),
        extract_json_string(text, "k_proj_path"),
        extract_json_string(text, "v_proj_path"),
        extract_json_string(text, "o_proj_path"),
        extract_json_string(text, "ffn_up_path"),
        extract_json_string(text, "ffn_down_path"),
    };

    if (config.d_model <= 0 || config.max_seq_len <= 0 || config.vocab_size <= 0 ||
        config.n_heads <= 0 || config.ffn_hidden <= 0) {
        std::cerr << "Config dimensions must be positive\n";
        std::exit(1);
    }

    if (config.d_model % config.n_heads != 0) {
        std::cerr << "d_model must be divisible by n_heads\n";
        std::exit(1);
    }

    return config;
}

inline EmbeddingWeights load_embedding_weights(const char* config_path) {
    const ModelConfig config = load_model_config(config_path);
    const std::filesystem::path base_dir = std::filesystem::path(config_path).parent_path();
    const std::filesystem::path token_path = base_dir / config.token_embedding_path;
    const std::filesystem::path pos_path = base_dir / config.positional_embedding_path;

    EmbeddingWeights weights{
        config,
        load_floats(token_path),
        load_floats(pos_path),
    };

    const std::size_t expected_token_values =
        static_cast<std::size_t>(config.vocab_size) * static_cast<std::size_t>(config.d_model);
    const std::size_t expected_pos_values =
        static_cast<std::size_t>(config.max_seq_len) * static_cast<std::size_t>(config.d_model);

    if (weights.token_embedding_table.size() != expected_token_values) {
        std::cerr << "token embedding table size mismatch\n";
        std::exit(1);
    }

    if (weights.positional_embedding_table.size() != expected_pos_values) {
        std::cerr << "positional embedding table size mismatch\n";
        std::exit(1);
    }

    return weights;
}

inline FFNWeights load_ffn_weights(const char* config_path) {
    const ModelConfig config = load_model_config(config_path);
    const std::filesystem::path base_dir = std::filesystem::path(config_path).parent_path();

    FFNWeights weights{
        config,
        load_floats(base_dir / config.ffn_up_path),
        load_floats(base_dir / config.ffn_down_path),
    };

    const std::size_t expected_up_values =
        static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.ffn_hidden);
    const std::size_t expected_down_values =
        static_cast<std::size_t>(config.ffn_hidden) * static_cast<std::size_t>(config.d_model);

    if (weights.ffn_up_weight.size() != expected_up_values ||
        weights.ffn_down_weight.size() != expected_down_values) {
        std::cerr << "ffn projection size mismatch\n";
        std::exit(1);
    }

    return weights;
}

inline AttentionWeights load_attention_weights(const char* config_path) {
    const ModelConfig config = load_model_config(config_path);
    const std::filesystem::path base_dir = std::filesystem::path(config_path).parent_path();

    AttentionWeights weights{
        config,
        load_floats(base_dir / config.q_proj_path),
        load_floats(base_dir / config.k_proj_path),
        load_floats(base_dir / config.v_proj_path),
        load_floats(base_dir / config.o_proj_path),
    };

    const std::size_t expected_proj_values =
        static_cast<std::size_t>(config.d_model) * static_cast<std::size_t>(config.d_model);

    if (weights.q_proj_weight.size() != expected_proj_values ||
        weights.k_proj_weight.size() != expected_proj_values ||
        weights.v_proj_weight.size() != expected_proj_values ||
        weights.o_proj_weight.size() != expected_proj_values) {
        std::cerr << "attention projection size mismatch\n";
        std::exit(1);
    }

    return weights;
}
