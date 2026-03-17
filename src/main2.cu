#include <cuda_runtime.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "model_loader.h"
#include "kernels.cu"

namespace {

void check_cuda(cudaError_t err, const char* context) {
    if (err != cudaSuccess) {
        std::cerr << context << ": " << cudaGetErrorString(err) << '\n';
        std::exit(1);
    }
}

int next_power_of_two(int value) {
    int power = 1;
    while (power < value) {
        power <<= 1;
    }
    return power;
}

struct LayerDeviceWeights {
    float* ln1_weight = nullptr;
    float* ln1_bias = nullptr;
    float* q_proj_weight = nullptr;
    float* q_proj_bias = nullptr;
    float* k_proj_weight = nullptr;
    float* k_proj_bias = nullptr;
    float* v_proj_weight = nullptr;
    float* v_proj_bias = nullptr;
    float* o_proj_weight = nullptr;
    float* o_proj_bias = nullptr;
    float* ln2_weight = nullptr;
    float* ln2_bias = nullptr;
    float* ffn_up_weight = nullptr;
    float* ffn_up_bias = nullptr;
    float* ffn_down_weight = nullptr;
    float* ffn_down_bias = nullptr;
};

struct DeviceBuffers {
    int* token_ids = nullptr;

    float* token_embedding_table = nullptr;
    float* positional_embedding_table = nullptr;
    float* final_norm_weight = nullptr;
    float* final_norm_bias = nullptr;
    float* lm_head_weight = nullptr;

    std::vector<LayerDeviceWeights> layers;

    float* hidden = nullptr;
    float* norm = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_scores = nullptr;
    float* attn_context = nullptr;
    float* attn_output = nullptr;
    float* residual = nullptr;
    float* ffn_up = nullptr;
    float* ffn_down = nullptr;
    float* final_norm_output = nullptr;
    float* logits = nullptr;
};

void free_layer(LayerDeviceWeights& layer) {
    cudaFree(layer.ln1_weight);
    cudaFree(layer.ln1_bias);
    cudaFree(layer.q_proj_weight);
    cudaFree(layer.q_proj_bias);
    cudaFree(layer.k_proj_weight);
    cudaFree(layer.k_proj_bias);
    cudaFree(layer.v_proj_weight);
    cudaFree(layer.v_proj_bias);
    cudaFree(layer.o_proj_weight);
    cudaFree(layer.o_proj_bias);
    cudaFree(layer.ln2_weight);
    cudaFree(layer.ln2_bias);
    cudaFree(layer.ffn_up_weight);
    cudaFree(layer.ffn_up_bias);
    cudaFree(layer.ffn_down_weight);
    cudaFree(layer.ffn_down_bias);
}

void free_buffers(DeviceBuffers& buffers) {
    cudaFree(buffers.token_ids);
    cudaFree(buffers.token_embedding_table);
    cudaFree(buffers.positional_embedding_table);
    cudaFree(buffers.final_norm_weight);
    cudaFree(buffers.final_norm_bias);
    cudaFree(buffers.lm_head_weight);
    for (LayerDeviceWeights& layer : buffers.layers) {
        free_layer(layer);
    }
    cudaFree(buffers.hidden);
    cudaFree(buffers.norm);
    cudaFree(buffers.q);
    cudaFree(buffers.k);
    cudaFree(buffers.v);
    cudaFree(buffers.attn_scores);
    cudaFree(buffers.attn_context);
    cudaFree(buffers.attn_output);
    cudaFree(buffers.residual);
    cudaFree(buffers.ffn_up);
    cudaFree(buffers.ffn_down);
    cudaFree(buffers.final_norm_output);
    cudaFree(buffers.logits);
}

void allocate_layer(LayerDeviceWeights& layer, int d_model, int ffn_hidden) {
    check_cuda(cudaMalloc(&layer.ln1_weight, d_model * sizeof(float)), "cudaMalloc ln1_weight");
    check_cuda(cudaMalloc(&layer.ln1_bias, d_model * sizeof(float)), "cudaMalloc ln1_bias");
    check_cuda(cudaMalloc(&layer.q_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc q_proj_weight");
    check_cuda(cudaMalloc(&layer.q_proj_bias, d_model * sizeof(float)), "cudaMalloc q_proj_bias");
    check_cuda(cudaMalloc(&layer.k_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc k_proj_weight");
    check_cuda(cudaMalloc(&layer.k_proj_bias, d_model * sizeof(float)), "cudaMalloc k_proj_bias");
    check_cuda(cudaMalloc(&layer.v_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc v_proj_weight");
    check_cuda(cudaMalloc(&layer.v_proj_bias, d_model * sizeof(float)), "cudaMalloc v_proj_bias");
    check_cuda(cudaMalloc(&layer.o_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc o_proj_weight");
    check_cuda(cudaMalloc(&layer.o_proj_bias, d_model * sizeof(float)), "cudaMalloc o_proj_bias");
    check_cuda(cudaMalloc(&layer.ln2_weight, d_model * sizeof(float)), "cudaMalloc ln2_weight");
    check_cuda(cudaMalloc(&layer.ln2_bias, d_model * sizeof(float)), "cudaMalloc ln2_bias");
    check_cuda(cudaMalloc(&layer.ffn_up_weight, d_model * ffn_hidden * sizeof(float)),
               "cudaMalloc ffn_up_weight");
    check_cuda(cudaMalloc(&layer.ffn_up_bias, ffn_hidden * sizeof(float)), "cudaMalloc ffn_up_bias");
    check_cuda(cudaMalloc(&layer.ffn_down_weight, ffn_hidden * d_model * sizeof(float)),
               "cudaMalloc ffn_down_weight");
    check_cuda(cudaMalloc(&layer.ffn_down_bias, d_model * sizeof(float)), "cudaMalloc ffn_down_bias");
}

void allocate_buffers(DeviceBuffers& buffers,
                      int max_seq_len,
                      int d_model,
                      int vocab_size,
                      int n_heads,
                      int ffn_hidden,
                      int n_layers) {
    const int max_total = max_seq_len * d_model;
    const int max_ffn_total = max_seq_len * ffn_hidden;
    const int max_scores = n_heads * max_seq_len * max_seq_len;
    const int max_logits = max_seq_len * vocab_size;

    check_cuda(cudaMalloc(&buffers.token_ids, max_seq_len * sizeof(int)), "cudaMalloc token_ids");
    check_cuda(cudaMalloc(&buffers.token_embedding_table, vocab_size * d_model * sizeof(float)),
               "cudaMalloc token_embedding_table");
    check_cuda(cudaMalloc(&buffers.positional_embedding_table, max_seq_len * d_model * sizeof(float)),
               "cudaMalloc positional_embedding_table");
    check_cuda(cudaMalloc(&buffers.final_norm_weight, d_model * sizeof(float)),
               "cudaMalloc final_norm_weight");
    check_cuda(cudaMalloc(&buffers.final_norm_bias, d_model * sizeof(float)),
               "cudaMalloc final_norm_bias");
    check_cuda(cudaMalloc(&buffers.lm_head_weight, d_model * vocab_size * sizeof(float)),
               "cudaMalloc lm_head_weight");

    buffers.layers.resize(n_layers);
    for (LayerDeviceWeights& layer : buffers.layers) {
        allocate_layer(layer, d_model, ffn_hidden);
    }

    check_cuda(cudaMalloc(&buffers.hidden, max_total * sizeof(float)), "cudaMalloc hidden");
    check_cuda(cudaMalloc(&buffers.norm, max_total * sizeof(float)), "cudaMalloc norm");
    check_cuda(cudaMalloc(&buffers.q, max_total * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&buffers.k, max_total * sizeof(float)), "cudaMalloc k");
    check_cuda(cudaMalloc(&buffers.v, max_total * sizeof(float)), "cudaMalloc v");
    check_cuda(cudaMalloc(&buffers.attn_scores, max_scores * sizeof(float)), "cudaMalloc attn_scores");
    check_cuda(cudaMalloc(&buffers.attn_context, max_total * sizeof(float)), "cudaMalloc attn_context");
    check_cuda(cudaMalloc(&buffers.attn_output, max_total * sizeof(float)), "cudaMalloc attn_output");
    check_cuda(cudaMalloc(&buffers.residual, max_total * sizeof(float)), "cudaMalloc residual");
    check_cuda(cudaMalloc(&buffers.ffn_up, max_ffn_total * sizeof(float)), "cudaMalloc ffn_up");
    check_cuda(cudaMalloc(&buffers.ffn_down, max_total * sizeof(float)), "cudaMalloc ffn_down");
    check_cuda(cudaMalloc(&buffers.final_norm_output, max_total * sizeof(float)),
               "cudaMalloc final_norm_output");
    check_cuda(cudaMalloc(&buffers.logits, max_logits * sizeof(float)), "cudaMalloc logits");
}

void copy_layer_weights(const TransformerLayerWeights& src, LayerDeviceWeights& dst,
                        int d_model, int ffn_hidden) {
    check_cuda(cudaMemcpy(dst.ln1_weight, src.ln1_weight.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ln1_weight");
    check_cuda(cudaMemcpy(dst.ln1_bias, src.ln1_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ln1_bias");
    check_cuda(cudaMemcpy(dst.q_proj_weight, src.q_proj_weight.data(), d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy q_proj_weight");
    check_cuda(cudaMemcpy(dst.q_proj_bias, src.q_proj_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy q_proj_bias");
    check_cuda(cudaMemcpy(dst.k_proj_weight, src.k_proj_weight.data(), d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy k_proj_weight");
    check_cuda(cudaMemcpy(dst.k_proj_bias, src.k_proj_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy k_proj_bias");
    check_cuda(cudaMemcpy(dst.v_proj_weight, src.v_proj_weight.data(), d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy v_proj_weight");
    check_cuda(cudaMemcpy(dst.v_proj_bias, src.v_proj_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy v_proj_bias");
    check_cuda(cudaMemcpy(dst.o_proj_weight, src.o_proj_weight.data(), d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy o_proj_weight");
    check_cuda(cudaMemcpy(dst.o_proj_bias, src.o_proj_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy o_proj_bias");
    check_cuda(cudaMemcpy(dst.ln2_weight, src.ln2_weight.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ln2_weight");
    check_cuda(cudaMemcpy(dst.ln2_bias, src.ln2_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ln2_bias");
    check_cuda(cudaMemcpy(dst.ffn_up_weight, src.ffn_up_weight.data(), d_model * ffn_hidden * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_up_weight");
    check_cuda(cudaMemcpy(dst.ffn_up_bias, src.ffn_up_bias.data(), ffn_hidden * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_up_bias");
    check_cuda(cudaMemcpy(dst.ffn_down_weight, src.ffn_down_weight.data(), ffn_hidden * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_down_weight");
    check_cuda(cudaMemcpy(dst.ffn_down_bias, src.ffn_down_bias.data(), d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_down_bias");
}

void copy_weights_to_device(const GPT2Weights& weights, DeviceBuffers& buffers) {
    const int d_model = weights.config.d_model;
    const int vocab_size = weights.config.vocab_size;
    const int max_seq_len = weights.config.max_seq_len;
    const int ffn_hidden = weights.config.ffn_hidden;

    check_cuda(cudaMemcpy(buffers.token_embedding_table, weights.token_embedding_table.data(),
                          vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy token_embedding_table");
    check_cuda(cudaMemcpy(buffers.positional_embedding_table, weights.positional_embedding_table.data(),
                          max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy positional_embedding_table");
    check_cuda(cudaMemcpy(buffers.final_norm_weight, weights.final_norm_weight.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy final_norm_weight");
    check_cuda(cudaMemcpy(buffers.final_norm_bias, weights.final_norm_bias.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy final_norm_bias");
    check_cuda(cudaMemcpy(buffers.lm_head_weight, weights.lm_head_weight.data(),
                          d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy lm_head_weight");

    for (int layer_idx = 0; layer_idx < weights.config.n_layers; ++layer_idx) {
        copy_layer_weights(weights.layers[layer_idx], buffers.layers[layer_idx], d_model, ffn_hidden);
    }
}

void run_forward(const std::vector<int>& token_ids,
                 const ModelConfig& config,
                 DeviceBuffers& buffers,
                 std::vector<float>& logits_out) {
    const int seq_len = static_cast<int>(token_ids.size());
    const int d_model = config.d_model;
    const int vocab_size = config.vocab_size;
    const int n_heads = config.n_heads;
    const int ffn_hidden = config.ffn_hidden;
    const int d_head = d_model / n_heads;
    const int total = seq_len * d_model;
    const int ffn_total = seq_len * ffn_hidden;

    check_cuda(cudaMemcpy(buffers.token_ids, token_ids.data(), seq_len * sizeof(int), cudaMemcpyHostToDevice),
               "cudaMemcpy token_ids");

    const int threads = 256;
    const int total_blocks = (total + threads - 1) / threads;
    const int ffn_blocks = (ffn_total + threads - 1) / threads;
    const dim3 matmul_threads(16, 16);
    const dim3 dmodel_blocks((d_model + 15) / 16, (seq_len + 15) / 16);
    const dim3 ffn_up_blocks((ffn_hidden + 15) / 16, (seq_len + 15) / 16);
    const dim3 logits_blocks((vocab_size + 15) / 16, (seq_len + 15) / 16);

    token_embedding_lookup_kernel<<<total_blocks, threads>>>(
        buffers.token_ids,
        buffers.token_embedding_table,
        buffers.positional_embedding_table,
        buffers.hidden,
        seq_len,
        d_model);
    check_cuda(cudaGetLastError(), "launch embeddings");

    const int layernorm_threads = next_power_of_two(d_model);
    const std::size_t layernorm_shared_bytes =
        static_cast<std::size_t>(layernorm_threads) * 2 * sizeof(float);

    for (int layer_idx = 0; layer_idx < config.n_layers; ++layer_idx) {
        LayerDeviceWeights& layer = buffers.layers[layer_idx];

        layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
            buffers.hidden, layer.ln1_weight, layer.ln1_bias, buffers.norm, seq_len, d_model, 1e-5f);
        check_cuda(cudaGetLastError(), "launch ln1");

        fused_matmul_bias<<<dmodel_blocks, matmul_threads>>>(
            buffers.norm, layer.q_proj_weight, buffers.q, seq_len, d_model, d_model, layer.q_proj_bias);
        fused_matmul_bias<<<dmodel_blocks, matmul_threads>>>(
            buffers.norm, layer.k_proj_weight, buffers.k, seq_len, d_model, d_model, layer.k_proj_bias);
        fused_matmul_bias<<<dmodel_blocks, matmul_threads>>>(
            buffers.norm, layer.v_proj_weight, buffers.v, seq_len, d_model, d_model, layer.v_proj_bias);
        check_cuda(cudaGetLastError(), "launch qkv");

        const dim3 attn_score_threads(16, 16);
        const dim3 attn_score_blocks((seq_len + 15) / 16, (seq_len + 15) / 16, n_heads);
        attention_scores_kernel<<<attn_score_blocks, attn_score_threads>>>(
            buffers.q, buffers.k, buffers.attn_scores, seq_len, d_model, n_heads);
        check_cuda(cudaGetLastError(), "launch attention scores");

        const int softmax_threads = next_power_of_two(seq_len);
        const std::size_t softmax_shared_bytes =
            static_cast<std::size_t>(softmax_threads) * 2 * sizeof(float);
        row_softmax_kernel<<<n_heads * seq_len, softmax_threads, softmax_shared_bytes>>>(
            buffers.attn_scores, n_heads * seq_len, seq_len);
        check_cuda(cudaGetLastError(), "launch attention softmax");

        const dim3 attn_value_threads(16, 16);
        const dim3 attn_value_blocks((d_head + 15) / 16, (seq_len + 15) / 16, n_heads);
        attention_weighted_sum_kernel<<<attn_value_blocks, attn_value_threads>>>(
            buffers.attn_scores, buffers.v, buffers.attn_context, seq_len, d_model, n_heads);
        check_cuda(cudaGetLastError(), "launch attention weighted sum");

        fused_matmul_bias<<<dmodel_blocks, matmul_threads>>>(
            buffers.attn_context, layer.o_proj_weight, buffers.attn_output,
            seq_len, d_model, d_model, layer.o_proj_bias);
        check_cuda(cudaGetLastError(), "launch o_proj");

        residual_add_kernel<<<total_blocks, threads>>>(
            buffers.attn_output, buffers.hidden, buffers.residual, total);
        check_cuda(cudaGetLastError(), "launch attention residual");

        layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
            buffers.residual, layer.ln2_weight, layer.ln2_bias, buffers.norm, seq_len, d_model, 1e-5f);
        check_cuda(cudaGetLastError(), "launch ln2");

        fused_matmul_bias<<<ffn_up_blocks, matmul_threads>>>(
            buffers.norm, layer.ffn_up_weight, buffers.ffn_up,
            seq_len, ffn_hidden, d_model, layer.ffn_up_bias);
        check_cuda(cudaGetLastError(), "launch ffn_up");

        gelu_kernel<<<ffn_blocks, threads>>>(buffers.ffn_up, ffn_total);
        check_cuda(cudaGetLastError(), "launch gelu");

        fused_matmul_bias<<<dmodel_blocks, matmul_threads>>>(
            buffers.ffn_up, layer.ffn_down_weight, buffers.ffn_down,
            seq_len, d_model, ffn_hidden, layer.ffn_down_bias);
        check_cuda(cudaGetLastError(), "launch ffn_down");

        residual_add_kernel<<<total_blocks, threads>>>(
            buffers.ffn_down, buffers.residual, buffers.hidden, total);
        check_cuda(cudaGetLastError(), "launch ffn residual");
    }

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        buffers.hidden, buffers.final_norm_weight, buffers.final_norm_bias,
        buffers.final_norm_output, seq_len, d_model, 1e-5f);
    check_cuda(cudaGetLastError(), "launch final norm");

    matmul_tiled<<<logits_blocks, matmul_threads>>>(
        buffers.final_norm_output, buffers.lm_head_weight, buffers.logits,
        seq_len, vocab_size, d_model);
    check_cuda(cudaGetLastError(), "launch lm_head");

    check_cuda(cudaDeviceSynchronize(), "sync forward");

    logits_out.resize(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(vocab_size));
    check_cuda(cudaMemcpy(logits_out.data(), buffers.logits,
                          seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy logits");
}

int argmax_last_token(const std::vector<float>& logits, int seq_len, int vocab_size) {
    int best_token_id = 0;
    float best = logits[(seq_len - 1) * vocab_size];
    for (int i = 1; i < vocab_size; ++i) {
        const float value = logits[(seq_len - 1) * vocab_size + i];
        if (value > best) {
            best = value;
            best_token_id = i;
        }
    }
    return best_token_id;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <model_config.json> <token_ids.txt> [max_new_tokens]\n";
        return 1;
    }

    const GPT2Weights weights = load_gpt2_weights(argv[1]);
    std::vector<int> token_ids = load_ints(argv[2]);
    const int max_new_tokens = (argc == 4) ? std::atoi(argv[3]) : 0;

    if (weights.config.d_model > 256) {
        std::cerr << "current minimal kernels expect d_model <= 256\n";
        return 1;
    }

    if (static_cast<int>(token_ids.size()) > weights.config.max_seq_len) {
        std::cerr << "seq_len exceeds max_seq_len\n";
        return 1;
    }

    if (static_cast<int>(token_ids.size()) > 256) {
        std::cerr << "current minimal kernels expect active seq_len <= 256\n";
        return 1;
    }

    for (int token_id : token_ids) {
        if (token_id < 0 || token_id >= weights.config.vocab_size) {
            std::cerr << "token id out of range: " << token_id << '\n';
            return 1;
        }
    }

    DeviceBuffers buffers;
    allocate_buffers(buffers,
                     weights.config.max_seq_len,
                     weights.config.d_model,
                     weights.config.vocab_size,
                     weights.config.n_heads,
                     weights.config.ffn_hidden,
                     weights.config.n_layers);
    copy_weights_to_device(weights, buffers);

    std::vector<float> logits;
    for (int step = 0; step <= max_new_tokens; ++step) {
        run_forward(token_ids, weights.config, buffers, logits);
        const int seq_len = static_cast<int>(token_ids.size());

        std::cout << "Current token ids:\n";
        for (int token_id : token_ids) {
            std::cout << token_id << ' ';
        }
        std::cout << "\n\nLast-position logits:\n";
        for (int token_id = 0; token_id < weights.config.vocab_size; ++token_id) {
            std::cout << logits[(seq_len - 1) * weights.config.vocab_size + token_id] << ' ';
        }

        const int next_token_id = argmax_last_token(logits, seq_len, weights.config.vocab_size);
        std::cout << "\nNext token argmax: " << next_token_id << "\n";

        if (step == max_new_tokens || static_cast<int>(token_ids.size()) >= weights.config.max_seq_len ||
            static_cast<int>(token_ids.size()) >= 256) {
            break;
        }

        token_ids.push_back(next_token_id);
        std::cout << "\n";
    }

    std::cout << "\nFinal token ids:\n";
    for (int token_id : token_ids) {
        std::cout << token_id << ' ';
    }
    std::cout << "\n";

    free_buffers(buffers);
    return 0;
}
