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

struct DeviceBuffers {
    int* token_ids = nullptr;

    float* token_embedding_table = nullptr;
    float* positional_embedding_table = nullptr;
    float* layernorm_weight = nullptr;
    float* layernorm_bias = nullptr;
    float* q_proj_weight = nullptr;
    float* k_proj_weight = nullptr;
    float* v_proj_weight = nullptr;
    float* o_proj_weight = nullptr;
    float* ffn_up_weight = nullptr;
    float* ffn_down_weight = nullptr;
    float* final_norm_weight = nullptr;
    float* final_norm_bias = nullptr;
    float* lm_head_weight = nullptr;

    float* embedding_output = nullptr;
    float* pre_attn_norm = nullptr;
    float* q = nullptr;
    float* k = nullptr;
    float* v = nullptr;
    float* attn_scores = nullptr;
    float* attn_context = nullptr;
    float* attn_output = nullptr;
    float* attn_residual = nullptr;
    float* pre_ffn_norm = nullptr;
    float* ffn_up = nullptr;
    float* ffn_down = nullptr;
    float* final_hidden = nullptr;
    float* final_norm_output = nullptr;
    float* logits = nullptr;
};

void free_buffers(DeviceBuffers& buffers) {
    cudaFree(buffers.token_ids);
    cudaFree(buffers.token_embedding_table);
    cudaFree(buffers.positional_embedding_table);
    cudaFree(buffers.layernorm_weight);
    cudaFree(buffers.layernorm_bias);
    cudaFree(buffers.q_proj_weight);
    cudaFree(buffers.k_proj_weight);
    cudaFree(buffers.v_proj_weight);
    cudaFree(buffers.o_proj_weight);
    cudaFree(buffers.ffn_up_weight);
    cudaFree(buffers.ffn_down_weight);
    cudaFree(buffers.final_norm_weight);
    cudaFree(buffers.final_norm_bias);
    cudaFree(buffers.lm_head_weight);
    cudaFree(buffers.embedding_output);
    cudaFree(buffers.pre_attn_norm);
    cudaFree(buffers.q);
    cudaFree(buffers.k);
    cudaFree(buffers.v);
    cudaFree(buffers.attn_scores);
    cudaFree(buffers.attn_context);
    cudaFree(buffers.attn_output);
    cudaFree(buffers.attn_residual);
    cudaFree(buffers.pre_ffn_norm);
    cudaFree(buffers.ffn_up);
    cudaFree(buffers.ffn_down);
    cudaFree(buffers.final_hidden);
    cudaFree(buffers.final_norm_output);
    cudaFree(buffers.logits);
}

void allocate_buffers(DeviceBuffers& buffers,
                      int max_seq_len,
                      int d_model,
                      int vocab_size,
                      int n_heads,
                      int ffn_hidden) {
    const int max_total = max_seq_len * d_model;
    const int max_ffn_total = max_seq_len * ffn_hidden;
    const int max_scores = n_heads * max_seq_len * max_seq_len;
    const int max_logits = max_seq_len * vocab_size;

    check_cuda(cudaMalloc(&buffers.token_ids, max_seq_len * sizeof(int)), "cudaMalloc token_ids");

    check_cuda(cudaMalloc(&buffers.token_embedding_table, vocab_size * d_model * sizeof(float)),
               "cudaMalloc token_embedding_table");
    check_cuda(cudaMalloc(&buffers.positional_embedding_table, max_seq_len * d_model * sizeof(float)),
               "cudaMalloc positional_embedding_table");
    check_cuda(cudaMalloc(&buffers.layernorm_weight, d_model * sizeof(float)),
               "cudaMalloc layernorm_weight");
    check_cuda(cudaMalloc(&buffers.layernorm_bias, d_model * sizeof(float)),
               "cudaMalloc layernorm_bias");
    check_cuda(cudaMalloc(&buffers.q_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc q_proj_weight");
    check_cuda(cudaMalloc(&buffers.k_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc k_proj_weight");
    check_cuda(cudaMalloc(&buffers.v_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc v_proj_weight");
    check_cuda(cudaMalloc(&buffers.o_proj_weight, d_model * d_model * sizeof(float)),
               "cudaMalloc o_proj_weight");
    check_cuda(cudaMalloc(&buffers.ffn_up_weight, d_model * ffn_hidden * sizeof(float)),
               "cudaMalloc ffn_up_weight");
    check_cuda(cudaMalloc(&buffers.ffn_down_weight, ffn_hidden * d_model * sizeof(float)),
               "cudaMalloc ffn_down_weight");
    check_cuda(cudaMalloc(&buffers.final_norm_weight, d_model * sizeof(float)),
               "cudaMalloc final_norm_weight");
    check_cuda(cudaMalloc(&buffers.final_norm_bias, d_model * sizeof(float)),
               "cudaMalloc final_norm_bias");
    check_cuda(cudaMalloc(&buffers.lm_head_weight, d_model * vocab_size * sizeof(float)),
               "cudaMalloc lm_head_weight");

    check_cuda(cudaMalloc(&buffers.embedding_output, max_total * sizeof(float)),
               "cudaMalloc embedding_output");
    check_cuda(cudaMalloc(&buffers.pre_attn_norm, max_total * sizeof(float)),
               "cudaMalloc pre_attn_norm");
    check_cuda(cudaMalloc(&buffers.q, max_total * sizeof(float)), "cudaMalloc q");
    check_cuda(cudaMalloc(&buffers.k, max_total * sizeof(float)), "cudaMalloc k");
    check_cuda(cudaMalloc(&buffers.v, max_total * sizeof(float)), "cudaMalloc v");
    check_cuda(cudaMalloc(&buffers.attn_scores, max_scores * sizeof(float)),
               "cudaMalloc attn_scores");
    check_cuda(cudaMalloc(&buffers.attn_context, max_total * sizeof(float)),
               "cudaMalloc attn_context");
    check_cuda(cudaMalloc(&buffers.attn_output, max_total * sizeof(float)),
               "cudaMalloc attn_output");
    check_cuda(cudaMalloc(&buffers.attn_residual, max_total * sizeof(float)),
               "cudaMalloc attn_residual");
    check_cuda(cudaMalloc(&buffers.pre_ffn_norm, max_total * sizeof(float)),
               "cudaMalloc pre_ffn_norm");
    check_cuda(cudaMalloc(&buffers.ffn_up, max_ffn_total * sizeof(float)),
               "cudaMalloc ffn_up");
    check_cuda(cudaMalloc(&buffers.ffn_down, max_total * sizeof(float)),
               "cudaMalloc ffn_down");
    check_cuda(cudaMalloc(&buffers.final_hidden, max_total * sizeof(float)),
               "cudaMalloc final_hidden");
    check_cuda(cudaMalloc(&buffers.final_norm_output, max_total * sizeof(float)),
               "cudaMalloc final_norm_output");
    check_cuda(cudaMalloc(&buffers.logits, max_logits * sizeof(float)), "cudaMalloc logits");
}

void copy_static_weights_to_device(const EmbeddingWeights& embedding_weights,
                                   const AttentionWeights& attention_weights,
                                   const FFNWeights& ffn_weights,
                                   const OutputWeights& output_weights,
                                   const std::vector<float>& layernorm_weight,
                                   const std::vector<float>& layernorm_bias,
                                   DeviceBuffers& buffers) {
    const int d_model = embedding_weights.config.d_model;
    const int vocab_size = embedding_weights.config.vocab_size;
    const int max_seq_len = embedding_weights.config.max_seq_len;
    const int ffn_hidden = embedding_weights.config.ffn_hidden;

    check_cuda(cudaMemcpy(buffers.token_embedding_table, embedding_weights.token_embedding_table.data(),
                          vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy token_embedding_table");
    check_cuda(cudaMemcpy(buffers.positional_embedding_table, embedding_weights.positional_embedding_table.data(),
                          max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy positional_embedding_table");
    check_cuda(cudaMemcpy(buffers.layernorm_weight, layernorm_weight.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy layernorm_weight");
    check_cuda(cudaMemcpy(buffers.layernorm_bias, layernorm_bias.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy layernorm_bias");
    check_cuda(cudaMemcpy(buffers.q_proj_weight, attention_weights.q_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy q_proj_weight");
    check_cuda(cudaMemcpy(buffers.k_proj_weight, attention_weights.k_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy k_proj_weight");
    check_cuda(cudaMemcpy(buffers.v_proj_weight, attention_weights.v_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy v_proj_weight");
    check_cuda(cudaMemcpy(buffers.o_proj_weight, attention_weights.o_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy o_proj_weight");
    check_cuda(cudaMemcpy(buffers.ffn_up_weight, ffn_weights.ffn_up_weight.data(),
                          d_model * ffn_hidden * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_up_weight");
    check_cuda(cudaMemcpy(buffers.ffn_down_weight, ffn_weights.ffn_down_weight.data(),
                          ffn_hidden * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_down_weight");
    check_cuda(cudaMemcpy(buffers.final_norm_weight, output_weights.final_norm_weight.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy final_norm_weight");
    check_cuda(cudaMemcpy(buffers.final_norm_bias, output_weights.final_norm_bias.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy final_norm_bias");
    check_cuda(cudaMemcpy(buffers.lm_head_weight, output_weights.lm_head_weight.data(),
                          d_model * vocab_size * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy lm_head_weight");
}

void run_forward(const std::vector<int>& token_ids,
                 int d_model,
                 int vocab_size,
                 int n_heads,
                 int ffn_hidden,
                 DeviceBuffers& buffers,
                 std::vector<float>& logits_out) {
    const int seq_len = static_cast<int>(token_ids.size());
    const int total = seq_len * d_model;
    const int ffn_total = seq_len * ffn_hidden;
    const int d_head = d_model / n_heads;

    check_cuda(cudaMemcpy(buffers.token_ids, token_ids.data(), seq_len * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy token_ids");

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const dim3 matmul_threads(16, 16);
    const dim3 dmodel_blocks((d_model + 15) / 16, (seq_len + 15) / 16);
    const dim3 ffn_up_blocks((ffn_hidden + 15) / 16, (seq_len + 15) / 16);
    const dim3 logits_blocks((vocab_size + 15) / 16, (seq_len + 15) / 16);

    token_embedding_lookup_kernel<<<blocks, threads>>>(
        buffers.token_ids,
        buffers.token_embedding_table,
        buffers.positional_embedding_table,
        buffers.embedding_output,
        seq_len,
        d_model);
    check_cuda(cudaGetLastError(), "launch token_embedding_lookup_kernel");

    const int layernorm_threads = next_power_of_two(d_model);
    const std::size_t layernorm_shared_bytes =
        static_cast<std::size_t>(layernorm_threads) * 2 * sizeof(float);

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        buffers.embedding_output,
        buffers.layernorm_weight,
        buffers.layernorm_bias,
        buffers.pre_attn_norm,
        seq_len,
        d_model,
        1e-5f);
    check_cuda(cudaGetLastError(), "launch pre-attn layernorm");

    matmul_tiled<<<dmodel_blocks, matmul_threads>>>(
        buffers.pre_attn_norm, buffers.q_proj_weight, buffers.q, seq_len, d_model, d_model);
    matmul_tiled<<<dmodel_blocks, matmul_threads>>>(
        buffers.pre_attn_norm, buffers.k_proj_weight, buffers.k, seq_len, d_model, d_model);
    matmul_tiled<<<dmodel_blocks, matmul_threads>>>(
        buffers.pre_attn_norm, buffers.v_proj_weight, buffers.v, seq_len, d_model, d_model);
    check_cuda(cudaGetLastError(), "launch qkv projections");

    const dim3 attn_score_threads(16, 16);
    const dim3 attn_score_blocks((seq_len + 15) / 16, (seq_len + 15) / 16, n_heads);
    attention_scores_kernel<<<attn_score_blocks, attn_score_threads>>>(
        buffers.q, buffers.k, buffers.attn_scores, seq_len, d_model, n_heads);
    check_cuda(cudaGetLastError(), "launch attention_scores_kernel");

    const int softmax_threads = next_power_of_two(seq_len);
    const std::size_t softmax_shared_bytes =
        static_cast<std::size_t>(softmax_threads) * 2 * sizeof(float);
    row_softmax_kernel<<<n_heads * seq_len, softmax_threads, softmax_shared_bytes>>>(
        buffers.attn_scores, n_heads * seq_len, seq_len);
    check_cuda(cudaGetLastError(), "launch row_softmax_kernel");

    const dim3 attn_value_threads(16, 16);
    const dim3 attn_value_blocks((d_head + 15) / 16, (seq_len + 15) / 16, n_heads);
    attention_weighted_sum_kernel<<<attn_value_blocks, attn_value_threads>>>(
        buffers.attn_scores, buffers.v, buffers.attn_context, seq_len, d_model, n_heads);
    check_cuda(cudaGetLastError(), "launch attention_weighted_sum_kernel");

    matmul_tiled<<<dmodel_blocks, matmul_threads>>>(
        buffers.attn_context, buffers.o_proj_weight, buffers.attn_output, seq_len, d_model, d_model);
    check_cuda(cudaGetLastError(), "launch o_proj");

    residual_add_kernel<<<blocks, threads>>>(
        buffers.attn_output, buffers.embedding_output, buffers.attn_residual, total);
    check_cuda(cudaGetLastError(), "launch attention residual");

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        buffers.attn_residual,
        buffers.layernorm_weight,
        buffers.layernorm_bias,
        buffers.pre_ffn_norm,
        seq_len,
        d_model,
        1e-5f);
    check_cuda(cudaGetLastError(), "launch pre-ffn layernorm");

    matmul_tiled<<<ffn_up_blocks, matmul_threads>>>(
        buffers.pre_ffn_norm, buffers.ffn_up_weight, buffers.ffn_up, seq_len, ffn_hidden, d_model);
    check_cuda(cudaGetLastError(), "launch ffn_up");

    relu_kernel<<<(ffn_total + 255) / 256, 256>>>(buffers.ffn_up, ffn_total);
    check_cuda(cudaGetLastError(), "launch relu");

    matmul_tiled<<<dmodel_blocks, matmul_threads>>>(
        buffers.ffn_up, buffers.ffn_down_weight, buffers.ffn_down, seq_len, d_model, ffn_hidden);
    check_cuda(cudaGetLastError(), "launch ffn_down");

    residual_add_kernel<<<blocks, threads>>>(
        buffers.ffn_down, buffers.attn_residual, buffers.final_hidden, total);
    check_cuda(cudaGetLastError(), "launch final residual");

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        buffers.final_hidden,
        buffers.final_norm_weight,
        buffers.final_norm_bias,
        buffers.final_norm_output,
        seq_len,
        d_model,
        1e-5f);
    check_cuda(cudaGetLastError(), "launch final norm");

    matmul_tiled<<<logits_blocks, matmul_threads>>>(
        buffers.final_norm_output, buffers.lm_head_weight, buffers.logits, seq_len, vocab_size, d_model);
    check_cuda(cudaGetLastError(), "launch lm_head");

    check_cuda(cudaDeviceSynchronize(), "sync forward pass");

    logits_out.resize(static_cast<std::size_t>(seq_len) * static_cast<std::size_t>(vocab_size));
    check_cuda(cudaMemcpy(logits_out.data(), buffers.logits,
                          seq_len * vocab_size * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy logits");
}

int argmax_last_token(const std::vector<float>& logits, int seq_len, int vocab_size) {
    int best_token_id = 0;
    float best_logit = logits[(seq_len - 1) * vocab_size];
    for (int token_id = 1; token_id < vocab_size; ++token_id) {
        const float value = logits[(seq_len - 1) * vocab_size + token_id];
        if (value > best_logit) {
            best_logit = value;
            best_token_id = token_id;
        }
    }
    return best_token_id;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3 && argc != 4) {
        std::cerr
            << "Usage: " << argv[0]
            << " <model_config.json> <token_ids.txt> [max_new_tokens]\n";
        return 1;
    }

    const EmbeddingWeights embedding_weights = load_embedding_weights(argv[1]);
    const AttentionWeights attention_weights = load_attention_weights(argv[1]);
    const FFNWeights ffn_weights = load_ffn_weights(argv[1]);
    const OutputWeights output_weights =
        load_output_weights(argv[1], embedding_weights.token_embedding_table);

    std::vector<int> token_ids = load_ints(argv[2]);
    const int d_model = embedding_weights.config.d_model;
    const int max_seq_len = embedding_weights.config.max_seq_len;
    const int vocab_size = embedding_weights.config.vocab_size;
    const int n_heads = embedding_weights.config.n_heads;
    const int ffn_hidden = embedding_weights.config.ffn_hidden;
    const int max_new_tokens = (argc == 4) ? std::atoi(argv[3]) : 0;

    if (static_cast<int>(token_ids.size()) > max_seq_len) {
        std::cerr << "seq_len exceeds max_seq_len\n";
        return 1;
    }

    if (d_model > 256 || max_seq_len > 256) {
        std::cerr << "current minimal kernels expect d_model <= 256 and max_seq_len <= 256\n";
        return 1;
    }

    for (int token_id : token_ids) {
        if (token_id < 0 || token_id >= vocab_size) {
            std::cerr << "token id out of range: " << token_id << '\n';
            return 1;
        }
    }

    std::vector<float> layernorm_weight(d_model, 1.0f);
    std::vector<float> layernorm_bias(d_model, 0.0f);

    DeviceBuffers buffers;
    allocate_buffers(buffers, max_seq_len, d_model, vocab_size, n_heads, ffn_hidden);
    copy_static_weights_to_device(embedding_weights, attention_weights, ffn_weights, output_weights,
                                  layernorm_weight, layernorm_bias, buffers);

    std::vector<float> logits;
    for (int step = 0; step <= max_new_tokens; ++step) {
        run_forward(token_ids, d_model, vocab_size, n_heads, ffn_hidden, buffers, logits);

        const int seq_len = static_cast<int>(token_ids.size());
        std::cout << "Current token ids:\n";
        for (int token_id : token_ids) {
            std::cout << token_id << ' ';
        }
        std::cout << "\n\nLast-position logits:\n";
        for (int vocab_idx = 0; vocab_idx < vocab_size; ++vocab_idx) {
            std::cout << logits[(seq_len - 1) * vocab_size + vocab_idx] << ' ';
        }

        const int next_token_id = argmax_last_token(logits, seq_len, vocab_size);
        std::cout << "\nNext token argmax: " << next_token_id << "\n";

        if (step == max_new_tokens || static_cast<int>(token_ids.size()) >= max_seq_len) {
            break;
        }

        token_ids.push_back(next_token_id);
        std::cout << "\n";
    }

    free_buffers(buffers);
    return 0;
}
