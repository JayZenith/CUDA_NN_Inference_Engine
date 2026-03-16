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

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr
            << "Usage: " << argv[0]
            << " <model_config.json> <token_ids.txt>\n";
        return 1;
    }

    const EmbeddingWeights weights = load_embedding_weights(argv[1]);
    const AttentionWeights attention_weights = load_attention_weights(argv[1]);
    const FFNWeights ffn_weights = load_ffn_weights(argv[1]);
    const std::vector<int> h_token_ids = load_ints(argv[2]);
    const int d_model = weights.config.d_model;
    const int max_seq_len = weights.config.max_seq_len;
    const int n_heads = weights.config.n_heads;
    const int d_head = d_model / n_heads;
    const int ffn_hidden = weights.config.ffn_hidden;
    const int seq_len = static_cast<int>(h_token_ids.size());
    const int total = seq_len * d_model;
    const int ffn_total = seq_len * ffn_hidden;

    if (seq_len > max_seq_len) {
        std::cerr << "seq_len exceeds max_seq_len\n";
        return 1;
    }

    if (d_model > 256) {
        std::cerr << "layernorm path currently expects d_model <= 256\n";
        return 1;
    }

    if (seq_len > 256) {
        std::cerr << "attention path currently expects seq_len <= 256\n";
        return 1;
    }

    const int vocab_size = weights.config.vocab_size;
    for (int token_id : h_token_ids) {
        if (token_id < 0 || token_id >= vocab_size) {
            std::cerr << "token id out of range: " << token_id << '\n';
            return 1;
        }
    }

    std::vector<float> h_output(total);
    std::vector<float> h_layernorm_weight(d_model, 1.0f);
    std::vector<float> h_layernorm_bias(d_model, 0.0f);
    std::vector<float> h_layernorm_output(total);
    std::vector<float> h_ffn_layernorm_output(total);
    std::vector<float> h_q(total);
    std::vector<float> h_k(total);
    std::vector<float> h_v(total);
    std::vector<float> h_attn_output(total);
    std::vector<float> h_ffn_up(ffn_total);
    std::vector<float> h_ffn_down(total);
    std::vector<float> h_final_output(total);

    int* d_token_ids = nullptr;
    float* d_token_embedding_table = nullptr;
    float* d_positional_embedding_table = nullptr;
    float* d_embedding_output = nullptr;
    float* d_layernorm_weight = nullptr;
    float* d_layernorm_bias = nullptr;
    float* d_layernorm_output = nullptr;
    float* d_q_proj_weight = nullptr;
    float* d_k_proj_weight = nullptr;
    float* d_v_proj_weight = nullptr;
    float* d_o_proj_weight = nullptr;
    float* d_q = nullptr;
    float* d_k = nullptr;
    float* d_v = nullptr;
    float* d_attn_scores = nullptr;
    float* d_attn_context = nullptr;
    float* d_attn_output = nullptr;
    float* d_attn_residual_output = nullptr;
    float* d_ffn_layernorm_output = nullptr;
    float* d_ffn_up_weight = nullptr;
    float* d_ffn_down_weight = nullptr;
    float* d_ffn_up = nullptr;
    float* d_ffn_down = nullptr;
    float* d_final_output = nullptr;

    check_cuda(cudaMalloc(&d_token_ids, seq_len * sizeof(int)), "cudaMalloc d_token_ids");
    check_cuda(cudaMalloc(&d_token_embedding_table, vocab_size * d_model * sizeof(float)),
               "cudaMalloc d_token_embedding_table");
    check_cuda(cudaMalloc(&d_positional_embedding_table, max_seq_len * d_model * sizeof(float)),
               "cudaMalloc d_positional_embedding_table");
    check_cuda(cudaMalloc(&d_embedding_output, total * sizeof(float)), "cudaMalloc d_embedding_output");
    check_cuda(cudaMalloc(&d_layernorm_weight, d_model * sizeof(float)), "cudaMalloc d_layernorm_weight");
    check_cuda(cudaMalloc(&d_layernorm_bias, d_model * sizeof(float)), "cudaMalloc d_layernorm_bias");
    check_cuda(cudaMalloc(&d_layernorm_output, total * sizeof(float)), "cudaMalloc d_layernorm_output");
    check_cuda(cudaMalloc(&d_q_proj_weight, d_model * d_model * sizeof(float)), "cudaMalloc d_q_proj_weight");
    check_cuda(cudaMalloc(&d_k_proj_weight, d_model * d_model * sizeof(float)), "cudaMalloc d_k_proj_weight");
    check_cuda(cudaMalloc(&d_v_proj_weight, d_model * d_model * sizeof(float)), "cudaMalloc d_v_proj_weight");
    check_cuda(cudaMalloc(&d_o_proj_weight, d_model * d_model * sizeof(float)), "cudaMalloc d_o_proj_weight");
    check_cuda(cudaMalloc(&d_q, total * sizeof(float)), "cudaMalloc d_q");
    check_cuda(cudaMalloc(&d_k, total * sizeof(float)), "cudaMalloc d_k");
    check_cuda(cudaMalloc(&d_v, total * sizeof(float)), "cudaMalloc d_v");
    check_cuda(cudaMalloc(&d_attn_scores, n_heads * seq_len * seq_len * sizeof(float)),
               "cudaMalloc d_attn_scores");
    check_cuda(cudaMalloc(&d_attn_context, total * sizeof(float)), "cudaMalloc d_attn_context");
    check_cuda(cudaMalloc(&d_attn_output, total * sizeof(float)), "cudaMalloc d_attn_output");
    check_cuda(cudaMalloc(&d_attn_residual_output, total * sizeof(float)),
               "cudaMalloc d_attn_residual_output");
    check_cuda(cudaMalloc(&d_ffn_layernorm_output, total * sizeof(float)),
               "cudaMalloc d_ffn_layernorm_output");
    check_cuda(cudaMalloc(&d_ffn_up_weight, d_model * ffn_hidden * sizeof(float)),
               "cudaMalloc d_ffn_up_weight");
    check_cuda(cudaMalloc(&d_ffn_down_weight, ffn_hidden * d_model * sizeof(float)),
               "cudaMalloc d_ffn_down_weight");
    check_cuda(cudaMalloc(&d_ffn_up, ffn_total * sizeof(float)), "cudaMalloc d_ffn_up");
    check_cuda(cudaMalloc(&d_ffn_down, total * sizeof(float)), "cudaMalloc d_ffn_down");
    check_cuda(cudaMalloc(&d_final_output, total * sizeof(float)), "cudaMalloc d_final_output");

    check_cuda(cudaMemcpy(d_token_ids, h_token_ids.data(), seq_len * sizeof(int),
                          cudaMemcpyHostToDevice),
               "cudaMemcpy token_ids");
    check_cuda(cudaMemcpy(d_token_embedding_table, weights.token_embedding_table.data(),
                          vocab_size * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy token_embedding_table");
    check_cuda(cudaMemcpy(d_positional_embedding_table, weights.positional_embedding_table.data(),
                          max_seq_len * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy positional_embedding_table");
    check_cuda(cudaMemcpy(d_layernorm_weight, h_layernorm_weight.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy layernorm_weight");
    check_cuda(cudaMemcpy(d_layernorm_bias, h_layernorm_bias.data(),
                          d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy layernorm_bias");
    check_cuda(cudaMemcpy(d_q_proj_weight, attention_weights.q_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy q_proj_weight");
    check_cuda(cudaMemcpy(d_k_proj_weight, attention_weights.k_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy k_proj_weight");
    check_cuda(cudaMemcpy(d_v_proj_weight, attention_weights.v_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy v_proj_weight");
    check_cuda(cudaMemcpy(d_o_proj_weight, attention_weights.o_proj_weight.data(),
                          d_model * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy o_proj_weight");
    check_cuda(cudaMemcpy(d_ffn_up_weight, ffn_weights.ffn_up_weight.data(),
                          d_model * ffn_hidden * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_up_weight");
    check_cuda(cudaMemcpy(d_ffn_down_weight, ffn_weights.ffn_down_weight.data(),
                          ffn_hidden * d_model * sizeof(float), cudaMemcpyHostToDevice),
               "cudaMemcpy ffn_down_weight");

    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    const dim3 matmul_threads(16, 16);
    const dim3 matmul_blocks((d_model + 15) / 16, (seq_len + 15) / 16);
    const dim3 ffn_up_blocks((ffn_hidden + 15) / 16, (seq_len + 15) / 16);

    token_embedding_lookup_kernel<<<blocks, threads>>>(
        d_token_ids,
        d_token_embedding_table,
        d_positional_embedding_table,
        d_embedding_output,
        seq_len,
        d_model);
    check_cuda(cudaGetLastError(), "launch token_embedding_lookup_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync token_embedding_lookup_kernel");

    const int layernorm_threads = next_power_of_two(d_model);
    const std::size_t layernorm_shared_bytes =
        static_cast<std::size_t>(layernorm_threads) * 2 * sizeof(float);

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        d_embedding_output,
        d_layernorm_weight,
        d_layernorm_bias,
        d_layernorm_output,
        seq_len,
        d_model,
        1e-5f);
    check_cuda(cudaGetLastError(), "launch layernorm_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync layernorm_kernel");

    matmul_tiled<<<matmul_blocks, matmul_threads>>>(
        d_layernorm_output, d_q_proj_weight, d_q, seq_len, d_model, d_model);
    matmul_tiled<<<matmul_blocks, matmul_threads>>>(
        d_layernorm_output, d_k_proj_weight, d_k, seq_len, d_model, d_model);
    matmul_tiled<<<matmul_blocks, matmul_threads>>>(
        d_layernorm_output, d_v_proj_weight, d_v, seq_len, d_model, d_model);
    check_cuda(cudaGetLastError(), "launch qkv matmuls");
    check_cuda(cudaDeviceSynchronize(), "sync qkv matmuls");

    const dim3 attn_score_threads(16, 16);
    const dim3 attn_score_blocks((seq_len + 15) / 16, (seq_len + 15) / 16, n_heads);
    attention_scores_kernel<<<attn_score_blocks, attn_score_threads>>>(
        d_q, d_k, d_attn_scores, seq_len, d_model, n_heads);
    check_cuda(cudaGetLastError(), "launch attention_scores_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync attention_scores_kernel");

    const int softmax_threads = next_power_of_two(seq_len);
    const std::size_t softmax_shared_bytes =
        static_cast<std::size_t>(softmax_threads) * 2 * sizeof(float);
    row_softmax_kernel<<<n_heads * seq_len, softmax_threads, softmax_shared_bytes>>>(
        d_attn_scores, n_heads * seq_len, seq_len);
    check_cuda(cudaGetLastError(), "launch row_softmax_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync row_softmax_kernel");

    const dim3 attn_value_threads(16, 16);
    const dim3 attn_value_blocks((d_head + 15) / 16, (seq_len + 15) / 16, n_heads);
    attention_weighted_sum_kernel<<<attn_value_blocks, attn_value_threads>>>(
        d_attn_scores, d_v, d_attn_context, seq_len, d_model, n_heads);
    check_cuda(cudaGetLastError(), "launch attention_weighted_sum_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync attention_weighted_sum_kernel");

    matmul_tiled<<<matmul_blocks, matmul_threads>>>(
        d_attn_context, d_o_proj_weight, d_attn_output, seq_len, d_model, d_model);
    check_cuda(cudaGetLastError(), "launch o_proj matmul");
    check_cuda(cudaDeviceSynchronize(), "sync o_proj matmul");

    residual_add_kernel<<<blocks, threads>>>(
        d_attn_output, d_embedding_output, d_attn_residual_output, total);
    check_cuda(cudaGetLastError(), "launch attention residual_add_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync attention residual_add_kernel");

    layernorm_kernel<<<seq_len, layernorm_threads, layernorm_shared_bytes>>>(
        d_attn_residual_output,
        d_layernorm_weight,
        d_layernorm_bias,
        d_ffn_layernorm_output,
        seq_len,
        d_model,
        1e-5f);
    check_cuda(cudaGetLastError(), "launch ffn layernorm_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync ffn layernorm_kernel");

    matmul_tiled<<<ffn_up_blocks, matmul_threads>>>(
        d_ffn_layernorm_output, d_ffn_up_weight, d_ffn_up, seq_len, ffn_hidden, d_model);
    check_cuda(cudaGetLastError(), "launch ffn_up matmul");
    check_cuda(cudaDeviceSynchronize(), "sync ffn_up matmul");

    relu_kernel<<<(ffn_total + 255) / 256, 256>>>(d_ffn_up, ffn_total);
    check_cuda(cudaGetLastError(), "launch relu_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync relu_kernel");

    matmul_tiled<<<matmul_blocks, matmul_threads>>>(
        d_ffn_up, d_ffn_down_weight, d_ffn_down, seq_len, d_model, ffn_hidden);
    check_cuda(cudaGetLastError(), "launch ffn_down matmul");
    check_cuda(cudaDeviceSynchronize(), "sync ffn_down matmul");

    residual_add_kernel<<<blocks, threads>>>(
        d_ffn_down, d_attn_residual_output, d_final_output, total);
    check_cuda(cudaGetLastError(), "launch final residual_add_kernel");
    check_cuda(cudaDeviceSynchronize(), "sync final residual_add_kernel");

    check_cuda(cudaMemcpy(h_output.data(), d_embedding_output, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy embedding_output");
    check_cuda(cudaMemcpy(h_layernorm_output.data(), d_layernorm_output, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy layernorm_output");
    check_cuda(cudaMemcpy(h_q.data(), d_q, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy q");
    check_cuda(cudaMemcpy(h_k.data(), d_k, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy k");
    check_cuda(cudaMemcpy(h_v.data(), d_v, total * sizeof(float), cudaMemcpyDeviceToHost),
               "cudaMemcpy v");
    check_cuda(cudaMemcpy(h_attn_output.data(), d_attn_output, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy attn_output");
    check_cuda(cudaMemcpy(h_ffn_layernorm_output.data(), d_ffn_layernorm_output, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy ffn_layernorm_output");
    check_cuda(cudaMemcpy(h_ffn_up.data(), d_ffn_up, ffn_total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy ffn_up");
    check_cuda(cudaMemcpy(h_ffn_down.data(), d_ffn_down, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy ffn_down");
    check_cuda(cudaMemcpy(h_final_output.data(), d_final_output, total * sizeof(float),
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy final_output");

    std::cout << "Token ids:\n";
    for (int i = 0; i < seq_len; ++i) {
        std::cout << h_token_ids[i] << '\n';
    }

    std::cout << "\nEmbedded sequence:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_output[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nLayerNorm output:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_layernorm_output[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nQ projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_q[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nK projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_k[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nV projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_v[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nAttention output projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_attn_output[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nFFN LayerNorm output:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_ffn_layernorm_output[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nFFN up projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < ffn_hidden; ++d) {
            std::cout << h_ffn_up[t * ffn_hidden + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nFFN down projection:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_ffn_down[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    std::cout << "\nFinal block output:\n";
    for (int t = 0; t < seq_len; ++t) {
        std::cout << "position " << t << ": ";
        for (int d = 0; d < d_model; ++d) {
            std::cout << h_final_output[t * d_model + d] << ' ';
        }
        std::cout << '\n';
    }

    cudaFree(d_token_ids);
    cudaFree(d_token_embedding_table);
    cudaFree(d_positional_embedding_table);
    cudaFree(d_embedding_output);
    cudaFree(d_layernorm_weight);
    cudaFree(d_layernorm_bias);
    cudaFree(d_layernorm_output);
    cudaFree(d_q_proj_weight);
    cudaFree(d_k_proj_weight);
    cudaFree(d_v_proj_weight);
    cudaFree(d_o_proj_weight);
    cudaFree(d_q);
    cudaFree(d_k);
    cudaFree(d_v);
    cudaFree(d_attn_scores);
    cudaFree(d_attn_context);
    cudaFree(d_attn_output);
    cudaFree(d_attn_residual_output);
    cudaFree(d_ffn_layernorm_output);
    cudaFree(d_ffn_up_weight);
    cudaFree(d_ffn_down_weight);
    cudaFree(d_ffn_up);
    cudaFree(d_ffn_down);
    cudaFree(d_final_output);
    return 0;
}
