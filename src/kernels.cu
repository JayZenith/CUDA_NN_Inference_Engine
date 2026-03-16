#include <cuda.h>
#include <stdio.h>

#define TILE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                             int M, int N, int K) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N)
        C[row * N + col] = acc;
}

__global__ void fused_matmul_bias(const float* A, const float* B, float* C,
                             int M, int N, int K, const float* bias) {
    __shared__ float sA[TILE][TILE];
    __shared__ float sB[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;
    float acc = 0.0f;

    for (int t = 0; t < (K + TILE - 1) / TILE; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        sA[threadIdx.y][threadIdx.x] =
            (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        sB[threadIdx.y][threadIdx.x] =
            (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;

        __syncthreads();

        #pragma unroll
        for (int k = 0; k < TILE; ++k)
            acc += sA[threadIdx.y][k] * sB[k][threadIdx.x];

        __syncthreads();
    }

    if (row < M && col < N){
        C[row * N + col] = acc + bias[col]; // Add bias, avoid extra mem read/write
    }
}

__global__ void relu_kernel(float* x, int size){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) x[i] = fmaxf(0.0f, x[i]);
}

__global__ void softmax_kernel(float* x, int size){
    float max_val = -1e20;
    for(int i=0;i<size;i++) max_val = fmaxf(max_val, x[i]);

    float sum = 0.0f;
    for(int i=0;i<size;i++){
        x[i] = expf(x[i] - max_val);
        sum += x[i];
    }
    for(int i=0;i<size;i++) x[i] /= sum;
}

__global__ void softmax_kernel2(float* x, int size){
    extern __shared__ float sdata[]; //dynamically allocate shared memory
    int tid = threadIdx.x;

    //load data into shared memory
    float val = (tid < size) ? x[tid] : -1e20f; //for max_val
    sdata[tid] = val;

    //wait for all threads to finish loading
    __syncthreads();

    //1. Parallel max reduction
    for(int s = blockDim.x/2; s>0; s>>=1){
        if(tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid+s]);
        __syncthreads();
    }
    float max_val = sdata[0];
    __syncthreads();

    //2. compute exp(x-max_val)
    float exp_val = (tid < size) ? expf(x[tid] - max_val) : 0.0f;
    sdata[tid] = exp_val;
    __syncthreads();

    //3. Parallel sum reduction
    for(int s = blockDim.x/2; s > 0; s >>= 1) {
        if(tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    float sum = sdata[0];
    __syncthreads();

    // Step 4: Normalize
    if(tid < size) x[tid] = exp_val / sum;
}

__global__ void token_embedding_lookup_kernel(const int* token_ids,
                                              const float* token_embedding_table,
                                              const float* positional_embedding_table,
                                              float* output,
                                              int seq_len,
                                              int d_model) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * d_model;

    if (idx >= total) {
        return;
    }

    int token_position = idx / d_model;
    int hidden_idx = idx % d_model;
    int token_id = token_ids[token_position];

    output[idx] =
        token_embedding_table[token_id * d_model + hidden_idx] +
        positional_embedding_table[token_position * d_model + hidden_idx];
}

__global__ void residual_add_kernel(const float* x,
                                    const float* residual,
                                    float* output,
                                    int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = x[idx] + residual[idx];
    }
}

__global__ void layernorm_kernel(const float* input,
                                 const float* gamma,
                                 const float* beta,
                                 float* output,
                                 int seq_len,
                                 int d_model,
                                 float eps) {
    extern __shared__ float shared[];
    float* sum_shared = shared;
    float* sq_sum_shared = shared + blockDim.x;

    int token_position = blockIdx.x;
    int feature_idx = threadIdx.x;
    int offset = token_position * d_model;

    float value = 0.0f;
    if (feature_idx < d_model) {
        value = input[offset + feature_idx];
    }

    sum_shared[feature_idx] = value;
    sq_sum_shared[feature_idx] = value * value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (feature_idx < stride) {
            sum_shared[feature_idx] += sum_shared[feature_idx + stride];
            sq_sum_shared[feature_idx] += sq_sum_shared[feature_idx + stride];
        }
        __syncthreads();
    }

    float mean = sum_shared[0] / d_model;
    float variance = sq_sum_shared[0] / d_model - mean * mean;
    float inv_std = rsqrtf(variance + eps);

    if (feature_idx < d_model) {
        float normalized = (value - mean) * inv_std;
        output[offset + feature_idx] = normalized * gamma[feature_idx] + beta[feature_idx];
    }
}

__global__ void attention_scores_kernel(const float* q,
                                        const float* k,
                                        float* scores,
                                        int seq_len,
                                        int d_model,
                                        int n_heads) {
    int head_idx = blockIdx.z;
    int query_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int key_pos = blockIdx.x * blockDim.x + threadIdx.x;
    int d_head = d_model / n_heads;

    if (query_pos >= seq_len || key_pos >= seq_len) {
        return;
    }

    float acc = 0.0f;
    for (int i = 0; i < d_head; ++i) {
        int feature_idx = head_idx * d_head + i;
        acc += q[query_pos * d_model + feature_idx] * k[key_pos * d_model + feature_idx];
    }

    float scale = rsqrtf(static_cast<float>(d_head));
    if (key_pos > query_pos) {
        scores[(head_idx * seq_len + query_pos) * seq_len + key_pos] = -1e20f;
    } else {
        scores[(head_idx * seq_len + query_pos) * seq_len + key_pos] = acc * scale;
    }
}

__global__ void row_softmax_kernel(float* x, int rows, int cols) {
    extern __shared__ float shared[];
    float* max_shared = shared;
    float* sum_shared = shared + blockDim.x;

    int row = blockIdx.x;
    int col = threadIdx.x;
    int offset = row * cols;

    float value = (col < cols) ? x[offset + col] : -1e20f;
    max_shared[col] = value;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            max_shared[col] = fmaxf(max_shared[col], max_shared[col + stride]);
        }
        __syncthreads();
    }

    float max_val = max_shared[0];
    float exp_val = (col < cols) ? expf(value - max_val) : 0.0f;
    sum_shared[col] = exp_val;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (col < stride) {
            sum_shared[col] += sum_shared[col + stride];
        }
        __syncthreads();
    }

    if (col < cols) {
        x[offset + col] = exp_val / sum_shared[0];
    }
}

__global__ void attention_weighted_sum_kernel(const float* attn_weights,
                                              const float* v,
                                              float* output,
                                              int seq_len,
                                              int d_model,
                                              int n_heads) {
    int head_idx = blockIdx.z;
    int token_pos = blockIdx.y * blockDim.y + threadIdx.y;
    int feature_in_head = blockIdx.x * blockDim.x + threadIdx.x;
    int d_head = d_model / n_heads;

    if (token_pos >= seq_len || feature_in_head >= d_head) {
        return;
    }

    float acc = 0.0f;
    for (int source_pos = 0; source_pos < seq_len; ++source_pos) {
        float weight = attn_weights[(head_idx * seq_len + token_pos) * seq_len + source_pos];
        float value = v[source_pos * d_model + head_idx * d_head + feature_in_head];
        acc += weight * value;
    }

    output[token_pos * d_model + head_idx * d_head + feature_in_head] = acc;
}
