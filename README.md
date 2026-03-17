# CUDA NN Inference Engine

A minimal CUDA transformer inference engine for small GPT-2-family models.

It currently supports:
- token embeddings + positional embeddings
- multi-layer causal self-attention
- GELU MLP blocks
- final norm + LM head
- argmax token generation
- decoding generated token IDs back to text with the matching Hugging Face tokenizer

## What This Project Proves

This project shows the core shape of decoder-only transformer inference on GPU:

1. prompt -> token ids
2. token ids -> embeddings
3. run transformer blocks
4. produce logits
5. pick next token id
6. repeat

The code is intentionally minimal. It is not an optimized production inference engine.

## Current Scope

Working well:
- small GPT-2-style models
- exported Hugging Face weights
- real GPU runtime testing

Not implemented:
- KV cache
- sampling beyond argmax
- Llama-style architectures (`RoPE`, `RMSNorm`, `GQA`, `SwiGLU`)
- wide-model support beyond the current small-kernel assumptions
- full tensor-by-tensor numerical validation against PyTorch

## Main Files

- [src/main2.cu](/home/jay-zenith/Desktop/CUDA_NN_Inference_Engine/src/main2.cu): main CUDA inference path and decode loop
- [src/kernels.cu](/home/jay-zenith/Desktop/CUDA_NN_Inference_Engine/src/kernels.cu): CUDA kernels
- [src/model_loader.h](/home/jay-zenith/Desktop/CUDA_NN_Inference_Engine/src/model_loader.h): exported GPT-2 bundle loader
- [scripts/export_gpt2_hf.py](/home/jay-zenith/Desktop/CUDA_NN_Inference_Engine/scripts/export_gpt2_hf.py): export Hugging Face GPT-2-family weights into this project’s flat-text bundle format
- [scripts/decode_with_hf_tokenizer.py](/home/jay-zenith/Desktop/CUDA_NN_Inference_Engine/scripts/decode_with_hf_tokenizer.py): decode generated token IDs back to text

## Build

```bash
mkdir -p build
nvcc -arch=sm_75 -O3 src/main2.cu -o build/main2
```

Adjust `sm_75` to match your GPU.

## Export A Model

Example with the tiny GPT-2-family test model:

```bash
python3 scripts/export_gpt2_hf.py \
  --model sshleifer/tiny-gpt2 \
  --prompt "Hey how are you" \
  --output-dir /tmp/tiny_gpt2_bundle
```

This writes:
- `model_config.json`
- `token_ids.txt`
- embedding weights
- per-layer transformer weights
- final norm / LM head files

## Run Generation

```bash
./build/main2 /tmp/tiny_gpt2_bundle/model_config.json /tmp/tiny_gpt2_bundle/token_ids.txt 8
```

That prints:
- current token ids
- last-position logits
- argmax next token
- final token id sequence

## Decode To Text

```bash
python3 scripts/decode_with_hf_tokenizer.py \
  --config /tmp/tiny_gpt2_bundle/model_config.json \
  --token-ids-file /tmp/tiny_gpt2_bundle/token_ids.txt
```

Or decode generated IDs directly:

```bash
python3 scripts/decode_with_hf_tokenizer.py \
  --config /tmp/tiny_gpt2_bundle/model_config.json \
  --token-ids "10814 703 389 345"
```

## Summary

This repo now contains a real, minimal CUDA transformer inference path for small GPT-2-family models:
- real Hugging Face weights
- real GPU execution
- real token generation
- real tokenizer decode back to strings
