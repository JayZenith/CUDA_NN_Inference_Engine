#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def write_floats(path: Path, tensor: torch.Tensor) -> None:
    tensor = tensor.detach().cpu().to(torch.float32).contiguous().view(-1)
    path.write_text("\n".join(f"{x:.9g}" for x in tensor.tolist()) + "\n")


def write_ints(path: Path, values: list[int]) -> None:
    path.write_text("\n".join(str(v) for v in values) + "\n")


def export_model(model_name: str, prompt: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
    model.eval()

    config = model.config
    if config.model_type != "gpt2":
        raise ValueError(f"Expected GPT-2 family model, got {config.model_type}")

    token_ids = tokenizer.encode(prompt, add_special_tokens=False)
    if not token_ids:
        raise ValueError("Prompt produced no token ids")

    n_inner = config.n_inner if config.n_inner is not None else 4 * config.n_embd
    exported_config = {
        "d_model": config.n_embd,
        "max_seq_len": config.n_positions,
        "vocab_size": config.vocab_size,
        "n_heads": config.n_head,
        "ffn_hidden": n_inner,
        "n_layers": config.n_layer,
        "tie_word_embeddings": 1 if getattr(config, "tie_word_embeddings", True) else 0,
        "token_embedding_path": "token_embeddings.txt",
        "positional_embedding_path": "pos_embeddings.txt",
        "final_norm_weight_path": "final_norm_weight.txt",
        "final_norm_bias_path": "final_norm_bias.txt",
        "lm_head_path": "lm_head_weight.txt",
        "model_name": model_name,
        "prompt": prompt,
    }

    (output_dir / "model_config.json").write_text(json.dumps(exported_config, indent=2) + "\n")
    write_ints(output_dir / "token_ids.txt", token_ids)

    transformer = model.transformer
    write_floats(output_dir / "token_embeddings.txt", transformer.wte.weight)
    write_floats(output_dir / "pos_embeddings.txt", transformer.wpe.weight)
    write_floats(output_dir / "final_norm_weight.txt", transformer.ln_f.weight)
    write_floats(output_dir / "final_norm_bias.txt", transformer.ln_f.bias)

    if getattr(config, "tie_word_embeddings", True):
        (output_dir / "lm_head_weight.txt").write_text("0.0\n")
    else:
        write_floats(output_dir / "lm_head_weight.txt", model.lm_head.weight.t())

    for layer_idx, block in enumerate(transformer.h):
        layer_dir = output_dir / f"layer_{layer_idx}"
        layer_dir.mkdir(exist_ok=True)

        q_weight, k_weight, v_weight = block.attn.c_attn.weight.detach().cpu().split(config.n_embd, dim=1)
        q_bias, k_bias, v_bias = block.attn.c_attn.bias.detach().cpu().split(config.n_embd, dim=0)

        write_floats(layer_dir / "ln1_weight.txt", block.ln_1.weight)
        write_floats(layer_dir / "ln1_bias.txt", block.ln_1.bias)
        write_floats(layer_dir / "q_proj_weight.txt", q_weight)
        write_floats(layer_dir / "q_proj_bias.txt", q_bias)
        write_floats(layer_dir / "k_proj_weight.txt", k_weight)
        write_floats(layer_dir / "k_proj_bias.txt", k_bias)
        write_floats(layer_dir / "v_proj_weight.txt", v_weight)
        write_floats(layer_dir / "v_proj_bias.txt", v_bias)
        write_floats(layer_dir / "o_proj_weight.txt", block.attn.c_proj.weight)
        write_floats(layer_dir / "o_proj_bias.txt", block.attn.c_proj.bias)
        write_floats(layer_dir / "ln2_weight.txt", block.ln_2.weight)
        write_floats(layer_dir / "ln2_bias.txt", block.ln_2.bias)
        write_floats(layer_dir / "ffn_up_weight.txt", block.mlp.c_fc.weight)
        write_floats(layer_dir / "ffn_up_bias.txt", block.mlp.c_fc.bias)
        write_floats(layer_dir / "ffn_down_weight.txt", block.mlp.c_proj.weight)
        write_floats(layer_dir / "ffn_down_bias.txt", block.mlp.c_proj.bias)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="hf-internal-testing/tiny-random-GPT2LMHeadModel")
    parser.add_argument("--prompt", default="Hey how are you")
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    export_model(args.model, args.prompt, Path(args.output_dir))


if __name__ == "__main__":
    main()
