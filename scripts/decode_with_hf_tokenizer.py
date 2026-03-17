#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer


def load_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_token_ids_from_file(path: Path) -> list[int]:
    return [int(line.strip()) for line in path.read_text().splitlines() if line.strip()]


def parse_inline_ids(text: str) -> list[int]:
    return [int(part) for part in text.strip().split() if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to exported model_config.json")
    parser.add_argument("--token-ids-file", help="Path to a file containing one token id per line")
    parser.add_argument("--token-ids", help="Inline token ids, space separated")
    args = parser.parse_args()

    if not args.token_ids_file and not args.token_ids:
        raise SystemExit("Provide --token-ids-file or --token-ids")

    config = load_config(Path(args.config))
    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])

    if args.token_ids_file:
        token_ids = load_token_ids_from_file(Path(args.token_ids_file))
    else:
        token_ids = parse_inline_ids(args.token_ids)

    print(tokenizer.decode(token_ids, skip_special_tokens=False))


if __name__ == "__main__":
    main()
