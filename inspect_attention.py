"""Report attention shape, row normalization, and causal/padding mass."""
import json
import os
from pathlib import Path
import socket

import torch
import lab1


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "data/manifest.json").read_text())
    rows = [json.loads(line) for line in
            (root / "data/passages.jsonl").read_text().splitlines()]
    weights = lab1.load_weights()
    wte, wpe = lab1.load_embedding_layers(weights)
    device = wte.weight.device
    report = {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "hostname": socket.gethostname(),
        "device": str(device),
        "dataset": manifest["dataset"]["id"],
        "config": manifest["dataset"]["config"],
        "corpus_sha256": manifest["corpus_sha256"],
        "passage_count": len(rows),
        "shape_order": ["batch", "layer", "head", "query", "key"],
        "max_row_sum_error": 0.0,
        "max_future_mass": 0.0,
        "max_padding_key_mass": 0.0,
        "max_padding_query_mass": 0.0,
    }
    eos = 50256
    with torch.inference_mode():
        for start in range(0, len(rows), 16):
            sequences = [row["token_ids"] for row in rows[start:start + 16]]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), eos, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for index, sequence in enumerate(sequences):
                ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
                mask[index, :len(sequence)] = True
            attention = lab1.gpt2_attentions(ids, mask, weights, wte, wpe)
            if start == 0:
                report["first_batch_shape"] = list(attention.shape)
            row_error = (attention.sum(-1) - mask[:, None, None, :].float()).abs().max().item()
            future = attention.triu(diagonal=1).abs().max().item()
            padding_keys = attention.masked_select((~mask)[:, None, None, None, :])
            padding_queries = attention.masked_select((~mask)[:, None, None, :, None])
            values = {
                "max_row_sum_error": row_error,
                "max_future_mass": future,
                "max_padding_key_mass": padding_keys.abs().max().item() if padding_keys.numel() else 0.0,
                "max_padding_query_mass": padding_queries.abs().max().item() if padding_queries.numel() else 0.0,
            }
            for key, value in values.items():
                report[key] = max(report[key], value)
            if start % 160 == 0:
                print(f"Inspected {start + len(sequences)}/{len(rows)} passages", flush=True)
    (root / "results").mkdir(exist_ok=True)
    (root / "results/attention_diagnostics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
