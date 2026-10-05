"""Mean attention to fixed backward offsets on the frozen Wikipedia corpus."""
import csv
import importlib.metadata
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import lab1

OFFSETS = [0, 1, 2, 4, 8, 16, 32]
BATCH_SIZE = 16


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / "data/manifest.json").read_text())
    passages = [json.loads(line) for line in
                (root / "data/passages.jsonl").read_text().splitlines()]
    weights = lab1.load_weights()
    wte, wpe = lab1.load_embedding_layers(weights)
    wte.eval()
    wpe.eval()
    device = wte.weight.device
    sums = torch.zeros(12, 12, len(OFFSETS), dtype=torch.float64)
    query_count = 0
    baseline_sum = 0.0

    with torch.inference_mode():
        for start in range(0, len(passages), BATCH_SIZE):
            sequences = [p["token_ids"] for p in passages[start:start + BATCH_SIZE]]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), 50256, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for index, sequence in enumerate(sequences):
                ids[index, :len(sequence)] = torch.tensor(sequence, device=device)
                mask[index, :len(sequence)] = True

            attention = lab1.gpt2_attentions(ids, mask, weights, wte, wpe)
            positions = torch.arange(max(OFFSETS), width, device=device)
            valid = mask[:, positions]
            for column, distance in enumerate(OFFSETS):
                values = attention[:, :, :, positions, positions - distance]
                sums[:, :, column] += (
                    values.double() * valid[:, None, None, :]
                ).sum(dim=(0, 3)).cpu()
            query_count += valid.sum().item()
            baseline_sum += (valid.double() / (positions + 1)).sum().item()
            if start % 160 == 0:
                print(f"Processed {start + len(sequences)}/{len(passages)} passages", flush=True)

    means = sums / query_count
    baseline = baseline_sum / query_count
    results = root / "results"
    figures = root / "figures"
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    with (results / "relative_position.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["layer", "head", "offset", "mean_attention", "uniform_baseline", "n_queries"])
        for layer in range(12):
            for head in range(12):
                for column, distance in enumerate(OFFSETS):
                    writer.writerow([layer, head, distance, means[layer, head, column].item(),
                                     baseline, query_count])

    top_heads = {}
    for column, distance in enumerate(OFFSETS):
        values, indices = means[:, :, column].reshape(-1).topk(5)
        ranking = [{"layer": index // 12, "head": index % 12,
                    "mean_attention": value, "enrichment_over_uniform": value / baseline}
                   for value, index in zip(values.tolist(), indices.tolist())]
        top_heads[str(distance)] = ranking
        print(f"Offset {distance}: " + "; ".join(
            f"L{row['layer']}H{row['head']}={row['mean_attention']:.4f}" for row in ranking), flush=True)

    metadata = {
        "dataset": manifest["dataset"], "corpus_sha256": manifest["corpus_sha256"],
        "model": manifest["model"], "passage_count": len(passages),
        "offsets": OFFSETS, "query_positions": "real tokens with zero-based i >= 32",
        "averaging": "equal weight per eligible query token across all passages",
        "n_queries": query_count, "uniform_baseline": baseline,
        "indexing": "zero-based layers and heads; offset d means key position i-d",
        "batch_size": BATCH_SIZE, "weight_dtype": str(wte.weight.dtype),
        "attention_dtype": "torch.float32", "accumulation_dtype": "torch.float64",
        "job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname(),
        "device": str(device), "top_heads": top_heads,
        "versions": {p: importlib.metadata.version(p) for p in ("torch", "matplotlib")},
    }
    (results / "relative_position_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    fig, ax = plt.subplots(figsize=(8, 23))
    image = ax.imshow(means.reshape(144, len(OFFSETS)).numpy(),
                      aspect="auto", interpolation="nearest", cmap="viridis", vmin=0)
    ax.set_xticks(range(len(OFFSETS)), [str(d) for d in OFFSETS])
    ax.set_yticks(range(144), [f"L{layer}H{head}" for layer in range(12) for head in range(12)], fontsize=7)
    for boundary in range(12, 144, 12):
        ax.axhline(boundary - 0.5, color="white", linewidth=0.6)
    ax.set_xlabel("Backward offset d (0 = self, 1 = previous token)")
    ax.set_ylabel("Layer and head (zero-based)")
    ax.set_title(f"GPT-2 relative-position attention\n{len(passages):,} Wikipedia openings; "
                 f"{query_count:,} queries with i ≥ 32\nUniform baseline = {baseline:.4f}")
    fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04, label="Mean attention probability")
    fig.tight_layout()
    for extension in ("png", "svg"):
        fig.savefig(figures / f"relative_position.{extension}", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved CSV and heatmap. Queries={query_count:,}; uniform baseline={baseline:.6f}", flush=True)


if __name__ == "__main__":
    main()
