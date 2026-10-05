# GPT-2 attention analysis

`lab1.py` contains our course GPT-2 implementation. `gpt2_attentions()` returns
post-softmax causal probabilities shaped `[batch, layer, head, query, key]`.
There are 12 layers and 12 heads; layer/head indices start at zero. Padded queries
and keys have zero attention. Generation continues to use the original SDPA path.
Weights and tokenizer are pinned to GPT-2 revision
`607a30d783dfa663caf39e06633721c8d4cfcd7e`.

## Run with Slurm

Run inference and data preparation only on allocated compute nodes. From this
workspace on the login node:

```bash
mkdir -p logs
sbatch --exclude=25a-hgpn001 prepare_inspect.sbatch
```

The job requests one GPU, four CPUs, 32 GB RAM, and 30 minutes on `dev`, account
`acd115198`. Dependencies are managed by `uv` using `pyproject.toml` and
`uv.lock`, with Python 3.14.7 and CUDA 13.0 PyTorch wheels. Batch scripts run
`uv sync --locked` on the compute node and execute scripts using `uv run` in
the workspace's `.venv`. Install packages only inside a compute allocation.
The scripts are ordinary research scripts, without runtime allocation guards
or a test suite. The batch script handles placement through `srun`.

To regenerate the lockfile after changing dependencies, submit
`sbatch --exclude=25a-hgpn001 setup.sbatch`. This runs `uv lock` and `uv sync`
inside a compute allocation. Ordinary experiment jobs use the existing lockfile.

## Frozen corpus

Source: [wikimedia/wikipedia, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia/tree/ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619/20231101.en),
English, train split. The immutable revision is
`ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619`.

`prepare_data.py` selects one of the 41 English shards with
`random.Random(42).randrange(41)`. It takes each article's first nonempty
paragraph, splitting on double newlines, and strips only surrounding whitespace.
It excludes empty openings, openings containing EOS, and openings shorter than
33 GPT-2 tokens. It samples 1,000 eligible openings by reservoir sampling using
a separate `random.Random(42)`, then sorts them by source row. Each input is
truncated to the first 128 GPT-2 token IDs, without inserting special tokens.

`data/passages.jsonl` freezes exact token IDs, source text, article IDs, titles,
URLs, paragraph indices, and source row indices. `data/manifest.json` records
preprocessing, source revision and shard, selected rows, package versions, hashes,
and token counts. `data/tokenizer.json` stores the tokenizer itself. Existing
frozen data is reused when submitting the job again.

This sample covers one shard and article openings only; it does not represent
the complete Wikipedia snapshot. The 33-token minimum supports common query
positions `i >= 32` for offsets `0, 1, 2, 4, 8, 16, 32`, while excluding short
openings. Report these sampling limits in the experiments.

Credit Wikipedia contributors; per-article URLs are included in the corpus.
The dataset card lists CC-BY-SA-3.0 and GFDL.
The superseded WikiText corpus and its validation results are in
`archive/wikitext/` and are not used by the current scripts.

## Attention diagnostics

`inspect_attention.py` runs a forward pass over the frozen corpus and reports
attention shape, maximum row-sum error, future-token mass, and padding mass in
`results/attention_diagnostics.json`. This is a diagnostic summary, with no
assertions, synthetic test cases, or model-parity tests. Right-padding uses EOS
IDs as masked storage filler; those IDs receive no attention. The later
entropy, gradient, and clustering experiments are not run yet.

The frozen Wikipedia corpus contains 1,000 openings from shard 40, sampled from
123,794 eligible articles: 78,679 input tokens and 46,679 offset-32 query positions.
Preparation ran in Slurm job `498998`. Diagnostics ran in job `499032`, on
`25a-hgpn003`: first-batch shape `[16, 12, 12, 128, 128]`, maximum normalization
error `4.77e-7`, and exactly zero future/padding attention.

## Experiment 1: relative positions

```bash
sbatch --exclude=25a-hgpn001 experiment1.sbatch
```

`experiment1.py` measures attention at backward offsets `0, 1, 2, 4, 8, 16, 32`.
Every offset uses the same real query positions `i >= 32`, with equal weight per
query token across passages. The causal-uniform baseline is the average of
`1/(i+1)` over those same queries. Padding queries are excluded.

Outputs: `results/relative_position.csv`,
`results/relative_position_metadata.json`, and `figures/relative_position.png`
and `.svg`. Metadata includes the corpus hash, model revision, actual query
count, averaging convention, execution details, and the top five heads per
offset. Layer/head numbering starts at zero. Matplotlib is included in the uv
project dependencies.

Experiment 1 completed in Slurm job `499296` on `25a-hgpn012`, using all 1,000
passages and 46,679 eligible queries per head/offset. The causal-uniform baseline
was `0.017407` (1.74%). The strongest self-attention head was `L0H3` (84.91%),
and the strongest previous-token head was `L4H11` (99.86%). These describe the
frozen sample and the shared `i >= 32` query set; they do not establish causal
importance for model predictions. All reported head labels are zero-based.
