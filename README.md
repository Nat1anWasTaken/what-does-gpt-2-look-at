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

## Experiment 2: lexical no-op candidates

The [lab specification](https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16)
defines Experiment 2 as lexical attention plus gradient importance; entropy is
Experiment 3. Run both stages with:

```bash
sbatch --exclude=25a-hgpn001 experiment2.sbatch
```

`experiment2.py` predefines periods, commas, and articles (`the`, `a`, `an`,
case-insensitive), with other real tokens as the comparison class. Classification
uses the frozen tokenizer's source offsets. Punctuation must occupy an entire
whitespace-stripped token; articles must occupy a complete word span in a single
BPE token. Bundled punctuation and word fragments remain in `other`. No artificial
EOS tokens are inserted. Exact membership rules are recorded in metadata.

Both stages use real query positions `i >= 32` with a real next-token target,
excluding each passage's last token. Attention mass is summed over visible keys
in each class and averaged with equal weight per query token across passages.
The causal-uniform baseline is the average visible class frequency; enrichment
is mean attention divided by that baseline.

The gradient stage uses the course transformer blocks, final layer normalization,
and tied vocabulary projection in float32. Its loss is **summed** next-token
cross-entropy over all real transitions within each passage, including early
positions. This makes edge gradients independent of batch-size reduction. The
reported gradient is the absolute partial derivative with respect to each
post-softmax attention probability, including downstream losses in that passage.
It is not a softmax-logit gradient or an isolated query's loss gradient. Parameters
are frozen; a differentiable input hidden state enables attention gradients.

Class gradient importance is the sum of absolute gradients divided by the number
of visible class edges, using the same selected queries as attention. A secondary
attention-weighted gradient divides `sum(A * abs(grad))` by `sum(A)`. The CSV
also reports each class's mean edge gradient relative to `other` in that head.
Gradients are local, unconstrained edge sensitivities; they do not by themselves
establish the effect of a finite, probability-preserving attention intervention.

Outputs: `results/token_class_attention.csv`,
`results/token_class_gradient_importance.csv`,
`results/token_class_metadata.json`, and PNG/SVG figures
`figures/token_class_attention.*` and `figures/token_class_gradient_importance.*`.
The figures show all heads by zero-based layer and a layer mean. Metadata records
corpus hash, model revision, query/edge counts, loss convention, class definitions,
execution details, diagnostics, and the top five heads per candidate class.
High enrichment paired with lower gradient than ordinary keys is consistent with
no-op-like behavior, but remains descriptive evidence without significance tests.

Experiment 2 completed in Slurm job `499363` on `25a-hgpn009` (NVIDIA H200),
using all 1,000 passages, 45,679 analysis queries, and 77,679 next-token loss
terms. This query count is 1,000 fewer than Experiment 1 because each passage's
last token has no within-passage next-token target. Future and padding attention
were exactly zero, maximum row-sum error was `4.77e-7`, and all gradients were
finite.

| Candidate class | Highest-enrichment head | Mean attention | Uniform baseline | Enrichment | Edge gradient / other |
| --- | --- | --- | --- | --- | --- |
| Period | L2H6 | 19.63% | 2.65% | 7.41× | 0.241 |
| Comma | L2H5 | 10.38% | 4.22% | 2.46× | 0.717 |
| Articles | L7H2 | 22.40% | 7.62% | 2.94× | 0.953 |

L2H6's period preference combines strong enrichment with substantially lower
local gradient sensitivity, consistent with a lexical no-op-like destination.
Comma results are mixed: for example, L1H5's comma enrichment is 2.33× while its
gradient ratio is 1.18, so an attention preference does not uniformly imply low
sensitivity. The five most article-enriched heads have gradient ratios of
0.84–0.95, a much smaller reduction; the no-op interpretation there is inconclusive.
These comparisons describe this frozen sample and are not causal ablation tests.

## Experiment 3: focused versus broad attention

```bash
sbatch --exclude=25a-hgpn001 experiment3.sbatch
```

`experiment3.py` follows Experiment 3 on the lab Notion page. For every head and
real query, it computes `H = -sum_j A[i,j] * ln(A[i,j])` over visible keys, with
`0 * ln(0) = 0`. Raw entropy is measured in nats and averaged over **all real
queries**, including position zero and each passage's final token. Position zero
has one visible key and hence zero entropy. Padding is excluded.

The normalized metric divides each query's entropy by `ln(i+1)` **before**
averaging. Position zero is excluded because its denominator is zero. Equal
weight is assigned to each eligible query token across passages. The raw
causal-uniform baseline is the average `ln(i+1)` on the same query set; the
normalized uniform baseline is one.

Primary figures show one point per head by zero-based layer, plus the layer mean:
`figures/entropy_raw.png`/`.svg` and `figures/entropy_normalized.png`/`.svg`.
`results/entropy_raw.csv` and `results/entropy_normalized.csv` contain 144 heads
for each of three query sets: the primary `all_real` set, Experiment 1's real
`i >= 32` set, and Experiment 2's real `i >= 32` queries with next-token targets.
The latter two permit comparisons without changing query eligibility.

`results/entropy_behavior_comparison.csv` joins the matched-query entropy values
with all positional and lexical attention results from Experiments 1 and 2.
`results/entropy_metadata.json` records corpus/model identity, definitions,
counts, baselines, top five focused/broad heads, comparison-source hashes,
execution details, and attention/entropy diagnostics. Like Experiment 1,
Experiment 3 uses float16 weights and float32 attention; Experiment 2 uses
float32 weights, which limits exact cross-experiment numerical comparisons.
Entropy describes attention concentration and does not establish causal
importance for predictions.

Experiment 3 completed in Slurm job `499426` on `25a-hgpn009`, using all 1,000
passages. Raw entropy used 78,679 real queries; normalized entropy used 77,679
queries after excluding position zero. The matched Experiment 1/2 sets contained
46,679 and 45,679 queries respectively. The all-query raw uniform baseline was
3.47909 nats. All entropy values were finite and within causal bounds, future
and padding attention were zero, and maximum attention row-sum error was
`4.77e-7`. Output consistency checks passed in compute job `499427`.

### Our observations from Experiments 1–3

The following are **our observations from the experiments** on the frozen
1,000-passage Wikipedia sample. We rank heads by normalized entropy over all
real queries with `i > 0`: lower values indicate focused attention, while values
near one indicate broad, nearly uniform attention. Raw entropy identifies the
same top-five focused and broad groups, although the order of L1H7 and L1H10
reverses within the broad group. All layer/head labels are zero-based.

| Most focused head | Normalized entropy | Observed behavior in Experiments 1–2 |
| --- | --- | --- |
| L4H11 | 0.00168 | Previous-token attention is 99.86%; little lexical enrichment. |
| L5H1 | 0.03294 | Articles receive 21.88% attention, 2.87× uniform; none of the tested offsets dominates. |
| L7H2 | 0.04699 | Articles receive 22.40%, 2.94× uniform, the strongest article enrichment. |
| L6H9 | 0.06608 | Articles receive 21.04%, 2.76× uniform; none of the tested offsets dominates. |
| L0H1 | 0.09515 | Self-attention is 83.02%; little lexical enrichment. |

We observe different behaviors among the most focused heads. L4H11 and L0H1
concentrate on fixed relative positions. L5H1, L7H2, and L6H9 have highly
concentrated attention, but the offsets tested in Experiment 1 do not identify
a dominant relative position. Their article enrichment is a clue about their
lexical preferences; however, articles account for only about 21–22% of their
attention. We therefore cannot conclude that articles explain their overall
concentration or identify the remaining preferred keys from these aggregates.

| Broadest head | Normalized entropy | Observed behavior in Experiments 1–2 |
| --- | --- | --- |
| L0H11 | 0.95567 | Tested offsets receive roughly 1.4–1.7% attention, near the 1.74% causal-uniform baseline. |
| L0H9 | 0.95129 | Broad attention with a modest recent-token preference; previous-token attention is 3.56%. |
| L1H8 | 0.92383 | Nearly flat attention across tested offsets; mild article enrichment of 1.59×. |
| L1H10 | 0.91885 | Broad attention biased toward nearby tokens: self 4.82%, previous token 4.48%. |
| L1H7 | 0.91876 | Mild recent-token preference and period enrichment of 1.97×. |

We observe that broad attention can still have positional or lexical preferences.
L0H11 is close to uniform, whereas L0H9 and L1H10 distribute attention widely
while favoring nearby positions. High entropy does not imply an absence of
structure in the attention distribution.

Experiment 2B also separates attention concentration from no-op-like evidence.
The article/other mean absolute edge-gradient ratios for L5H1, L7H2, and L6H9
are 0.866, 0.953, and 0.844 respectively. These are only modest reductions in
local loss sensitivity, so the heads' low entropy and article enrichment do not
establish a no-op role. Among the broad heads, L0H9 and L1H7 have higher gradients
on periods, commas, and articles than on other tokens, despite some lexical
enrichment. For example, their period/other gradient ratios are 1.91 and 1.41.

In contrast, Experiment 2's clearest period candidate, L2H6, combines period
attention enrichment of 7.41× with a period/other gradient ratio of 0.241, yet
its normalized entropy on the matched Experiment 2 queries is 0.65831. This is
consistent with no-op-like behavior without placing the head among the most
focused heads. A preference for a token class need not concentrate attention on
a single key.

The raw and normalized layer trends agree qualitatively: layer 1 is broadest on
average (normalized layer mean 0.761), layer 7 is most focused (0.294), and layer
11 rises again (0.490). In our sample, attention concentration therefore does
not change monotonically with depth, and heads within a layer differ substantially.

The rankings above use 77,679 queries. Positional comparisons use Experiment 1's
46,679 real queries with `i >= 32`; lexical and gradient comparisons use
Experiment 2's 45,679 queries with `i >= 32` and a real next-token target.
Matched-query entropy values are available in
`results/entropy_behavior_comparison.csv`: for example, L4H11 has normalized
entropy 0.00133 on the Experiment 1 set, while L5H1, L7H2, and L6H9 have values
0.03998, 0.05334, and 0.08037 on the Experiment 2 set. Experiment 2 uses float32
weights, whereas Experiments 1 and 3 use float16 weights. These comparisons are
our descriptive observations, with no significance tests; attention and local
gradients do not establish causal importance or prove no-op behavior.
