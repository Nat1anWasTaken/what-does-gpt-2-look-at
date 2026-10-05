# What Does GPT-2 Look At? An Analysis of Attention in a Causal Language Model

## Introduction

Pretrained language models acquire linguistic regularities through prediction objectives, but predictive performance alone does not reveal how these regularities are represented. Research on model analysis therefore examines complementary sources of evidence: predictions on controlled inputs, linguistic information recoverable from hidden representations, and the structure of attention distributions. Attention analysis makes token-to-token interactions observable, although these interactions do not by themselves explain a prediction. This study follows the attention-based approach of [Clark et al. (2019)](https://aclanthology.org/W19-4828/).

We investigate which attention patterns transfer from bidirectional BERT to GPT-2 Small. Four experiments characterize positional preferences, potential lexical no-op destinations, attention concentration, and similarity between heads. Following the [October 1 lab revision](https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16), we examine punctuation and articles rather than inserting artificial `<|endoftext|>` tokens. Our central question is whether GPT-2 exhibits comparable behavioral organization despite its restriction to preceding context.

## Experimental setup

We use the course implementation of GPT-2 Small (124M parameters), with 12 layers and 12 heads per layer, in evaluation mode. All head labels are **zero-based**: L4H11 denotes layer 4, head 11. We analyze post-softmax attention probabilities, denoted by $A^{(h)}_{x,i,j}$ for head $h$, passage $x$, query position $i$, and key position $j$. Causal masking permits only $j\leq i$; padding is excluded.

The frozen corpus contains 1,000 English Wikipedia article openings from `wikimedia/wikipedia`, snapshot `20231101.en`, train split. Seed 42 selects shard 40 of 41; a separate reservoir sample with the same seed selects 1,000 of its 123,794 eligible openings. Each opening is the first nonempty paragraph, contains at least 33 GPT-2 tokens, and is truncated to 128 tokens. No boundary tokens are inserted. The resulting corpus contains 78,679 tokens. Exact passages, token IDs, article attribution, source revisions, and preprocessing are preserved in [the data manifest](data/manifest.json) and [frozen corpus](data/passages.jsonl).

Each statistic is computed for individual queries before aggregation. Eligible query tokens receive equal weight across passages; passages therefore contribute in proportion to their eligible lengths. Query sets differ according to the measurement:

| Analysis | Eligible queries | Count per head or head pair |
| --- | --- | ---: |
| Relative positions | $i\geq32$ | 46,679 |
| Lexical attention and gradients | $i\geq32$ with a within-passage next-token target | 45,679 |
| Raw entropy and head similarity | All real tokens | 78,679 |
| Normalized entropy | All real tokens except $i=0$ | 77,679 |

Matched-query entropy estimates are also available for the first two sets. Experiments 1, 3, and 4 use float16 model weights and float32 attention; Experiment 2 uses float32 weights. Saved diagnostics report zero attention to future and padded positions and a maximum attention normalization error of $4.77\times10^{-7}$. All reported results come from completed Slurm runs.

## Results

### 1. Relative-position specialization

For backward offsets $d\in\{0,1,2,4,8,16,32\}$, we measure

$$
R_h(d)=\operatorname{mean}_{x,i\geq32} A^{(h)}_{x,i,i-d}.
$$

Using the same queries for every offset avoids differences caused by query eligibility. Uniform attention over each visible prefix gives a mean baseline of 1.74% per tested position.

![Mean attention at selected backward offsets for all heads](figures/relative_position.png)

*Figure 1. Positional preferences on the common query set. Offset zero denotes the current token; positive distances denote earlier tokens.*

The strongest self-attending head, **L0H3**, assigns 84.91% of its attention to the current token. Five heads exceed 50% self-attention. Two heads exceed 50% previous-token attention: **L2H2** (55.41%) and **L4H11** (99.86%). No head assigns at least 25% to any tested offset beyond one token. This excludes strong specialization at those particular offsets under our threshold, not attention to distant context in general.

Previous-token specialization agrees qualitatively with BERT. Strong self-attention is also prominent in this GPT-2 sample, whereas Clark et al. describe little self-attention in most BERT heads. Future-token attention is structurally impossible in GPT-2; its absence is not evidence of a failed learned behavior.

### 2. Lexical destinations and the no-op hypothesis

We test whether frequently attended lexical classes also have low local loss sensitivity. Candidate classes are periods, commas, and the articles *the*, *a*, and *an*. Punctuation must occupy a complete token after stripping whitespace; articles must be complete, case-insensitive words represented by one token. All remaining real tokens form the comparison class, `other`.

For class $C$, attention mass and its causal-uniform baseline are

$$
M_C(i,h)=\sum_{j\leq i:\,x_j\in C} A^{(h)}_{x,i,j},
\qquad
U_C(i)=\frac{\#\{j\leq i:x_j\in C\}}{i+1}.
$$

Enrichment is $E_C(h)=\mathbb{E}[M_C]/\mathbb{E}[U_C]$. This controls for class availability under uniform attention. We then measure $|\partial L/\partial A^{(h)}_{x,i,j}|$, where $L$ is summed next-token cross-entropy over all 77,679 within-passage transitions. The loss includes early positions and downstream predictions; gradient summaries use the selected queries above. Class importance is the mean absolute gradient per visible class edge, reported relative to `other` within the same head.

![Attention to periods, commas, articles, and other tokens by layer](figures/token_class_attention.png)

*Figure 2. Mean attention to lexical classes. Points represent individual heads; lines summarize layers.*

![Mean absolute attention-edge gradients by token class and layer](figures/token_class_gradient_importance.png)

*Figure 3. Local sensitivity of next-token loss to attention probabilities. High attention and low gradient sensitivity provide complementary evidence when evaluating a no-op interpretation.*

| Class | Most enriched head | Mean attention | Uniform baseline | Enrichment | Gradient / other |
| --- | --- | ---: | ---: | ---: | ---: |
| Period | L2H6 | 19.63% | 2.65% | 7.41× | 0.241 |
| Comma | L2H5 | 10.38% | 4.22% | 2.46× | 0.717 |
| Articles | L7H2 | 22.40% | 7.62% | 2.94× | 0.953 |

**L2H6 provides the clearest no-op-like evidence:** periods attract strongly enriched attention while their mean edge gradient is approximately one quarter of that for other tokens. This is consistent with a relatively insensitive attention destination. It does not establish that periods are semantically empty or that this head has no effect on predictions.

Other classes provide weaker evidence. L1H5 has comma enrichment of 2.33× but a gradient ratio of 1.18. The five most article-enriched heads have ratios of 0.84–0.95, indicating only modest reductions in sensitivity. Thus, lexical preference alone does not support a general no-op interpretation. Our findings offer a limited analogue of BERT's separator result, rather than evidence for a widespread lexical substitute for `[SEP]`.

### 3. Focused and broad attention

We compute entropy over each visible prefix using natural logarithms:

$$
H_{x,i,h}=-\sum_{j=0}^{i}A^{(h)}_{x,i,j}\log A^{(h)}_{x,i,j}.
$$

Because the maximum is $\log(i+1)$, we also compute $H_{x,i,h}/\log(i+1)$ for $i>0$, normalizing **before** averaging. Raw entropy retains the original measurement; normalized entropy expresses concentration relative to available context. The mean raw uniform baseline is 3.47909 nats, and the normalized uniform baseline is one.

![Raw attention entropy by layer](figures/entropy_raw.png)

*Figure 4. Raw entropy for every head and layer means. Lower values indicate more concentrated attention.*

![Attention entropy normalized by visible context size](figures/entropy_normalized.png)

*Figure 5. Entropy normalized by the maximum at each query position, excluding the single-key first position.*

| Head | Mean normalized entropy | Relationship to the preceding experiments |
| --- | ---: | --- |
| L4H11 | 0.00168 | Almost exclusive previous-token attention |
| L5H1 | 0.03294 | Article enrichment of 2.87× |
| L7H2 | 0.04699 | Article enrichment of 2.94× |
| L6H9 | 0.06608 | Article enrichment of 2.76× |
| L0H1 | 0.09515 | Self-attention of 83.02% |
| L0H11 | 0.95567 | Broadest head; tested offsets near the uniform baseline |
| L0H9 | 0.95129 | Broad attention with a modest previous-token preference |

The five broadest heads occur in layers 0–1. Layer 1 has the highest mean normalized entropy (0.761), layer 7 the lowest (0.294), and layer 11 rises to 0.490. Concentration therefore varies non-monotonically with depth, with substantial variation within layers.

Entropy also distinguishes different aspects of specialization. L4H11's low entropy is explained by its positional preference. In contrast, L5H1, L7H2, and L6H9 concentrate sharply without a dominant tested offset; articles account for only approximately 21–22% of their attention. Their remaining preferred keys cannot be identified from these summaries. Conversely, L2H6 has normalized entropy 0.65831 on the matched lexical query set: the strongest no-op-like candidate is not among the most focused heads. Concentration, lexical preference, and local sensitivity are distinct properties.

### 4. Similarity between heads

For each pair of heads, we average Jensen–Shannon divergence between distributions for the **same passage and query**:

$$
D(h_a,h_b)=\mathbb{E}_{x,i}[\operatorname{JS}(A^{(h_a)}_{x,i,:},A^{(h_b)}_{x,i,:})],
\qquad
\operatorname{JS}(P,Q)=\tfrac12\operatorname{KL}(P\|M)+\tfrac12\operatorname{KL}(Q\|M),
\quad M=(P+Q)/2.
$$

We use divergence in nats, without taking its square root. Metric multidimensional scaling (MDS) embeds the resulting 144 × 144 matrix in two dimensions, using seed 42 and eight initializations. Both views below use identical coordinates.

![MDS embedding of attention heads colored by layer](figures/mds_by_layer.png)

*Figure 6. Head similarity by layer. Proximity approximates mean same-query divergence; axes have no intrinsic interpretation.*

![The same MDS embedding annotated by attention behavior](figures/mds_by_behavior.png)

*Figure 7. The same coordinates labeled by behaviors from Experiments 1–3. Labels are descriptive categories, not fitted clusters, and can overlap in the underlying data.*

Heads in the same layer have lower mean divergence (0.17680) than heads in different layers (0.25449). The nearest neighbor shares its layer for 20.83% of heads, compared with 7.69% under uniform random neighbor selection. This indicates a same-layer tendency, without complete separation by depth or a statistical significance claim.

| Behavioral group | Heads | Within-group JSD | Group-to-other JSD |
| --- | ---: | ---: | ---: |
| Self-attending | 5 | 0.13630 | 0.51051 |
| Previous-token | 2 | 0.19924 | 0.49087 |
| Broad | 16 | 0.10779 | 0.32470 |
| Focused | 5 | 0.47589 | 0.38257 |
| Article-enriched | 77 | 0.13863 | 0.29521 |
| Punctuation-enriched | 11 | 0.24816 | 0.30265 |

Groups use exploratory thresholds: positional attention ≥ 0.5, normalized entropy ≥ 0.8 for broad or ≤ 0.1 for focused, and lexical enrichment ≥ 2. A no-op-like candidate additionally requires a gradient ratio ≤ 0.5. Only L2H6 meets that joint criterion, so a no-op-like cluster cannot be assessed.

Broad and positional groups are relatively coherent. Focused heads, however, are more divergent from one another than from other heads on average: distributions can be equally concentrated while selecting different keys. Within this heterogeneous group, **L5H1–L7H2 is the closest pair in the entire matrix** (0.01550), and both are close to L6H9 (0.01791 and 0.02124). Their agreement on the same queries strengthens the evidence for a shared attention pattern beyond their similar entropy and article enrichment. It does not identify that pattern's causal function.

These comparisons use the original divergence matrix. The MDS visualization has Stress-1 of 0.11744 and pairwise distance correlation of 0.97648, preserving substantial structure while distorting individual distances.

## Integrated discussion

The experiments show that GPT-2 attention has several forms of organization also reported for BERT: previous-token specialization, broad early-layer attention, and a tendency for heads within a layer to behave similarly. These correspondences are qualitative; we did not rerun BERT under our preprocessing and averaging conventions. The comparison concerns recurring patterns rather than matched effect sizes. [Clark et al. (2019)](https://aclanthology.org/W19-4828/)

Taken together, the measurements distinguish **where attention goes, how concentrated it is, and how sensitive the loss is to its weights**. L4H11 combines a stable positional target with extremely low entropy. L5H1, L7H2, and L6H9 share concentrated same-query distributions across layers, but their modest article-gradient reductions do not establish a no-op role. L2H6 exhibits the clearest low-sensitivity lexical preference without extreme concentration. No single metric captures all these behaviors, and a visually coherent group need not correspond to a common predictive function.

The causal architecture explains some differences exactly. Attention to a future token is always zero. A broad GPT-2 head can aggregate only the available prefix, and its entropy ceiling grows with query position. A terminal document-boundary token cannot receive attention from preceding queries. These facts motivate the shared positional query set, prefix-normalized entropy, and revised lexical analysis.

Other differences cannot be attributed to masking alone. The mask does not force strong self-attention, determine which lexical classes attract attention, or require a particular depth profile. GPT-2 and BERT also differ in training objectives, tokenization, training data, and input formatting. In particular, our lexical results do not reproduce BERT's widespread separator-associated no-op pattern, but omitting artificial boundary tokens makes a direct separator comparison unavailable. The evidence supports a limited lexical analogue, not the absence of no-op behavior in GPT-2 generally.

Causal attention may favor backward-looking strategies, but quantifying its contribution would require controlled models differing in attention direction while holding other factors fixed. Our study therefore identifies architectural constraints and plausible interpretations without estimating what fraction of the empirical differences they explain. BERT's reported syntactic and coreference specializations also remain untested here; surface preferences cannot substitute for annotated linguistic evaluation.

## Limitations

**Dataset and context length.** The sample covers one English Wikipedia shard and article openings only. Excluding openings shorter than 33 tokens and retaining at most 128 tokens favors particular passage lengths and discourse positions. The results may differ for other genres, languages, later paragraphs, or longer contexts. Short windows also restrict evidence about distant dependencies and document-level behavior.

**Model coverage and comparison.** We study one pretrained GPT-2 Small checkpoint, with no replication across model sizes or training seeds. Comparison with published BERT results is confounded by differences in input construction, tokenization, objectives, and training corpora. Shared qualitative patterns do not demonstrate a universal property of Transformer attention.

**Averaging and operational definitions.** Equal weighting of query tokens gives longer passages more influence. Gradient means instead weight visible edges, and class enrichment is a ratio of aggregate means. These summaries can conceal passage-specific behavior and differences in token position. Query eligibility varies across experiments; matched entropy estimates reduce, but do not remove, this issue. Single-token lexical definitions exclude split words and bundled punctuation, and the uniform baseline does not control for syntax or positional preferences. Position zero contributes zero entropy and zero pairwise divergence. Weight precision also differs in Experiment 2.

**Uncertainty and visualization.** We report descriptive estimates without confidence intervals, resampling across passages, or corrections for selecting extreme heads. Behavioral thresholds are exploratory. MDS compresses the divergence matrix, and apparent separation does not establish discrete functional clusters.

**Attention and causality.** Attention weights describe how value vectors are combined, not their content or their ultimate contribution through residual and downstream computations. Gradients add local sensitivity evidence but are unconstrained derivatives of summed passage loss; they do not measure a finite redistribution that preserves attention normalization. Low gradients may reflect local insensitivity rather than a true no-op mechanism. Head ablations or controlled attention interventions would be needed to assess causal contributions to prediction.

## Reproducibility

The frozen [manifest](data/manifest.json) pins dataset revision `ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619` and model/tokenizer revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Per-head measurements and methodological metadata are in [results](results/); figures are available as PNG and SVG in [figures](figures/). The earlier WikiText archive is not part of this report. Credit for source text belongs to Wikipedia contributors; article URLs and dataset licensing are recorded with the corpus.

All computation must run on Slurm-allocated compute nodes. From the repository root, submit `prepare_inspect.sbatch` if preparing the corpus, then `experiment1.sbatch`, `experiment2.sbatch`, and `experiment3.sbatch`. After these complete, submit `experiment4.sbatch`, which uses their outputs. For example:

```bash
mkdir -p logs
sbatch --exclude=25a-hgpn001 experiment1.sbatch
```

The batch scripts request one GPU, four CPUs, 32 GB RAM, and 30 minutes, install the locked dependencies with `uv`, and execute through `srun`. Cluster account and partition settings may require adaptation.

## Partner contributions

## References

Clark, K., Khandelwal, U., Levy, O., and Manning, C. D. (2019). [What Does BERT Look at? An Analysis of BERT's Attention](https://aclanthology.org/W19-4828/). *Proceedings of the 2019 ACL Workshop BlackboxNLP*, 276–286. [Original analysis code](https://github.com/clarkkev/attention-analysis).

[Lab 3 — GPT-2 Attention Analysis](https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16). Assignment and experiment specification, including the October 1, 2026 revision.

Wikimedia/Wikipedia contributors. [English Wikipedia snapshot, 20231101.en](https://huggingface.co/datasets/wikimedia/wikipedia/tree/ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619/20231101.en). Frozen source revision recorded above.

## Conclusions

GPT-2 Small exhibits positional specialization, broad early-layer attention, and structured similarity among heads, extending several qualitative BERT findings to a causal language model. The four experiments also show that concentration, lexical preference, and loss sensitivity must be interpreted separately: focused heads need not share targets, and enriched lexical attention need not be a no-op. L2H6 offers the clearest but still provisional lexical no-op analogue. Causal masking fully explains the absence of future attention and the changing entropy ceiling; it does not by itself explain the remaining learned patterns. The main contribution is a reproducible characterization of attention behavior, with causal function remaining an open question.
