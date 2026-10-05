"""Focused versus broad GPT-2 attention: raw and causal-normalized entropy."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import lab1

BATCH_SIZE = 16
SCOPES = ['all_real', 'experiment1_queries', 'experiment2_queries']


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    manifest = json.loads((root/'data/manifest.json').read_text())
    corpus = (root/'data/passages.jsonl').read_bytes()
    if hashlib.sha256(corpus).hexdigest() != manifest['corpus_sha256']:
        raise ValueError('Frozen corpus hash mismatch')
    passages = [json.loads(line) for line in corpus.splitlines()]
    weights = lab1.load_weights()
    wte, wpe = lab1.load_embedding_layers(weights)
    wte.eval()
    wpe.eval()
    device = wte.weight.device
    sums = torch.zeros(3, 2, 12, 12, dtype=torch.float64)
    counts = torch.zeros(3, 2, dtype=torch.long)
    uniform_sums = torch.zeros(3, dtype=torch.float64)
    diagnostics = dict(max_row_sum_error=0.0, max_future_attention=0.0,
                       max_padding_attention=0.0, min_entropy=float('inf'),
                       max_entropy_above_uniform=0.0, finite_entropy=True)
    with torch.inference_mode():
        for start in range(0, len(passages), BATCH_SIZE):
            sequences = [p['token_ids'] for p in passages[start:start+BATCH_SIZE]]
            width = max(map(len, sequences))
            ids = torch.full((len(sequences), width), 50256, dtype=torch.long, device=device)
            mask = torch.zeros_like(ids, dtype=torch.bool)
            for b, sequence in enumerate(sequences):
                ids[b, :len(sequence)] = torch.tensor(sequence, device=device)
                mask[b, :len(sequence)] = True
            attention = lab1.gpt2_attentions(ids, mask, weights, wte, wpe)
            # 0*log(0) = 0, including masked future and padding edges.
            entropy = -(attention * attention.clamp_min(torch.finfo(attention.dtype).tiny).log()).sum(-1).double()
            positions = torch.arange(width, device=device)
            max_entropy = (positions.double()+1).log()
            normalized = entropy / max_entropy.clamp_min(1)[None, None, None, :]
            # Use the actual log(i+1) for i>0, including log(2)<1.
            normalized[..., 1:] = entropy[..., 1:] / max_entropy[None, None, None, 1:]
            comparable2 = mask & (positions >= 32)
            comparable2 = comparable2.clone()
            comparable2[:, :-1] &= mask[:, 1:]
            comparable2[:, -1] = False
            query_sets = [mask, mask & (positions >= 32), comparable2]
            for s, valid in enumerate(query_sets):
                positive = valid & (positions > 0)
                sums[s, 0] += (entropy * valid[:, None, None, :]).sum((0, 3)).cpu()
                sums[s, 1] += (normalized * positive[:, None, None, :]).sum((0, 3)).cpu()
                counts[s, 0] += valid.sum().cpu()
                counts[s, 1] += positive.sum().cpu()
                uniform_sums[s] += (max_entropy * valid).sum().cpu()
            visible = positions[None, :] <= positions[:, None]
            diagnostics['max_row_sum_error'] = max(diagnostics['max_row_sum_error'],
                (attention.sum(-1)-mask[:, None, None, :].float()).abs().max().item())
            diagnostics['max_future_attention'] = max(diagnostics['max_future_attention'],
                attention.masked_fill(visible[None, None, None], 0).abs().max().item())
            diagnostics['max_padding_attention'] = max(diagnostics['max_padding_attention'],
                attention.masked_fill(mask[:, None, None, :, None] & mask[:, None, None, None, :], 0).abs().max().item())
            real_entropy = entropy.masked_select(mask[:, None, None, :])
            diagnostics['min_entropy'] = min(diagnostics['min_entropy'], real_entropy.min().item())
            diagnostics['max_entropy_above_uniform'] = max(diagnostics['max_entropy_above_uniform'],
                (entropy-max_entropy).masked_select(mask[:, None, None, :]).max().item())
            diagnostics['finite_entropy'] &= bool(torch.isfinite(entropy).all())
            if start % 160 == 0:
                print(f'Processed {start+len(sequences)}/{len(passages)} passages', flush=True)
    expected = [[sum(len(p['token_ids']) for p in passages), sum(len(p['token_ids'])-1 for p in passages)],
                [sum(len(p['token_ids'])-32 for p in passages)]*2,
                [sum(len(p['token_ids'])-33 for p in passages)]*2]
    if counts.tolist() != expected:
        raise ValueError(f'Query count mismatch: {counts.tolist()} versus {expected}')
    if (not diagnostics['finite_entropy'] or diagnostics['min_entropy'] < -1e-6 or
        diagnostics['max_entropy_above_uniform'] > 1e-5 or diagnostics['max_row_sum_error'] > 1e-5 or
        diagnostics['max_future_attention'] or diagnostics['max_padding_attention']):
        raise ValueError(f'Invalid entropy/attention: {diagnostics}')
    means = sums / counts[:, :, None, None]
    baseline = uniform_sums / counts[:, 0]
    results, figures = root/'results', root/'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    rankings = {}
    for metric, name in enumerate(['raw', 'normalized']):
        rows = []
        for s, scope in enumerate(SCOPES):
            for l in range(12):
                for h in range(12):
                    rows.append(dict(layer=l, head=h, query_set=scope,
                                     mean_entropy=means[s,metric,l,h].item(),
                                     uniform_baseline=baseline[s].item() if metric == 0 else 1.0,
                                     n_queries=int(counts[s,metric])))
        write_csv(results/f'entropy_{name}.csv', rows)
        for s, scope in enumerate(SCOPES):
            ranking = means[s,metric].flatten().argsort().tolist()
            rankings[f'{name}/{scope}'] = {
                label: [dict(layer=i//12, head=i%12, mean_entropy=means[s,metric,i//12,i%12].item()) for i in indices]
                for label, indices in [('focused', ranking[:5]), ('broad', ranking[-5:][::-1])]}
        print(f'{name}: {rankings[f"{name}/all_real"]}', flush=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        for h in range(12):
            ax.scatter(range(12), means[0,metric,:,h].numpy(), color='tab:blue', alpha=.65, s=22)
        ax.plot(range(12), means[0,metric].mean(1).numpy(), color='black', label='Layer mean')
        ax.axhline(baseline[0] if metric == 0 else 1, color='gray', linestyle='--', label='Causal uniform')
        ax.set(xlabel='Layer (zero-based)', ylabel='Mean entropy (nats)' if metric == 0 else 'Mean normalized entropy',
               title=f'GPT-2 {name} attention entropy\n{len(passages):,} Wikipedia openings; {int(counts[0,metric]):,} queries')
        ax.set_xticks(range(12))
        ax.set_ylim(bottom=0)
        ax.legend()
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(figures/f'entropy_{name}.{ext}', dpi=180)
        plt.close(fig)
    comparisons = []
    sources = {}
    for experiment, filename, scope, metafile in [
        (1, 'relative_position.csv', 1, 'relative_position_metadata.json'),
        (2, 'token_class_attention.csv', 2, 'token_class_metadata.json')]:
        source = (results/filename).read_bytes()
        previous_meta = json.loads((results/metafile).read_text())
        if previous_meta['corpus_sha256'] != manifest['corpus_sha256'] or previous_meta['model'] != manifest['model']:
            raise ValueError('Comparison results do not share corpus/model')
        sources[filename] = hashlib.sha256(source).hexdigest()
        prior = list(csv.DictReader(source.decode().splitlines()))
        for row in prior:
            l, h = int(row['layer']), int(row['head'])
            if int(row['n_queries']) != int(counts[scope,0]):
                raise ValueError('Comparison query counts differ')
            comparisons.append(dict(layer=l, head=h, experiment=experiment,
                category=f"offset_{row['offset']}" if experiment == 1 else row['token_class'],
                mean_attention=float(row['mean_attention']),
                enrichment=float(row['mean_attention'])/float(row['uniform_baseline']),
                raw_entropy=means[scope,0,l,h].item(), normalized_entropy=means[scope,1,l,h].item(),
                n_queries=int(counts[scope,0])))
    write_csv(results/'entropy_behavior_comparison.csv', comparisons)
    metadata = dict(dataset=manifest['dataset'], model=manifest['model'], corpus_sha256=manifest['corpus_sha256'],
        passage_count=len(passages), source_specification='https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16',
        formula='H = -sum_j A[i,j]*ln(A[i,j]); 0*ln(0)=0; normalized H = H/ln(i+1)',
        entropy_units='nats (natural logarithm)',
        query_sets={'all_real': 'raw: every real query i>=0; normalized: every real query i>0',
                    'experiment1_queries': 'every real query i>=32, including final token',
                    'experiment2_queries': 'every real query i>=32 with a real next-token target'},
        n_queries={scope: dict(raw=int(counts[s,0]), normalized=int(counts[s,1])) for s, scope in enumerate(SCOPES)},
        averaging='entropy computed per query before averaging; equal weight per eligible token across passages; normalization before averaging',
        uniform_baseline={scope: dict(raw=baseline[s].item(), normalized=1.0) for s, scope in enumerate(SCOPES)},
        indexing='zero-based layers and heads', batch_size=BATCH_SIZE,
        weight_dtype=str(wte.weight.dtype), attention_dtype='torch.float32', accumulation_dtype='torch.float64',
        diagnostics=diagnostics, rankings=rankings, comparison_source_sha256=sources,
        comparison_note='Same corpus/model and matching query sets; Experiment 1/3 weights float16, Experiment 2 float32. These are descriptive cross-experiment comparisons.',
        job_id=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(), device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        versions={p: importlib.metadata.version(p) for p in ['torch', 'matplotlib']},
        limitations=manifest['limitations']+['Entropy measures concentration, not causal importance for predictions.'])
    (results/'entropy_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    print(f'Saved entropy outputs; counts={counts.tolist()}; diagnostics={diagnostics}', flush=True)


if __name__ == '__main__':
    main()
