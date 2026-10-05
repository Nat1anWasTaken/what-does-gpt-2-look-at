"""Lexical attention destinations and next-token loss gradients (Lab Experiment 2)."""
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer
import lab1

BATCH_SIZE = 4
CLASSES = ['period', 'comma', 'articles', 'other']


def key_classes(passage, tokenizer):
    # Require a complete lexical span; do not mistake BPE word fragments for articles.
    encoded = tokenizer.encode(passage['text'], add_special_tokens=False)
    ids = passage['token_ids']
    if encoded.ids[:len(ids)] != ids:
        raise ValueError('Frozen text and tokenizer do not reproduce token IDs')
    spans = {(m.start(), m.end()) for m in re.finditer(r'\b(?:the|a|an)\b', passage['text'], re.I)}
    labels = []
    for start, end in encoded.offsets[:len(ids)]:
        piece = passage['text'][start:end]
        stripped = piece.strip()
        left = start + len(piece) - len(piece.lstrip())
        right = end - (len(piece) - len(piece.rstrip()))
        labels.append(0 if stripped == '.' else 1 if stripped == ',' else
                      2 if (left, right) in spans else 3)
    return labels


def write_csv(path, columns, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main():
    torch.set_num_threads(4)
    root = Path(__file__).resolve().parent
    corpus = (root / 'data/passages.jsonl').read_bytes()
    manifest = json.loads((root / 'data/manifest.json').read_text())
    if hashlib.sha256(corpus).hexdigest() != manifest['corpus_sha256']:
        raise ValueError('Corpus hash mismatch')
    passages = [json.loads(line) for line in corpus.splitlines()]
    tokenizer = Tokenizer.from_file(str(root / 'data/tokenizer.json'))
    labels = [key_classes(p, tokenizer) for p in passages]
    weights = {k: v.float() for k, v in lab1.load_weights().items()}
    wte, wpe = lab1.load_embedding_layers(weights)
    for module in (wte, wpe):
        module.eval()
        module.requires_grad_(False)
    device = wte.weight.device
    sums = torch.zeros(12, 12, 4, 3, dtype=torch.float64)
    baseline_sum = torch.zeros(4, dtype=torch.float64)
    edges = torch.zeros(4, dtype=torch.float64)
    count = loss_count = 0
    loss_sum = 0.0
    diagnostics = {'max_row_sum_error': 0.0, 'future_mass': 0.0, 'padding_mass': 0.0,
                   'finite_gradients': True}
    for start in range(0, len(passages), BATCH_SIZE):
        batch = passages[start:start + BATCH_SIZE]
        width = max(len(p['token_ids']) for p in batch)
        ids = torch.full((len(batch), width), 50256, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids, dtype=torch.bool)
        categories = torch.full_like(ids, -1)
        for b, p in enumerate(batch):
            n = len(p['token_ids'])
            ids[b, :n] = torch.tensor(p['token_ids'], device=device)
            mask[b, :n] = True
            categories[b, :n] = torch.tensor(labels[start+b], device=device)
        hidden = lab1.embed_token_ids(ids, mask, wte, wpe).detach().requires_grad_(True)
        attentions = []
        for layer in range(12):
            hidden, attention = lab1.transformer_block(hidden, mask, weights, layer, return_attention=True)
            attentions.append(attention)
        hidden = torch.nn.functional.layer_norm(hidden, (768,), weights['ln_f.weight'], weights['ln_f.bias'])
        logits = lab1.project_to_vocabulary(hidden[:, :-1], wte)
        targets = ids[:, 1:].masked_fill(~mask[:, 1:], -100)
        # Sum avoids changing gradient scale with batch size or padding.
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 50257), targets.reshape(-1),
                                                 ignore_index=-100, reduction='sum')
        gradients = torch.autograd.grad(loss, attentions)
        loss_sum += loss.item()
        loss_count += mask[:, 1:].sum().item()
        positions = torch.arange(width, device=device)
        queries = mask & (positions[None, :] >= 32)
        queries[:, :-1] &= mask[:, 1:]
        queries[:, -1] = False
        visible = positions[None, :] <= positions[:, None]
        count += queries.sum().item()
        for c in range(4):
            selected = (categories == c)[:, None, :] & visible[None, :, :] & queries[:, :, None]
            edges[c] += selected.sum().item()
            baseline_sum[c] += (selected.sum(-1).double() / (positions + 1)).sum().item()
            for layer, (attention, gradient) in enumerate(zip(attentions, gradients)):
                a, g = attention.detach(), gradient.detach().abs()
                chosen = selected[:, None, :, :]
                sums[layer, :, c, 0] += (a.double() * chosen).sum((0, 2, 3)).cpu()
                sums[layer, :, c, 1] += (g.double() * chosen).sum((0, 2, 3)).cpu()
                sums[layer, :, c, 2] += (a.double() * g.double() * chosen).sum((0, 2, 3)).cpu()
        for a, g in zip(attentions, gradients):
            a = a.detach()
            diagnostics['max_row_sum_error'] = max(diagnostics['max_row_sum_error'],
                (a.sum(-1) - mask[:, None, :].float()).abs().max().item())
            diagnostics['future_mass'] = max(diagnostics['future_mass'], a.masked_select(~visible[None, None]).abs().max().item())
            diagnostics['padding_mass'] = max(diagnostics['padding_mass'],
                a.masked_fill(mask[:, None, :, None] & mask[:, None, None, :], 0).abs().max().item())
            diagnostics['finite_gradients'] &= bool(torch.isfinite(g).all())
        del gradients, attentions, logits, hidden, loss
        if start % 80 == 0:
            print(f'Processed {start+len(batch)}/{len(passages)} passages', flush=True)
    if not diagnostics['finite_gradients'] or diagnostics['future_mass'] or diagnostics['padding_mass']:
        raise ValueError(f'Invalid attention/gradients: {diagnostics}')
    if diagnostics['max_row_sum_error'] > 1e-5:
        raise ValueError(f'Attention row normalization failed: {diagnostics}')
    baseline = baseline_sum / count
    mass = sums[..., 0] / count
    edge_gradient = sums[..., 1] / edges
    weighted_gradient = sums[..., 2] / sums[..., 0]
    rows, gradient_rows = [], []
    for l in range(12):
        for h in range(12):
            for c, name in enumerate(CLASSES):
                rows.append(dict(layer=l, head=h, token_class=name, mean_attention=mass[l,h,c].item(),
                                 uniform_baseline=baseline[c].item(), enrichment=(mass[l,h,c]/baseline[c]).item(), n_queries=count))
                gradient_rows.append(dict(layer=l, head=h, token_class=name,
                    mean_abs_gradient_per_edge=edge_gradient[l,h,c].item(),
                    attention_weighted_abs_gradient=weighted_gradient[l,h,c].item(),
                    gradient_ratio_to_other=(edge_gradient[l,h,c]/edge_gradient[l,h,3]).item(),
                    n_edges=int(edges[c]), n_queries=count))
    results, figures = root/'results', root/'figures'
    results.mkdir(exist_ok=True)
    figures.mkdir(exist_ok=True)
    write_csv(results/'token_class_attention.csv', list(rows[0]), rows)
    write_csv(results/'token_class_gradient_importance.csv', list(gradient_rows[0]), gradient_rows)
    top = {}
    for c, name in enumerate(CLASSES[:3]):
        indices = (mass[:,:,c]/baseline[c]).flatten().argsort(descending=True)[:5].tolist()
        top[name] = [dict(layer=i//12, head=i%12, mean_attention=mass[i//12,i%12,c].item(),
                         enrichment=(mass[i//12,i%12,c]/baseline[c]).item(),
                         gradient_ratio_to_other=(edge_gradient[i//12,i%12,c]/edge_gradient[i//12,i%12,3]).item()) for i in indices]
        print(f'{name}: {top[name]}', flush=True)
    metadata = dict(corpus_sha256=manifest['corpus_sha256'], model=manifest['model'], dataset=manifest['dataset'],
        source_specification='https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16',
        passage_count=len(passages), n_queries=count, n_loss_tokens=loss_count,
        query_set='real zero-based i >= 32 with a real next-token target; final real token excluded',
        averaging='equal weight per eligible query token across passages for attention and uniform baseline',
        token_classes={'period': 'single token with whitespace-stripped source span exactly .',
                       'comma': 'single token with whitespace-stripped source span exactly ,',
                       'articles': 'complete case-insensitive word the/a/an in one token, with Unicode word boundaries',
                       'other': 'all remaining real tokens; includes multi-character punctuation tokens'},
        loss='summed next-token cross-entropy over all real within-passage transitions; final layer norm applied',
        gradient='absolute derivative of summed corpus loss with respect to post-softmax attention probabilities; unconstrained edge partial derivatives',
        gradient_averaging='sum absolute gradients / visible class edge count; attention-weighted gradient = sum(A*abs(grad))/sum(A)',
        gradient_context='includes downstream loss terms within each passage; not an isolated per-query loss derivative',
        indexing='zero-based layers/heads', batch_size=BATCH_SIZE, weight_dtype='torch.float32',
        accumulation_dtype='torch.float64', mean_next_token_loss=loss_sum/loss_count,
        uniform_baseline=dict(zip(CLASSES, baseline.tolist())), n_edges=dict(zip(CLASSES, edges.tolist())),
        diagnostics=diagnostics, top_five_heads_by_enrichment=top,
        job_id=os.environ.get('SLURM_JOB_ID'), hostname=socket.gethostname(), device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        versions={p: importlib.metadata.version(p) for p in ['torch', 'tokenizers', 'matplotlib']},
        interpretation='High enrichment plus lower gradient than ordinary keys is consistent with no-op-like behavior, not proof of causal irrelevance. Comparisons are descriptive, without significance tests.',
        limitations=manifest['limitations'])
    (results/'token_class_metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    for filename, values, ylabel in [('token_class_attention', mass, 'Mean attention mass'),
                                    ('token_class_gradient_importance', edge_gradient, 'Mean absolute loss gradient per visible edge')]:
        fig, axes = plt.subplots(1, 4, figsize=(17, 4.5))
        for c, ax in enumerate(axes):
            for h in range(12):
                ax.scatter(range(12), values[:,h,c].numpy(), s=12, alpha=.65)
            ax.plot(range(12), values[:,:,c].mean(1).numpy(), color='black', label='Layer mean')
            if filename == 'token_class_attention':
                ax.axhline(baseline[c], color='gray', linestyle='--', label='Causal uniform')
            ax.set(title=CLASSES[c], xlabel='Layer (zero-based)', ylabel=ylabel)
            ax.set_xticks(range(12))
            ax.legend(fontsize=8)
        fig.suptitle(f'GPT-2 lexical attention: {len(passages):,} openings; {count:,} queries')
        fig.tight_layout()
        for ext in ['png', 'svg']:
            fig.savefig(figures/f'{filename}.{ext}', dpi=180)
        plt.close(fig)
    print(f'Saved Experiment 2 outputs; queries={count}; loss tokens={loss_count}; diagnostics={diagnostics}', flush=True)


if __name__ == '__main__':
    main()
