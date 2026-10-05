"""Same-query mean Jensen-Shannon divergence between GPT-2 heads and 2D MDS."""
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import MDS
import torch
import lab1

BATCH_SIZE = 16
QUERY_CHUNK = 128
PAIR_CHUNK = 256
SEED = 42


def entropy(p):
    return -(p * p.clamp_min(torch.finfo(p.dtype).tiny).log()).sum(-1)


def js_pairs(p, left, right):
    a, b = p[:, left, :], p[:, right, :]
    return entropy((a+b)*0.5) - (entropy(a)+entropy(b))*0.5


def validate_formula(device):
    p = torch.tensor([[[1., 0.], [0., 1.], [.5, .5]]], device=device)
    left = torch.tensor([0, 0, 1], device=device)
    right = torch.tensor([0, 1, 0], device=device)
    values = js_pairs(p, left, right)
    if not torch.allclose(values, torch.tensor([[0., math.log(2), math.log(2)]], device=device), atol=1e-6):
        raise ValueError('JSD identity/disjoint/symmetry checks failed')
    # Independent direct KL formula on overlapping distributions.
    a = torch.tensor([[[.2, .8], [.7, .3]]], dtype=torch.float64, device=device)
    mix = a.mean(1)
    reference = .5 * (a[:,0]*(a[:,0]/mix).log() + a[:,1]*(a[:,1]/mix).log()).sum(-1)
    if not torch.allclose(js_pairs(a, torch.tensor([0],device=device), torch.tensor([1],device=device))[:,0], reference, atol=1e-12):
        raise ValueError('JSD direct-KL check failed')


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def behaviors(root, manifest):
    results = root/'results'
    sources = {}
    for filename in ['relative_position_metadata.json', 'token_class_metadata.json', 'entropy_metadata.json']:
        m = json.loads((results/filename).read_text())
        if m['corpus_sha256'] != manifest['corpus_sha256'] or m['model'] != manifest['model']:
            raise ValueError('Behavior sources must share corpus and model')
    def read(filename):
        raw = (results/filename).read_bytes()
        sources[filename] = hashlib.sha256(raw).hexdigest()
        return list(csv.DictReader(raw.decode().splitlines()))
    offsets = {(int(r['layer'])*12+int(r['head']),int(r['offset'])):float(r['mean_attention']) for r in read('relative_position.csv')}
    lexical = {(int(r['layer'])*12+int(r['head']),r['token_class']):float(r['enrichment']) for r in read('token_class_attention.csv')}
    gradient = {(int(r['layer'])*12+int(r['head']),r['token_class']):float(r['gradient_ratio_to_other']) for r in read('token_class_gradient_importance.csv')}
    ent = {int(r['layer'])*12+int(r['head']):float(r['mean_entropy']) for r in read('entropy_normalized.csv') if r['query_set']=='all_real'}
    rows = []
    for i in range(144):
        flags = []
        if offsets[i,1] >= .5: flags.append('previous-token')
        if offsets[i,0] >= .5: flags.append('self')
        if max(offsets[i,d] for d in [2,4,8,16,32]) >= .25: flags.append('longer-offset')
        for c in ['period','comma','articles']:
            if lexical[i,c] >= 2 and gradient[i,c] <= .5:
                flags.append('no-op-like candidate')
                break
        if lexical[i,'articles'] >= 2: flags.append('article-enriched')
        if max(lexical[i,'period'],lexical[i,'comma']) >= 2: flags.append('punctuation-enriched')
        if ent[i] >= .8: flags.append('broad')
        if ent[i] <= .1: flags.append('focused')
        rows.append(dict(layer=i//12, head=i%12, category=flags[0] if flags else 'other',
                         behavior_flags=';'.join(flags), normalized_entropy=ent[i],
                         self_attention=offsets[i,0], previous_attention=offsets[i,1],
                         article_enrichment=lexical[i,'articles'], period_enrichment=lexical[i,'period']))
    return rows, sources


def summarize(distance, rows):
    a,b = np.triu_indices(144,1)
    same = a//12 == b//12
    summary = dict(mean_same_layer_jsd=float(distance[a[same],b[same]].mean()),
                   mean_different_layer_jsd=float(distance[a[~same],b[~same]].mean()))
    nearest = distance.copy()
    np.fill_diagonal(nearest, np.inf)
    summary['nearest_neighbor_same_layer_fraction'] = float(np.mean(nearest.argmin(1)//12 == np.arange(144)//12))
    summary['same_layer_random_neighbor_baseline'] = 11/143
    groups = {}
    for flag in ['self','previous-token','longer-offset','broad','focused','article-enriched','punctuation-enriched','no-op-like candidate']:
        ids = [i for i,r in enumerate(rows) if flag in r['behavior_flags'].split(';')]
        outside = [i for i in range(144) if i not in ids]
        group = dict(n_heads=len(ids), heads=[f'L{i//12}H{i%12}' for i in ids])
        if len(ids)>1:
            within = distance[np.ix_(ids,ids)]
            group['mean_within_jsd'] = float(within[np.triu_indices(len(ids),1)].mean())
        if ids and outside:
            group['mean_to_other_heads_jsd'] = float(distance[np.ix_(ids,outside)].mean())
        groups[flag] = group
    summary['behavior_groups'] = groups
    order = np.argsort(distance[a,b])[:10]
    summary['closest_pairs'] = [dict(head_a=f'L{a[k]//12}H{a[k]%12}', head_b=f'L{b[k]//12}H{b[k]%12}', jsd=float(distance[a[k],b[k]])) for k in order]
    return summary


def main():
    torch.set_num_threads(4)
    root=Path(__file__).resolve().parent
    manifest=json.loads((root/'data/manifest.json').read_text())
    corpus=(root/'data/passages.jsonl').read_bytes()
    if hashlib.sha256(corpus).hexdigest()!=manifest['corpus_sha256']: raise ValueError('Corpus hash mismatch')
    passages=[json.loads(line) for line in corpus.splitlines()]
    rows, sources=behaviors(root,manifest)
    weights=lab1.load_weights()
    wte,wpe=lab1.load_embedding_layers(weights)
    wte.eval(); wpe.eval()
    device=wte.weight.device
    validate_formula(device)
    pairs=torch.triu_indices(144,144,offset=1,device=device)
    total=torch.zeros(pairs.shape[1],dtype=torch.float64,device=device)
    count=0
    diagnostics=dict(max_row_sum_error=0., max_future_attention=0., max_padding_attention=0.,
                     min_query_jsd=float('inf'),max_query_jsd=0.)
    with torch.inference_mode():
        for start in range(0,len(passages),BATCH_SIZE):
            sequences=[p['token_ids'] for p in passages[start:start+BATCH_SIZE]]
            width=max(map(len,sequences))
            ids=torch.full((len(sequences),width),50256,dtype=torch.long,device=device)
            mask=torch.zeros_like(ids,dtype=torch.bool)
            for b,seq in enumerate(sequences):
                ids[b,:len(seq)]=torch.tensor(seq,device=device)
                mask[b,:len(seq)]=True
            attention=lab1.gpt2_attentions(ids,mask,weights,wte,wpe)
            positions=torch.arange(width,device=device)
            visible=positions[None,:]<=positions[:,None]
            diagnostics['max_row_sum_error']=max(diagnostics['max_row_sum_error'],(attention.sum(-1)-mask[:,None,None,:].float()).abs().max().item())
            diagnostics['max_future_attention']=max(diagnostics['max_future_attention'],attention.masked_fill(visible[None,None,None],0).abs().max().item())
            diagnostics['max_padding_attention']=max(diagnostics['max_padding_attention'],attention.masked_fill(mask[:,None,None,:,None]&mask[:,None,None,None,:],0).abs().max().item())
            p=attention.reshape(len(sequences),144,width,width).permute(0,2,1,3)[mask]
            count+=p.shape[0]
            for q in range(0,p.shape[0],QUERY_CHUNK):
                query=p[q:q+QUERY_CHUNK]
                h=entropy(query)
                for k in range(0,pairs.shape[1],PAIR_CHUNK):
                    left,right=pairs[:,k:k+PAIR_CHUNK]
                    mixture=(query[:,left,:]+query[:,right,:])*.5
                    js=entropy(mixture)-(h[:,left]+h[:,right])*.5
                    diagnostics['min_query_jsd']=min(diagnostics['min_query_jsd'],js.min().item())
                    diagnostics['max_query_jsd']=max(diagnostics['max_query_jsd'],js.max().item())
                    total[k:k+PAIR_CHUNK]+=js.double().sum(0)
            if start%80==0: print(f'Processed {start+len(sequences)}/{len(passages)} passages',flush=True)
    if count!=sum(len(p['token_ids']) for p in passages): raise ValueError('Query count mismatch')
    if diagnostics['min_query_jsd'] < -2e-6 or diagnostics['max_query_jsd'] > math.log(2)+2e-6 or diagnostics['max_row_sum_error']>1e-5 or diagnostics['max_future_attention'] or diagnostics['max_padding_attention']:
        raise ValueError(f'Invalid probabilities/JSD: {diagnostics}')
    values=(total/count).cpu().numpy()
    if not np.isfinite(values).all(): raise ValueError('Nonfinite JSD')
    distance=np.zeros((144,144),dtype=np.float64)
    left,right=pairs.cpu().numpy()
    distance[left,right]=np.maximum(values,0)
    distance[right,left]=distance[left,right]
    np.save(root/'results/head_jsd.npy',distance)
    params=dict(n_components=2,dissimilarity='precomputed',n_init=8,init='random',max_iter=1000,eps=1e-9,n_jobs=1,random_state=SEED,normalized_stress=False)
    mds=MDS(**params)
    coordinates=mds.fit_transform(distance)
    embedding_distance=np.linalg.norm(coordinates[:,None]-coordinates[None,:],axis=-1)
    a,b=np.triu_indices(144,1)
    stress1=float(np.sqrt(np.sum((embedding_distance[a,b]-distance[a,b])**2)/np.sum(embedding_distance[a,b]**2)))
    for i,r in enumerate(rows): r.update(x=float(coordinates[i,0]), y=float(coordinates[i,1]))
    write_csv(root/'results/mds_coordinates.csv',rows)
    summary=summarize(distance,rows)
    categories=list(dict.fromkeys(r['category'] for r in rows))
    for view in ['layer','behavior']:
        fig,ax=plt.subplots(figsize=(10,8))
        if view=='layer':
            for l in range(12):
                selected=np.arange(l*12,(l+1)*12)
                ax.scatter(*coordinates[selected].T,color=plt.get_cmap('tab20')(l),label=f'Layer {l}',s=45,alpha=.8)
        else:
            for c,name in enumerate(categories):
                selected=[i for i,r in enumerate(rows) if r['category']==name]
                ax.scatter(*coordinates[selected].T,color=plt.get_cmap('tab10')(c),label=name,s=45,alpha=.8)
        for i in [3,1,11,9,23,22,19,35,30,59,61,86,81]:
            ax.annotate(f'L{i//12}H{i%12}',coordinates[i],xytext=(4,4),textcoords='offset points',fontsize=7)
        ax.set(xlabel='MDS dimension 1',ylabel='MDS dimension 2',title=f'GPT-2 heads by {view}: mean JSD (nats)\n{count:,} real queries; fixed seed {SEED}; Stress-1 = {stress1:.3f}')
        ax.set_aspect('equal',adjustable='datalim')
        ax.legend(loc='center left',bbox_to_anchor=(1, .5),fontsize=8)
        fig.tight_layout()
        for ext in ['png','svg']: fig.savefig(root/f'figures/mds_by_{view}.{ext}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    metadata=dict(corpus_sha256=manifest['corpus_sha256'],model=manifest['model'],dataset=manifest['dataset'],
        source_specification='https://app.notion.com/p/3ef4aa3c576581478d9de382c4083d16',passage_count=len(passages),n_queries=count,
        query_set='all real query tokens including i=0 and final token; identical for every pair',
        averaging='equal weight per real query across passages; compute JSD for same-query distributions before averaging',
        divergence='JS(P,Q) = H((P+Q)/2) - (H(P)+H(Q))/2; natural logarithm; divergence in nats, no square root',
        support='visible real keys j<=i; masked future/padding zeros contribute zero',
        head_order='index = layer*12 + head; zero-based layers/heads',batch_size=BATCH_SIZE,query_chunk=QUERY_CHUNK,pair_chunk=PAIR_CHUNK,
        weight_dtype=str(wte.weight.dtype),attention_dtype='torch.float32',accumulation_dtype='torch.float64',
        diagnostics=diagnostics,formula_checks='identical distributions zero; disjoint point masses ln(2); symmetry; independent direct KL agreement',
        mds=dict(parameters=params,raw_stress=float(mds.stress_),stress1=stress1,n_iter=int(mds.n_iter_),pairwise_distance_correlation=float(np.corrcoef(distance[a,b],embedding_distance[a,b])[0,1])),
        behavior_rules='Exploratory descriptive thresholds: self/previous attention>=0.5; tested offsets 2/4/8/16/32>=0.25; lexical enrichment>=2; no-op-like candidate also gradient/other<=0.5; broad normalized entropy>=0.8; focused<=0.1. Flags may overlap; plot category uses this listed priority, with candidate before lexical groups.',
        behavior_source_sha256=sources,summary=summary,job_id=os.environ.get('SLURM_JOB_ID'),hostname=socket.gethostname(),device=str(device),
        gpu=torch.cuda.get_device_name() if device.type=='cuda' else None,
        versions={p:importlib.metadata.version(p) for p in ['torch','numpy','scikit-learn','matplotlib']},
        limitations=manifest['limitations']+['2D MDS distorts divergence; groups are descriptive labels, not fitted clusters or significance tests.','Position-zero distributions are identical and dilute mean JSD uniformly.','Behavior sources use different documented query sets; Experiment 2 weights float32 versus float16 here.','Attention similarity does not establish causal importance.'])
    (root/'results/head_jsd_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(dict(n_queries=count,diagnostics=diagnostics,mds=metadata['mds'],summary=summary),indent=2),flush=True)


if __name__=='__main__': main()
