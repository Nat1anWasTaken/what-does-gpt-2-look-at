"""Freeze 1000 article-opening passages from pinned wikimedia/wikipedia."""


def main():
    import os
    import socket
    execution = {"job_id": os.environ.get("SLURM_JOB_ID"), "hostname": socket.gethostname()}
    import hashlib
    import importlib.metadata
    import json
    from pathlib import Path
    import random
    import shutil
    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq
    from tokenizers import Tokenizer
    import lab1

    dataset_id = "wikimedia/wikipedia"
    revision = "ad5752b5e625abfcdeefe5ae0ad2c3721c4b2619"
    config = "20231101.en"
    shard_index = random.Random(42).randrange(41)
    source_name = f"{config}/train-{shard_index:05d}-of-00041.parquet"
    out = Path(__file__).resolve().parent / "data"
    out.mkdir(exist_ok=True)
    if (out / "manifest.json").exists() or (out / "passages.jsonl").exists():
        raise FileExistsError("Frozen corpus already exists; use inspect_attention.py")
    print(f"Downloading pinned Wikipedia shard: {source_name}", flush=True)
    source = hf_hub_download(dataset_id, source_name, repo_type="dataset",
                             revision=revision, cache_dir=out / "download_cache")
    tokenizer_source = hf_hub_download(lab1.MODEL_ID, "tokenizer.json",
                                       revision=lab1.MODEL_REVISION)
    shutil.copyfile(tokenizer_source, out / "tokenizer.json")
    tokenizer = Tokenizer.from_file(str(out / "tokenizer.json"))
    eos = tokenizer.token_to_id("<|endoftext|>")
    rejected = {"empty": 0, "too_short": 0, "eos": 0}
    chosen = []
    eligible_count = 0
    source_count = 0
    rng = random.Random(42)
    parquet = pq.ParquetFile(source)
    for batch in parquet.iter_batches(batch_size=1024, columns=["id", "url", "title", "text"]):
        for article in batch.to_pylist():
            index = source_count
            source_count += 1
            paragraph_index = None
            text = ""
            for pindex, paragraph in enumerate(article["text"].split("\n\n")):
                if paragraph.strip():
                    paragraph_index = pindex
                    text = paragraph.strip()
                    break
            if not text:
                rejected["empty"] += 1
                continue
            ids = tokenizer.encode(text, add_special_tokens=False).ids
            if eos in ids:
                rejected["eos"] += 1
                continue
            if len(ids) < 33:
                rejected["too_short"] += 1
                continue
            eligible_count += 1
            row = {"source_row": index, "article_id": article["id"],
                   "article_title": article["title"], "article_url": article["url"],
                   "paragraph_index": paragraph_index, "text": text,
                   "original_token_count": len(ids), "token_ids": ids[:128]}
            if len(chosen) < 1000:
                chosen.append(row)
            else:
                slot = rng.randrange(eligible_count)
                if slot < 1000:
                    chosen[slot] = row
        if source_count % 10240 == 0:
            print(f"Processed {source_count} articles; {eligible_count} eligible", flush=True)
    if eligible_count < 1000:
        raise ValueError(f"Only {eligible_count} eligible passages")
    chosen.sort(key=lambda row: row["source_row"])
    for index, row in enumerate(chosen):
        row["passage_id"] = index
    corpus = out / "passages.jsonl"
    corpus.write_text("".join(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n"
                               for r in chosen), encoding="utf-8")

    def sha(path):
        digest = hashlib.sha256()
        with Path(path).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    manifest = {
        "schema_version": 2,
        "dataset": {"id": dataset_id, "revision": revision,
                    "config": config, "split": "train", "language": "en",
                    "snapshot_date": "2023-11-01", "file": source_name,
                    "source_sha256": sha(source), "shard_count": 41,
                    "shard_selection": "random.Random(42).randrange(41)",
                    "selected_shard": shard_index,
                    "url": f"https://huggingface.co/datasets/{dataset_id}/tree/{revision}/{config}",
                    "licenses": ["CC-BY-SA-3.0", "GFDL"],
                    "attribution": "Wikipedia contributors; per-article URLs in passages.jsonl"},
        "model": {"id": lab1.MODEL_ID, "revision": lab1.MODEL_REVISION},
        "tokenizer": {"file": "tokenizer.json", "sha256": sha(out / "tokenizer.json"),
                      "add_special_tokens": False},
        "preprocessing": {
            "unit": "first nonempty paragraph per article, splitting text on double newline",
            "text": "strip leading/trailing whitespace only; preserve internal whitespace and case",
            "exclude": "empty article openings, openings containing EOS, openings under 33 tokens",
            "tokenization": "encode complete stripped opening, then retain first 128 IDs",
            "insert_bos_or_eos": False, "minimum_tokens": 33, "maximum_tokens": 128,
            "sampling": "1000-item reservoir over eligible openings in source-row order; independent random.Random(42); sort by source_row",
            "seed": 42},
        "limitations": ["One of 41 English shards; not a sample across the entire snapshot",
                        "Article openings only; exclude openings shorter than 33 GPT-2 tokens"],
        "source_rows": source_count, "eligible_rows": eligible_count, "rejected": rejected,
        "passage_count": len(chosen),
        "selected_source_rows": [r["source_row"] for r in chosen],
        "selected_article_ids": [r["article_id"] for r in chosen],
        "total_tokens": sum(len(r["token_ids"]) for r in chosen),
        "offset32_query_count": sum(len(r["token_ids"]) - 32 for r in chosen),
        "corpus_file": "passages.jsonl", "corpus_sha256": sha(corpus),
        "versions": {p: importlib.metadata.version(p) for p in
                     ("torch", "huggingface-hub", "tokenizers", "safetensors", "pyarrow")},
        "execution": execution,
    }
    temporary_manifest = out / "manifest.json.tmp"
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(out / "manifest.json")
    print(json.dumps({k: manifest[k] for k in
                     ("passage_count", "eligible_rows", "total_tokens", "offset32_query_count", "corpus_sha256")}), flush=True)


if __name__ == "__main__":
    main()
