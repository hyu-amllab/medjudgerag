#!/usr/bin/env python3
"""
build_test_context.py — Build retrieved-passage context for MedMCQA / MedQA test sets.

For MedMCQA: full validation split (4,183).
For MedQA: full test split (1,273).

Output format matches textbooks_kg_context.jsonl:
  {"id", "question", "label", "choices", "retrieved_ids", "retrieved_passages"}

Usage:
  python build_test_context.py --benchmark medmcqa   --gpu 0
  python build_test_context.py --benchmark medqa     --gpu 0
"""

import os
import sys
import json
import random
import argparse
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "MedRAG" / "src"))


# ── Constants ──────────────────────────────────────────────────────────────
SEED = 42
K = 5
DB_DIR = str(REPO_ROOT / "MedRAG" / "corpus")
OUTPUT_DIR = REPO_ROOT / "data"


# ── Git LFS patch (reuse from build_rag_train_data.py) ────────────────────
_HF_CORPUS_REPOS = {
    "pubmed": "MedRAG/pubmed",
    "textbooks": "MedRAG/textbooks",
    "statpearls": "MedRAG/statpearls",
    "wikipedia": "MedRAG/wikipedia",
}
_lfs_checked: set = set()


def _is_lfs_pointer(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(27) == b"version https://git-lfs.git"
    except OSError:
        return False


def _fetch_chunk_file(fpath: str, db_dir: str) -> None:
    if fpath in _lfs_checked:
        return
    _lfs_checked.add(fpath)
    if not _is_lfs_pointer(fpath):
        return
    from huggingface_hub import hf_hub_download
    rel_to_db = Path(fpath).relative_to(db_dir)
    corpus_name = rel_to_db.parts[0]
    file_in_repo = str(Path(*rel_to_db.parts[1:]))
    hf_repo = _HF_CORPUS_REPOS.get(corpus_name)
    if not hf_repo:
        raise ValueError(f"Unknown corpus '{corpus_name}'")
    print(f"  [LFS] {corpus_name}/{file_in_repo} — downloading from {hf_repo}")
    os.unlink(fpath)
    hf_hub_download(
        repo_id=hf_repo, filename=file_in_repo,
        repo_type="dataset", local_dir=str(Path(db_dir) / corpus_name),
    )


def _patch_retriever_lfs(db_dir: str):
    from utils import Retriever
    _original_idx2txt = Retriever.idx2txt

    def _lfs_aware_idx2txt(self, indices):
        for idx in indices:
            fpath = os.path.join(self.chunk_dir, idx["source"] + ".jsonl")
            _fetch_chunk_file(fpath, db_dir)
        return _original_idx2txt(self, indices)

    Retriever.idx2txt = _lfs_aware_idx2txt
    print("Retriever patched for git-lfs.")


# ── Data loaders ───────────────────────────────────────────────────────────

def load_medmcqa_test():
    """Full MedMCQA validation (= MedRAG test) split (4,183)."""
    from datasets import load_dataset

    ds = load_dataset("medmcqa", split="validation")
    answer_map = {0: "A", 1: "B", 2: "C", 3: "D"}

    samples = []
    for item in ds:
        if item.get("cop") is None:
            continue
        options = {
            "A": item["opa"], "B": item["opb"],
            "C": item["opc"], "D": item["opd"],
        }
        samples.append({
            "id": str(item.get("id", "")),
            "question": item["question"],
            "choices": options,
            "label": answer_map.get(item["cop"], ""),
            "subject": item.get("subject_name", "Unknown"),
        })
    print(f"  MedMCQA: {len(samples)} test samples")
    return samples


def load_medqa_test():
    """Full MedQA test split (1,273)."""
    from datasets import load_dataset

    ds = load_dataset("GBaker/MedQA-USMLE-4-options", split="test")
    samples = []
    for i, item in enumerate(ds):
        samples.append({
            "id": f"medqa_test_{i}",
            "question": item["question"],
            "choices": item["options"],
            "label": item["answer_idx"],
        })
    print(f"  MedQA: {len(samples)} test samples")
    return samples


# ── Checkpoint ─────────────────────────────────────────────────────────────

def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    done = set()
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return done


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Build test context (retrieved passages) for MedMCQA / MedQA"
    )
    parser.add_argument("--benchmark", required=True, choices=["medmcqa", "medqa"])
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--db_dir", default=DB_DIR)
    parser.add_argument("--gpu", type=int, default=0,
                        help="GPU for retriever (default: 0)")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    _patch_retriever_lfs(args.db_dir)

    from utils import RetrievalSystem

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load samples
    if args.benchmark == "medmcqa":
        samples = load_medmcqa_test()
    elif args.benchmark == "medqa":
        samples = load_medqa_test()
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")
    out_path = OUTPUT_DIR / f"full_context_{args.benchmark}.jsonl"

    # Checkpoint
    done = load_done_ids(out_path)
    pending = [s for s in samples if s["id"] not in done]
    print(f"\n[{args.benchmark}] total={len(samples)}  done={len(done)}  pending={len(pending)}")

    if not pending:
        print("Nothing to do — already complete.")
        return

    # Init retrieval
    print("Initialising RetrievalSystem [Contriever + PubMedTextbooks]...")
    retrieval_system = RetrievalSystem(
        retriever_name="Contriever",
        corpus_name="PubMedTextbooks",
        db_dir=args.db_dir,
    )
    print("RetrievalSystem ready.\n")

    with out_path.open("a", encoding="utf-8") as f:
        for idx, item in enumerate(pending, 1):
            try:
                docs, scores = retrieval_system.retrieve(item["question"], k=K)
                passages = [d.get("content", "") for d in docs]
                doc_ids = [d.get("id", "") for d in docs]

                record = {
                    "id": item["id"],
                    "question": item["question"],
                    "choices": item["choices"],
                    "label": item["label"],
                    "retrieved_ids": doc_ids,
                    "retrieved_passages": passages,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
                f.flush()

                if idx % 100 == 0:
                    print(f"  [{args.benchmark}] {idx}/{len(pending)}")

            except Exception as e:
                print(f"  ERROR on {item['id']}: {e}")

    print(f"[{args.benchmark}] Done — {len(done) + len(pending)} records in {out_path}")


if __name__ == "__main__":
    main()
