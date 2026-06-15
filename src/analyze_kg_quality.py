#!/usr/bin/env python3
"""
analyze_kg_quality.py — KG quality analysis across λ_g sweep.

Parses generated KGs from explicit MedJudgeRAG eval JSONL files (one per λ_g
checkpoint) and computes intrinsic structural metrics, paired with the
answer accuracy from the same files. Used to test whether KG surface-form
quality and downstream answer accuracy anti-correlate as λ_g changes.

The benchmark test set has no gold KG, so we report intrinsic metrics
rather than F1 vs. gold:
  - empty_rate     : fraction of samples whose generated KG block is empty
  - n_entities     : average #entities extracted per sample
  - n_relations    : average #relations extracted per sample
  - valid_type     : fraction of entities whose Type ∈ allowed entity types
  - valid_reltype  : fraction of relations whose RelationType ∈ allowed set
  - orphan_rel     : fraction of relations whose source/target is not in
                     the entity list of the same sample
  - answer_acc     : answer accuracy from the eval file's `correct` field

Usage:
    # Explicit file list (one per λ_g)
    python analyze_kg_quality.py \\
      --eval_files \\
        outputs/eval_llama3_kgw00_explicit_medqa.jsonl \\
        outputs/eval_llama3_kgw01_explicit_medqa.jsonl \\
        outputs/eval_llama3_kgw03_explicit_medqa.jsonl \\
        outputs/eval_llama3_kgw05_explicit_medqa.jsonl \\
      --output results/kg_quality/llama3_medqa

    # Auto-discover (one backbone + benchmark pair)
    python analyze_kg_quality.py \\
      --auto_discover --backbone llama --benchmark medqa \\
      --output results/kg_quality/llama3_medqa
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path


# ── Allowed entity / relation types (must match training prompt) ──
VALID_ENTITY_TYPES = {
    "Activities & Behaviors", "Anatomy", "Chemicals & Drugs",
    "Concepts & Ideas", "Devices", "Disorders",
    "Genes & Molecular Sequences", "Geographic Areas", "Living Beings",
    "Objects", "Occupations", "Organizations", "Phenomena",
    "Physiology", "Procedures",
}
VALID_RELATION_TYPES = {
    "part_of", "located_in", "connected_to", "adjacent_to", "performs",
    "uses", "affects", "causes", "result_of", "indicates", "measures",
    "diagnoses", "manifestation_of", "precedes", "co_occurs_with",
}

ANALYSIS_DELIMITER_PATTERNS = ("<ANALYSIS>", "<ANALYSIS")


def extract_kg_block(raw_generation):
    """Return only the KG portion of raw_generation (text before <ANALYSIS>)."""
    if not raw_generation:
        return ""
    text = str(raw_generation)
    cut = len(text)
    for pat in ANALYSIS_DELIMITER_PATTERNS:
        idx = text.find(pat)
        if idx != -1 and idx < cut:
            cut = idx
    return text[:cut].strip()


def parse_entities(kg_text):
    """Extract (name, type) pairs from KG text."""
    pattern = r'\(\s*"Entity"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"'
    return re.findall(pattern, kg_text)


def parse_relations(kg_text):
    """Extract (source, rel_type, target) triples from KG text."""
    pattern = r'\(\s*"Relation"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"\s*,\s*"([^"]*)"'
    return re.findall(pattern, kg_text)


def per_sample_metrics(raw_generation):
    """Compute per-sample structural KG metrics."""
    kg_text = extract_kg_block(raw_generation)
    ents = parse_entities(kg_text)
    rels = parse_relations(kg_text)

    n_ents = len(ents)
    n_rels = len(rels)
    is_empty = (n_ents == 0 and n_rels == 0)

    valid_type = sum(1 for _, t in ents if t in VALID_ENTITY_TYPES)
    valid_reltype = sum(1 for _, rt, _ in rels if rt in VALID_RELATION_TYPES)

    ent_names = {name for name, _ in ents}
    orphan_rel = sum(
        1 for src, _, tgt in rels
        if src not in ent_names or tgt not in ent_names
    )

    return {
        "is_empty": is_empty,
        "n_ents": n_ents,
        "n_rels": n_rels,
        "valid_type": valid_type,
        "valid_reltype": valid_reltype,
        "orphan_rel": orphan_rel,
        "kg_chars": len(kg_text),
    }


def safe_div(num, den):
    return num / den if den > 0 else 0.0


def aggregate(samples):
    """Aggregate per-sample dicts into summary metrics."""
    n = len(samples)
    if n == 0:
        return {"n": 0}

    n_empty = sum(s["is_empty"] for s in samples)
    sum_ents = sum(s["n_ents"] for s in samples)
    sum_rels = sum(s["n_rels"] for s in samples)
    sum_valid_type = sum(s["valid_type"] for s in samples)
    sum_valid_reltype = sum(s["valid_reltype"] for s in samples)
    sum_orphan = sum(s["orphan_rel"] for s in samples)
    n_correct = sum(1 for s in samples if s.get("correct"))
    sum_kg_chars = sum(s["kg_chars"] for s in samples)

    return {
        "n": n,
        "empty_rate": 100.0 * n_empty / n,
        "avg_n_ents": sum_ents / n,
        "avg_n_rels": sum_rels / n,
        "valid_type_rate": 100.0 * safe_div(sum_valid_type, sum_ents),
        "valid_reltype_rate": 100.0 * safe_div(sum_valid_reltype, sum_rels),
        "orphan_rel_rate": 100.0 * safe_div(sum_orphan, sum_rels),
        "answer_acc": 100.0 * n_correct / n,
        "avg_kg_chars": sum_kg_chars / n,
    }


def parse_filename_metadata(path):
    """Infer (backbone, benchmark, lambda_g) from eval filename.

    Examples it tries to match:
      eval_{backbone}_kgw{XX}_explicit_{bench}.jsonl
    """
    name = Path(path).stem

    # Backbone
    backbone = None
    for cand in ["llama3", "llama", "mistral"]:
        if f"_{cand}_" in name or name.startswith(f"eval_{cand}"):
            backbone = cand
            break

    # Benchmark (last token in the stem usually)
    benchmark = None
    for cand in ["medqa", "medmcqa"]:
        if name.endswith(f"_{cand}_{cand}") or name.endswith(f"_{cand}"):
            benchmark = cand
            break

    # Lambda_g — match kgwNN
    m = re.search(r"_kgw(\d{2})(?:_|$)", name)
    if m:
        digits = m.group(1)
        lambda_g = int(digits) / 10.0
    else:
        lambda_g = None  # unknown / default

    return backbone, benchmark, lambda_g


def auto_discover(eval_dir, backbone=None, benchmark=None):
    """Glob eval JSONL files matching backbone+benchmark, sorted by λ_g."""
    pattern = "eval_*.jsonl"
    candidates = sorted(glob.glob(str(Path(eval_dir) / pattern)))
    selected = []
    for p in candidates:
        bb, bm, lg = parse_filename_metadata(p)
        if backbone and bb != backbone:
            continue
        if benchmark and bm != benchmark:
            continue
        if lg is None:
            continue  # skip files where we can't tell λ_g
        selected.append((lg, p))
    selected.sort(key=lambda x: x[0])
    return selected


def load_eval_file(path):
    """Yield dicts from a JSONL eval file, attaching per-sample KG metrics."""
    samples = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            metrics = per_sample_metrics(d.get("raw_generation", ""))
            metrics["id"] = d.get("id")
            metrics["correct"] = bool(d.get("correct"))
            metrics["decision"] = d.get("decision")
            samples.append(metrics)
    return samples


def format_table(rows, columns, col_widths):
    """Render a fixed-width text table."""
    lines = []
    header = " | ".join(c.ljust(w) for c, w in zip(columns, col_widths))
    lines.append(header)
    lines.append("-" * len(header))
    for row in rows:
        cells = []
        for c, w in zip(columns, col_widths):
            v = row.get(c, "")
            if isinstance(v, float):
                cells.append(f"{v:>.2f}".ljust(w))
            else:
                cells.append(str(v).ljust(w))
        lines.append(" | ".join(cells))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="KG structural quality vs λ_g ablation analyzer."
    )
    parser.add_argument(
        "--eval_files", nargs="+", default=None,
        help="Explicit list of eval JSONL files (one per λ_g checkpoint).",
    )
    parser.add_argument(
        "--labels", nargs="+", default=None,
        help="Optional λ_g labels for each --eval_files entry "
             "(e.g., 0.0 0.1 0.3 0.5 0.7). Falls back to filename parsing.",
    )
    parser.add_argument(
        "--auto_discover", action="store_true",
        help="Auto-discover eval files from --eval_dir matching --backbone+--benchmark.",
    )
    parser.add_argument(
        "--eval_dir", default="outputs",
        help="Directory to scan for eval JSONL files (with --auto_discover).",
    )
    parser.add_argument("--backbone", choices=["llama3", "llama", "mistral"], default=None)
    parser.add_argument("--benchmark", choices=["medqa", "medmcqa"], default=None)
    parser.add_argument(
        "--output", default="results/kg_quality/summary",
        help="Output prefix (writes .txt table and .json results).",
    )
    args = parser.parse_args()

    # Resolve file list
    file_label_pairs = []
    if args.auto_discover:
        if not (args.backbone and args.benchmark):
            raise SystemExit(
                "Auto-discover requires --backbone and --benchmark."
            )
        discovered = auto_discover(
            args.eval_dir, backbone=args.backbone, benchmark=args.benchmark
        )
        if not discovered:
            raise SystemExit(
                f"No eval files found under {args.eval_dir} for "
                f"backbone={args.backbone} benchmark={args.benchmark}."
            )
        for lg, p in discovered:
            file_label_pairs.append((p, lg))
    else:
        if not args.eval_files:
            raise SystemExit("Provide --eval_files or --auto_discover.")

        if args.labels:
            if len(args.labels) != len(args.eval_files):
                raise SystemExit(
                    "--labels length must match --eval_files length."
                )
            for p, lab in zip(args.eval_files, args.labels):
                try:
                    lg = float(lab)
                except ValueError:
                    raise SystemExit(f"--labels must be numeric, got {lab!r}")
                file_label_pairs.append((p, lg))
        else:
            for p in args.eval_files:
                _, _, lg = parse_filename_metadata(p)
                if lg is None:
                    raise SystemExit(
                        f"Could not infer λ_g from {p}; pass --labels explicitly."
                    )
                file_label_pairs.append((p, lg))
        # Sort by λ_g for consistent presentation
        file_label_pairs.sort(key=lambda x: x[1])

    # Process each file, group by (backbone, benchmark)
    grouped = defaultdict(list)  # (backbone, benchmark) -> list of (lambda_g, summary, file)
    all_results = []

    for path, lambda_g in file_label_pairs:
        bb, bm, _ = parse_filename_metadata(path)
        bb = bb or args.backbone or "?"
        bm = bm or args.benchmark or "?"

        samples = load_eval_file(path)
        summary = aggregate(samples)
        summary["lambda_g"] = lambda_g
        summary["file"] = path
        summary["backbone"] = bb
        summary["benchmark"] = bm

        grouped[(bb, bm)].append(summary)
        all_results.append(summary)
        print(f"[loaded] {Path(path).name}  "
              f"backbone={bb} benchmark={bm} λ_g={lambda_g}  n={summary['n']}")

    # Build text report grouped by (backbone, benchmark)
    columns = [
        "lambda_g", "n", "answer_acc", "empty_rate",
        "avg_n_ents", "avg_n_rels",
        "valid_type_rate", "valid_reltype_rate", "orphan_rel_rate",
        "avg_kg_chars",
    ]
    col_widths = [10, 6, 10, 10, 10, 10, 14, 16, 14, 12]

    report_lines = []
    for (bb, bm), entries in sorted(grouped.items()):
        report_lines.append(f"\n{'='*100}")
        report_lines.append(f" Backbone = {bb}   Benchmark = {bm}   "
                            f"({len(entries)} checkpoints)")
        report_lines.append("=" * 100)
        rows = sorted(entries, key=lambda e: e["lambda_g"])
        report_lines.append(format_table(rows, columns, col_widths))

    full_report = "\n".join(report_lines)

    # Save outputs
    out_prefix = Path(args.output)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    txt_path = out_prefix.with_suffix(".txt")
    json_path = out_prefix.with_suffix(".json")

    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(full_report + "\n")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)

    print(full_report)
    print(f"\n[saved] {txt_path}")
    print(f"[saved] {json_path}")


if __name__ == "__main__":
    main()
