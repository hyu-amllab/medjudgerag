#!/usr/bin/env python3
"""
Build strict no-KG control SFT data from existing SFT targets.

Default behavior (strict):
- item["kg"] is emptied
- "KG Entities:" / "KG Relations:" lines are removed from answer_target
- KG mentions inside Evidence/Summary text are sanitized
- full_output is set to analysis-only text (answer_target only)

All non-KG fields are preserved.
"""

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict

def _sanitize_line_text(text: str) -> str:
    # Remove explicit KG artifacts (IDs, KG mentions, relation/entity meta spans)
    text = re.sub(r"\([^)]*\b(?:KG|entities?|relations?|R\d+)\b[^)]*\)", "", text, flags=re.I)
    text = re.sub(r"\bR\d+\b", "", text)
    text = re.sub(r"\bKG\s+(?:relation|relations|entity|entities)\b", "document evidence", text, flags=re.I)
    text = re.sub(r"\bKG\b", "document", text)
    text = re.sub(r"\b(?:and|or)\s+relations?\b", "", text, flags=re.I)
    text = re.sub(r"\b(?:and|or)\s+entities?\b", "", text, flags=re.I)
    text = re.sub(r"\brelations?\b", "", text, flags=re.I)
    text = re.sub(r"\bentities?\b", "", text, flags=re.I)
    text = re.sub(r"\band\s+relation\b", "", text, flags=re.I)
    text = re.sub(r"\band\s+relations\b", "", text, flags=re.I)
    text = re.sub(r"\bthe\s+relation\b", "", text, flags=re.I)
    text = re.sub(r"\bthe\s+relations\b", "", text, flags=re.I)
    text = re.sub(r"\(\s*[,;:\-]*\s*\)", "", text)
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,.;:]){2,}", r"\1", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip(" ,;\t")


def normalize_answer_target(
    answer_target: Any,
    strip_kg_from_text: bool = False,
    drop_kg_fields: bool = False,
) -> Any:
    if isinstance(answer_target, str):
        lines = answer_target.splitlines()
        out = []
        for line in lines:
            stripped = line.lstrip()
            indent = line[: len(line) - len(stripped)]
            if stripped.startswith("KG Entities:"):
                if not drop_kg_fields:
                    out.append(f"{indent}KG Entities: []")
            elif stripped.startswith("KG Relations:"):
                if not drop_kg_fields:
                    out.append(f"{indent}KG Relations: []")
            elif strip_kg_from_text and stripped.startswith("Evidence:"):
                body = _sanitize_line_text(stripped[len("Evidence:"):].strip())
                out.append(f"{indent}Evidence: {body or 'No relevant evidence found.'}")
            elif strip_kg_from_text and stripped.startswith("Summary:"):
                body = _sanitize_line_text(stripped[len("Summary:"):].strip())
                out.append(f"{indent}Summary: {body}")
            else:
                out.append(line)
        return "\n".join(out).strip()

    if isinstance(answer_target, dict):
        out = dict(answer_target)
        oa = out.get("option_analysis")
        if isinstance(oa, dict):
            oa2: Dict[str, Any] = {}
            for k, v in oa.items():
                if isinstance(v, dict):
                    vv = dict(v)
                    vv["kg_entities"] = []
                    vv["kg_relations"] = []
                    oa2[k] = vv
                else:
                    oa2[k] = v
            out["option_analysis"] = oa2
        return out

    return answer_target


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input jsonl")
    ap.add_argument("--output", required=True, help="Output jsonl")
    ap.add_argument(
        "--no_strip_kg_from_text",
        action="store_true",
        help="Disable KG mention sanitization in Evidence/Summary text.",
    )
    ap.add_argument(
        "--no_drop_kg_fields",
        action="store_true",
        help="Keep KG Entities/KG Relations lines in answer_target.",
    )
    ap.add_argument(
        "--no_analysis_only_full_output",
        action="store_true",
        help="Do not overwrite full_output with analysis-only answer_target text.",
    )
    ap.add_argument(
        "--no_empty_kg_field",
        action="store_true",
        help="Keep original item['kg'] value.",
    )
    args = ap.parse_args()

    strip_kg_from_text = not args.no_strip_kg_from_text
    drop_kg_fields = not args.no_drop_kg_fields
    analysis_only_full_output = not args.no_analysis_only_full_output
    empty_kg_field = not args.no_empty_kg_field

    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    changed_kg = 0
    changed_entities_line = 0
    changed_relations_line = 0
    residual_kg_mentions = 0

    with in_path.open("r", encoding="utf-8") as fin, out_path.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            obj = json.loads(line)
            n += 1

            old_kg = obj.get("kg", "")
            if old_kg:
                changed_kg += 1
            if empty_kg_field:
                obj["kg"] = ""

            at_old = obj.get("answer_target", "")
            at_new = normalize_answer_target(
                at_old,
                strip_kg_from_text=strip_kg_from_text,
                drop_kg_fields=drop_kg_fields,
            )

            if isinstance(at_old, str):
                changed_entities_line += len(re.findall(r"^\s*KG Entities:\s*(?!\[\])", at_old, flags=re.M))
                changed_relations_line += len(re.findall(r"^\s*KG Relations:\s*(?!\[\])", at_old, flags=re.M))

            obj["answer_target"] = at_new

            # strict no-KG control should train analysis-only completion
            if isinstance(at_new, str):
                if analysis_only_full_output:
                    obj["full_output"] = at_new
                if re.search(r"\bKG\b|KG Entities:|KG Relations:|\bR\d+\b", at_new):
                    residual_kg_mentions += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"Input : {in_path}")
    print(f"Output: {out_path}")
    print(f"Samples: {n}")
    print(f"kg changed: {changed_kg}")
    print(f"KG Entities lines normalized: {changed_entities_line}")
    print(f"KG Relations lines normalized: {changed_relations_line}")
    print(f"residual KG mentions in answer_target: {residual_kg_mentions}")
    print(f"strip_kg_from_text: {strip_kg_from_text}")
    print(f"drop_kg_fields: {drop_kg_fields}")
    print(f"analysis_only_full_output: {analysis_only_full_output}")
    print(f"empty_kg_field: {empty_kg_field}")


if __name__ == "__main__":
    main()
