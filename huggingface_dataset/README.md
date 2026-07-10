---
license: cc-by-nc-sa-4.0
language:
- en
task_categories:
- question-answering
- text-generation
tags:
- medical
- mcqa
- retrieval-augmented-generation
- knowledge-graph
- chain-of-thought
- distillation
- gpt-5.1
pretty_name: MedJudgeRAG SFT Traces and Evaluation Contexts
size_categories:
- 10K<n<100K
configs:
- config_name: teacher_traces_postprocessed
  data_files:
  - split: train
    path: teacher_traces_postprocessed.jsonl
- config_name: teacher_traces_mistral
  data_files:
  - split: train
    path: teacher_traces_mistral_8192.jsonl
- config_name: teacher_traces_llama3
  data_files:
  - split: train
    path: teacher_traces_llama3_8192.jsonl
- config_name: medqa_eval_retrieval_ids
  data_files:
  - split: test
    path: medqa_eval_retrieval_ids.jsonl
- config_name: medmcqa_eval_retrieval_ids
  data_files:
  - split: validation
    path: medmcqa_eval_retrieval_ids.jsonl
---

# MedJudgeRAG: SFT Traces and Evaluation Contexts

This dataset accompanies the paper **[MedJudgeRAG: Option-Wise Evidence Judgment with Dynamic Knowledge Graphs for Medical MCQA](https://openreview.net/forum?id=lrZIKfoRfz)**.
The paper is presented at [The Workshop on Graph Foundation Models: A New Era for Graph Machine Learning](https://sites.google.com/view/gfmicml-2026), held at the 43rd International Conference on Machine Learning (ICML 2026), Seoul, South Korea.
It contains the GPT-5.1 teacher reasoning traces used to train the student models and the held-out evaluation contexts used to report results in the paper.

To preserve the licensing terms of the underlying retrieval corpus, **raw retrieved passage text is not redistributed**.
The released files keep only the retrieval document identifiers and their ranking order.
A helper script at the GitHub repository reconstructs the full passages from a local MedRAG-style corpus.

- Paper: see the GitHub repository
- Code: https://github.com/hyu-amllab/medjudgerag

## Dataset Configurations

There are five configurations. Each configuration is a single JSONL file with one record per line.
When creating the Hugging Face dataset repository, place these five JSONL files at the dataset repository root next to this `README.md`.

| Config | Split | Rows | Purpose |
| --- | --- | --- | --- |
| `teacher_traces_postprocessed` | `train` | 3,562 | Tokenizer-agnostic, schema-validated teacher traces. Source for backbone-specific length filtering. |
| `teacher_traces_mistral` | `train` | 3,479 | Teacher traces filtered to <= 8,192 tokens under the Mistral-7B-Instruct-v0.3 tokenizer and chat template. |
| `teacher_traces_llama3` | `train` | 3,559 | Teacher traces filtered to <= 8,192 tokens under the Meta-Llama-3-8B-Instruct tokenizer and chat template. |
| `medqa_eval_retrieval_ids` | `test` | 1,273 | MedQA test split with top-5 retrieval document IDs (no passage text). |
| `medmcqa_eval_retrieval_ids` | `validation` | 4,183 | MedMCQA validation split with top-5 retrieval document IDs (no passage text). |

Token counts in `teacher_traces_mistral` and `teacher_traces_llama3` were computed by rendering the chat template the corresponding backbone uses at training and measuring the total assistant target length.

## Loading

```python
from datasets import load_dataset

# Mistral teacher traces (3,479 rows)
mistral_train = load_dataset(
    "youarethewon/medjudgerag",
    "teacher_traces_mistral",
    split="train",
)

# Llama 3 teacher traces (3,559 rows)
llama3_train = load_dataset(
    "youarethewon/medjudgerag",
    "teacher_traces_llama3",
    split="train",
)

# MedQA evaluation context (1,273 rows, test split)
medqa_test = load_dataset(
    "youarethewon/medjudgerag",
    "medqa_eval_retrieval_ids",
    split="test",
)

# MedMCQA evaluation context (4,183 rows, validation split)
medmcqa_val = load_dataset(
    "youarethewon/medjudgerag",
    "medmcqa_eval_retrieval_ids",
    split="validation",
)
```

## Schema

### Teacher trace configurations

Each row contains the inputs to the teacher model, the validated teacher output that becomes the SFT target, and the list of retrieval IDs that were prepended as documents.

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable instance ID (e.g. `medqa_7526`). |
| `benchmark` | string | One of `medqa`, `medmcqa`. |
| `question` | string | Clinical question text. |
| `options` | object | `{"A": str, "B": str, "C": str, "D": str}` |
| `answer` | string | Gold answer letter, used only for post-hoc validation of the teacher output. |
| `split` | string | Internal SFT split assignment (`train` or `val`). |
| `kg` | string | Structured knowledge-graph block produced by the teacher. |
| `answer_target` | string | Per-option judgment block, decision mode, summary, and final answer choice. |
| `full_output` | string | Concatenation `<kg>\n<ANALYSIS>\n<answer_target>` used as the assistant target. |
| `retrieved_doc_ids` | list of string | Top-5 MedRAG document IDs in retrieval order. Replaces the original retrieved passage text. |

In the teacher output, evidence markers such as `[1]`, `[2]`, ... refer to 1-based positions inside `retrieved_doc_ids`:

```text
[1] -> retrieved_doc_ids[0]
[2] -> retrieved_doc_ids[1]
...
```

### Evaluation context configurations

| Field | Type | Description |
| --- | --- | --- |
| `id` | string | Stable instance ID (e.g. `medqa_test_42`). |
| `question` | string | Clinical question text. |
| `choices` | object | `{"A": str, "B": str, "C": str, "D": str}` |
| `label` | string | Gold answer letter. |
| `retrieved_ids` | list of string | Top-5 MedRAG document IDs in retrieval order. |

The evaluation files use `choices` and `retrieved_ids` rather than `options` and `retrieved_doc_ids`, matching the MedRAG-style benchmark convention.

## Reconstructing Full Retrieval Contexts

Training and document-augmented evaluation require the actual passage text, not just the IDs.
We provide a helper script at the GitHub repository that takes a MedRAG-style corpus directory and rewrites each row with full `retrieved_docs` records (for SFT) or `retrieved_passages` strings (for evaluation).

```bash
# SFT teacher traces -> dict-list under `retrieved_docs`
python src/rehydrate_retrieved_docs.py \
  --input  teacher_traces_mistral_8192.jsonl \
  --output full_teacher_traces_mistral_8192.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus

# Evaluation contexts -> string-list under `retrieved_passages`
python src/rehydrate_retrieved_docs.py \
  --input  medqa_eval_retrieval_ids.jsonl \
  --output full_context_medqa.jsonl \
  --medrag_corpus_dir /path/to/medrag/corpus \
  --output_key retrieved_passages
```

The script fails by default if any retrieval ID cannot be resolved; pass `--allow_missing` only if empty document text is acceptable.

## Source Datasets and Retrieval

- Teacher-trace rows are derived from the public training splits of MedQA (`GBaker/MedQA-USMLE-4-options`) and MedMCQA (`medmcqa`).
- Evaluation rows use the public MedQA `test` split and the MedMCQA `validation` split (used as the test set following the MedRAG convention).
- Retrieval was performed with Contriever over the MedRAG corpus (PubMed abstracts plus medical textbooks), keeping the top 5 documents per question.
- Teacher traces were generated with GPT-5.1 via the OpenAI Batch API using greedy decoding (`temperature=0`).
The gold answer is **not** included in the teacher prompt and is used only as a post-hoc validation filter.

## Data Splits Used in the Paper

The training data was drawn exclusively from the **training** splits of the source datasets:

- 2,750 random samples from the MedQA `train` split (seed 42).
- 2,750 random samples from the MedMCQA `train` split (seed 42).

Out of these 5,500 generated traces, **3,562 (64.8%)** passed both structural and semantic validation.
After tokenizer-specific length filtering at 8,192 tokens:

- **3,479** samples were retained for Mistral-7B-Instruct-v0.3 (train: 3,148 / val: 331).
- **3,559** samples were retained for Meta-Llama-3-8B-Instruct (train: 3,222 / val: 337).

Evaluation uses the held-out MedQA `test` split (1,273 questions) and the MedMCQA `validation` split (4,183 questions).
These are predefined, mutually exclusive partitions of the source corpora, so no evaluation instance can appear in the training pipeline.
The SFT validation rows used for checkpoint selection are also carved from the training split via a disjoint offset and never overlap with any evaluation set.

## License and Redistribution Notice

This dataset is released under **CC BY-NC-SA 4.0**.

The included `question`, `options`, `answer`/`label`, `choices`, and `id` fields are derived from MedQA (Jin et al., 2021) and MedMCQA (Pal et al., 2022) and inherit their respective licenses.
Users are responsible for complying with the terms of the original datasets when using these fields.

The `retrieved_doc_ids` / `retrieved_ids` fields are stable identifiers into the MedRAG corpus.
We do not redistribute the raw retrieved passage text because portions of the MedRAG corpus, particularly the medical textbook chunks, may be subject to third-party copyright.
Reconstruction of the actual passages requires the user to set up the MedRAG corpus locally, following the original MedRAG release.

GPT-5.1 teacher outputs (`kg`, `answer_target`, `full_output`) are released under CC BY-NC-SA 4.0 as a contribution of this work.
The teacher prompt and decoding configuration that produced these traces are documented in the GitHub repository.

## Citation

If you use this dataset, please cite the MedJudgeRAG paper, MedQA, MedMCQA, and the MedRAG corpus.
The MedJudgeRAG entry below is a temporary pre-camera-ready citation and will be updated once the final camera-ready metadata is available.

```bibtex
@inproceedings{
anonymous2026medjudgerag,
title={MedJudge{RAG}: Option-Wise Evidence Judgment with Dynamic Knowledge Graphs for Medical {MCQA}},
author={Anonymous},
booktitle={Workshop on Graph Foundation Models: A New Era for Graph Machine Learning},
year={2026},
url={https://openreview.net/forum?id=lrZIKfoRfz}
}
```
