---
language:
  - en
  - ko
tags:
- sentence-transformers
- sentence-similarity
- sparse-encoder
- sparse
- splade
- retrieval
- multimodal
- multi-modal
- crossmodal
- cross-modal
- feature-extraction
- aerospace
- telepix
pipeline_tag: feature-extraction
library_name: sentence-transformers
license: apache-2.0
---
<p align="center">
    <img src="https://cdn-uploads.huggingface.co/production/uploads/61d6f4a4d49065ee28a1ee7e/V8n2En7BlMNHoi1YXVv8Q.png" width="400"/>
<p>
  
# PIXIE-Splade-v1.0
**PIXIE-Splade-v1.0** is a **bilingual (ko, en)** [SPLADE](https://arxiv.org/abs/2403.06789) retriever, developed by [TelePIX Co., Ltd](https://telepix.net/). 
**PIXIE** stands for Tele**PIX** **I**ntelligent **E**mbedding, representing TelePIX’s high-performance embedding technology.
This model is specifically optimized for retrieval tasks in Korean and English, and demonstrates strong performance in aerospace domain. Through extensive fine-tuning and domain-specific evaluation, PIXIE shows robust retrieval quality for real-world use cases such as document understanding, technical QA, and information retrieval in aerospace and related high-precision fields.
PIXIE-Splade-v1.0 outputs sparse lexical vectors that are directly 
compatible with inverted indexing (e.g., Lucene/Elasticsearch). 
Because each non-zero weight corresponds to a Ko-En subword/token, 
interpretability is built-in: you can inspect which tokens drive retrieval.

## Why SPLADE for Search?
- **Inverted Index Ready**: Directly index weighted tokens in standard IR stacks (Lucene/Elasticsearch).
- **Interpretable by Design**: Top-k contributing tokens per query/document explain *why* a hit matched.
- **Production-Friendly**: Fast candidate generation at web scale; memory/latency tunable via sparsity thresholds.
- **Hybrid-Retrieval Friendly**: Combine with dense retrievers via score fusion.

## Model Description
- **Model Type:** SPLADE Sparse Encoder
<!-- - **Base model:** [Unknown](https://huggingface.co/unknown) -->
- **Maximum Sequence Length:** 5632 tokens
- **Output Dimensionality:** 50000 dimensions
- **Similarity Function:** Dot Product
- **Language:** Bilingual — Korean and English
- **Domain Specialization:** Aerospace Information Retrieval
- **License:** apache-2.0 

### Full Model Architecture

```
SparseEncoder(
  (0): MLMTransformer({'max_seq_length': 5632, 'do_lower_case': False, 'architecture': 'ModernBertForMaskedLM'})
  (1): SpladePooling({'pooling_strategy': 'max', 'activation_function': 'relu', 'word_embedding_dimension': 50000})
)
```

## Quality Benchmarks
**PIXIE-Splade-v1.0** is a bilingual embedding model specialized for Korean and English retrieval tasks. 
It delivers consistently strong performance across a diverse set of domain-specific and open-domain benchmarks in both languages, demonstrating its effectiveness in real-world search applications.
The table below presents the retrieval performance of several sparse embedding models evaluated on a variety of Korean and English benchmarks.
We report **Normalized Discounted Cumulative Gain (nDCG@10)** scores, which measure how well a ranked list of documents aligns with ground truth relevance. Higher values indicate better retrieval quality.  
  
All evaluations were conducted using the open-source **[Korean-MTEB-Retrieval-Evaluators](https://github.com/BM-K/Korean-MTEB-Retrieval-Evaluators)** codebase to ensure consistent dataset handling, indexing, retrieval, and nDCG@10 computation across models.

### Benchmark Overview and Dataset Descriptions
| Model Name | # params | STELLA (ko-en) | STELLA (en-en) | MTEB (ko) | BEIR (en) |
|------|:---:|:---:|:---:|:---:|:---:|
| telepix/PIXIE-Rune-v1.0 (dense baseline) | 0.5B | 0.5972 | 0.7627 | 0.7603 | 0.5872 |
| **telepix/PIXIE-Splade-v1.0** | **0.1B** | **0.4148** | **0.6741** | **0.7025** | **0.3760** |
| | | | | | |
| opensearch-project/opensearch-neural-sparse-encoding-multilingual-v1 | 0.2B | 0.2618 | 0.7055 | 0.5358 | 0.3756 |
| naver/splade-v3 | 0.1B | N/A | 0.7836 | 0.0685 | 0.3680 |
| BM25 | N/A | N/A | 0.6589 | 0.5071 | 0.4074 |

To better interpret the evaluation results above, we briefly describe the characteristics and evaluation intent of each benchmark suite used in this comparison.
Each benchmark is designed to assess different aspects of retrieval capability, ranging from domain-specific technical understanding to open-domain and multilingual generalization.

#### STELLA
[STELLA](https://arxiv.org/abs/2601.03496) is an aerospace-domain Information Retrieval (IR) benchmark constructed from NASA Technical Reports Server (NTRS) documents. It is designed to evaluate both:

- **Lexical matching** ability (does the retriever benefit from exact technical terms? | TCQ)
- **Semantic matching** ability (can the retriever match concepts even when technical terms are not explicitly used? | TAQ).

STELLA provides **dual-type synthetic queries** and a **cross-lingual extension** for multilingual evaluation while keeping the corpus in English.

#### 6 Datasets of MTEB (Korean)
Descriptions of the benchmark datasets used for evaluation are as follows:
- **Ko-StrategyQA**  
  A Korean multi-hop open-domain question answering dataset designed for complex reasoning over multiple documents.
- **AutoRAGRetrieval**  
  A domain-diverse retrieval dataset covering finance, government, healthcare, legal, and e-commerce sectors.
- **MIRACLRetrieval**  
  A document retrieval benchmark built on Korean Wikipedia articles.
- **PublicHealthQA**  
  A retrieval dataset focused on medical and public health topics.
- **BelebeleRetrieval**  
  A dataset for retrieving relevant content from web and news articles in Korean.
- **MultiLongDocRetrieval**  
  A long-document retrieval benchmark based on Korean Wikipedia and mC4 corpus.

#### 7 Datasets of BEIR (English)
Descriptions of the benchmark datasets used for evaluation are as follows:
- **ArguAna**  
  A dataset for argument retrieval based on claim-counterclaim pairs from online debate forums.
- **FEVER**  
  A fact verification dataset using Wikipedia for evidence-based claim validation.
- **FiQA-2018**  
  A retrieval benchmark tailored to the finance domain with real-world questions and answers.
- **HotpotQA**  
  A multi-hop open-domain QA dataset requiring reasoning across multiple documents.
- **MSMARCO**  
  A large-scale benchmark using real Bing search queries and corresponding web documents.
- **NQ**  
  A Google QA dataset where user questions are answered using Wikipedia articles.
- **SCIDOCS**  
  A citation-based document retrieval dataset focused on scientific papers.

## Direct Use (Inverted-Index Retrieval)

```python
import torch
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple
from transformers import AutoTokenizer
from sentence_transformers import SparseEncoder

model_name= 'telepix/PIXIE-Splade-v1.0'
device = "cuda" if torch.cuda.is_available() else "cpu"

def _to_dense_numpy(x) -> np.ndarray:
    if hasattr(x, "to_dense"):
        return x.to_dense().float().cpu().numpy()
    if isinstance(x, torch.Tensor):
        return x.float().cpu().numpy()
    return np.asarray(x)

def _filter_special_ids(ids: List[int], tokenizer) -> List[int]:
    special = set(getattr(tokenizer, "all_special_ids", []) or [])
    return [i for i in ids if i not in special]

def build_inverted_index(
    model: SparseEncoder,
    tokenizer,
    documents: List[str],
    batch_size: int = 8,
    min_weight: float = 0.0,
) -> Tuple[Dict[int, List[Tuple[int, float]]], List[str]]:
    with torch.no_grad():
        doc_emb = model.encode_document(documents, batch_size=batch_size)
    doc_dense = _to_dense_numpy(doc_emb)

    index: Dict[int, List[Tuple[int, float]]] = defaultdict(list)

    for doc_idx, vec in enumerate(doc_dense):
        nz = np.flatnonzero(vec > min_weight)
        nz = _filter_special_ids(nz.tolist(), tokenizer)

        for token_id in nz:
            index[token_id].append((doc_idx, float(vec[token_id])))

    return index

def splade_token_overlap_inverted(
    model: SparseEncoder,
    tokenizer,
    inverted_index: Dict[int, List[Tuple[int, float]]],
    documents: List[str],
    queries: List[str],
    top_k_docs: int = 3,
    top_k_tokens: int = 5,
    min_weight: float = 0.0,
):
    for qi, qtext in enumerate(queries):
        with torch.no_grad():
            q_vec = model.encode_query(qtext)
        q_vec = _to_dense_numpy(q_vec).ravel()

        q_nz = np.flatnonzero(q_vec > min_weight).tolist()
        q_nz = _filter_special_ids(q_nz, tokenizer)

        scores: Dict[int, float] = defaultdict(float)
        per_doc_contrib: Dict[int, Dict[int, Tuple[float, float, float]]] = defaultdict(dict)

        for tid in q_nz:
            qw = float(q_vec[tid])
            postings = inverted_index.get(tid, [])
            for doc_idx, dw in postings:
                prod = qw * dw
                scores[doc_idx] += prod
                per_doc_contrib[doc_idx][tid] = (qw, dw, prod)

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k_docs]

        print("\n" + "="*60)
        print(f"[Query {qi + 1}] {qtext}")
        print("="*60)

        if not ranked:
            print("→ No matching documents found.")
            continue

        for rank, (doc_idx, score) in enumerate(ranked, start=1):
            doc = documents[doc_idx]
            print(f"\n→ Rank {rank} | Score: {score:.4f}")
            print(f"  Document: \"{doc}\"")

            contrib = per_doc_contrib[doc_idx]
            if not contrib:
                print("  (No overlapping tokens)")
                continue

            top = sorted(contrib.items(), key=lambda kv: kv[1][2], reverse=True)[:top_k_tokens]
            token_ids = [tid for tid, _ in top]
            tokens = tokenizer.convert_ids_to_tokens(token_ids)

            print(f"  [Top {top_k_tokens} Contributing Tokens]")
            print(f"  {'Token':<20} {'Score (qw*dw)':>15}")
            print(f"  {'-'*35}")
            for (tid, (qw, dw, prod)), tok in zip(top, tokens):
                clean_tok = tok.replace("##", "")
                print(f"  {clean_tok:<20} {prod:15.4f}")

if __name__ == "__main__":
    print(f"Loading model: {model_name}...")
    model = SparseEncoder(model_name).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    documents = [
        "텔레픽스는 위성 데이터를 분석하여 해양, 농업 등 다양한 분야에 솔루션을 제공합니다.",
        "고해상도 광학 위성 영상은 국방 및 정찰 목적으로 중요하게 활용됩니다.",
        "TelePIX provides advanced solutions by analyzing satellite data for ocean and agriculture.",
        "High-resolution optical satellite imagery is critical for defense and reconnaissance.",
        "Space economy creates new value through the utilization of space-based data."
    ]

    # Cross-lingual test queries :)
    queries = [
        "텔레픽스는 어떤 산업 분야에서 위성 데이터를 활용하나요?",
        "Utilization of satellite imagery for defense",
    ]

    print("Building inverted index...")
    inverted_index = build_inverted_index(
        model=model,
        tokenizer=tokenizer,
        documents=documents,
        batch_size=4,
        min_weight=0.01, # 노이즈 제거를 위해 약간의 threshold를 줄 수 있습니다.
    )

    splade_token_overlap_inverted(
        model=model,
        tokenizer=tokenizer,
        inverted_index=inverted_index,
        documents=documents,
        queries=queries,
        top_k_docs=2,
        top_k_tokens=5
    )
```

## License
The PIXIE-Splade-v1.0 model is licensed under Apache License 2.0.

## Citation
```
@misc{TelePIX-PIXIE-Splade-v1.0,
  title={PIXIE-Splade-v1.0},
  author={TelePIX AI Research Team and Bongmin Kim},
  year={2026},
  url={https://huggingface.co/telepix/PIXIE-Splade-v1.0}
}
```

## Contact

If you have any suggestions or questions about the PIXIE, please reach out to the authors at bmkim@telepix.net.