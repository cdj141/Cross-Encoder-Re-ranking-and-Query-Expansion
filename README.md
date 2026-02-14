# Neural Re-ranking and Query Expansion

Fine-tuning cross-encoder models for document re-ranking on TREC DL 2019
and evaluating ensemble fusion and query expansion strategies.

## Overview

This project investigates neural re-ranking using cross-encoder
architectures and evaluates:

1.  Fine-tuning MiniLM, TinyBERT, and DistilRoBERTa
2.  Ensemble fusion methods for ranking improvement
3.  LLM-based query expansion
4.  Pseudo-relevance feedback (PRF)

The task focuses on re-ranking candidate documents using NDCG@10,
Recall@100, and MAP@1000.

## Dataset

-   TREC DL 2019 benchmark
-   Fine-tuning data: MS MARCO passage ranking dataset

Evaluation metrics: - NDCG@10 - Recall@100 - MAP@1000

## Task 1: Cross-Encoder Fine-Tuning

Models evaluated:

MiniLM\
NDCG@10: 66.96

TinyBERT\
NDCG@10: 69.28

DistilRoBERTa\
NDCG@10: 58.71

TinyBERT achieved the best overall performance due to higher training
steps and efficient optimization.

## Task 2: Ensemble Fusion

Five ensemble methods were evaluated:

-   RRF
-   WMNZ
-   LogISR
-   BordaFuse
-   wCondorcet

Best result:

WMNZ\
NDCG@10: 69.64

Ensemble methods generally improved stability and smoothed
model-specific biases.

## Task 3: Pairwise Fusion

Best pair:

MiniLM + TinyBERT\
NDCG@10: 69.43

Adding weaker models did not always improve performance, indicating that
ensemble quality depends on component strength.

## Task 4: LLM-based Query Expansion

Model used: HuggingFaceH4/zephyr-7b-beta

Chain-of-Thought prompting was applied to generate expanded queries.
Expanded queries added semantic context but introduced verbosity.

Result: Original queries outperformed expanded queries.

## Task 5: Pseudo-Relevance Feedback (PRF)

Top-3 retrieved documents were used as context for expansion.

PRF expansion further reduced performance due to semantic drift and
verbosity effects on smaller re-rankers.

## Key Findings

-   TinyBERT provides strong efficiency-performance trade-off.
-   WMNZ fusion yields the best ensemble performance.
-   Query expansion does not guarantee ranking improvement.
-   Verbose expansions can harm lightweight cross-encoders.

## Limitations

-   Limited training time (1 hour per model)
-   No domain adaptation beyond MS MARCO
-   Query expansion sensitive to prompt design
-   Small re-rankers struggle with long expanded inputs

## Future Improvements

-   Longer fine-tuning schedules
-   Query-aware dynamic ensemble weighting
-   Adaptive expansion filtering
-   Larger cross-encoder models

## Repository Structure

Fine_tuning.ipynb\
Evaluating.ipynb\
ranking_fusion_and_expansion.ipynb

## Run

Install dependencies:

pip install transformers ranx torch

Run notebooks sequentially for: 1. Fine-tuning 2. Evaluation 3. Fusion
and query expansion

## Technologies

Python\
PyTorch\
Transformers\
ranx (fusion library)\
LLMs (Zephyr-7B)

## Author

Dongjie Chen\
MSc Computer Science (Data Science), Leiden University
