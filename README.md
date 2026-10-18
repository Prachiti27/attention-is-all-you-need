# Scaled Dot-Product Attention

A implementation of the **Scaled Dot-Product Attention** mechanism from the research paper [Attention Is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf).

## Overview

Attention allows each token to assign weights to other tokens and combine their information to generate contextual representations.

## Implementation

The implementation covers:

- Query (Q), Key (K), and Value (V) projections
- Attention score calculation
- Scaling and softmax normalization
- Weighted sum of value vectors

## Formula

```text
Attention(Q, K, V) = softmax(QKᵀ / √dₖ) V
```

## Tech Stack

- Python
- PyTorch

## Run Locally

Install the dependency:

```bash
pip install torch
```

Run the implementation:

```bash
python main.py
```
