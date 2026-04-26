# MASC 515 Assignment 3: AI and microgpt

This repository contains an enhanced version of the original `microgpt` by Andrej Karpathy. Using AI collaboration, I have implemented four key modern transformer architectures.

---

## 1. GELUs (Gaussian Error Linear Units)
**Core Idea:** Unlike ReLU which simply clips negative values to zero, GELU weights inputs by their magnitude via a Gaussian cumulative distribution. 
- **Why it matters:** It provides a smoother gradient and allows a small amount of negative information to pass through, which helps the model learn complex patterns more effectively. In the code, I replaced the standard `relu()` method with the GELU approximation formula.

## 2. LoRA (Low-Rank Adaptation)
**Core Idea:** Instead of retraining the entire weight matrix $W$ (which is computationally expensive), LoRA adds two small, low-rank matrices $A$ and $B$ alongside it. 
- **Why it matters:** During training, only $A$ and $B$ are updated ($W$ remains frozen). This drastically reduces the number of parameters to train while maintaining performance. I modified the linear layer logic to compute both the standard path and the LoRA side-path.

## 3. RoPE (Rotary Positional Embedding)
**Core Idea:** RoPE encodes position by rotating the Query and Key vectors in a complex plane at specific angles based on their position index.
- **Why it matters:** It captures the *relative* distance between tokens naturally through the rotation difference, rather than using absolute position labels. In this implementation, I removed the absolute positional embedding matrix (`wpe`) and injected RoPE during the attention calculation.

## 4. Mixture of Experts (MoE)
**Core Idea:** Instead of one large Feed-Forward Network (FFN) for every token, MoE uses a "Gate" to route each token to a subset of specialized "Expert" layers.
- **Why it matters:** This allows the model to have a massive total parameter count (high capacity) while only using a fraction of those parameters for any single calculation (high efficiency). I replaced the standard MLP block with a Gating mechanism and multiple expert networks.

---

## How to Run
```bash
python microgpt.py
