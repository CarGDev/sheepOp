# SheepOp LLM - Complete Mathematical Formulation

Complete mathematical derivation and step-by-step solutions for every component of the SheepOp Language Model.

## Table of Contents

1. [Data Processing and Tokenization](#1-data-processing-and-tokenization)
2. [Token Embedding](#2-token-embedding)
3. [Positional Encoding](#3-positional-encoding)
4. [Multi-Head Self-Attention](#4-multi-head-self-attention)
5. [Feed-Forward Network](#5-feed-forward-network)
6. [Layer Normalization](#6-layer-normalization)
7. [Transformer Block](#7-transformer-block)
8. [Complete Forward Pass](#8-complete-forward-pass)
9. [Loss Computation](#9-loss-computation)
10. [Backpropagation](#10-backpropagation)
11. [AdamW Optimizer Update](#11-adamw-optimizer-update)
12. [Learning Rate Scheduling](#12-learning-rate-scheduling)
13. [Text Generation](#13-text-generation)

---

## 1. Data Processing and Tokenization

### 1.1 Text Extraction

Given a text file with lines, we extract text samples:

**Input:** Raw text files, PDFs, images, code files  
**Output:** List of text strings $S = \{s_1, s_2, \ldots, s_N\}$ where each $s_i$ is a text line

**Example:**
```
Input: "Hello world\nMachine learning is cool."
Output: S = ["Hello world", "Machine learning is cool."]
```

### 1.2 Character-Level Tokenization

**Vocabulary Construction:**

For character-level tokenization, we create a vocabulary $V$ mapping characters to token IDs:

```math
V = \{(\text{<pad>}, 0), (\text{<unk>}, 1), (\text{<bos>}, 2), (\text{<eos>}, 3), (\text{space}, 4), (\text{!}, 5), \ldots, (\text{z}, 129)\}
```

Or more formally:

```math
V: \mathcal{C} \rightarrow \mathbb{N}, \quad V(c) = \begin{cases}
0 & \text{if } c = \text{<pad>} \\
1 & \text{if } c = \text{<unk>} \\
2 & \text{if } c = \text{<bos>} \\
3 & \text{if } c = \text{<eos>} \\
4 & \text{if } c = \text{space} \\
\vdots & \\
129 & \text{if } c = \text{z}
\end{cases}
```

where $\mathcal{C}$ is the set of all characters in the vocabulary.

**Encoding Function:**

For a text string $s = c_1 c_2 \ldots c_n$ where $c_i$ are characters:

```math
\text{encode}(s) = [V[c_1], V[c_2], \ldots, V[c_n]]
```

**Example:**
```
Input: "Hi"
s = ['H', 'i']
V = {'H': 72, 'i': 105}  # ASCII values
encode("Hi") = [72, 105]
```

**Decoding Function:**

```math
\text{decode}([t_1, t_2, ..., t_n]) = V^{-1}[t_1] \cdot V^{-1}[t_2] \cdot \ldots \cdot V^{-1}[t_n]
```

where $V^{-1}$ is the inverse mapping from token IDs to characters.

### 1.3 Sequence Chunking

For a token sequence $T = [t_1, t_2, ..., t_L]$ and maximum length $M$:

**Chunking:**

```math
\text{chunks} = \{[t_{i\cdot S}, t_{i\cdot S+1}, ..., t_{\min(i\cdot S+M, L)}] : i \in \{0, 1, ..., \lfloor\frac{L-M}{S}\rfloor\}\}
```

where $S$ is the stride (default $S = M$).

**Padding:**

For a chunk $C$ with length $|C| < M$:

```math
\text{padded}(C) = C \oplus [\text{pad\_token}]^{(M - |C|)}
```

**Example:**
```
M = 5, S = 5
T = [72, 105, 44, 32, 119, 111, 114, 108, 100]
Chunk 1: [72, 105, 44, 32, 119]
Chunk 2: [111, 114, 108, 100, <pad>]
```

---

## 2. Token Embedding

### 2.1 Embedding Matrix

We have an embedding matrix $E \in \mathbb{R}^{V \times d}$ where:
- $V$ = vocabulary size
- $d$ = embedding dimension (d_model)

### 2.2 Embedding Lookup

For input token IDs $\mathbf{t} = [t_1, t_2, ..., t_n]$:

```math
\mathbf{X} = E[\mathbf{t}] = \begin{bmatrix} E[t_1] \\ E[t_2] \\ \vdots \\ E[t_n] \end{bmatrix} \in \mathbb{R}^{n \times d}
```

**Example:**
```
V = 128, d = 512
t = [72, 105]
E[72] = [0.1, -0.2, ..., 0.05]  (512-dim vector)
E[105] = [-0.1, 0.3, ..., 0.02]  (512-dim vector)

X = [[0.1, -0.2, ..., 0.05],
     [-0.1, 0.3, ..., 0.02]]
```

**Batch Processing:**

For batch size $B$:

```math
\mathbf{X} = E[\mathbf{T}] \in \mathbb{R}^{B \times n \times d}
```

where $\mathbf{T} \in \mathbb{N}^{B \times n}$ is the batch of token IDs.

---

## 3. Positional Encoding

### 3.1 Sinusoidal Positional Encoding

For position $pos$ and dimension $i$:

```math
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right)
```

**Origin of the 10000 Constant:**

The constant $10000$ is a **hyperparameter** introduced in the original "Attention Is All You Need" paper (Vaswani et al., 2017). This value controls the **frequency** (or wavelength) of the sinusoidal functions used for positional encoding.

**What 10000 Controls:**

The term $10000^{2i/d}$ creates a **geometric progression** of frequencies across different dimensions:

- **Lower dimensions** (small $i$): Higher frequencies (faster oscillation)
- **Higher dimensions** (large $i$): Lower frequencies (slower oscillation)

**Mathematical Interpretation:**

The wavelength $\lambda_i$ for dimension pair $(2i, 2i+1)$ is:

```math
\lambda_i = 2\pi \cdot 10000^{2i/d}
```

This means:
- When $i = 0$: $\lambda_0 = 2\pi \cdot 10000^{0} = 2\pi \approx 6.28$ (short wavelength)
- When $i = d/2 - 1$: $\lambda_{d/2-1} = 2\pi \cdot 10000^{(d-2)/d} \approx 2\pi \cdot 10000$ (long wavelength)

**Why 10000?**

1. **Scale Balance**: It provides a good balance between:
   - Being large enough to create distinguishable patterns across positions
   - Being small enough to prevent numerical issues

2. **Empirical Choice**: The authors found this value works well for typical sequence lengths (up to ~5000 tokens)

3. **Frequency Range**: For $d = 512$:
   - Lowest frequency: $\frac{1}{10000^{512/512}} = \frac{1}{10000} = 0.0001$ cycles per position
   - Highest frequency: $\frac{1}{10000^{0/512}} = 1$ cycle per position
   - This covers a wide range allowing the model to capture both local and long-range positional patterns

**What Happens if We Change It?**

- **Smaller values** (e.g., 100): Higher frequencies overall → better for short sequences, but may cause aliasing for long sequences
- **Larger values** (e.g., 100000): Lower frequencies overall → better for very long sequences, but may lose fine-grained positional information
- **Different values** are sometimes used: Some models use 10000, others use 5000 or 20000 depending on their typical sequence lengths

**Example Frequency Analysis:**

For $d = 512$:

```
i = 0:   10000^(0/512) = 1.0      → wavelength ≈ 6.28 positions
i = 64:  10000^(128/512) = 10     → wavelength ≈ 62.8 positions  
i = 128: 10000^(256/512) = 100    → wavelength ≈ 628 positions
i = 256: 10000^(512/512) = 10000  → wavelength ≈ 62,832 positions
```

This creates a **multi-scale representation** where different dimensions encode positional information at different resolutions.

**Simplified Form:**

```math
PE_{(pos, 2i)} = \sin\left(pos \cdot \exp\left(-\frac{2i \log(10000)}{d}\right)\right)
```

```math
PE_{(pos, 2i+1)} = \cos\left(pos \cdot \exp\left(-\frac{2i \log(10000)}{d}\right)\right)
```

### 3.2 Positional Encoding Matrix

For sequence length $n$ and model dimension $d$:

```math
PE = \begin{bmatrix}
PE_{(0,0)} & PE_{(0,1)} & \cdots & PE_{(0,d-1)} \\
PE_{(1,0)} & PE_{(1,1)} & \cdots & PE_{(1,d-1)} \\
\vdots & \vdots & \ddots & \vdots \\
PE_{(n-1,0)} & PE_{(n-1,1)} & \cdots & PE_{(n-1,d-1)}
\end{bmatrix} \in \mathbb{R}^{n \times d}
```

### 3.3 Adding Positional Encoding

```math
\mathbf{X}' = \mathbf{X} + PE
```

**Example Calculation:**

```
d = 512, pos = 0, i = 0:
PE(0,0) = sin(0 / 10000^(0/512)) = sin(0) = 0
PE(0,1) = cos(0 / 10000^(0/512)) = cos(0) = 1

pos = 0, i = 1:
PE(0,2) = sin(0 / 10000^(2/512)) = sin(0) = 0
PE(0,3) = cos(0 / 10000^(2/512)) = cos(0) = 1

pos = 1, i = 0:
PE(1,0) = sin(1 / 10000^(0/512)) = sin(1) ≈ 0.8415
PE(1,1) = cos(1 / 10000^(0/512)) = cos(1) ≈ 0.5403
```

### 3.4 Dropout Application

```math
\mathbf{X}'' = \text{Dropout}(\mathbf{X}', p)
```

where $p$ is the dropout probability (typically 0.1).

---

## 4. Multi-Head Self-Attention

### 4.1 Query, Key, Value Projections

For input $\mathbf{X} \in \mathbb{R}^{B \times n \times d}$:

```math
\mathbf{Q} = \mathbf{X} W_Q, \quad \mathbf{K} = \mathbf{X} W_K, \quad \mathbf{V} = \mathbf{X} W_V
```

where:
- $W_Q, W_K, W_V \in \mathbb{R}^{d \times d}$ are learnable weight matrices
- $\mathbf{Q}, \mathbf{K}, \mathbf{V} \in \mathbb{R}^{B \times n \times d}$

**Example:**
```
B = 2, n = 5, d = 512
X shape: [2, 5, 512]
W_Q shape: [512, 512]
Q = X @ W_Q  →  [2, 5, 512]
```

### 4.2 Multi-Head Splitting

For $h$ heads:

```math
d_k = \frac{d}{h}
```

```math
\mathbf{Q}_i = \mathbf{Q}[:, :, i \cdot d_k : (i+1) \cdot d_k] \in \mathbb{R}^{B \times n \times d_k}
```

```math
\mathbf{K}_i = \mathbf{K}[:, :, i \cdot d_k : (i+1) \cdot d_k] \in \mathbb{R}^{B \times n \times d_k}
```

```math
\mathbf{V}_i = \mathbf{V}[:, :, i \cdot d_k : (i+1) \cdot d_k] \in \mathbb{R}^{B \times n \times d_k}
```

**Reshaping:**

```math
\mathbf{Q}_i \in \mathbb{R}^{B \times h \times n \times d_k}
```

**Example:**
```
d = 512, h = 8, d_k = 64
Q shape: [2, 5, 512]
After reshape: [2, 8, 5, 64]
```

### 4.3 Scaled Dot-Product Attention

**Attention Scores:**

```math
\mathbf{S} = \frac{\mathbf{Q}_i \mathbf{K}_i^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times h \times n \times n}
```

**Example Calculation:**

```
For head i, one example:
Q_i[0,0] = [0.1, -0.2, 0.3, ..., 0.05]  (64-dim)
K_i[0,0] = [0.2, 0.1, -0.1, ..., 0.1]  (64-dim)

Dot product: Q_i[0,0] · K_i[0,0] = 0.1×0.2 + (-0.2)×0.1 + ... = 0.15
Scale: 0.15 / √64 = 0.15 / 8 = 0.01875

Score matrix S[i,j] = Q_i[i] · K_i[j] / √d_k
```

### 4.4 Causal Masking

For causal (autoregressive) attention:

```math
M_{causal} = \begin{bmatrix}
1 & -\infty & -\infty & \cdots \\
1 & 1 & -\infty & \cdots \\
1 & 1 & 1 & \cdots \\
\vdots & \vdots & \vdots & \ddots
\end{bmatrix}
```

```math
\mathbf{S}_{masked} = \mathbf{S} + M_{causal}
```

**Example:**
```
n = 3
M_causal = [[0, -inf, -inf],
           [0, 0, -inf],
           [0, 0, 0]]

S = [[0.2, 0.1, 0.3],
     [0.1, 0.4, 0.2],
     [0.3, 0.2, 0.5]]

S_masked = [[0.2, -inf, -inf],
            [0.1, 0.4, -inf],
            [0.3, 0.2, 0.5]]
```

### 4.5 Softmax Normalization

```math
\mathbf{A} = \text{softmax}(\mathbf{S}_{masked}) = \frac{\exp(\mathbf{S}_{masked})}{\sum_{j=1}^{n} \exp(\mathbf{S}_{masked}[i,j])}
```

**Element-wise:**

```math
A_{ij} = \frac{\exp(S_{masked,ij})}{\sum_{k=1}^{n} \exp(S_{masked,ik})}
```

**Example:**
```
S_masked = [[0.2, -inf, -inf],
            [0.1, 0.4, -inf],
            [0.3, 0.2, 0.5]]

For row 0:
exp(0.2) = 1.221, exp(-inf) = 0, exp(-inf) = 0
sum = 1.221
A[0,0] = 1.221/1.221 = 1.0
A[0,1] = 0/1.221 = 0
A[0,2] = 0/1.221 = 0

For row 1:
exp(0.1) = 1.105, exp(0.4) = 1.492, exp(-inf) = 0
sum = 2.597
A[1,0] = 1.105/2.597 ≈ 0.426
A[1,1] = 1.492/2.597 ≈ 0.574
A[1,2] = 0/2.597 = 0

A = [[1.0, 0.0, 0.0],
     [0.426, 0.574, 0.0],
     [0.268, 0.263, 0.469]]
```

### 4.6 Attention Application

```math
\mathbf{O}_i = \mathbf{A}_i \mathbf{V}_i \in \mathbb{R}^{B \times h \times n \times d_k}
```

**Example:**
```
A[0] = [1.0, 0.0, 0.0]
V[0] = [[0.1, 0.2, ...],
        [0.3, 0.4, ...],
        [0.5, 0.6, ...]]

O[0] = 1.0×[0.1,0.2,...] + 0.0×[0.3,0.4,...] + 0.0×[0.5,0.6,...]
     = [0.1, 0.2, ...]
```

### 4.7 Concatenation and Output Projection

**Concatenate heads:**

```math
\mathbf{O} = \text{Concat}(\mathbf{O}_1, \mathbf{O}_2, ..., \mathbf{O}_h) \in \mathbb{R}^{B \times n \times d}
```

**Output projection:**

```math
\text{Attention}(\mathbf{X}) = \mathbf{O} W_O \in \mathbb{R}^{B \times n \times d}
```

where $W_O \in \mathbb{R}^{d \times d}$ is the output projection weight matrix.

---

## 5. Feed-Forward Network

### 5.1 Feed-Forward Computation

```math
\text{FFN}(\mathbf{X}) = \text{ReLU}(\mathbf{X} W_1 + \mathbf{b}_1) W_2 + \mathbf{b}_2
```

Using GELU activation (default):

```math
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)
```

where $\Phi(x)$ is the standard normal CDF.

**Approximation:**

```math
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}\left(x + 0.044715x^3\right)\right)\right)
```

**Complete FFN:**

```math
\mathbf{H} = \mathbf{X} W_1 \in \mathbb{R}^{B \times n \times d_{ff}}
```

```math
\mathbf{H}' = \text{GELU}(\mathbf{H}) \in \mathbb{R}^{B \times n \times d_{ff}}
```

```math
\mathbf{H}'' = \text{Dropout}(\mathbf{H}', p)
```

```math
\text{FFN}(\mathbf{X}) = \mathbf{H}'' W_2 \in \mathbb{R}^{B \times n \times d}
```

**Example:**
```
d = 512, d_ff = 2048
X shape: [2, 5, 512]
W1 shape: [512, 2048]
H = X @ W1 → [2, 5, 2048]
H' = GELU(H) → [2, 5, 2048]
H'' = Dropout(H', 0.1) → [2, 5, 2048]
W2 shape: [2048, 512]
FFN(X) = H'' @ W2 → [2, 5, 512]
```

---

## 6. Layer Normalization

### 6.1 Layer Normalization Formula

For input $\mathbf{x} \in \mathbb{R}^d$:

```math
\mu = \frac{1}{d} \sum_{i=1}^{d} x_i
```

```math
\sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2
```

```math
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

```math
\text{LayerNorm}(\mathbf{x}) = \gamma \odot \hat{\mathbf{x}} + \beta
```

where:
- $\epsilon$ = small constant (default 1e-5)
- $\gamma$ = learnable scale parameter
- $\beta$ = learnable shift parameter
- $\odot$ = element-wise multiplication

**Example:**
```
x = [1.0, 2.0, 3.0, 4.0]
d = 4
μ = (1.0 + 2.0 + 3.0 + 4.0) / 4 = 2.5
σ² = ((1-2.5)² + (2-2.5)² + (3-2.5)² + (4-2.5)²) / 4
   = (2.25 + 0.25 + 0.25 + 2.25) / 4 = 1.25
σ = √1.25 ≈ 1.118

ε = 1e-5
x̂ = [(1-2.5)/(1.118+1e-5), (2-2.5)/(1.118+1e-5), ...]
   = [-1.341, -0.447, 0.447, 1.341]

γ = [1.0, 1.0, 1.0, 1.0]  (initialized)
β = [0.0, 0.0, 0.0, 0.0]  (initialized)
LayerNorm(x) = γ ⊙ x̂ + β = x̂
```

---

## 7. Transformer Block

### 7.1 Pre-Norm Architecture

**Self-Attention Block:**

```math
\mathbf{X}_1 = \mathbf{X} + \text{Dropout}(\text{Attention}(\text{LayerNorm}(\mathbf{X})), p)
```

**Feed-Forward Block:**

```math
\mathbf{X}_2 = \mathbf{X}_1 + \text{Dropout}(\text{FFN}(\text{LayerNorm}(\mathbf{X}_1)), p)
```

**Complete Transformer Block:**

```math
\mathbf{X}_{out} = \text{TransformerBlock}(\mathbf{X}_{in})
```

**Step-by-step:**

```math
1. \mathbf{X}_{norm1} = \text{LayerNorm}(\mathbf{X}_{in})
2. \mathbf{X}_{attn} = \text{Attention}(\mathbf{X}_{norm1})
3. \mathbf{X}_{attn\_drop} = \text{Dropout}(\mathbf{X}_{attn}, p)
4. \mathbf{X}_1 = \mathbf{X}_{in} + \mathbf{X}_{attn\_drop}$ (residual connection)
5. \mathbf{X}_{norm2} = \text{LayerNorm}(\mathbf{X}_1)
6. \mathbf{X}_{ffn} = \text{FFN}(\mathbf{X}_{norm2})
7. \mathbf{X}_{ffn\_drop} = \text{Dropout}(\mathbf{X}_{ffn}, p)
8. \mathbf{X}_{out} = \mathbf{X}_1 + \mathbf{X}_{ffn\_drop}$ (residual connection)
```

---

## 8. Complete Forward Pass

### 8.1 Full Model Forward Pass

Given input token IDs $\mathbf{T} \in \mathbb{N}^{B \times n}$:

**Step 1: Token Embedding**
```math
\mathbf{X}_0 = E[\mathbf{T}] \in \mathbb{R}^{B \times n \times d}
```

**Step 2: Positional Encoding**
```math
\mathbf{X}_1 = \mathbf{X}_0 + PE \in \mathbb{R}^{B \times n \times d}
```
```math
\mathbf{X}_2 = \text{Dropout}(\mathbf{X}_1, p)
```

**Step 3: Transformer Layers**

For $L$ layers:

```math
\mathbf{X}_{l+1} = \text{TransformerBlock}_l(\mathbf{X}_l), \quad l = 2, 3, ..., L+1
```

**Step 4: Final Layer Norm**

```math
\mathbf{X}_{final} = \text{LayerNorm}(\mathbf{X}_{L+1})
```

**Step 5: Output Projection**

```math
\mathbf{L} = \mathbf{X}_{final} W_{out} \in \mathbb{R}^{B \times n \times V}
```

where $W_{out} \in \mathbb{R}^{d \times V}$ is the output projection matrix.

**Output logits:**

```math
\text{logits}[b, t, v] = \text{log probability of token } v \text{ at position } t \text{ in batch } b
```

---

## 9. Loss Computation

### 9.1 Cross-Entropy Loss

For logits $\mathbf{L} \in \mathbb{R}^{B \times n \times V}$ and labels $\mathbf{Y} \in \mathbb{N}^{B \times n}$:

**Reshape for loss:**

```math
\mathbf{L}_{flat} = \mathbf{L}.view(B \cdot n, V) \in \mathbb{R}^{(B \cdot n) \times V}
```

```math
\mathbf{Y}_{flat} = \mathbf{Y}.view(B \cdot n) \in \mathbb{N}^{B \cdot n}
```

**Softmax probabilities:**

```math
p_i = \frac{\exp(L_{flat}[i, y_i])}{\sum_{v=1}^{V} \exp(L_{flat}[i, v])}
```

**Cross-entropy loss:**

```math
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \log(p_i)
```

where $N$ is the number of valid (non-padding) tokens.

**Masked loss (ignoring padding):**

```math
\mathcal{L} = -\frac{1}{N} \sum_{i: y_i \neq \text{pad\_id}} \log(p_i)
```

**Example:**
```
B = 2, n = 3, V = 128
L shape: [2, 3, 128]
Y = [[72, 105, -100], [44, 32, 119]]  (-100 is padding)

L_flat shape: [6, 128]
Y_flat = [72, 105, -100, 44, 32, 119]

For i=0 (y_i=72):
  logits = L_flat[0] = [0.1, -0.2, ..., 0.5, ...]  (128 values)
  p_0 = exp(0.5) / sum(exp(logits)) ≈ 0.8  (assuming 0.5 was max)
  log(p_0) = log(0.8) ≈ -0.223

For i=2 (y_i=-100):
  Skip (padding token)

Total loss = -1/5 * (log(p_0) + log(p_1) + log(p_3) + log(p_4) + log(p_5))
```

### 9.2 Perplexity

```math
\text{Perplexity} = \exp(\mathcal{L}) = \exp\left(-\frac{1}{N} \sum_{i=1}^{N} \log(p_i)\right)
```

**Example:**
```
If L = 2.0, then Perplexity = exp(2.0) ≈ 7.39
```

---

## 10. Backpropagation

### 10.1 Gradient Flow

**Loss gradient:**

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{L}_{flat}} = \frac{\partial}{\partial \mathbf{L}_{flat}} \left(-\frac{1}{N} \sum_{i=1}^{N} \log(p_i)\right)
```

**Chain rule through output projection:**

```math
\frac{\partial \mathcal{L}}{\partial W_{out}} = \frac{\partial \mathcal{L}}{\partial \mathbf{L}} \cdot \frac{\partial \mathbf{L}}{\partial W_{out}}
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{final}} = \frac{\partial \mathcal{L}}{\partial \mathbf{L}} \cdot W_{out}^T
```

**Through transformer layers (backward):**

For layer $l$ from $L$ to $1$:

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{X}_l} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{l+1}} \cdot \frac{\partial \mathbf{X}_{l+1}}{\partial \mathbf{X}_l}
```

**Residual connection gradient:**

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{X}_{in}} = \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{out}} + \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{residual}}
```

### 10.2 Attention Gradients

**Attention weight gradients:**

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{A}} = \frac{\partial \mathcal{L}}{\partial \mathbf{O}} \cdot \mathbf{V}^T
```

**Query, Key, Value gradients:**

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{Q}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \cdot \mathbf{K} \cdot \frac{1}{\sqrt{d_k}}
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{K}} = \frac{\partial \mathcal{L}}{\partial \mathbf{S}} \cdot \mathbf{Q}^T \cdot \frac{1}{\sqrt{d_k}}
```

```math
\frac{\partial \mathcal{L}}{\partial \mathbf{V}} = \mathbf{A}^T \cdot \frac{\partial \mathcal{L}}{\partial \mathbf{O}}
```

### 10.3 Gradient Clipping

**Gradient norm:**

```math
||\mathbf{g}|| = \sqrt{\sum_{i} g_i^2}
```

**Clipped gradient:**

```math
\mathbf{g}_{clipped} = \begin{cases}
\mathbf{g} & \text{if } ||\mathbf{g}|| \leq \theta \\
\mathbf{g} \cdot \frac{\theta}{||\mathbf{g}||} & \text{if } ||\mathbf{g}|| > \theta
\end{cases}
```

where $\theta$ is the max gradient norm (default 1.0).

**Example:**
```
g = [0.5, 0.8, 1.2]
||g|| = √(0.5² + 0.8² + 1.2²) = √(0.25 + 0.64 + 1.44) = √2.33 ≈ 1.526
θ = 1.0
Since ||g|| > θ:
g_clipped = g × (1.0 / 1.526) = [0.328, 0.524, 0.786]
```

---

## 11. AdamW Optimizer Update

### 11.1 AdamW Algorithm

For parameter $\theta_t$ at step $t$:

**Momentum update:**

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t
```

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2
```

**Bias correction:**

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

**Parameter update:**

```math
\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
```

where:
- $\beta_1 = 0.9$ (momentum decay)
- $\beta_2 = 0.999$ (variance decay)
- $\eta_t$ = learning rate at step $t$
- $\lambda$ = weight decay coefficient (default 0.01)
- $\epsilon = 10^{-8}$ (numerical stability)

### 11.2 Step-by-Step Example

**Initialization:**
```
t = 0
θ₀ = 0.5  (initial parameter value)
m₀ = 0, v₀ = 0
β₁ = 0.9, β₂ = 0.999
η = 0.001 (learning rate)
λ = 0.01  (weight decay)
ε = 1e-8
```

**Step 1:**
```
t = 1
g₁ = 0.3  (gradient)

m₁ = 0.9 × 0 + 0.1 × 0.3 = 0.03
v₁ = 0.999 × 0 + 0.001 × 0.3² = 0.001 × 0.09 = 0.00009

m̂₁ = 0.03 / (1 - 0.9¹) = 0.03 / 0.1 = 0.3
v̂₁ = 0.00009 / (1 - 0.999¹) = 0.00009 / 0.001 = 0.09

θ₁ = 0.5 - 0.001 × (0.3 / (√0.09 + 1e-8) + 0.01 × 0.5)
   = 0.5 - 0.001 × (0.3 / 0.3 + 0.005)
   = 0.5 - 0.001 × (1.005)
   = 0.5 - 0.001005
   = 0.498995
```

**Step 2:**
```
t = 2
g₂ = -0.2

m₂ = 0.9 × 0.03 + 0.1 × (-0.2) = 0.027 - 0.02 = 0.007
v₂ = 0.999 × 0.00009 + 0.001 × (-0.2)² = 0.00008991 + 0.00004 = 0.00012991

m̂₂ = 0.007 / (1 - 0.9²) = 0.007 / 0.19 = 0.0368
v̂₂ = 0.00012991 / (1 - 0.999²) = 0.00012991 / 0.001999 ≈ 0.06496

θ₂ = 0.498995 - 0.001 × (0.0368 / (√0.06496 + 1e-8) + 0.01 × 0.498995)
   = 0.498995 - 0.001 × (0.0368 / 0.2549 + 0.00498995)
   = 0.498995 - 0.001 × (0.1444 + 0.00498995)
   = 0.498995 - 0.001 × 0.1494
   = 0.498995 - 0.0001494
   = 0.498846
```

### 11.3 AdamW vs Adam

The key difference is the weight decay term:

**Adam:**
```math
\theta_t = \theta_{t-1} - \eta_t \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

Then separately apply weight decay:
```math
\theta_t = \theta_t (1 - \lambda)
```

**AdamW:**
```math
\theta_t = \theta_{t-1} - \eta_t \left(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1}\right)
```

AdamW decouples weight decay from gradient-based updates, leading to better generalization.

---

## 12. Learning Rate Scheduling

### 12.1 Cosine Annealing Schedule

```math
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\pi \cdot \frac{t}{T_{max}})}{2}
```

where:
- $\eta_{max}$ = initial learning rate
- $\eta_{min}$ = minimum learning rate (default 0)
- $T_{max}$ = total number of steps
- $t$ = current step

**Example:**
```
η_max = 0.001
η_min = 0
T_max = 10000
t = 0: η₀ = 0 + (0.001 - 0) × (1 + cos(0)) / 2 = 0.001 × 1 = 0.001
t = 2500: η = 0 + 0.001 × (1 + cos(π/4)) / 2 = 0.001 × (1 + 0.707) / 2 ≈ 0.000854
t = 5000: η = 0 + 0.001 × (1 + cos(π/2)) / 2 = 0.001 × (1 + 0) / 2 = 0.0005
t = 7500: η = 0 + 0.001 × (1 + cos(3π/4)) / 2 ≈ 0.000146
t = 10000: η = 0 + 0.001 × (1 + cos(π)) / 2 = 0.001 × (1 + (-1)) / 2 = 0
```

### 12.2 Learning Rate Schedule Visualization

The cosine annealing schedule creates a smooth decay from maximum to minimum learning rate following a cosine curve.

---

## 13. Text Generation

### 13.1 Autoregressive Generation

Given prompt tokens $\mathbf{P} = [p_1, p_2, ..., p_k]$:

**Initialization:**
```math
\mathbf{T}_0 = \mathbf{P}
```

**For each generation step $t$ from $k+1$ to $k+n$:**

1. **Forward pass:**
```math
\mathbf{L}_t = \text{Model}(\mathbf{T}_{t-1})
```

2. **Get next token logits:**
```math
\mathbf{l}_t = \mathbf{L}_t[:, -1, :] \in \mathbb{R}^{B \times V}
```

3. **Apply temperature:**
```math
\mathbf{l}_t' = \frac{\mathbf{l}_t}{T}
```
   where $T$ is the temperature (default 1.0).

4. **Top-k filtering (optional):**
```math
\mathbf{l}_t''[v] = \begin{cases}
\mathbf{l}_t'[v] & \text{if } v \in \text{top-k}(\mathbf{l}_t') \\
-\infty & \text{otherwise}
\end{cases}
```

5. **Top-p (nucleus) sampling (optional):**
   - Sort tokens by probability
   - Find smallest set $S$ where $\sum_{v \in S} p(v) \geq p$
   - Set probabilities outside $S$ to 0

6. **Sample token:**
```math
p_t = \text{softmax}(\mathbf{l}_t'') \in \mathbb{R}^V
t_t \sim \text{Categorical}(p_t)
```

7. **Append token:**
```math
\mathbf{T}_t = [\mathbf{T}_{t-1}, t_t]
```

### 13.2 Generation Example

**Input:**
```
Prompt: "Hello"
P = [72, 101, 108, 108, 111]  ("Hello")
```

**Step 1:**
```
T₀ = [72, 101, 108, 108, 111]
Forward pass → L₁ shape: [1, 5, 128]
l₁ = L₁[0, -1, :] = [0.1, -0.2, ..., 0.8, ...]  (logits for next token)

Apply temperature T=1.0:
l₁' = l₁ / 1.0 = l₁

Softmax:
p₁ = softmax(l₁) = [0.001, 0.0005, ..., 0.15, ...]

Sample (let's say token 32 = ' '):
t₁ = 32
T₁ = [72, 101, 108, 108, 111, 32]
```

**Step 2:**
```
T₁ = [72, 101, 108, 108, 111, 32]
Forward pass → L₂ shape: [1, 6, 128]
l₂ = L₂[0, -1, :]

Continue until max_length reached...
```

### 13.3 Top-k Sampling

**Example:**
```
V = 128, k = 50
l = [0.5, 0.3, ..., -0.1, ...]  (128 logits)

Sort and get top 50:
top_k_indices = [0, 5, 12, ..., 87]  (50 tokens)

l' = [-inf, -inf, ..., 0.5, -inf, ..., 0.3, ...]
     (only top-k kept, others set to -inf)
```

### 13.4 Top-p (Nucleus) Sampling

**Example:**
```
p = 0.95 (threshold)
p_sorted = [0.3, 0.2, 0.15, 0.1, 0.05, 0.03, ...]  (sorted probabilities)

Cumulative: [0.3, 0.5, 0.65, 0.75, 0.8, 0.83, ...]

Find where cumulative ≥ 0.95:
At index 20: cumulative = 0.96 ≥ 0.95
Keep first 20 tokens, set others to 0
```

---

## Summary

This document provides complete mathematical formulations for:

1. **Data Processing**: Tokenization, chunking, padding
2. **Embeddings**: Token embeddings and positional encodings
3. **Attention**: Multi-head self-attention with scaling and masking
4. **Feed-Forward**: GELU activation and linear transformations
5. **Normalization**: Layer normalization with learnable parameters
6. **Training**: Loss computation, backpropagation, gradient clipping
7. **Optimization**: AdamW update rule with momentum and variance tracking
8. **Scheduling**: Cosine annealing learning rate schedule
9. **Generation**: Autoregressive sampling with temperature, top-k, and top-p

Each section includes:
- Mathematical formulations
- Step-by-step calculations
- Worked examples with numerical values
- Implementation details

All equations are directly implementable in PyTorch and match the actual implementation in the SheepOp codebase.

