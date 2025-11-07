# SheepOp LLM - Mathematical Control System Model

Complete mathematical control system formulation of the SheepOp Language Model, treating the entire system as a unified mathematical control system with state-space representations, transfer functions, and step-by-step explanations.

## Table of Contents

1. [System Overview](#1-system-overview)
2. [State-Space Representation](#2-state-space-representation)
3. [Tokenizer as Input Encoder](#3-tokenizer-as-input-encoder)
4. [Seed Control System](#4-seed-control-system)
5. [Embedding Layer Control](#5-embedding-layer-control)
6. [Positional Encoding State](#6-positional-encoding-state)
7. [Self-Attention Control System](#7-self-attention-control-system)
8. [Feed-Forward Control](#8-feed-forward-control)
9. [Layer Normalization Feedback](#9-layer-normalization-feedback)
10. [Complete System Dynamics](#10-complete-system-dynamics)
11. [Training as Optimization Control](#11-training-as-optimization-control)
12. [Inference Control Loop](#12-inference-control-loop)

---

## 1. System Overview

### 1.1 Control System Architecture

The SheepOp LLM can be modeled as a **nonlinear dynamical control system** with:

- **Input**: Character sequence $\mathbf{c} = [c_1, c_2, ..., c_n]$
- **State**: Hidden representations $\mathbf{h}\_t $at each layer and time step
- **Control**: Model parameters $\theta = \{W_Q, W_K, W_V, W_1, W_2, ...\}
 $
- **Output**: Probability distribution over vocabulary $\mathbf{p}\_t \in \mathbb{R}^V$

**System Block Diagram:**

```
Input Sequence → Tokenizer → Embeddings → Positional Encoding →
    ↓
    [Transformer Layer 1] → [Transformer Layer 2] → ... → [Transformer Layer L]
    ↓
    Output Projection → Logits → Softmax → Output Probabilities
```

### 1.2 Mathematical System Formulation

The complete system can be expressed as:

```math

\mathbf{y}_t = \mathcal{F}(\mathbf{x}_t, \mathbf{h}_t, \theta, \mathbf{s})

```

where:

- $\mathbf{x}\_t $= input at time$ t$
- $\mathbf{h}\_t $= hidden state at time$ t$
- $\theta $= system parameters (weights)
- $\mathbf{s} $= seed for randomness
- $\mathcal{F} $= complete forward function

---

## 2. State-Space Representation

### 2.1 Discrete-Time State-Space Model

For a transformer with L layers and sequence length n :

**State Vector:**

```math
\mathbf{H}_t = \begin{bmatrix}
\mathbf{h}_t^{(1)} \\
\mathbf{h}_t^{(2)} \\
\vdots \\
\mathbf{h}_t^{(L)}
\end{bmatrix} \in \mathbb{R}^{L \times n \times d}
```

where

$\mathbf{h}_t^{(l)} \in \mathbb{R}^{n \times d}  is the hidden state at layer  l .$

**State Update Equation:**

```math

\mathbf{h}_t^{(l+1)} = f_l(\mathbf{h}_t^{(l)}, \theta_l), \quad l = 0, 1, ..., L-1


where  f_l  is the transformation at layer  l .
```

**Output Equation:**

```math

\mathbf{y}_t = g(\mathbf{h}_t^{(L)}, \theta_{out})

```

### 2.2 System Linearity Analysis

The system is **nonlinear** due to:

- Attention mechanism (softmax)
- Activation functions (GELU)
- Layer normalization

However, individual components can be analyzed as **piecewise linear** systems.

---

## 3. Tokenizer as Input Encoder

### 3.1 Tokenizer Control Function

The tokenizer maps a character sequence to a discrete token sequence:

```math

\mathcal{T}: \mathcal{C}^* \rightarrow \mathbb{N}^*

```

**Mathematical Formulation:**

For input sequence $\mathbf{c} = [c_1, c_2, ..., c_n] $:

```math

\mathbf{t} = \mathcal{T}(\mathbf{c}) = [V(c_1), V(c_2), ..., V(c_n)]


where  V: \mathcal{C} \rightarrow \mathbb{N}  is the vocabulary mapping function.
```

### 3.2 Vocabulary Mapping Function

```math

V(c) = \begin{cases}
0 & \text{if } c = \text{<pad>} \\
1 & \text{if } c = \text{<unk>} \\
2 & \text{if } c = \text{<bos>} \\
3 & \text{if } c = \text{<eos>} \\
v & \text{if } c \in \mathcal{C}_{vocab}
\end{cases}

```

**Control Properties:**

- **Deterministic**: Same input always produces same output
- **Invertible**: For most tokens, $V^{-1}$ exists
- **Bijective**: Each character maps to unique token ID

### 3.3 Tokenizer State Space

The tokenizer maintains internal state:

```math

\Sigma_{\mathcal{T}} = \{V, V^{-1}, \text{padding\_strategy}, \text{max\_length}\}

```

**State Transition:**

```math

\Sigma_{\mathcal{T}}' = \Sigma_{\mathcal{T}} \quad \text{(static during operation)}

```

### 3.4 Step-by-Step Explanation

**Step 1: Character Extraction**

- Input: Raw text string "Hello"
- Process: Extract each character $c \in \{'H', 'e', 'l', 'l', 'o'\}$
- Meaning: Break down text into atomic units

**Step 2: Vocabulary Lookup**

- Process: Apply $V(c)$ to each character
- Example: $V('H') = 72, V('e') = 101, V('l') = 108, V('o') = 111$
- Meaning: Convert characters to numerical indices

**Step 3: Sequence Formation**

- Output: $\mathbf{t} = [72, 101, 108, 108, 111]$
- Meaning: Numerical representation ready for embedding

**Control Impact**: Tokenizer creates the **foundation** for all subsequent processing. Any error here propagates through the entire system.

---

## 4. Seed Control System

### 4.1 Seed as System Initialization

The seed $s \in \mathbb{N}$ controls **randomness** throughout the system:

```math

\mathcal{R}(\mathbf{x}, s) = \text{deterministic\_random}(\mathbf{x}, s)

```

### 4.2 Seed Propagation Function

**Initialization:**

```math

\text{seed\_torch}(s): \text{torch.manual\_seed}(s)


\text{seed\_cuda}(s): \text{torch.cuda.manual\_seed\_all}(s)


\text{seed\_cudnn}(s): \text{torch.backends.cudnn.deterministic} = \text{True}

```

**Mathematical Model:**

```math

\mathbb{P}(\mathbf{W} | s) = \begin{cases}
\delta(\mathbf{W} - \mathbf{W}_s) & \text{if deterministic} \\
\text{some distribution} & \text{if stochastic}
\end{cases}


where  \delta  is the Dirac delta and  \mathbf{W}_s  is the weight initialization given seed  s .
```

### 4.3 Seed Control Equation

For weight initialization:

```math

\mathbf{W}_0 = \mathcal{I}(\mathbf{s}, \text{init\_method})


where  \mathcal{I}  is the initialization function.
```

**Example - Normal Initialization:**

```math

\mathbf{W}_0 \sim \mathcal{N}(0, \sigma^2) \quad \text{with random state } r(s)



W_{ij} = \sigma \cdot \Phi^{-1}(U_{ij}(s))


where:
-  \mathcal{N}(0, \sigma^2)  = normal distribution
-  \Phi^{-1}  = inverse CDF
-  U_{ij}(s)  = uniform random number from seed  s
-  \sigma = 0.02  (typical value)
```

### 4.4 Step-by-Step Explanation

**Step 1: Seed Input**

- Input: $s = 42$
- Meaning: Provides reproducibility guarantee

**Step 2: RNG State Initialization**

- Process: Set all random number generators to state based on $s$
- Meaning: Ensures deterministic behavior

**Step 3: Weight Initialization**

- Process: Generate all weights using RNG with seed $s$
- Example: $W\_{ij} = \text{normal}(0, 0.02, \text{seed}=42)$
- Meaning: Starting point for optimization

**Step 4: Training Determinism**

- Process: Same seed + same data → same gradients → same updates
- Meaning: Complete reproducibility

**Control Impact**: Seed controls **initial conditions** and **stochastic processes** throughout training. It's the **control parameter** for reproducibility.

---

## 5. Embedding Layer Control

### 5.1 Embedding as Linear Transformation

The embedding layer performs a **lookup operation**:

```math

\mathcal{E}: \mathbb{N} \rightarrow \mathbb{R}^d

```

**Mathematical Formulation:**

```math

\mathbf{E} \in \mathbb{R}^{V \times d} \quad \text{(embedding matrix)}



\mathbf{x}_t = \mathbf{E}[\mathbf{t}_t] = \mathbf{E}_t \in \mathbb{R}^d


where  \mathbf{t}_t \in \mathbb{N}  is the token ID at position  t .
```

### 5.2 Embedding Control System

**Batch Processing:**

```math

\mathbf{X} = \mathbf{E}[\mathbf{T}] \in \mathbb{R}^{B \times n \times d}


where  \mathbf{T} \in \mathbb{N}^{B \times n}  is the batch of token IDs.
```

**Control Function:**

```math

\mathbf{X} = \mathcal{E}(\mathbf{T}, \mathbf{E})

```

**Gradient Flow:**

```math

\frac{\partial \mathcal{L}}{\partial \mathbf{E}} = \sum_{b,t} \frac{\partial \mathcal{L}}{\partial \mathbf{X}_{b,t}} \cdot \mathbf{1}[\mathbf{T}_{b,t}]


where  \mathbf{1}[\mathbf{T}_{b,t}]  is a one-hot indicator.
```

### 5.3 Step-by-Step Explanation

**Step 1: Token ID Input**

- Input: $t = 72$ (token ID for 'H')
- Meaning: Discrete index into vocabulary

**Step 2: Matrix Lookup**

- Process: $\mathbf{x} = \mathbf{E}[72]$
- Example: $\mathbf{x} = [0.1, -0.2, 0.3, ..., 0.05] \in \mathbb{R}^{512}$
- Meaning: Continuous vector representation

**Step 3: Semantic Encoding**

- Property: Similar tokens have similar embeddings (after training)
- Meaning: Embeddings capture semantic relationships

**Control Impact**: Embedding layer **projects** discrete tokens into continuous space, enabling gradient-based optimization.

---

## 6. Positional Encoding State

### 6.1 Positional Encoding as Additive Control

```math

\mathbf{X}_{pos} = \mathbf{X} + \mathbf{PE} \in \mathbb{R}^{B \times n \times d}


where  \mathbf{PE} \in \mathbb{R}^{n \times d}  is the positional encoding matrix.
```

### 6.2 Positional Encoding Function

```math

PE_{(pos, i)} = \begin{cases}
\sin\left(\frac{pos}{10000^{2i/d}}\right) & \text{if } i \text{ is even} \\
\cos\left(\frac{pos}{10000^{2(i-1)/d}}\right) & \text{if } i \text{ is odd}
\end{cases}

```

### 6.3 Control System Interpretation

**Additive Control:**

```math

\mathbf{X}_{out} = \mathbf{X}_{in} + \mathbf{U}_{pos}


where  \mathbf{U}_{pos}  is the **control input** representing position information.
```

**Meaning**: Positional encoding **injects** positional information into the embeddings.

### 6.4 Step-by-Step Explanation

**Step 1: Position Index**

- Input: Position $pos = 0, 1, 2, ..., n-1$
- Meaning: Absolute position in sequence

**Step 2: Encoding Generation**

- Process: Compute $PE\_{(pos, i)}$ for each dimension $ i$
- Example: $PE*{(0, 0)} = 0, PE*{(0, 1)} = 1, PE\_{(1, 0)} \approx 0.84$
- Meaning: Unique pattern for each position

**Step 3: Addition Operation**

- Process: $\mathbf{X}\_{pos} = \mathbf{X} + PE$
- Meaning: Position information added to embeddings

**Step 4: Multi-Scale Representation**

- Property: Different dimensions encode different frequency scales
- Meaning: Model can learn both local and global positional patterns

**Control Impact**: Positional encoding provides **temporal/spatial awareness** to the model, enabling it to understand sequence order.

---

## 7. Self-Attention Control System

### 7.1 Attention as Information Routing

Self-attention can be modeled as a **dynamical control system** that routes information:

```math

\mathbf{O} = \text{Attention}(\mathbf{X}, \mathbf{W}_Q, \mathbf{W}_K, \mathbf{W}_V)

```

### 7.2 State-Space Model for Attention

**Query, Key, Value Generation:**

```math

\mathbf{Q} = \mathbf{X} \mathbf{W}_Q \in \mathbb{R}^{B \times n \times d}


\mathbf{K} = \mathbf{X} \mathbf{W}_K \in \mathbb{R}^{B \times n \times d}


\mathbf{V} = \mathbf{X} \mathbf{W}_V \in \mathbb{R}^{B \times n \times d}

```

**Attention Scores (Transfer Function):**

```math

\mathbf{S} = \frac{\mathbf{Q} \mathbf{K}^T}{\sqrt{d_k}} \in \mathbb{R}^{B \times h \times n \times n}

```

**Attention Weights (Control Signal):**

```math

\mathbf{A} = \text{softmax}(\mathbf{S}) \in \mathbb{R}^{B \times h \times n \times n}

```

**Output (Controlled Response):**

```math

\mathbf{O} = \mathbf{A} \mathbf{V} \in \mathbb{R}^{B \times h \times n \times d_k}

```

### 7.3 Control System Interpretation

**Attention as Feedback Control:**

```math

\mathbf{O}_i = \sum_{j=1}^{n} A_{ij} \mathbf{V}_j


where  A_{ij}  is the **control gain** determining how much information flows from position  j  to position  i .
```

**Meaning**: Attention acts as a **learnable routing mechanism** controlled by similarities between queries and keys.

### 7.4 Multi-Head Attention Control

**Head Splitting:**

```math

\mathbf{Q}_h = \mathbf{Q}[:, :, h \cdot d_k : (h+1) \cdot d_k] \in \mathbb{R}^{B \times n \times d_k}

```

**Parallel Processing:**

```math

\mathbf{O}_h = \text{Attention}(\mathbf{Q}_h, \mathbf{K}_h, \mathbf{V}_h), \quad h = 1, ..., H

```

**Concatenation:**

```math

\mathbf{O} = \text{Concat}[\mathbf{O}_1, \mathbf{O}_2, ..., \mathbf{O}_H] \in \mathbb{R}^{B \times n \times d}

```

### 7.5 Causal Masking Control

**Causal Mask:**

```math

M_{ij} = \begin{cases}
0 & \text{if } i \geq j \text{ (allowed)} \\
-\infty & \text{if } i < j \text{ (masked)}
\end{cases}

```

**Masked Attention:**

```math

\mathbf{S}_{masked} = \mathbf{S} + M

```

**Effect**: Prevents information flow from future positions.

### 7.6 Step-by-Step Explanation

**Step 1: Query, Key, Value Generation**

- Process: Linear transformations of input
- Meaning: Create three representations: what to look for (Q), what to match (K), what to retrieve (V)

**Step 2: Similarity Computation**

- Process: $S\_{ij} = Q_i \cdot K_j / \sqrt{d_k}$
- Meaning: Measure similarity/relevance between positions $i$ and $ j
 $

**Step 3: Softmax Normalization**

- Process: $A*{ij} = \exp(S*{ij}) / \sum*k \exp(S*{ik})$
- Meaning: Convert similarities to probability distribution (attention weights)

**Step 4: Weighted Aggregation**

- Process: $O*i = \sum_j A*{ij} V_j$
- Meaning: Combine values weighted by attention probabilities

**Step 5: Information Flow**

- Property: Each position receives information from all other positions (with causal masking)
- Meaning: Enables long-range dependencies and context understanding

**Control Impact**: Self-attention is the **core control mechanism** that determines **what information flows where** in the sequence.

---

## 8. Feed-Forward Control

### 8.1 Feed-Forward as Nonlinear Transformation

```math

\text{FFN}(\mathbf{X}) = \text{GELU}(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2

```

### 8.2 Control System Model

**Two-Stage Transformation:**

```math

\mathbf{H} = \mathbf{X} \mathbf{W}_1 \in \mathbb{R}^{B \times n \times d_{ff}}



\mathbf{H}' = \text{GELU}(\mathbf{H}) \in \mathbb{R}^{B \times n \times d_{ff}}



\mathbf{O} = \mathbf{H}' \mathbf{W}_2 \in \mathbb{R}^{B \times n \times d}

```

### 8.3 GELU Activation Control

```math

\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right)

```

**Control Interpretation**: GELU applies **smooth gating** - values near zero are suppressed, positive values pass through.

### 8.4 Step-by-Step Explanation

**Step 1: Expansion**

- Process: $\mathbf{H} = \mathbf{X} \mathbf{W}_1 expands to d_{ff} > d$
- Example: $d = 512 \rightarrow d\_{ff} = 2048$
- Meaning: Increases capacity for complex transformations

**Step 2: Nonlinear Activation**

- Process: $\mathbf{H}' = \text{GELU}(\mathbf{H})$
- Meaning: Introduces nonlinearity, enabling complex function approximation

**Step 3: Compression**

- Process: $\mathbf{O} = \mathbf{H}' \mathbf{W}\_2 $compresses back to$ d$
- Meaning: Projects back to original dimension

**Control Impact**: FFN provides **nonlinear processing power** and **feature transformation** at each position.

---

## 9. Layer Normalization Feedback

### 9.1 Normalization as Feedback Control

```math

\text{LayerNorm}(\mathbf{x}) = \gamma \odot \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta


where:
-  \mu = \frac{1}{d} \sum_{i=1}^{d} x_i  (mean)
-  \sigma^2 = \frac{1}{d} \sum_{i=1}^{d} (x_i - \mu)^2  (variance)
-  \gamma, \beta  = learnable parameters (scale and shift)
```

### 9.2 Control System Interpretation

**Normalization as State Regulation:**

```math

\mathbf{x}_{norm} = \gamma \odot \frac{\mathbf{x} - \mu(\mathbf{x})}{\sigma(\mathbf{x})} + \beta

```

**Meaning**: Normalization **regulates** the distribution of activations, preventing saturation and improving gradient flow.

### 9.3 Pre-Norm Architecture

**Transformer Block with Pre-Norm:**

```math

\mathbf{x}_{norm} = \text{LayerNorm}(\mathbf{x}_{in})


\mathbf{x}_{attn} = \text{Attention}(\mathbf{x}_{norm})


\mathbf{x}_{out} = \mathbf{x}_{in} + \mathbf{x}_{attn} \quad \text{(residual connection)}

```

**Control Impact**: Pre-norm architecture provides **stability** and **better gradient flow**.

### 9.4 Step-by-Step Explanation

**Step 1: Mean Computation**

- Process: $\mu = \frac{1}{d} \sum x_i$
- Meaning: Find center of distribution

**Step 2: Variance Computation**

- Process: $\sigma^2 = \frac{1}{d} \sum (x_i - \mu)^2$
- Meaning: Measure spread of distribution

**Step 3: Normalization**

- Process: $\hat{x}\_i = (x_i - \mu) / \sqrt{\sigma^2 + \epsilon}$
- Meaning: Standardize to zero mean, unit variance

**Step 4: Scale and Shift**

- Process: $x\_{out} = \gamma \odot \hat{x} + \beta$
- Meaning: Allow model to learn optimal scale and shift

**Control Impact**: Layer normalization provides **stability** and **faster convergence** by maintaining consistent activation distributions.

---

## 10. Complete System Dynamics

### 10.1 Complete Forward Pass

**System State Evolution:**

```math

\mathbf{h}_0 = \mathcal{E}(\mathbf{T}) + \mathbf{PE} \quad \text{(embedding + positional)}



\mathbf{h}_l = \text{TransformerBlock}_l(\mathbf{h}_{l-1}), \quad l = 1, ..., L



\mathbf{y} = \mathbf{h}_L \mathbf{W}_{out} \in \mathbb{R}^{B \times n \times V}

```

### 10.2 Recursive System Equation

```math

\mathbf{h}_t^{(l)} = f_l(\mathbf{h}_t^{(l-1)}, \theta_l)


where:


f_l(\mathbf{x}, \theta_l) = \mathbf{x} + \text{Dropout}(\text{Attention}(\text{LayerNorm}(\mathbf{x}))) + \text{Dropout}(\text{FFN}(\text{LayerNorm}(\mathbf{x} + \text{Attention}(\text{LayerNorm}(\mathbf{x})))))

```

### 10.3 System Transfer Function

The complete system can be viewed as:

```math

\mathbf{Y} = \mathcal{F}(\mathbf{T}, \theta, \mathbf{s})


where:
-  \mathbf{T}  = input tokens
-  \theta  = all parameters
-  \mathbf{s}  = seed
```

**Properties:**

- **Nonlinear**: Due to softmax, GELU, normalization
- **Differentiable**: All operations have gradients
- **Compositional**: Built from simpler functions

### 10.4 Step-by-Step System Flow

**Step 1: Input Encoding**

- Input: Token sequence $\mathbf{T}$
- Process: Embedding + Positional Encoding
- Output: $\mathbf{h}\_0 \in \mathbb{R}^{B \times n \times d}$
- Meaning: Convert discrete tokens to continuous vectors with position info

**Step 2: Layer Processing**

- For each layer $l = 1, ..., L $:
  - Process: Self-attention + FFN with residual connections
  - Output: $\mathbf{h}\_l \in \mathbb{R}^{B \times n \times d}$
  - Meaning: Transform representations through attention and processing

**Step 3: Output Generation**

- Process: Final layer norm + output projection
- Output: $\mathbf{L} \in \mathbb{R}^{B \times n \times V} (logits)$
- Meaning: Predict probability distribution over vocabulary

**Step 4: Probability Computation**

- Process: Softmax over logits
- Output: $\mathbf{p} \in \mathbb{R}^{B \times n \times V}
  (probabilities)$
- Meaning: Normalized probability distribution for next token prediction

---

## 11. Training as Optimization Control

### 11.1 Training as Optimal Control Problem

**Objective Function:**

```math

J(\theta) = \frac{1}{N} \sum_{i=1}^{N} \mathcal{L}(\mathbf{y}_i, \hat{\mathbf{y}}_i(\theta))


where:
-  \mathcal{L}  = loss function (cross-entropy)
-  \mathbf{y}_i  = true labels
-  \hat{\mathbf{y}}_i(\theta)  = model predictions
```

**Optimization Problem:**

```math

\theta^* = \arg\min_{\theta} J(\theta)

```

### 11.2 Gradient-Based Control

**Gradient Computation:**

```math

\mathbf{g}_t = \nabla_\theta J(\theta_t) = \frac{\partial J}{\partial \theta_t}

```

**Parameter Update (AdamW):**

```math

\theta_{t+1} = \theta_t - \eta_t \left(\frac{\hat{\mathbf{m}}_t}{\sqrt{\hat{\mathbf{v}}_t} + \epsilon} + \lambda \theta_t\right)


where:
-  \hat{\mathbf{m}}_t  = biased-corrected momentum
-  \hat{\mathbf{v}}_t  = biased-corrected variance
-  \eta_t  = learning rate (controlled by scheduler)
-  \lambda  = weight decay coefficient
```

### 11.3 Learning Rate Control

**Cosine Annealing Schedule:**

```math

\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \cdot \frac{1 + \cos(\pi \cdot \frac{t}{T_{max}})}{2}

```

**Control Interpretation**: Learning rate acts as **gain scheduling** - high gain initially for fast convergence, low gain later for fine-tuning.

### 11.4 Gradient Clipping Control

**Clipping Function:**

```math

\mathbf{g}_{clipped} = \begin{cases}
\mathbf{g} & \text{if } ||\mathbf{g}|| \leq \theta \\
\mathbf{g} \cdot \frac{\theta}{||\mathbf{g}||} & \text{if } ||\mathbf{g}|| > \theta
\end{cases}

```

**Purpose**: Prevents **explosive gradients** that could destabilize training.

### 11.5 Step-by-Step Training Control

**Step 1: Forward Pass**

- Process: $\hat{\mathbf{y}} = \mathcal{F}(\mathbf{x}, \theta_t)$
- Meaning: Compute predictions with current parameters

**Step 2: Loss Computation**

- Process: $\mathcal{L} = \text{CrossEntropy}(\hat{\mathbf{y}}, \mathbf{y})$
- Meaning: Measure prediction error

**Step 3: Backward Pass**

- Process: $\mathbf{g} = \nabla\_\theta \mathcal{L}$
- Meaning: Compute gradients for all parameters

**Step 4: Gradient Clipping**

- Process: $\mathbf{g}\_{clipped} = \text{Clip}(\mathbf{g}, \theta)$
- Meaning: Prevent gradient explosion

**Step 5: Optimizer Update**

- Process: $\theta*{t+1} = \text{AdamW}(\theta_t, \mathbf{g}*{clipped}, \eta_t)$
- Meaning: Update parameters using adaptive learning rate

**Step 6: Learning Rate Update**

- Process: $\eta\_{t+1} = \text{Scheduler}(\eta_t, t)$
- Meaning: Adjust learning rate according to schedule

**Control Impact**: Training process is a **closed-loop control system** where:

- **Error signal**: Loss
- **Controller**: Optimizer (AdamW)
- **Actuator**: Parameter updates
- **Plant**: Model forward pass

---

## 12. Inference Control Loop

### 12.1 Autoregressive Generation as Control Loop

**State-Space Model:**

```math

\mathbf{h}_t = \mathcal{F}(\mathbf{x}_t, \mathbf{h}_{t-1}, \theta)



\mathbf{p}_t = \text{softmax}(\mathbf{h}_t \mathbf{W}_{out})



\mathbf{x}_{t+1} \sim \text{Categorical}(\mathbf{p}_t)

```

### 12.2 Generation Control Function

**Step-by-Step:**

1. **Current State**: $\mathbf{h}\_t$
2. **Output Generation**: $\mathbf{p}_t = \text{softmax}(\mathbf{h}\_t \mathbf{W}_{out})$
3. **Sampling**: $x\_{t+1} \sim \mathbf{p}\_t (with temperature, top-k, top-p)$
4. **State Update**: $\mathbf{h}_{t+1} = \mathcal{F}([\mathbf{h}\_t, x_{t+1}], \theta)$
5. **Repeat**: Until max length or stop token

### 12.3 Sampling Control Parameters

**Temperature Control:**

```math

\mathbf{p}_t^{temp} = \text{softmax}\left(\frac{\mathbf{h}_t \mathbf{W}_{out}}{T}\right)


-  T < 1 : More deterministic (sharp distribution)
-  T > 1 : More random (flat distribution)
-  T = 1 : Default
```

**Top-k Filtering:**

```math

\mathbf{p}_t^{topk}[v] = \begin{cases}
\mathbf{p}_t[v] & \text{if } v \in \text{top-k}(\mathbf{p}_t) \\
0 & \text{otherwise}
\end{cases}

```

**Top-p (Nucleus) Sampling:**

```math

\mathbf{p}_t^{topp}[v] = \begin{cases}
\mathbf{p}_t[v] & \text{if } v \in S_p \\
0 & \text{otherwise}
\end{cases}


where  S_p  is the smallest set such that  \sum_{v \in S_p} \mathbf{p}_t[v] \geq p .
```

### 12.4 Step-by-Step Inference Control

**Step 1: Initialization**

- Input: Prompt tokens $\mathbf{P} = [p_1, ..., p_k]$
- Process: Initialize state $\mathbf{h}\_0 = \mathcal{E}(\mathbf{P}) + \mathbf{PE}$
- Meaning: Set initial state from prompt

**Step 2: Forward Pass**

- Process: $\mathbf{h}_t = \text{Transformer}(\mathbf{h}_{t-1})$
- Output: Hidden state $\mathbf{h}\_t$
- Meaning: Process current sequence

**Step 3: Logit Generation**

- Process: $\mathbf{l}_t = \mathbf{h}\_t \mathbf{W}_{out}$
- Output: Logits $\mathbf{l}\_t \in \mathbb{R}^V$
- Meaning: Unnormalized scores for each token

**Step 4: Probability Computation**

- Process: $\mathbf{p}\_t = \text{softmax}(\mathbf{l}\_t / T)$
- Output: Probability distribution $\mathbf{p}\_t$
- Meaning: Normalized probabilities with temperature

**Step 5: Sampling**

- Process: $x\_{t+1} \sim \mathbf{p}\_t (with optional top-k/top-p)$
- Output: Next token $x\_{t+1}$
- Meaning: Stochastically select next token

**Step 6: State Update**

- Process: Append $x*{t+1}$ to sequence, update $\mathbf{h}*{t+1}$
- Meaning: Incorporate new token into state

**Step 7: Termination Check**

- Condition: $t < \text{max_length} and x\_{t+1} \neq \text{<eos>}$
- If true: Go to Step 2
- If false: Return generated sequence

**Control Impact**: Inference is a **recurrent control system** where:

- **State**: Current hidden representation
- **Control**: Sampling strategy (temperature, top-k, top-p)
- **Output**: Generated token sequence

---

## Summary: Unified Control System Model

### Complete System Equation

```math

\mathbf{Y} = \mathcal{G}(\mathbf{C}, \theta, \mathbf{s}, \mathbf{T}, \{k, p\})


where:
-  \mathbf{C}  = input characters
-  \theta  = model parameters
-  \mathbf{s}  = seed
-  \mathbf{T}  = temperature
-  \{k, p\}  = top-k and top-p parameters
```

### System Components as Control Elements

1. **Tokenizer**: Input encoder $\mathcal{T}$
2. **Seed**: Initialization control $\mathbf{s}$
3. **Embeddings**: State projection $\mathcal{E}$
4. **Positional Encoding**: Temporal control $\mathbf{PE}$
5. **Attention**: Information routing $\mathcal{A}$
6. **FFN**: Nonlinear transformation $\mathcal{F}$
7. **Normalization**: State regulation $\mathcal{N}$
8. **Optimizer**: Parameter control $\mathcal{O}$
9. **Scheduler**: Learning rate control $\mathcal{S}$
10. **Sampling**: Output control $\mathcal{P}$

### Control Flow Summary

```
Input Characters
    ↓ [Tokenizer Control]
Token IDs
    ↓ [Seed Control]
Initialized Parameters
    ↓ [Embedding Control]
Vector Representations
    ↓ [Positional Control]
Position-Aware Vectors
    ↓ [Attention Control]
Context-Aware Representations
    ↓ [FFN Control]
Transformed Features
    ↓ [Normalization Control]
Stabilized Activations
    ↓ [Output Control]
Probability Distributions
    ↓ [Sampling Control]
Generated Tokens
```

Each component acts as a **control element** in a unified dynamical system, working together to transform input text into meaningful language model outputs.

---

## 13. Block Diagram Analysis

### 13.1 Single Transformer Block Control System

**Block Diagram (a): Detailed Single Transformer Block**

```
Input X
    ↓
    ┌─────────────┐
    │ LayerNorm   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Multi-Head  │
    │ Attention   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Dropout    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── (Residual Connection from X)
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ LayerNorm   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Feed-Forward│
    │  Network    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Dropout    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── (Residual Connection)
    └──────┬──────┘
           ↓
    Output X'
```

**Mathematical Transfer Function:**

```math

\mathbf{X}_{out} = \mathbf{X}_{in} + \text{Dropout}(\text{FFN}(\text{LayerNorm}(\mathbf{X}_{in} + \text{Dropout}(\text{Attention}(\text{LayerNorm}(\mathbf{X}_{in})))))

```

### 13.2 Simplified Transformer Block

**Block Diagram (b): Simplified Single Block**

```
Input X
    ↓
    ┌─────────────────────────────────────┐
    │ TransformerBlock                    │
    │ G_block(X) = X + Attn(LN(X)) +      │
    │              FFN(LN(X + Attn(LN(X))))│
    └──────────────┬──────────────────────┘
                   ↓
              Output X'
```

**Transfer Function:**

```math

G_{block}(\mathbf{X}) = \mathbf{X} + G_{attn}(\text{LN}(\mathbf{X})) + G_{ffn}(\text{LN}(\mathbf{X} + G_{attn}(\text{LN}(\mathbf{X}))))


where:
-  G_{attn}  = Attention transfer function
-  G_{ffn}  = Feed-forward transfer function
-  \text{LN}  = Layer normalization
```

### 13.3 Complete Model with Multiple Layers

**Block Diagram (c): Cascaded Transformer Blocks**

```
Input Tokens T
    ↓
    ┌─────────────┐
    │ Embedding   │
    │   G_emb     │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Positional  │
    │ G_pos       │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Block 1     │
    │ G_block₁    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Block 2     │
    │ G_block₂    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │    ...      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Block L     │
    │ G_block_L   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Final Norm  │
    │ G_norm      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Output Proj │
    │ G_out       │
    └──────┬──────┘
           ↓
    Output Logits
```

**Overall Transfer Function:**

```math

\mathbf{Y} = G_{out} \circ G_{norm} \circ G_{block_L} \circ ... \circ G_{block_2} \circ G_{block_1} \circ G_{pos} \circ G_{emb}(\mathbf{T})

```

### 13.4 Closed-Loop Training System

**Block Diagram (d): Training Control Loop**

```
Input Data X
    ↓
    ┌─────────────┐
    │   Model     │
    │  Forward    │
    │     F       │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Output    │
    │     ŷ       │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │    Loss     │
    │  L(ŷ, y)    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Gradient   │
    │    ∇θ       │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Clipping    │
    │   Clip      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Optimizer   │
    │  AdamW      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Parameter  │
    │   Update    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      -      │ ←─── (Feedback to Model)
    └─────────────┘
```

**Closed-Loop Transfer Function:**

```math

\theta_{t+1} = \theta_t - \eta_t \cdot \text{AdamW}(\text{Clip}(\nabla_\theta L(\mathcal{F}(\mathbf{X}, \theta_t), \mathbf{y})))

```

---

## 14. Vector Visualization and Examples

### 14.1 Example Phrase: "Hello World"

We'll trace through the complete system with the phrase **"Hello World"**.

#### Step 1: Tokenization

**Input:** `"Hello World"`

**Process:**

```
Characters: ['H', 'e', 'l', 'l', 'o', ' ', 'W', 'o', 'r', 'l', 'd']
Token IDs:   [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]
```

**Mathematical:**

```math

\mathbf{c} = \text{"Hello World"}


\mathbf{t} = \mathcal{T}(\mathbf{c}) = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]

```

**Vector Representation:**

- Dimension: $n = 11$ tokens
- Token IDs: $\mathbf{t} \in \mathbb{N}^{11}$

#### Step 2: Embedding

**Embedding Matrix:** $\mathbf{E} \in \mathbb{R}^{128 \times 512}$

**Lookup Operation:**

```math

\mathbf{X} = \mathbf{E}[\mathbf{t}] = \begin{bmatrix}
\mathbf{E}[72] \\
\mathbf{E}[101] \\
\mathbf{E}[108] \\
\mathbf{E}[108] \\
\mathbf{E}[111] \\
\mathbf{E}[32] \\
\mathbf{E}[87] \\
\mathbf{E}[111] \\
\mathbf{E}[114] \\
\mathbf{E}[108] \\
\mathbf{E}[100]
\end{bmatrix} \in \mathbb{R}^{11 \times 512}

```

**Example Values (first 3 dimensions):**

```math

\mathbf{E}[72] = [0.1, -0.2, 0.3, ...]^T \\
\mathbf{E}[101] = [-0.1, 0.3, -0.1, ...]^T \\
\mathbf{E}[108] = [0.05, 0.15, -0.05, ...]^T

```

**Vector Visualization:**

```
Token 'H' (ID=72):   [0.10, -0.20,  0.30, ..., 0.05]  (512-dim vector)
Token 'e' (ID=101):  [-0.10,  0.30, -0.10, ..., 0.02]  (512-dim vector)
Token 'l' (ID=108):  [0.05,  0.15, -0.05, ..., 0.01]  (512-dim vector)
...
```

#### Step 3: Positional Encoding

**Positional Encoding Matrix:** $\mathbf{PE} \in \mathbb{R}^{11 \times 512}$

**Computation:**

```math

PE_{(0, 0)} = \sin(0 / 10000^0) = 0 \\
PE_{(0, 1)} = \cos(0 / 10000^0) = 1 \\
PE_{(1, 0)} = \sin(1 / 10000^0) = \sin(1) \approx 0.8415 \\
PE_{(1, 1)} = \cos(1 / 10000^0) = \cos(1) \approx 0.5403

```

**Addition:**

```math

\mathbf{X}_{pos} = \mathbf{X} + \mathbf{PE}

```

**Example (first token, first 3 dimensions):**

```math

\mathbf{X}_{pos}[0, :3] = \begin{bmatrix}
0.1 \\ -0.2 \\ 0.3
\end{bmatrix} + \begin{bmatrix}
0 \\ 1 \\ 0
\end{bmatrix} = \begin{bmatrix}
0.1 \\ 0.8 \\ 0.3
\end{bmatrix}

```

#### Step 4: Multi-Head Attention

**Query, Key, Value Projections:**

Let $\mathbf{W}\_Q, \mathbf{W}\_K, \mathbf{W}\_V \in \mathbb{R}^{512 \times 512}$

```math

\mathbf{Q} = \mathbf{X}_{pos} \mathbf{W}_Q \in \mathbb{R}^{11 \times 512}

```

**Example Calculation (head 0, token 0):**

For $h = 0 , d_k = 512/8 = 64 $:

```math

\mathbf{Q}[0, :64] = \mathbf{X}_{pos}[0] \mathbf{W}_Q[:, :64]

```

**Attention Score Computation:**

```math

S_{0,1} = \frac{\mathbf{Q}[0] \cdot \mathbf{K}[1]}{\sqrt{64}} = \frac{\sum_{i=0}^{63} Q_{0,i} \cdot K_{1,i}}{8}

```

**Example Numerical Calculation:**

Assume:

```math

\mathbf{Q}[0, :3] = [0.2, -0.1, 0.3] \\
\mathbf{K}[1, :3] = [0.1, 0.2, -0.1]



S_{0,1} = \frac{0.2 \times 0.1 + (-0.1) \times 0.2 + 0.3 \times (-0.1)}{8} \\
= \frac{0.02 - 0.02 - 0.03}{8} = \frac{-0.03}{8} = -0.00375

```

**Attention Weights:**

```math

A_{0,:} = \text{softmax}(S_{0,:}) = \frac{\exp(S_{0,:})}{\sum_{j=0}^{10} \exp(S_{0,j})}

```

**Example:**

If $S\_{0,:} = [-0.004, 0.05, 0.02, 0.02, 0.08, -0.01, 0.03, 0.08, 0.01, 0.02, 0.04]$

```math

\exp(S_{0,:}) = [0.996, 1.051, 1.020, 1.020, 1.083, 0.990, 1.030, 1.083, 1.010, 1.020, 1.041]



\sum = 11.335



A_{0,:} = [0.088, 0.093, 0.090, 0.090, 0.096, 0.087, 0.091, 0.096, 0.089, 0.090, 0.092]

```

**Output Calculation:**

```math

\mathbf{O}[0] = \sum_{j=0}^{10} A_{0,j} \mathbf{V}[j]

```

**Example (first dimension):**

```math

O_{0,0} = A_{0,0} V_{0,0} + A_{0,1} V_{1,0} + ... + A_{0,10} V_{10,0} \\
= 0.088 \times 0.2 + 0.093 \times 0.1 + ... + 0.092 \times 0.15 \\
\approx 0.12

```

#### Step 5: Feed-Forward Network

**Input:** $\mathbf{X}\_{attn} \in \mathbb{R}^{11 \times 512}$

**First Linear Transformation:**

```math

\mathbf{H} = \mathbf{X}_{attn} \mathbf{W}_1 \in \mathbb{R}^{11 \times 2048}

```

**Example (token 0, first dimension):**

```math

H_{0,0} = \sum_{i=0}^{511} X_{attn,0,i} \cdot W_{1,i,0}


Assuming  X_{attn}[0, :3] = [0.12, -0.05, 0.08]  and  W_1[:3, :3] = \begin{bmatrix} 0.1 & 0.2 \\ -0.1 & 0.1 \\ 0.05 & -0.05 \end{bmatrix}


H_{0,0} = 0.12 \times 0.1 + (-0.05) \times (-0.1) + 0.08 \times 0.05 \\
= 0.012 + 0.005 + 0.004 = 0.021

```

**GELU Activation:**

```math

\text{GELU}(0.021) = 0.021 \cdot \frac{1}{2}\left(1 + \text{erf}\left(\frac{0.021}{\sqrt{2}}\right)\right)



\text{erf}(0.021/\sqrt{2}) = \text{erf}(0.0148) \approx 0.0167



\text{GELU}(0.021) = 0.021 \times 0.5 \times (1 + 0.0167) = 0.021 \times 0.5084 \approx 0.0107

```

**Second Linear Transformation:**

```math

\mathbf{O}_{ffn} = \mathbf{H}' \mathbf{W}_2 \in \mathbb{R}^{11 \times 512}

```

#### Step 6: Complete Forward Pass Through One Layer

**Input:** $\mathbf{X}_{in} = \mathbf{X}_{pos} \in \mathbb{R}^{11 \times 512}$

**Step 6.1: Layer Normalization**

```math

\mu_0 = \frac{1}{512} \sum_{i=0}^{511} X_{in,0,i}

```

**Example:**

```math

\mu_0 = \frac{0.1 + 0.8 + 0.3 + ...}{512} \approx 0.02



\sigma_0^2 = \frac{1}{512} \sum_{i=0}^{511} (X_{in,0,i} - \mu_0)^2



\sigma_0^2 \approx \frac{(0.1-0.02)^2 + (0.8-0.02)^2 + ...}{512} \approx 0.15



\hat{X}_{0,0} = \frac{0.1 - 0.02}{\sqrt{0.15 + 1e-5}} = \frac{0.08}{0.387} \approx 0.207

```

**Step 6.2: Attention Output**

```math

\mathbf{X}_{attn} = \text{Attention}(\hat{\mathbf{X}})

```

**Step 6.3: Residual Connection**

```math

\mathbf{X}_{res1} = \mathbf{X}_{in} + \mathbf{X}_{attn}

```

**Example:**

```math

X_{res1,0,0} = 0.1 + 0.12 = 0.22

```

**Step 6.4: Second Layer Norm + FFN**

```math

\mathbf{X}_{ffn} = \text{FFN}(\text{LayerNorm}(\mathbf{X}_{res1}))

```

**Step 6.5: Final Residual**

```math

\mathbf{X}_{out} = \mathbf{X}_{res1} + \mathbf{X}_{ffn}

```

**Example:**

```math

X_{out,0,0} = 0.22 + 0.15 = 0.37

```

#### Step 7: Output Projection

**After L layers:**

```math

\mathbf{H}_{final} = \text{LayerNorm}(\mathbf{X}_{out}^{(L)}) \in \mathbb{R}^{11 \times 512}

```

**Output Projection:**

```math

\mathbf{L} = \mathbf{H}_{final} \mathbf{W}_{out} \in \mathbb{R}^{11 \times 128}

```

**Example (position 0):**

```math

L_{0,:} = \mathbf{H}_{final}[0] \mathbf{W}_{out} \in \mathbb{R}^{128}

```

**Softmax:**

```math

p_{0,v} = \frac{\exp(L_{0,v})}{\sum_{w=0}^{127} \exp(L_{0,w})}

```

**Example:**

If $L*{0,72} = 5.2 (logit for 'H'), L*{0,101} = 3.1 (logit for 'e'), etc.$

```math

\exp(5.2) = 181.27 \\
\exp(3.1) = 22.20 \\
\vdots



\sum_{w=0}^{127} \exp(L_{0,w}) \approx 250.0



p_{0,72} = \frac{181.27}{250.0} \approx 0.725 \quad \text{(72\% probability for H)}

```

---

## 15. Complete Numerical Example: "Hello"

Let's trace through the complete system with **"Hello"** step-by-step.

### Input: "Hello"

### Stage 1: Tokenization

```math

\mathbf{c} = \text{"Hello"} = ['H', 'e', 'l', 'l', 'o']



\mathbf{t} = [72, 101, 108, 108, 111]

```

### Stage 2: Embedding (d=512)

```math

\mathbf{E} \in \mathbb{R}^{128 \times 512}



\mathbf{X} = \begin{bmatrix}
\mathbf{E}[72] \\
\mathbf{E}[101] \\
\mathbf{E}[108] \\
\mathbf{E}[108] \\
\mathbf{E}[111]
\end{bmatrix} = \begin{bmatrix}
0.10 & -0.20 & 0.30 & ... & 0.05 \\
-0.10 & 0.30 & -0.10 & ... & 0.02 \\
0.05 & 0.15 & -0.05 & ... & 0.01 \\
0.05 & 0.15 & -0.05 & ... & 0.01 \\
-0.05 & 0.20 & 0.10 & ... & 0.03
\end{bmatrix} \in \mathbb{R}^{5 \times 512}

```

### Stage 3: Positional Encoding

```math

\mathbf{PE} = \begin{bmatrix}
0 & 1 & 0 & ... & 0 \\
0.84 & 0.54 & 0.01 & ... & 0.00 \\
0.91 & -0.42 & 0.02 & ... & 0.00 \\
0.14 & -0.99 & 0.03 & ... & 0.00 \\
-0.76 & -0.65 & 0.04 & ... & 0.00
\end{bmatrix} \in \mathbb{R}^{5 \times 512}



\mathbf{X}_{pos} = \mathbf{X} + \mathbf{PE} = \begin{bmatrix}
0.10 & 0.80 & 0.30 & ... & 0.05 \\
0.74 & 0.84 & -0.09 & ... & 0.02 \\
0.96 & -0.27 & -0.03 & ... & 0.01 \\
0.19 & -0.84 & -0.02 & ... & 0.01 \\
-0.81 & -0.45 & 0.14 & ... & 0.03
\end{bmatrix}

```

### Stage 4: Attention (h=8 heads, d_k=64)

**Query Generation:**

```math

\mathbf{Q} = \mathbf{X}_{pos} \mathbf{W}_Q \in \mathbb{R}^{5 \times 512}

```

**Score Matrix (head 0):**

```math

\mathbf{S}_0 = \frac{\mathbf{Q}_0 \mathbf{K}_0^T}{\sqrt{64}} \in \mathbb{R}^{5 \times 5}

```

**Example Values:**

```math

\mathbf{S}_0 = \begin{bmatrix}
0.50 & -0.10 & 0.20 & 0.15 & 0.30 \\
-0.05 & 0.45 & 0.10 & 0.08 & 0.25 \\
0.15 & 0.05 & 0.40 & 0.30 & 0.20 \\
0.12 & 0.08 & 0.28 & 0.35 & 0.18 \\
0.25 & 0.15 & 0.22 & 0.20 & 0.42
\end{bmatrix}

```

**Attention Weights:**

```math

\mathbf{A}_0 = \text{softmax}(\mathbf{S}_0) = \begin{bmatrix}
0.35 & 0.15 & 0.22 & 0.20 & 0.28 \\
0.15 & 0.38 & 0.20 & 0.18 & 0.27 \\
0.23 & 0.18 & 0.32 & 0.30 & 0.26 \\
0.21 & 0.19 & 0.28 & 0.33 & 0.25 \\
0.27 & 0.22 & 0.26 & 0.25 & 0.36
\end{bmatrix}

```

**Output (head 0):**

```math

\mathbf{O}_0 = \mathbf{A}_0 \mathbf{V}_0 \in \mathbb{R}^{5 \times 64}

```

**Concatenate All Heads:**

```math

\mathbf{O} = \text{Concat}[\mathbf{O}_0, ..., \mathbf{O}_7] \in \mathbb{R}^{5 \times 512}

```

### Stage 5: Feed-Forward

```math

\mathbf{H} = \mathbf{O} \mathbf{W}_1 \in \mathbb{R}^{5 \times 2048}



\mathbf{H}' = \text{GELU}(\mathbf{H}) \in \mathbb{R}^{5 \times 2048}



\mathbf{O}_{ffn} = \mathbf{H}' \mathbf{W}_2 \in \mathbb{R}^{5 \times 512}

```

### Stage 6: Output Logits

After processing through all L layers:

```math

\mathbf{L} = \mathbf{H}_{final} \mathbf{W}_{out} \in \mathbb{R}^{5 \times 128}

```

**Example (position 4, predicting next token):**

```math

L_{4,:} = [2.1, 1.5, ..., 5.2, ..., 3.1, ...]


Where:
-  L_{4,111} = 5.2  (high score for 'o')
-  L_{4,32} = 4.8  (high score for space)
-  L_{4,87} = 4.5  (high score for 'W')
```

**Probability Distribution:**

```math

\mathbf{p}_4 = \text{softmax}(L_{4,:}) = [0.01, 0.008, ..., 0.25, ..., 0.18, ...]



p_{4,111} \approx 0.25 \quad \text{(25\% for o)} \\
p_{4,32} \approx 0.22 \quad \text{(22\% for space)} \\
p_{4,87} \approx 0.18 \quad \text{(18\% for W)}

```

---

## 16. Vector Space Visualization

### 16.1 Embedding Space

**2D Projection Example:**

After embedding "Hello", tokens occupy positions in 512-dimensional space. Projected to 2D:

```
Token Positions (idealized 2D projection):

        'l' (0.05, 0.15)
          ●

                    'e' (-0.10, 0.30)
                      ●

Origin (0, 0)
    ●

                      'H' (0.10, -0.20)
                        ●

                            'o' (-0.05, 0.20)
                              ●
```

**Distance in Embedding Space:**

```math

d(\mathbf{E}[72], \mathbf{E}[101]) = ||\mathbf{E}[72] - \mathbf{E}[101]||_2



d = \sqrt{(0.1 - (-0.1))^2 + (-0.2 - 0.3)^2 + ...} \approx \sqrt{0.04 + 0.25 + ...} \approx 2.1

```

### 16.2 Attention Weight Visualization

**Attention Matrix Visualization:**

```
Position   0    1    2    3    4
        ┌─────┴─────┴─────┴─────┴──┐
Token 0 │ 0.35 0.15 0.22 0.20 0.28 │  'H'
        │                          │
Token 1 │ 0.15 0.38 0.20 0.18 0.27 │  'e'
        │                          │
Token 2 │ 0.23 0.18 0.32 0.30 0.26 │  'l'
        │                          │
Token 3 │ 0.21 0.19 0.28 0.33 0.25 │  'l'
        │                          │
Token 4 │ 0.27 0.22 0.26 0.25 0.36 │  'o'
        └──────────────────────────┘
```

**Interpretation:**

- Token 0 ('H') attends most to itself (0.35) and token 4 (0.28)
- Token 4 ('o') attends moderately to all positions
- Higher values indicate stronger attention

### 16.3 Probability Distribution Visualization

**Output Distribution for Position 5 (next token after "Hello"):**

```
Probability Distribution p[5, :]

Probability
    │
0.3 │           ●
    │
0.2 │      ●         ●
    │
0.1 │  ●       ●           ●     ●
    │
0.0 ├─┴───┴───┴───┴───┴───┴───┴───┴─── Token IDs
    32  72  87  101  108 111  ... 127
    ␣   H   W   e    l   o
```

**Meaning:**

- Highest probability for space (32) ≈ 0.28
- Next: 'o' (111) ≈ 0.23
- Then: 'W' (87) ≈ 0.18
- Model predicts space or continuation

---

## 17. Advanced Block Diagram Simplification

### 17.1 Complex Multi-Layer System Simplification

Following control system reduction techniques, we can simplify the transformer model step-by-step:

**Diagram (a): Original Complex System**

```
Input R (Tokens)
    ↓
    ┌─────────────┐
    │   Embedding │
    │    G_emb    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Positional  │
    │   Encoding  │
    │    G_pos    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── Feedback from Layer 2
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Layer 1    │
    │ G_block₁   │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── Feedback from Output
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Layer 2    │
    │ G_block₂    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── Feedback H₁
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │ Output Proj │
    │    G_out    │
    └──────┬──────┘
           ↓
    Output C (Logits)
```

**Diagram (b): First Simplification (Combine Embedding and Positional)**

```
Input R
    ↓
    ┌─────────────────────┐
    │ G_emb_pos =         │
    │ G_pos ∘ G_emb       │
    └──────┬──────────────┘
           ↓
    ┌─────────────┐
    │      +      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Layer 1    │
    │ G_block₁    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │  Layer 2    │
    │ G_block₂    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── H₁
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │    G_out    │
    └──────┬──────┘
           ↓
    Output C
```

**Diagram (c): Second Simplification (Combine Layers)**

```
Input R
    ↓
    ┌─────────────────────┐
    │ G_emb_pos           │
    └──────┬──────────────┘
           ↓
    ┌──────────────────────────────────┐
    │ G_layers = G_block₂ ∘ G_block₁   │
    │ Equivalent to:                   │
    │ X + Δ₁(X) + Δ₂(X + Δ₁(X))        │
    └──────┬───────────────────────────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── H₁
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │    G_out    │
    └──────┬──────┘
           ↓
    Output C
```

**Diagram (d): Third Simplification (Combine with Output)**

```
Input R
    ↓
    ┌──────────────────────────────┐
    │ G_forward =                  │
    │ G_out ∘ G_layers ∘ G_emb_pos │
    └──────┬───────────────────────┘
           ↓
    ┌─────────────┐
    │      +      │ ←─── H₁ (Feedback)
    └──────┬──────┘
           ↓
    Output C
```

**Diagram (e): Final Simplified Transfer Function**

```
Input R
    ↓
    ┌────────────────────────────────────────────┐
    │ Overall Transfer Function:                 │
    │                                            │
    │ C/R = G_forward / (1 + G_forward × H₁)     │
    │                                            │
    │ Where:                                     │
    │ G_forward = G_out ∘ G_layers ∘ G_emb_pos   │
    │                                            │
    └──────┬─────────────────────────────────────┘
           ↓
    Output C
```

**Mathematical Derivation:**

**Step 1:** Combine embedding and positional encoding:

```math

G_{emb\_pos}(\mathbf{T}) = G_{pos}(G_{emb}(\mathbf{T})) = \mathbf{E}[\mathbf{T}] + \mathbf{PE}

```

**Step 2:** Combine transformer layers:

```math

G_{layers}(\mathbf{X}) = G_{block_2}(G_{block_1}(\mathbf{X}))



G_{layers}(\mathbf{X}) = \mathbf{X} + \Delta_1(\mathbf{X}) + \Delta_2(\mathbf{X} + \Delta_1(\mathbf{X}))


where  \Delta_l  represents the transformation inside block  l .
```

**Step 3:** Combine with output projection:

```math

G_{forward}(\mathbf{T}) = G_{out}(G_{layers}(G_{emb\_pos}(\mathbf{T})))

```

**Step 4:** Apply feedback reduction:

```math

\frac{C}{R} = \frac{G_{forward}}{1 + G_{forward} \times H_1}

```

### 17.2 Attention Block Simplification

**Diagram (a): Detailed Attention**

```
Input X
    ↓
    ┌─────────────┐
    │      Q      │ ←─── W_Q
    │      K      │ ←─── W_K
    │      V      │ ←─── W_V
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Scores    │
    │ S = QK^T/√d │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Softmax   │
    │   A = σ(S)  │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │    Output   │
    │   O = AV    │
    └──────┬──────┘
           ↓
    ┌─────────────┐
    │   Out Proj  │
    │    W_O      │
    └──────┬──────┘
           ↓
    Output X'
```

**Diagram (b): Simplified Attention Transfer Function**

```
Input X
    ↓
    ┌──────────────────────────────┐
    │ G_attn(X) =                  │
    │ W_O · softmax(QK^T/√d) · V   │
    │                              │
    │ Where:                       │
    │ Q = XW_Q, K = XW_K, V = XW_V │
    └──────┬───────────────────────┘
           ↓
    Output X'
```

**Mathematical Transfer Function:**

```math

G_{attn}(\mathbf{X}) = \mathbf{X} \mathbf{W}_O \cdot \text{softmax}\left(\frac{(\mathbf{X} \mathbf{W}_Q)(\mathbf{X} \mathbf{W}_K)^T}{\sqrt{d_k}}\right) \cdot (\mathbf{X} \mathbf{W}_V)

```

---

## 18. Vector Trace: "Hello World" Complete Flow

### 18.1 Complete Vector Trace with Numerical Values

**Input:** `"Hello World"`

**Stage 1: Tokenization**

```math

\mathbf{t} = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]

```

**Stage 2: Embedding (showing first 4 dimensions)**

```math

\mathbf{X} = \begin{bmatrix}
[H] & 0.10 & -0.20 & 0.30 & 0.15 & ... \\
[e] & -0.10 & 0.30 & -0.10 & 0.08 & ... \\
[l] & 0.05 & 0.15 & -0.05 & 0.03 & ... \\
[l] & 0.05 & 0.15 & -0.05 & 0.03 & ... \\
[o] & -0.05 & 0.20 & 0.10 & 0.06 & ... \\
[ ] & 0.02 & 0.05 & 0.02 & 0.01 & ... \\
[W] & 0.15 & -0.15 & 0.25 & 0.12 & ... \\
[o] & -0.05 & 0.20 & 0.10 & 0.06 & ... \\
[r] & 0.08 & 0.10 & -0.08 & 0.04 & ... \\
[l] & 0.05 & 0.15 & -0.05 & 0.03 & ... \\
[d] & 0.12 & -0.08 & 0.18 & 0.09 & ...
\end{bmatrix} \in \mathbb{R}^{11 \times 512}

```

**Stage 3: Positional Encoding (first 4 dimensions)**

```math

\mathbf{PE} = \begin{bmatrix}
[0] & 0.00 & 1.00 & 0.00 & 0.00 & ... \\
[1] & 0.84 & 0.54 & 0.01 & 0.00 & ... \\
[2] & 0.91 & -0.42 & 0.02 & 0.00 & ... \\
[3] & 0.14 & -0.99 & 0.03 & 0.00 & ... \\
[4] & -0.76 & -0.65 & 0.04 & 0.00 & ... \\
[5] & -0.96 & 0.28 & 0.05 & 0.00 & ... \\
[6] & -0.28 & 0.96 & 0.06 & 0.00 & ... \\
[7] & 0.65 & 0.76 & 0.07 & 0.00 & ... \\
[8] & 0.99 & -0.14 & 0.08 & 0.00 & ... \\
[9] & 0.42 & -0.91 & 0.09 & 0.00 & ... \\
[10] & -0.54 & -0.84 & 0.10 & 0.00 & ...
\end{bmatrix}

```

**Stage 4: Combined Input**

```math

\mathbf{X}_{pos} = \mathbf{X} + \mathbf{PE}

```

**Example Row 0 (token 'H'):**

```math

\mathbf{X}_{pos}[0, :4] = [0.10, -0.20, 0.30, 0.15] + [0.00, 1.00, 0.00, 0.00] = [0.10, 0.80, 0.30, 0.15]

```

**Stage 5: Attention (Head 0, showing attention from token 0 to all tokens)**

```math

\mathbf{S}_0[0, :] = [0.50, -0.10, 0.20, 0.15, 0.30, -0.05, 0.18, 0.28, 0.12, 0.20, 0.22]



\mathbf{A}_0[0, :] = \text{softmax}(\mathbf{S}_0[0, :]) = [0.35, 0.15, 0.22, 0.20, 0.28, 0.14, 0.19, 0.26, 0.17, 0.21, 0.23]


**Meaning:** Token 'H' (position 0) attends:
- 35% to itself
- 28% to token 'o' (position 4)
- 26% to token 'o' (position 7)
- 23% to token 'd' (position 10)
```

**Stage 6: Attention Output**

```math

\mathbf{O}_0[0, :] = \sum_{j=0}^{10} A_{0,j} \mathbf{V}_0[j, :]

```

**Example (first dimension):**

```math

O_{0,0,0} = 0.35 \times 0.12 + 0.15 \times 0.08 + ... + 0.23 \times 0.15 \approx 0.115

```

**Stage 7: FFN Output**

```math

\mathbf{H}_{ffn}[0, :4] = [0.15, -0.08, 0.22, 0.18]

```

**Stage 8: Final Output (after all layers)**

```math

\mathbf{H}_{final}[0, :4] = [0.42, 0.25, 0.58, 0.31]

```

**Stage 9: Logits**

```math

\mathbf{L}[0, :] = [2.1, 1.8, ..., 5.2, ..., 3.4, ...]


Where  L[0, 72] = 5.2  is highest (predicting 'H' at position 1).
```

**Stage 10: Probabilities**

```math

\mathbf{p}[0, :] = \text{softmax}(\mathbf{L}[0, :]) = [0.01, 0.008, ..., 0.28, ..., 0.15, ...]



p[0, 72] \approx 0.28 \quad \text{(28\% probability for H)}

```

---

## 19. Vector Plots and Visualizations

### 19.1 Embedding Vector Trajectory

**Trajectory Plot:**

```
512-Dimensional Embedding Space (2D Projection)

     0.3 │                          'e' (pos 1)
         │                            ●
     0.2 │                    'r' (pos 8)
         │                      ●
     0.1 │         'l' (pos 2,3,9)      'o' (pos 4,7)
         │            ●                 ●
     0.0 ├───────────────────────────────────────────
         │    'H' (pos 0)
    -0.1 │       ●
         │
    -0.2 │
         │
    -0.3 │                              'W' (pos 6)
         │                                  ●
         └───────────────────────────────────────────
           -0.3  -0.2  -0.1  0.0  0.1  0.2  0.3
```

### 19.2 Attention Heatmap

**Attention Weight Matrix Visualization:**

```
Attention Weights A[i,j] for "Hello World"

         j →   0    1    2    3    4    5    6    7    8    9   10
         ↓  ['H'] ['e'] ['l'] ['l'] ['o'] [' '] ['W'] ['o'] ['r'] ['l'] ['d']
i=0 ['H'] │ 0.35 0.15 0.22 0.20 0.28 0.14 0.19 0.26 0.17 0.21 0.23 │
i=1 ['e'] │ 0.15 0.38 0.20 0.18 0.27 0.16 0.18 0.25 0.19 0.22 0.20 │
i=2 ['l'] │ 0.23 0.18 0.32 0.30 0.26 0.17 0.21 0.24 0.25 0.31 0.23 │
i=3 ['l'] │ 0.21 0.19 0.28 0.33 0.25 0.18 0.20 0.23 0.24 0.30 0.22 │
i=4 ['o'] │ 0.27 0.22 0.26 0.25 0.36 0.19 0.23 0.29 0.24 0.27 0.25 │
i=5 [' '] │ 0.18 0.20 0.19 0.21 0.24 0.40 0.22 0.25 0.21 0.20 0.22 │
i=6 ['W'] │ 0.22 0.21 0.23 0.24 0.26 0.20 0.45 0.28 0.27 0.23 0.25 │
i=7 ['o'] │ 0.26 0.25 0.24 0.23 0.29 0.21 0.28 0.38 0.26 0.24 0.26 │
i=8 ['r'] │ 0.19 0.21 0.25 0.24 0.24 0.19 0.27 0.26 0.42 0.27 0.28 │
i=9 ['l'] │ 0.21 0.22 0.31 0.30 0.27 0.20 0.23 0.24 0.27 0.35 0.24 │
i=10['d'] │ 0.23 0.20 0.23 0.22 0.25 0.22 0.25 0.26 0.28 0.24 0.48 │

Color Coding:
█ = 0.48-0.50 (very high attention)
█ = 0.35-0.48 (high attention)
█ = 0.25-0.35 (medium attention)
█ = 0.15-0.25 (low attention)
█ = 0.00-0.15 (very low attention)
```

### 19.3 Probability Distribution Plot

**Logits and Probabilities:**

```
Logits L[5, :] (predicting token after "Hello ")

Logit
Value │
  6.0 │                    ● (token 87 'W')
      │
  5.0 │           ● (token 111 'o')
      │
  4.0 │      ● (token 32 ' ')         ● (token 114 'r')
      │
  3.0 │  ●                         ●  ●
      │
  2.0 │  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
      │
  1.0 │  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●  ●
      │
  0.0 ├─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴── Token IDs
       32  72  87  101 108 111 114 ...
       ␣   H   W   e   l   o   r

Probabilities p[5, :]

Probability
    │
 0.3│                    ● ('W')
    │
 0.2│      ● (' ')              ● ('o')
    │
 0.1│  ●       ●  ●  ●  ●     ●  ●
    │
 0.0├─┴──┴──┴──┴──┴──┴──┴──┴──┴──┴──┴── Token IDs
     32  72  87  101 108 111 114 ...
```

### 19.4 Hidden State Evolution Through Layers

**Layer-by-Layer Transformation:**

```
Hidden State Evolution for Token 'H' (position 0)

Dimension 0:
Layer 0: 0.10  (embedding + positional)
Layer 1: 0.42  (after attention + FFN)
Layer 2: 0.58  (after second layer)
Layer 3: 0.65  (after third layer)
...      ...
Layer L: 0.72  (final hidden state)

Dimension 1:
Layer 0: 0.80  (embedding + positional)
Layer 1: 0.25  (after attention + FFN)
Layer 2: 0.18  (after second layer)
Layer 3: 0.22  (after third layer)
...      ...
Layer L: 0.15  (final hidden state)
```

**Visualization:**

```
Hidden State Magnitude ||h[l]|| Over Layers

Magnitude
    │
 1.0│ ●
    │   ●
 0.8│     ●
    │       ●
 0.6│         ●
    │           ●
 0.4│             ●
    │               ●
 0.2│                 ●
    │                   ●
 0.0├───────────────────────── Layer
    0  1  2  3  4  5  6
```

---

## 20. Summary: Complete Mathematical Trace

### Complete System Equation with Numerical Example

**Text:** `"Hello World"`

**Complete Mathematical Flow:**

1. **Tokenization:**

```math

   \mathbf{t} = \mathcal{T}(\text{"Hello World"}) = [72, 101, 108, 108, 111, 32, 87, 111, 114, 108, 100]

```

2. **Embedding:**

```math

   \mathbf{X} = \mathbf{E}[\mathbf{t}] \in \mathbb{R}^{11 \times 512}

```

3. **Positional Encoding:**

```math

   \mathbf{X}_{pos} = \mathbf{X} + \mathbf{PE} \in \mathbb{R}^{11 \times 512}

```

4. **Transformer Layers (L=6):**

```math

   \mathbf{h}_l = \text{TransformerBlock}_l(\mathbf{h}_{l-1}), \quad l = 1, ..., 6

```

5. **Output:**

```math

   \mathbf{L} = \mathbf{h}_6 \mathbf{W}_{out} \in \mathbb{R}^{11 \times 128}

```

6. **Probabilities:**

```math

   \mathbf{p} = \text{softmax}(\mathbf{L}) \in \mathbb{R}^{11 \times 128}

```

**Final Prediction:**

For position 5 (after "Hello "):

```math

p[5, 87] = 0.28 \quad \text{(28\% for W)} \\
p[5, 32] = 0.22 \quad \text{(22\% for space)} \\
p[5, 111] = 0.18 \quad \text{(18\% for o)}

```

**Most Likely:** `'W'` → Complete prediction: `"Hello World"`

---

_This document provides a complete mathematical control system formulation with block diagrams, vector visualizations, numerical examples, and step-by-step calculations for every component of the SheepOp LLM._
