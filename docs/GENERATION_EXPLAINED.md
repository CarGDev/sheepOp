# What is Generation? Step-by-Step Explanation

Complete step-by-step explanation of text generation: how models generate text using autoregressive generation, sampling, and decoding strategies.

## Table of Contents

1. [What is Generation?](#91-what-is-generation)
2. [Autoregressive Generation](#92-autoregressive-generation)
3. [Sampling Strategies](#93-sampling-strategies)
4. [Temperature](#94-temperature)
5. [Top-k Sampling](#95-top-k-sampling)
6. [Top-p (Nucleus) Sampling](#96-top-p-nucleus-sampling)
7. [Step-by-Step Generation Process](#97-step-by-step-generation-process)
8. [Exercise: Complete Generation Example](#98-exercise-complete-generation-example)
9. [Key Takeaways](#99-key-takeaways)

---

## 9.1 What is Generation?

### Simple Definition

**Generation** (text generation) is the process of using a trained model to produce new text, one token at a time, based on a given prompt.

### Visual Analogy

**Think of generation like writing a story:**

```
Prompt: "Once upon a time"

Model generates:
  "Once upon a time" → "there"
  "Once upon a time there" → "was"
  "Once upon a time there was" → "a"
  "Once upon a time there was a" → "princess"
  ...

Final: "Once upon a time there was a princess..."
```

**Model predicts next word, one at a time!**

### What Generation Does

**Generation:**
1. **Takes** a prompt (starting text)
2. **Predicts** next token probabilities
3. **Samples** a token from distribution
4. **Appends** token to sequence
5. **Repeats** until complete

**Result:** Generated text continuation!

---

## 9.2 Autoregressive Generation

### What is Autoregressive?

**Autoregressive** means the model uses its own previous outputs as inputs for the next prediction.

### How It Works

**Step 1: Initial Prompt**
```
Prompt: "Hello"
Sequence: ["Hello"]
```

**Step 2: First Prediction**
```
Input: ["Hello"]
Model output: Probabilities for next token
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  ...
```

**Step 3: Sample Token**
```
Sample: "World" (selected)
Sequence: ["Hello", "World"]
```

**Step 4: Second Prediction**
```
Input: ["Hello", "World"]
Model output: Probabilities for next token
  "!": 0.5
  ".": 0.3
  ",": 0.1
  ...
```

**Step 5: Continue**
```
Sample: "!"
Sequence: ["Hello", "World", "!"]
Continue until max length or stop token...
```

### Mathematical Formulation

**For prompt $\mathbf{P} = [p_1, ..., p_k]$:**

**Initialization:**
```math
\mathbf{T}_0 = \mathbf{P}
```

**For each step $t \geq k+1$:**

1. **Forward pass:**
   ```math
   \mathbf{L}_t = \text{Model}(\mathbf{T}_{t-1})
   ```

2. **Get next token probabilities:**
   ```math
   \mathbf{p}_t = \text{softmax}(\mathbf{L}_t[:, -1, :])
   ```

3. **Sample token:**
   ```math
   t_t \sim \text{Categorical}(\mathbf{p}_t)
   ```

4. **Append token:**
   ```math
   \mathbf{T}_t = [\mathbf{T}_{t-1}, t_t]
   ```

**Repeat until stop condition!**

---

## 9.3 Sampling Strategies

### Deterministic vs Stochastic

**Deterministic (Greedy):**
```
Always pick highest probability:
  "World": 0.4 ← Highest
  "there": 0.3
  "friend": 0.2
  
→ Always picks "World"
→ Same output every time
```

**Stochastic (Sampling):**
```
Sample from distribution:
  "World": 0.4 (40% chance)
  "there": 0.3 (30% chance)
  "friend": 0.2 (20% chance)
  
→ Different output each time
→ More diverse generations
```

### Why Sampling?

**Greedy (Deterministic):**
- Same output every time
- Can be repetitive
- Less creative

**Sampling:**
- Different outputs each time
- More diverse
- More creative
- Better for creative tasks

---

## 9.4 Temperature

### What is Temperature?

**Temperature** controls the randomness of sampling by scaling the logits before applying softmax.

### Formula

```math
\mathbf{p}_t = \text{softmax}\left(\frac{\mathbf{l}_t}{T}\right)
```

**Where:**
- $\mathbf{l}_t$ = logits (raw scores)
- $T$ = temperature
- $\mathbf{p}_t$ = probabilities

### How Temperature Works

**T = 0.5 (Low Temperature - More Deterministic):**
```
Logits: [2.0, 1.0, 0.5]
After scaling: [4.0, 2.0, 1.0]
After softmax: [0.88, 0.11, 0.01]
→ Sharp distribution (one token dominates)
→ More deterministic
```

**T = 1.0 (Standard Temperature):**
```
Logits: [2.0, 1.0, 0.5]
After scaling: [2.0, 1.0, 0.5]
After softmax: [0.66, 0.24, 0.10]
→ Moderate distribution
→ Balanced
```

**T = 2.0 (High Temperature - More Random):**
```
Logits: [2.0, 1.0, 0.5]
After scaling: [1.0, 0.5, 0.25]
After softmax: [0.52, 0.31, 0.17]
→ Flat distribution (more uniform)
→ More random
```

### Visual Comparison

```
Probability
    │
 1.0│  T=0.5: ●
    │
 0.8│
    │
 0.6│  T=1.0:       ●
    │
 0.4│
    │
 0.2│  T=2.0:                ●
    │
 0.0├───────────────────────── Token
    "World"  "there"  "friend"
```

**Lower T = Sharper distribution = More deterministic**  
**Higher T = Flatter distribution = More random**

### When to Use Different Temperatures

**Low Temperature (T < 1.0):**
- Factual tasks
- Reproducible outputs
- When you want consistent results

**Standard Temperature (T = 1.0):**
- Default setting
- Balanced behavior
- Good for most tasks

**High Temperature (T > 1.0):**
- Creative writing
- Diverse outputs
- When you want variety

---

## 9.5 Top-k Sampling

### What is Top-k?

**Top-k sampling** limits the sampling to only the top k most likely tokens.

### How It Works

**Step 1: Get Probabilities**
```
All tokens:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  "hello": 0.05
  "cat": 0.03
  "dog": 0.02
  ...
```

**Step 2: Select Top-k (e.g., k=3)**
```
Top 3:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
```

**Step 3: Remove Others**
```
Set others to 0:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  "hello": 0.0
  "cat": 0.0
  "dog": 0.0
  ...
```

**Step 4: Renormalize**
```
Sum = 0.4 + 0.3 + 0.2 = 0.9
Renormalize:
  "World": 0.4/0.9 = 0.44
  "there": 0.3/0.9 = 0.33
  "friend": 0.2/0.9 = 0.22
```

**Step 5: Sample from Top-k**
```
Sample from these 3 tokens only
```

### Mathematical Formulation

**Given probabilities $\mathbf{p}_t$ and top-k:**

```math
\mathbf{p}_t^{topk}[v] = \begin{cases}
\frac{\mathbf{p}_t[v]}{\sum_{u \in \text{top-k}} \mathbf{p}_t[u]} & \text{if } v \in \text{top-k} \\
0 & \text{otherwise}
\end{cases}
```

### Why Top-k?

**Benefits:**
- Removes low-probability tokens
- Focuses on likely candidates
- Reduces randomness from unlikely tokens
- Better quality generations

**Example:**
```
Without top-k: Might sample "xyz" (very unlikely)
With top-k=50: Only samples from top 50 tokens
→ Better quality!
```

---

## 9.6 Top-p (Nucleus) Sampling

### What is Top-p?

**Top-p (nucleus) sampling** keeps the smallest set of tokens whose cumulative probability is at least p.

### How It Works

**Step 1: Sort Probabilities**
```
Sorted (descending):
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  "hello": 0.05
  "cat": 0.03
  "dog": 0.02
  ...
```

**Step 2: Compute Cumulative Probabilities**
```
Cumulative:
  "World": 0.4
  "there": 0.7  (0.4 + 0.3)
  "friend": 0.9  (0.7 + 0.2)
  "hello": 0.95  (0.9 + 0.05)
  "cat": 0.98   (0.95 + 0.03)
  ...
```

**Step 3: Find Nucleus (e.g., p=0.9)**
```
Find smallest set where sum ≥ 0.9:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  Cumulative: 0.9 ✓
  
→ Keep these 3 tokens
```

**Step 4: Remove Others**
```
Keep:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  Others: 0.0
```

**Step 5: Renormalize and Sample**
```
Renormalize and sample
```

### Mathematical Formulation

**Given probabilities $\mathbf{p}_t$ and top-p:**

**Find smallest set S:**
```math
S = \arg\min \{ |S'| : \sum_{v \in S'} \mathbf{p}_t[v] \geq p \}
```

**Then:**
```math
\mathbf{p}_t^{topp}[v] = \begin{cases}
\frac{\mathbf{p}_t[v]}{\sum_{u \in S} \mathbf{p}_t[u]} & \text{if } v \in S \\
0 & \text{otherwise}
\end{cases}
```

### Why Top-p?

**Benefits:**
- Adapts to distribution shape
- Keeps relevant tokens dynamically
- Better than fixed k in some cases
- More flexible than top-k

**Example:**
```
Sharp distribution: Top-p=0.9 might keep 3 tokens
Flat distribution: Top-p=0.9 might keep 50 tokens
→ Adapts automatically!
```

---

## 9.7 Step-by-Step Generation Process

### Complete Process

**Given prompt: "Hello"**

#### Step 1: Encode Prompt

```
Prompt: "Hello"
Token IDs: [72]
```

#### Step 2: Forward Pass

```
Input: [72]
Model processes through layers
Output: Logits for all tokens
  Token 72: 5.2
  Token 87: 4.8 ← "World"
  Token 101: 3.2 ← "there"
  Token 108: 2.1 ← "friend"
  ...
```

#### Step 3: Apply Temperature

```
Temperature: T = 1.0
Scaled logits: Same as above
```

#### Step 4: Apply Top-k (Optional)

```
Top-k: k = 50
Keep top 50 tokens, remove others
```

#### Step 5: Apply Top-p (Optional)

```
Top-p: p = 0.95
Keep tokens with cumulative prob ≥ 0.95
```

#### Step 6: Compute Probabilities

```
Apply softmax:
  "World": 0.4
  "there": 0.3
  "friend": 0.2
  ...
```

#### Step 7: Sample Token

```
Sample from distribution:
Selected: "World" (token 87)
```

#### Step 8: Append Token

```
Sequence: [72, 87]
Text: "Hello World"
```

#### Step 9: Repeat

```
Input: [72, 87]
→ Predict next token
→ Sample
→ Append
→ Repeat...
```

---

## 9.8 Exercise: Complete Generation Example

### Problem

**Given:**
- Prompt: "The"
- Model logits for next token: `[10.0, 8.0, 5.0, 2.0, 1.0, 0.5, ...]` (for tokens: "cat", "dog", "car", "house", "tree", "book", ...)
- Temperature: T = 1.0
- Top-k: k = 3
- Top-p: p = 0.9

**Generate the next token step-by-step.**

### Step-by-Step Solution

#### Step 1: Initial Setup

**Prompt:**
```
"The"
Token IDs: [32] (assuming "The" = token 32)
```

**Logits:**
```
Token "cat":   10.0
Token "dog":   8.0
Token "car":   5.0
Token "house": 2.0
Token "tree":  1.0
Token "book":  0.5
...
```

#### Step 2: Apply Temperature

**Temperature: T = 1.0**

**Scaled logits (divide by T):**
```
Token "cat":   10.0 / 1.0 = 10.0
Token "dog":   8.0 / 1.0 = 8.0
Token "car":   5.0 / 1.0 = 5.0
Token "house": 2.0 / 1.0 = 2.0
Token "tree":  1.0 / 1.0 = 1.0
Token "book":  0.5 / 1.0 = 0.5
```

**No change (T=1.0 is identity)**

#### Step 3: Apply Top-k Filtering

**Top-k: k = 3**

**Select top 3 tokens:**
```
Top 3:
  "cat":   10.0
  "dog":   8.0
  "car":   5.0
```

**Set others to -∞:**
```
Token "cat":   10.0
Token "dog":   8.0
Token "car":   5.0
Token "house": -∞
Token "tree":  -∞
Token "book":  -∞
```

#### Step 4: Apply Top-p Filtering

**First, compute probabilities from top-k tokens:**

**Apply softmax:**
```
exp(10.0) = 22026.47
exp(8.0) = 2980.96
exp(5.0) = 148.41
Sum = 25155.84

P("cat") = 22026.47 / 25155.84 ≈ 0.875
P("dog") = 2980.96 / 25155.84 ≈ 0.119
P("car") = 148.41 / 25155.84 ≈ 0.006
```

**Cumulative probabilities:**
```
"cat":   0.875
"dog":   0.994  (0.875 + 0.119)
"car":   1.000  (0.994 + 0.006)
```

**Find smallest set where sum ≥ 0.9:**
```
"cat": 0.875 < 0.9
"cat" + "dog": 0.994 ≥ 0.9 ✓

→ Keep "cat" and "dog"
→ Remove "car"
```

**Result:**
```
Token "cat":   10.0
Token "dog":   8.0
Token "car":   -∞  (removed)
```

#### Step 5: Compute Final Probabilities

**Apply softmax to remaining tokens:**
```
exp(10.0) = 22026.47
exp(8.0) = 2980.96
Sum = 25007.43

P("cat") = 22026.47 / 25007.43 ≈ 0.881
P("dog") = 2980.96 / 25007.43 ≈ 0.119
```

#### Step 6: Sample Token

**Sample from distribution:**
```
Random number: 0.75

Cumulative:
  "cat": 0.881 ← 0.75 falls here
  "dog": 1.000

→ Selected: "cat"
```

### Answer

**Generated token: "cat"**

**Final sequence:**
```
Prompt: "The"
Generated: "cat"
Full text: "The cat"
```

### Summary

| Step | Operation | Result |
|------|-----------|--------|
| 1 | Initial logits | [10.0, 8.0, 5.0, 2.0, ...] |
| 2 | Apply temperature (T=1.0) | [10.0, 8.0, 5.0, 2.0, ...] |
| 3 | Top-k filtering (k=3) | Keep top 3: [10.0, 8.0, 5.0] |
| 4 | Top-p filtering (p=0.9) | Keep cumulative ≥0.9: [10.0, 8.0] |
| 5 | Compute probabilities | [0.881, 0.119] |
| 6 | Sample | "cat" selected |

**The model generated "cat" following "The"!**

---

## 9.9 Key Takeaways

### Generation

✅ **Generation produces text one token at a time**  
✅ **Autoregressive: uses previous outputs as inputs**  
✅ **Iterative process: predict → sample → append → repeat**

### Sampling Strategies

✅ **Temperature: Controls randomness (lower = deterministic, higher = random)**  
✅ **Top-k: Limits to top k tokens**  
✅ **Top-p: Keeps smallest set with cumulative probability ≥ p**  
✅ **Combined: Often use temperature + top-k or top-p**

### Why Important

✅ **Enables text generation from trained models**  
✅ **Different strategies produce different outputs**  
✅ **Essential for language model deployment**

---

## Mathematical Summary

### Generation Process

**Initialization:**
```math
\mathbf{T}_0 = \mathbf{P}
```

**For each step $t$:**
```math
\mathbf{l}_t = \text{Model}(\mathbf{T}_{t-1})[:, -1, :]
```

```math
\mathbf{l}_t' = \frac{\mathbf{l}_t}{T} \quad \text{(temperature)}
```

```math
\mathbf{l}_t'' = \text{Top-k}(\mathbf{l}_t') \quad \text{(optional)}
```

```math
\mathbf{l}_t''' = \text{Top-p}(\mathbf{l}_t'') \quad \text{(optional)}
```

```math
\mathbf{p}_t = \text{softmax}(\mathbf{l}_t''')
```

```math
t_t \sim \text{Categorical}(\mathbf{p}_t)
```

```math
\mathbf{T}_t = [\mathbf{T}_{t-1}, t_t]
```

---

*This document provides a comprehensive explanation of text generation, including autoregressive generation, sampling strategies, temperature, top-k, and top-p with mathematical formulations and solved exercises.*

