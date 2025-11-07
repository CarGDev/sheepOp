# What is a Neural Network? Step-by-Step Explanation

Complete step-by-step explanation of neural networks: what neurons are, what weights are, how calculations work, why they're important, with mathematical derivations and solved exercises.

## Table of Contents

1. [What is a Neural Network?](#61-what-is-a-neural-network)
2. [What is a Neuron?](#62-what-is-a-neuron)
3. [What are Weights?](#63-what-are-weights)
4. [How Neurons Calculate](#64-how-neurons-calculate)
5. [Why Weights are Important](#65-why-weights-are-important)
6. [Complete Mathematical Formulation](#66-complete-mathematical-formulation)
7. [Multi-Layer Neural Networks](#67-multi-layer-neural-networks)
8. [Exercise 1: Single Neuron Calculation](#68-exercise-1-single-neuron-calculation)
9. [Exercise 2: Multi-Layer Network](#69-exercise-2-multi-layer-network)
10. [Exercise 3: Learning Weights](#610-exercise-3-learning-weights)
11. [Key Takeaways](#611-key-takeaways)

---

## 6.1 What is a Neural Network?

### Simple Definition

A **neural network** is a computational model inspired by biological neurons that processes information through interconnected nodes (neurons) to make predictions or decisions.

### Visual Analogy

**Think of a neural network like a factory:**

```
Input → Worker 1 → Worker 2 → Worker 3 → Output
```

**Neural Network:**

```
Input → Neuron 1 → Neuron 2 → Neuron 3 → Output
```

**Each worker (neuron) does a specific job, and they work together to produce the final result.**

### Basic Structure

```
Input Layer      Hidden Layer      Output Layer
     ●               ●                 ●
     ●               ●                 ●
     ●               ●                 ●
     ●               ●
```

**Key Components:**

- **Input Layer:** Receives data
- **Hidden Layers:** Process information
- **Output Layer:** Produces predictions
- **Connections:** Weights between neurons

---

## 6.2 What is a Neuron?

### Simple Definition

A **neuron** (also called a node or unit) is the basic processing unit of a neural network. It receives inputs, performs calculations, and produces an output.

### Biological Inspiration

**Biological Neuron:**

```
Dendrites → Cell Body → Axon → Synapses
(inputs)    (process)   (output) (connections)
```

**Artificial Neuron:**

```
Inputs → Weighted Sum → Activation → Output
```

### Structure of a Neuron

```
Input 1 (x₁) ────┐
                 │
Input 2 (x₂) ────┼──→ [Σ] ─→ [f] ─→ Output (y)
                 │
Input 3 (x₃) ────┘
```

**Components:**

1. **Inputs:** Values fed into the neuron
2. **Weights:** Strength of connections
3. **Weighted Sum:** Sum of inputs × weights
4. **Bias:** Added constant
5. **Activation Function:** Applies nonlinearity
6. **Output:** Final result

### Visual Representation

```
Neuron:
    ┌─────────────────────┐
    │  Inputs: x₁, x₂, x₃ │
    │  Weights: w₁, w₂, w₃│
    │                     │
    │  z = Σ(xᵢ × wᵢ) + b │
    │  y = f(z)           │
    │                     │
    │  Output: y          │
    └─────────────────────┘
```

**Where:**

- `z` = weighted sum (before activation)
- `f` = activation function
- `y` = output (after activation)

---

## 6.3 What are Weights?

### Simple Definition

**Weights** are numerical values that determine the strength of connections between neurons. They control how much each input contributes to the output.

### Visual Analogy

**Think of weights like volume controls:**

```
Music Source 1 ──[Volume: 0.8]──→ Speakers
Music Source 2 ──[Volume: 0.3]──→ Speakers
Music Source 3 ──[Volume: 0.5]──→ Speakers
```

**Higher weight = Louder contribution**

**Neural Network:**

```
Input 1 ──[Weight: 0.8]──→ Neuron
Input 2 ──[Weight: 0.3]──→ Neuron
Input 3 ──[Weight: 0.5]──→ Neuron
```

**Higher weight = Stronger influence**

### What Weights Do

**Weights determine:**

1. **How much each input matters**
2. **The relationship between inputs and outputs**
3. **What patterns the neuron learns**

**Example:**

**Weight = 0.1:**

- Input has small influence
- Weak connection

**Weight = 5.0:**

- Input has large influence
- Strong connection

**Weight = -2.0:**

- Input has negative influence
- Inverts the relationship

**Weight = 0.0:**

- Input has no influence
- Connection is cut

### Weight Matrix

**In a layer with multiple neurons:**

```
Input Layer          Weights Matrix      Output Layer
x₁ ───────────────────┐
                      │   w₁₁  w₁₂       y₁
x₂ ───────────────────┼─  w₂₁  w₂₂  ──── y₂
                      │   w₃₁  w₃₂
x₃ ───────────────────┘
```

**Weight Matrix:**

```
W = [w₁₁  w₁₂]
    [w₂₁  w₂₂]
    [w₃₁  w₃₂]
```

**Each row:** Connections from one input  
**Each column:** Connections to one output

---

## 6.4 How Neurons Calculate

### Step-by-Step Calculation

#### Step 1: Weighted Sum

**Multiply each input by its weight:**

```math
z = x_1 \times w_1 + x_2 \times w_2 + x_3 \times w_3 + ... + b
```

**Or in vector form:**

```math
z = \mathbf{x} \cdot \mathbf{w} + b = \sum_{i=1}^{n} x_i w_i + b
```

**Where:**

- $x_i$ = input value
- $w_i$ = weight for input $i$
- $b$ = bias (constant)
- $n$ = number of inputs

#### Step 2: Add Bias

**Bias shifts the activation:**

```math
z = \sum_{i=1}^{n} x_i w_i + b
```

**Bias allows the neuron to:**

- Shift activation threshold
- Learn patterns independent of inputs
- Adjust baseline output

#### Step 3: Apply Activation Function

**Apply nonlinear function:**

```math
y = f(z)
```

**Common activation functions:**

**ReLU (Rectified Linear Unit):**

```math
f(z) = \max(0, z)
```

**Sigmoid:**

```math
f(z) = \frac{1}{1 + e^{-z}}
```

**Tanh:**

```math
f(z) = \tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}
```

**GELU (used in transformers):**

```math
f(z) = z \cdot \Phi(z)
```

**Where $\Phi(z)$ is the CDF of standard normal distribution**

### Complete Example

**Given:**

- Inputs: $x_1 = 0.5, x_2 = 0.3, x_3 = 0.8$
- Weights: $w_1 = 0.6, w_2 = 0.4, w_3 = 0.2$
- Bias: $b = 0.1$
- Activation: ReLU

**Step 1: Weighted Sum**

```
z = (0.5 × 0.6) + (0.3 × 0.4) + (0.8 × 0.2) + 0.1
  = 0.3 + 0.12 + 0.16 + 0.1
  = 0.68
```

**Step 2: Apply Activation**

```
y = ReLU(0.68)
  = max(0, 0.68)
  = 0.68
```

**Result:** Output = 0.68

---

## 6.5 Why Weights are Important

### Reason 1: They Determine What the Neuron Learns

**Different weights = Different patterns:**

**Pattern 1: Emphasis on Input 1**

```
w₁ = 5.0, w₂ = 0.1, w₃ = 0.1
→ Neuron cares mostly about input 1
```

**Pattern 2: Balanced Weights**

```
w₁ = 0.5, w₂ = 0.5, w₃ = 0.5
→ Neuron treats all inputs equally
```

**Pattern 3: Inverted Relationship**

```
w₁ = -2.0, w₂ = 1.0, w₃ = 1.0
→ Neuron inverses input 1's effect
```

### Reason 2: They Enable Learning

**Training adjusts weights:**

**Before Training:**

```
Weights: Random values
→ Random predictions
```

**After Training:**

```
Weights: Learned values
→ Accurate predictions
```

**Weights are what the model learns!**

### Reason 3: They Control Information Flow

**High weights:** Information flows easily  
**Low weights:** Information flows weakly  
**Zero weights:** Information blocked  
**Negative weights:** Information inverted

### Reason 4: They Enable Complex Patterns

**Multiple neurons with different weights:**

```
Neuron 1: w₁ = 1.0, w₂ = 0.0 → Detects pattern A
Neuron 2: w₁ = 0.0, w₂ = 1.0 → Detects pattern B
Neuron 3: w₁ = 0.5, w₂ = 0.5 → Detects pattern C
```

**Together:** Model learns complex relationships!

---

## 6.6 Complete Mathematical Formulation

### Single Neuron Formula

**Complete neuron calculation:**

```math
z = \sum_{i=1}^{n} x_i w_i + b
```

```math
y = f(z)
```

**Where:**

- $\mathbf{x} = [x_1, x_2, ..., x_n]$ = input vector
- $\mathbf{w} = [w_1, w_2, ..., w_n]$ = weight vector
- $b$ = bias (scalar)
- $f$ = activation function
- $z$ = weighted sum (before activation)
- $y$ = output (after activation)

### Matrix Formulation

**For multiple neurons:**

```math
\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}
```

```math
\mathbf{Y} = f(\mathbf{z})
```

**Where:**

- $\mathbf{X} \in \mathbb{R}^{B \times n}$ = input matrix (B samples, n features)
- $\mathbf{W} \in \mathbb{R}^{n \times m}$ = weight matrix (n inputs, m neurons)
- $\mathbf{b} \in \mathbb{R}^{1 \times m}$ = bias vector
- $\mathbf{z} \in \mathbb{R}^{B \times m}$ = weighted sums
- $\mathbf{Y} \in \mathbb{R}^{B \times m}$ = outputs

**Example:**

**Input Matrix:**

```
X = [x₁₁  x₁₂]  (2 samples, 2 features)
    [x₂₁  x₂₂]
```

**Weight Matrix:**

```
W = [w₁₁  w₁₂]  (2 inputs, 2 neurons)
    [w₂₁  w₂₂]
```

**Bias Vector:**

```
b = [b₁  b₂]  (2 neurons)
```

**Calculation:**

```
z = X × W + b

z₁₁ = x₁₁×w₁₁ + x₁₂×w₂₁ + b₁
z₁₂ = x₁₁×w₁₂ + x₁₂×w₂₂ + b₂
z₂₁ = x₂₁×w₁₁ + x₂₂×w₂₁ + b₁
z₂₂ = x₂₁×w₁₂ + x₂₂×w₂₂ + b₂
```

---

## 6.7 Multi-Layer Neural Networks

### Structure

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
    x₁            h₁₁               h₂₁             y₁
    x₂            h₁₂               h₂₂             y₂
    x₃            h₁₃               h₂₃
```

### Forward Pass

**Layer 1:**

```math
\mathbf{h}_1 = f_1(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)
```

**Layer 2:**

```math
\mathbf{h}_2 = f_2(\mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2)
```

**Output Layer:**

```math
\mathbf{Y} = f_3(\mathbf{h}_2 \mathbf{W}_3 + \mathbf{b}_3)
```

**Chained together:**

```math
\mathbf{Y} = f_3(f_2(f_1(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1) \mathbf{W}_2 + \mathbf{b}_2) \mathbf{W}_3 + \mathbf{b}_3)
```

**Each layer transforms the input!**

---

## 6.8 Exercise 1: Single Neuron Calculation

### Problem

**Given a single neuron with:**

- Inputs: $x_1 = 2.0, x_2 = -1.0, x_3 = 0.5$
- Weights: $w_1 = 0.5, w_2 = -0.3, w_3 = 0.8$
- Bias: $b = 0.2$
- Activation function: ReLU $f(z) = \max(0, z)$

**Calculate the output of this neuron.**

### Step-by-Step Solution

#### Step 1: Weighted Sum

**Compute:**

```math
z = \sum_{i=1}^{3} x_i w_i + b
```

**Substitute values:**

```math
z = (2.0 \times 0.5) + (-1.0 \times -0.3) + (0.5 \times 0.8) + 0.2
```

**Calculate each term:**

```math
z = (1.0) + (0.3) + (0.4) + 0.2
```

**Sum:**

```math
z = 1.0 + 0.3 + 0.4 + 0.2 = 1.9
```

#### Step 2: Apply Activation Function

**Apply ReLU:**

```math
y = \text{ReLU}(z) = \max(0, z) = \max(0, 1.9) = 1.9
```

### Answer

**The output of the neuron is $y = 1.9$.**

### Verification

**Check calculation:**

- Input contribution 1: $2.0 \times 0.5 = 1.0$
- Input contribution 2: $-1.0 \times -0.3 = 0.3$
- Input contribution 3: $0.5 \times 0.8 = 0.4$
- Bias: $0.2$
- Total: $1.0 + 0.3 + 0.4 + 0.2 = 1.9$ ✓
- ReLU(1.9) = 1.9 ✓

---

## 6.9 Exercise 2: Multi-Layer Network

### Problem

**Given a neural network with 2 layers:**

**Layer 1:**

- Inputs: $x_1 = 1.0, x_2 = 0.5$
- Weights: $W_1 = \begin{bmatrix} 0.6 & 0.4 \\ 0.2 & 0.8 \end{bmatrix}$
- Bias: $b_1 = [0.1, -0.1]$
- Activation: ReLU

**Layer 2:**

- Inputs: Outputs from Layer 1
- Weights: $W_2 = \begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix}$
- Bias: $b_2 = 0.2$
- Activation: ReLU

**Calculate the final output.**

### Step-by-Step Solution

#### Step 1: Layer 1 - Weighted Sum

**Input vector:**

```math
\mathbf{x} = [1.0, 0.5]
```

**Weight matrix:**

```math
\mathbf{W}_1 = \begin{bmatrix} 0.6 & 0.4 \\ 0.2 & 0.8 \end{bmatrix}
```

**Bias vector:**

```math
\mathbf{b}_1 = [0.1, -0.1]
```

**Calculate:**

```math
\mathbf{z}_1 = \mathbf{x} \mathbf{W}_1 + \mathbf{b}_1
```

**Matrix multiplication:**

```math
\mathbf{z}_1 = [1.0, 0.5] \begin{bmatrix} 0.6 & 0.4 \\ 0.2 & 0.8 \end{bmatrix} + [0.1, -0.1]
```

**Compute:**

```math
z_{1,1} = 1.0 \times 0.6 + 0.5 \times 0.2 + 0.1 = 0.6 + 0.1 + 0.1 = 0.8
```

```math
z_{1,2} = 1.0 \times 0.4 + 0.5 \times 0.8 + (-0.1) = 0.4 + 0.4 - 0.1 = 0.7
```

```math
\mathbf{z}_1 = [0.8, 0.7]
```

#### Step 2: Layer 1 - Apply Activation

**Apply ReLU:**

```math
\mathbf{h}_1 = \text{ReLU}(\mathbf{z}_1) = [\max(0, 0.8), \max(0, 0.7)] = [0.8, 0.7]
```

#### Step 3: Layer 2 - Weighted Sum

**Input (from Layer 1):**

```math
\mathbf{h}_1 = [0.8, 0.7]
```

**Weight matrix:**

```math
\mathbf{W}_2 = \begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix}
```

**Bias:**

```math
b_2 = 0.2
```

**Calculate:**

```math
z_2 = \mathbf{h}_1 \mathbf{W}_2 + b_2
```

**Matrix multiplication:**

```math
z_2 = [0.8, 0.7] \begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix} + 0.2
```

**Compute:**

```math
z_2 = 0.8 \times 0.5 + 0.7 \times 0.7 + 0.2 = 0.4 + 0.49 + 0.2 = 1.09
```

#### Step 4: Layer 2 - Apply Activation

**Apply ReLU:**

```math
y = \text{ReLU}(z_2) = \max(0, 1.09) = 1.09
```

### Answer

**The final output is $y = 1.09$.**

### Summary Table

<table>
  <tr>
    <th>Layer</th>
    <th>Input</th>
    <th>Weights</th>
    <th>Bias</th>
    <th>Weighted Sum</th>
    <th>Activation</th>
    <th>Output</th>
  </tr>

  <tr>
    <td>1</td>
    <td>[1.0, 0.5]</td>
    <td>$$\begin{bmatrix} 0.6 & 0.4 \\ 0.2 & 0.8 \end{bmatrix}$$</td>
    <td>[0.1, -0.1]</td>
    <td>[0.8, 0.7]</td>
    <td>ReLU</td>
    <td>[0.8, 0.7]</td>
  </tr>

  <tr>
    <td>2</td>
    <td>[0.8, 0.7]</td>
    <td>$$\begin{bmatrix} 0.5 \\ 0.7 \end{bmatrix}$$</td>
    <td>0.2</td>
    <td>1.09</td>
    <td>ReLU</td>
    <td><strong>1.09</strong></td>
  </tr>
</table>
---

## 6.10 Exercise 3: Learning Weights

### Problem

**Given a neuron that should output 1.0 when inputs are [1.0, 1.0] and output 0.0 when inputs are [0.0, 0.0], find appropriate weights and bias.**

**Use:**

- Activation: Sigmoid $f(z) = \frac{1}{1 + e^{-z}}$
- Desired behavior: AND gate (output 1 only when both inputs are 1)

### Step-by-Step Solution

#### Step 1: Set Up Equations

**For input [1.0, 1.0], desired output ≈ 1.0:**

```math
f(w_1 \times 1.0 + w_2 \times 1.0 + b) = 1.0
```

**For input [0.0, 0.0], desired output ≈ 0.0:**

```math
f(w_1 \times 0.0 + w_2 \times 0.0 + b) = 0.0
```

**Note:** Sigmoid outputs range from 0 to 1, so:

- $f(z) \approx 1.0$ when $z \gg 0$ (e.g., $z > 5$)
- $f(z) \approx 0.0$ when $z \ll 0$ (e.g., $z < -5$)

#### Step 2: Solve for Bias

**From equation 2:**

```math
f(b) = 0.0
```

**For sigmoid to output ≈ 0:**

```math
b < -5
```

**Let's use:**

```math
b = -10
```

#### Step 3: Solve for Weights

**From equation 1:**

```math
f(w_1 + w_2 - 10) = 1.0
```

**For sigmoid to output ≈ 1:**

```math
w_1 + w_2 - 10 > 5
```

```math
w_1 + w_2 > 15
```

**Let's use equal weights:**

```math
w_1 = w_2 = 8.0
```

**Check:**

```math
w_1 + w_2 = 8.0 + 8.0 = 16.0 > 15 \quad ✓
```

#### Step 4: Verify Solution

**Test Case 1: Input [1.0, 1.0]**

```math
z = 1.0 \times 8.0 + 1.0 \times 8.0 + (-10) = 8.0 + 8.0 - 10 = 6.0
```

```math
y = \frac{1}{1 + e^{-6.0}} = \frac{1}{1 + 0.0025} \approx 0.9975 \approx 1.0 \quad ✓
```

**Test Case 2: Input [0.0, 0.0]**

```math
z = 0.0 \times 8.0 + 0.0 \times 8.0 + (-10) = -10
```

```math
y = \frac{1}{1 + e^{10}} = \frac{1}{1 + 22026} \approx 0.00005 \approx 0.0 \quad ✓
```

**Test Case 3: Input [1.0, 0.0]**

```math
z = 1.0 \times 8.0 + 0.0 \times 8.0 + (-10) = 8.0 - 10 = -2.0
```

```math
y = \frac{1}{1 + e^{2.0}} = \frac{1}{1 + 7.39} \approx 0.12 < 0.5 \quad ✓
```

**Test Case 4: Input [0.0, 1.0]**

```math
z = 0.0 \times 8.0 + 1.0 \times 8.0 + (-10) = 8.0 - 10 = -2.0
```

```math
y = \frac{1}{1 + e^{2.0}} \approx 0.12 < 0.5 \quad ✓
```

### Answer

**Appropriate weights and bias:**

- $w_1 = 8.0$
- $w_2 = 8.0$
- $b = -10.0$

**The neuron implements an AND gate correctly!**

### Key Insight

**This demonstrates learning:**

- Training finds weights that produce desired behavior
- Different weights = Different logic functions
- Learning algorithms (like backpropagation) automatically find these weights from data!

---

## 6.11 Key Takeaways

### Neurons

✅ **Neurons are the basic processing units**  
✅ **Receive inputs, compute weighted sum, apply activation**  
✅ **Output is the result of activation function**

### Weights

✅ **Weights control connection strength**  
✅ **Determine what patterns neurons learn**  
✅ **Are what the model learns during training**  
✅ **Enable complex pattern recognition**

### Calculation

✅ **Weighted sum: $z = \sum x_i w_i + b$**  
✅ **Activation: $y = f(z)$**  
✅ **Matrix form enables efficient computation**

### Importance

✅ **Weights enable learning**  
✅ **Control information flow**  
✅ **Enable complex pattern recognition**  
✅ **Are adjusted during training to minimize error**

### Neural Networks

✅ **Multiple neurons form layers**  
✅ **Multiple layers form networks**  
✅ **Each layer transforms the input**  
✅ **Deep networks learn hierarchical features**

---

## Mathematical Summary

### Single Neuron

```math
z = \sum_{i=1}^{n} x_i w_i + b
```

```math
y = f(z)
```

### Multiple Neurons (Matrix Form)

```math
\mathbf{z} = \mathbf{X} \mathbf{W} + \mathbf{b}
```

```math
\mathbf{Y} = f(\mathbf{z})
```

### Multi-Layer Network

```math
\mathbf{h}_1 = f_1(\mathbf{X} \mathbf{W}_1 + \mathbf{b}_1)
```

```math
\mathbf{h}_2 = f_2(\mathbf{h}_1 \mathbf{W}_2 + \mathbf{b}_2)
```

```math
\mathbf{Y} = f_3(\mathbf{h}_2 \mathbf{W}_3 + \mathbf{b}_3)
```

---

_This document provides a comprehensive explanation of neural networks, neurons, weights, and calculations with mathematical derivations and solved exercises._
