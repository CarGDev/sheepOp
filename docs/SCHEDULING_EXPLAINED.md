# What is Scheduling? Step-by-Step Explanation

Complete step-by-step explanation of learning rate scheduling: how scheduling adjusts learning rates during training to improve convergence.

## Table of Contents

1. [What is Scheduling?](#81-what-is-scheduling)
2. [Why Do We Need Scheduling?](#82-why-do-we-need-scheduling)
3. [Fixed Learning Rate](#83-fixed-learning-rate)
4. [Cosine Annealing](#84-cosine-annealing)
5. [Other Scheduling Strategies](#85-other-scheduling-strategies)
6. [Why Scheduling Matters](#86-why-scheduling-matters)
7. [Complete Mathematical Formulation](#87-complete-mathematical-formulation)
8. [Exercise: Schedule Calculation](#88-exercise-schedule-calculation)
9. [Key Takeaways](#89-key-takeaways)

---

## 8.1 What is Scheduling?

### Simple Definition

**Scheduling** (learning rate scheduling) is the process of adjusting the learning rate during training to improve convergence and final model performance.

### Visual Analogy

**Think of scheduling like adjusting speed while driving:**

```
Fixed Learning Rate:
    ┌──────────────────────────┐
    │ Speed: 60 mph (constant) │
    └──────────────────────────┘
    → Hard to stop precisely!

Scheduled Learning Rate:
    ┌──────────────────────────┐
    │ Speed: 60 → 40 → 20 → 10 │
    └──────────────────────────┘
    → Smooth deceleration!
```

**Scheduling adjusts speed (learning rate) as you approach the destination (convergence)!**

### What Scheduling Does

**Scheduling:**
1. **Starts** with higher learning rate (fast learning)
2. **Gradually reduces** learning rate (precise fine-tuning)
3. **Converges** to optimal solution

**Result:** Better convergence and performance!

---

## 8.2 Why Do We Need Scheduling?

### The Problem with Fixed Learning Rate

**High Learning Rate:**
```
Learning Rate: 0.001 (constant)
→ Fast initial learning ✓
→ But overshoots minimum ✗
→ Bounces around ✗
→ Poor convergence ✗
```

**Low Learning Rate:**
```
Learning Rate: 0.0001 (constant)
→ Stable convergence ✓
→ But very slow learning ✗
→ Takes forever to converge ✗
```

**Can't have both!**

### The Solution: Scheduling

**Adaptive Learning Rate:**
```
Start: 0.001 (fast learning)
Middle: 0.0005 (moderate)
End: 0.0001 (fine-tuning)
→ Fast initial learning ✓
→ Stable convergence ✓
→ Best of both worlds!
```

### Benefits of Scheduling

**1. Faster Convergence**
- High initial rate = Fast progress
- Lower later rate = Precise convergence

**2. Better Final Performance**
- Fine-tuning at end = Better solution
- Avoids overshooting = More stable

**3. More Stable Training**
- Gradual reduction = Smooth optimization
- Less oscillation = More reliable

---

## 8.3 Fixed Learning Rate

### What is Fixed Learning Rate?

**Learning rate stays constant throughout training:**

```math
\eta_t = \eta_0 \quad \text{for all } t
```

**Where:**
- $\eta_0$ = initial learning rate
- $t$ = training step

### Example

**Fixed Rate:**
```
Step 0:    η = 0.001
Step 100:  η = 0.001
Step 1000: η = 0.001
Step 10000: η = 0.001
```

**Constant throughout!**

### Visualization

```
Learning Rate
     │
0.001│─────────────────────────────────────
     │
     │
     │
     │
     └───────────────────────────────────── Steps
```

### Problems

**1. Too High:**
- Overshoots minimum
- Oscillates around solution
- Never converges precisely

**2. Too Low:**
- Very slow training
- Takes forever to converge
- May get stuck

**Solution:** Use scheduling!

---

## 8.4 Cosine Annealing

### What is Cosine Annealing?

**Cosine Annealing** reduces the learning rate following a cosine curve from maximum to minimum.

### Formula

```math
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \times \frac{1 + \cos\left(\frac{\pi t}{T_{max}}\right)}{2}
```

**Where:**
- $\eta_t$ = learning rate at step $t$
- $\eta_{min}$ = minimum learning rate (default: 0)
- $\eta_{max}$ = initial/maximum learning rate
- $T_{max}$ = total number of steps
- $t$ = current step

### How It Works

**Step 1: Calculate Cosine Value**
```math
\cos\left(\frac{\pi t}{T_{max}}\right)
```

**Step 2: Shift to [0, 1] Range**
```math
\frac{1 + \cos\left(\frac{\pi t}{T_{max}}\right)}{2}
```

**Step 3: Scale to Learning Rate Range**
```math
\eta_{min} + (\eta_{max} - \eta_{min}) \times \text{scale}
```

### Example Calculation

**Given:**
- $\eta_{max} = 0.001$
- $\eta_{min} = 0$
- $T_{max} = 10000$

**At step $t = 0$:**
```math
\eta_0 = 0 + (0.001 - 0) \times \frac{1 + \cos(0)}{2} = 0.001 \times 1 = 0.001
```

**At step $t = 2500$:**
```math
\eta_{2500} = 0 + 0.001 \times \frac{1 + \cos(\pi/4)}{2} = 0.001 \times \frac{1 + 0.707}{2} \approx 0.000854
```

**At step $t = 5000$:**
```math
\eta_{5000} = 0 + 0.001 \times \frac{1 + \cos(\pi/2)}{2} = 0.001 \times \frac{1 + 0}{2} = 0.0005
```

**At step $t = 7500$:**
```math
\eta_{7500} = 0 + 0.001 \times \frac{1 + \cos(3\pi/4)}{2} = 0.001 \times \frac{1 + (-0.707)}{2} \approx 0.000146
```

**At step $t = 10000$:**
```math
\eta_{10000} = 0 + 0.001 \times \frac{1 + \cos(\pi)}{2} = 0.001 \times \frac{1 + (-1)}{2} = 0
```

### Visualization

```
Learning Rate
      │
0.001 │●───────────────\
      │                 \
      │                  \
0.0005│                   \
      │                    \
      │                     \
      │                      \
      │                       \
      │                        \
      │                         \
     0│                          ●─────
      └───────────────────────────────────── Steps
        0    2500  5000  7500  10000
```

**Smooth cosine curve!**

### Why Cosine Annealing?

**Benefits:**
1. **Smooth decay:** No abrupt changes
2. **Gradual reduction:** Better fine-tuning
3. **Works well:** Commonly used in practice
4. **High initial rate:** Fast learning
5. **Low final rate:** Precise convergence

---

## 8.5 Other Scheduling Strategies

### 1. Step Decay

**Reduce learning rate at fixed intervals:**

```math
\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}
```

**Where:**
- $\gamma$ = decay factor (e.g., 0.1)
- $s$ = step size (e.g., every 1000 steps)

**Example:**
```
Step 0-999:    η = 0.001
Step 1000-1999: η = 0.0001  (×0.1)
Step 2000-2999: η = 0.00001 (×0.1)
```

**Visualization:**
```
Learning Rate
      │
0.001 │───────┐
      │       │
      │       └───────┐
0.0001│               │
      │               └───────┐
      │                       │
      └───────────────────────── Steps
```

### 2. Exponential Decay

**Continuous exponential reduction:**

```math
\eta_t = \eta_0 \times \gamma^t
```

**Where:**
- $\gamma$ = decay rate (e.g., 0.9995)

**Visualization:**
```
Learning Rate
     │
0.001│●──────────────\
     │                \
     │                 \
     │                  \
     │                   \
     │                    \
     │                     \
     │                      \
     └──────────────────────── Steps
```

### 3. Warmup Scheduling

**Start with low rate, increase, then decrease:**

**Warmup Phase:**
```math
\eta_t = \eta_{max} \times \frac{t}{T_{warmup}}
```

**After Warmup:**
```math
\eta_t = \text{Cosine Annealing or other schedule}
```

**Visualization:**
```
Learning Rate
     │
0.001│      ╱───────\
     │     ╱         \
     │    ╱           \
     │   ╱             \
     │  ╱               \
     │ ╱                 \
     │╱                   \
     └───────────────────── Steps
```

### 4. One Cycle Learning Rate

**One cycle: increase then decrease:**

```math
\eta_t = \begin{cases}
\eta_{min} + (\eta_{max} - \eta_{min}) \times \frac{t}{T_1} & t \leq T_1 \\
\eta_{max} - (\eta_{max} - \eta_{min}) \times \frac{t - T_1}{T_2} & t > T_1
\end{cases}
```

**Visualization:**
```
Learning Rate
     │
0.001│      ╱─────\
     │     ╱       \
     │    ╱         \
     │   ╱           \
     │  ╱             \
     │ ╱               \
     │╱                 \
     └─────────────────── Steps
```

---

## 8.6 Why Scheduling Matters

### Benefit 1: Better Convergence

**Without Scheduling:**
```
Loss: 3.0 → 2.5 → 2.3 → 2.2 → 2.15 → 2.12 → ...
      (slow convergence at end)
```

**With Scheduling:**
```
Loss: 3.0 → 2.5 → 2.3 → 2.2 → 2.1 → 2.05 → ...
      (faster convergence, better final loss)
```

### Benefit 2: More Stable Training

**Fixed High Rate:**
```
Loss: 3.0 → 2.5 → 2.3 → 2.4 → 2.3 → 2.4 → ...
      (oscillating, unstable)
```

**Scheduled Rate:**
```
Loss: 3.0 → 2.5 → 2.3 → 2.2 → 2.15 → 2.12 → ...
      (smooth, stable)
```

### Benefit 3: Better Final Performance

**Comparison:**
```
Fixed LR:      Final Loss = 2.15
Scheduled LR:  Final Loss = 2.05

→ 5% improvement!
```

---

## 8.7 Complete Mathematical Formulation

### General Scheduling Formula

```math
\eta_t = f(t, \eta_0, \eta_{min}, T_{max}, ...)
```

**Where $f$ is the scheduling function**

### Cosine Annealing (Complete)

```math
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \times \frac{1 + \cos\left(\frac{\pi t}{T_{max}}\right)}{2}
```

**Boundary Conditions:**
- At $t = 0$: $\eta_0 = \eta_{max}$
- At $t = T_{max}$: $\eta_{T_{max}} = \eta_{min}$

### Step Decay

```math
\eta_t = \eta_0 \times \gamma^{\lfloor t / s \rfloor}
```

### Exponential Decay

```math
\eta_t = \eta_0 \times \gamma^t
```

### Warmup + Cosine Annealing

**Warmup Phase ($t \leq T_{warmup}$):**
```math
\eta_t = \eta_{max} \times \frac{t}{T_{warmup}}
```

**Annealing Phase ($t > T_{warmup}$):**
```math
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \times \frac{1 + \cos\left(\frac{\pi (t - T_{warmup})}{T_{max} - T_{warmup}}\right)}{2}
```

---

## 8.8 Exercise: Schedule Calculation

### Problem

**Given Cosine Annealing schedule:**

- $\eta_{max} = 0.002$
- $\eta_{min} = 0.0001$
- $T_{max} = 5000$ steps

**Calculate the learning rate at:**
1. Step $t = 0$
2. Step $t = 1250$
3. Step $t = 2500$
4. Step $t = 3750$
5. Step $t = 5000$

### Step-by-Step Solution

#### General Formula

```math
\eta_t = \eta_{min} + (\eta_{max} - \eta_{min}) \times \frac{1 + \cos\left(\frac{\pi t}{T_{max}}\right)}{2}
```

**Substitute values:**
```math
\eta_t = 0.0001 + (0.002 - 0.0001) \times \frac{1 + \cos\left(\frac{\pi t}{5000}\right)}{2}
```

```math
\eta_t = 0.0001 + 0.0019 \times \frac{1 + \cos\left(\frac{\pi t}{5000}\right)}{2}
```

#### Step 1: t = 0

```math
\eta_0 = 0.0001 + 0.0019 \times \frac{1 + \cos(0)}{2}
```

```math
\eta_0 = 0.0001 + 0.0019 \times \frac{1 + 1}{2}
```

```math
\eta_0 = 0.0001 + 0.0019 \times 1 = 0.0001 + 0.0019 = 0.002
```

**Answer:** $\eta_0 = 0.002$

#### Step 2: t = 1250

```math
\eta_{1250} = 0.0001 + 0.0019 \times \frac{1 + \cos(\pi/4)}{2}
```

```math
\eta_{1250} = 0.0001 + 0.0019 \times \frac{1 + 0.707}{2}
```

```math
\eta_{1250} = 0.0001 + 0.0019 \times 0.8535 = 0.0001 + 0.001621 = 0.001721
```

**Answer:** $\eta_{1250} \approx 0.001721$

#### Step 3: t = 2500

```math
\eta_{2500} = 0.0001 + 0.0019 \times \frac{1 + \cos(\pi/2)}{2}
```

```math
\eta_{2500} = 0.0001 + 0.0019 \times \frac{1 + 0}{2}
```

```math
\eta_{2500} = 0.0001 + 0.0019 \times 0.5 = 0.0001 + 0.00095 = 0.00105
```

**Answer:** $\eta_{2500} = 0.00105$

#### Step 4: t = 3750

```math
\eta_{3750} = 0.0001 + 0.0019 \times \frac{1 + \cos(3\pi/4)}{2}
```

```math
\eta_{3750} = 0.0001 + 0.0019 \times \frac{1 + (-0.707)}{2}
```

```math
\eta_{3750} = 0.0001 + 0.0019 \times 0.1465 = 0.0001 + 0.000278 = 0.000378
```

**Answer:** $\eta_{3750} \approx 0.000378$

#### Step 5: t = 5000

```math
\eta_{5000} = 0.0001 + 0.0019 \times \frac{1 + \cos(\pi)}{2}
```

```math
\eta_{5000} = 0.0001 + 0.0019 \times \frac{1 + (-1)}{2}
```

```math
\eta_{5000} = 0.0001 + 0.0019 \times 0 = 0.0001 + 0 = 0.0001
```

**Answer:** $\eta_{5000} = 0.0001$

### Summary Table

| Step | Cosine Value | Scale Factor | Learning Rate |
|------|--------------|--------------|---------------|
| 0 | 1.0 | 1.0 | 0.002 |
| 1250 | 0.707 | 0.854 | 0.001721 |
| 2500 | 0.0 | 0.5 | 0.00105 |
| 3750 | -0.707 | 0.146 | 0.000378 |
| 5000 | -1.0 | 0.0 | 0.0001 |

**Smooth decay from 0.002 to 0.0001!**

---

## 8.9 Key Takeaways

### Scheduling

✅ **Scheduling adjusts learning rate during training**  
✅ **Starts high (fast learning), ends low (fine-tuning)**  
✅ **Improves convergence and final performance**

### Cosine Annealing

✅ **Smooth cosine-based decay**  
✅ **Gradual reduction from max to min**  
✅ **Works well for transformers**

### Why Important

✅ **Faster convergence**  
✅ **More stable training**  
✅ **Better final performance**  
✅ **Essential for optimal training**

---

*This document provides a comprehensive explanation of learning rate scheduling, including cosine annealing and other strategies with mathematical formulations and solved exercises.*

