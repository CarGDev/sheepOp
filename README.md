# SheepOp LLM 🐑➡️🤖

**Author:** Carlos Gutierrez  
**Email:** carlos.gutierrez@carg.dev  
**License:** Dual License - Apache 2.0 (Research) + Commercial License (Commercial Use)

A modern language model implementation from scratch, incorporating insights from recent research papers.

---

## Purpose of the Project

SheepOp LLM is a comprehensive transformer-based language model implementation designed for:

- **Research & Education**: Understanding how large language models work from the ground up
- **Custom Training**: Training models on domain-specific data (PDFs, code, text files)
- **Production Deployment**: Optimized inference with KV caching and efficient attention mechanisms
- **Multi-Format Data Processing**: Support for various data types including PDFs, images (OCR), code files, and text

The project provides a complete toolkit for building, training, and deploying transformer language models with modern best practices.

---

## Documentation Index

All detailed documentation is available in the [`docs/`](docs/) folder:

### Core Concepts

- **[Complete Guide](docs/COMPLETE_GUIDE.md)** - Full project documentation with mathematical foundations, architecture, and usage
- **[Architecture](docs/ARCHITECTURE.md)** - System architecture and design patterns
- **[Mathematics](docs/MATHEMATICS.md)** - Complete mathematical derivations for all components

### Component Explanations

- **[Embeddings](docs/EMBEDDINGS_EXPLAINED.md)** - What are embeddings and how they work
- **[Attention](docs/ATTENTION_EXPLAINED.md)** - Attention mechanisms explained step-by-step
- **[Feed-Forward](docs/FEED_FORWARD_EXPLAINED.md)** - Feed-forward networks explained
- **[Normalization](docs/NORMALIZATION_EXPLAINED.md)** - Layer normalization explained
- **[Neural Networks](docs/NEURAL_NETWORK_EXPLAINED.md)** - Neural networks, neurons, and weights explained

### Training & Optimization

- **[Training](docs/TRAINING_EXPLAINED.md)** - What is training, why we need data, why more data is better, and how to interpret training metrics
- **[Optimization](docs/OPTIMIZATION_EXPLAINED.md)** - Optimizers (AdamW, gradient descent) explained
- **[Scheduling](docs/SCHEDULING_EXPLAINED.md)** - Learning rate scheduling explained
- **[Generation](docs/GENERATION_EXPLAINED.md)** - Text generation and sampling strategies

### Data & Processing

- **[Data Processing](docs/DATA_PROCESSING_EXPLAINED.md)** - How data processing works step-by-step
- **[Multi-Format Data Guide](docs/MULTI_FORMAT_DATA_GUIDE.md)** - Working with PDFs, images, code files
- **[Data Guide](docs/DATA_GUIDE.md)** - General data handling guide
- **[Database Extraction Guide](docs/DATABASE_EXTRACTION_GUIDE.md)** - Extracting data from databases
- **[Repository Download Guide](docs/REPOSITORY_DOWNLOAD_GUIDE.md)** - Automatically downloading GitHub repositories for code training

### Advanced Topics

- **[Control System Model](docs/CONTROL_SYSTEM_MODEL.md)** - Mathematical control system formulation
- **[Optimizations](docs/OPTIMIZATIONS.md)** - Performance optimizations
- **[Retraining Guide](docs/RETRAINING_GUIDE.md)** - How to retrain models

---

## Common Questions

### Getting Started

**Q: How do I get started with this project?**  
**A:** See [Complete Guide](docs/COMPLETE_GUIDE.md) - Quick Start section

**Q: What do I need to install?**  
**A:** See [Complete Guide](docs/COMPLETE_GUIDE.md) - Installation section

**Q: How do I train my first model?**  
**A:** See [Complete Guide](docs/COMPLETE_GUIDE.md) - Usage section

### Understanding Concepts

**Q: What are embeddings?**  
**A:** See [Embeddings Explained](docs/EMBEDDINGS_EXPLAINED.md)

**Q: How does attention work?**  
**A:** See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

**Q: What is a feed-forward network?**  
**A:** See [Feed-Forward Explained](docs/FEED_FORWARD_EXPLAINED.md)

**Q: Why do we need normalization?**  
**A:** See [Normalization Explained](docs/NORMALIZATION_EXPLAINED.md)

**Q: How do neural networks work?**  
**A:** See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Q: What is a neuron and what are weights?**  
**A:** See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

### Training Questions

**Q: What is training and why do we need it?**  
**A:** See [Training Explained](docs/TRAINING_EXPLAINED.md)

**Q: Why do we need data for training?**  
**A:** See [Training Explained](docs/TRAINING_EXPLAINED.md) - Why Do We Need Data section

**Q: Why is more data better?**  
**A:** See [Training Explained](docs/TRAINING_EXPLAINED.md) - Why More Data is Better section

**Q: How does the optimizer work?**  
**A:** See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Q: What is learning rate scheduling?**  
**A:** See [Scheduling Explained](docs/SCHEDULING_EXPLAINED.md)

### Data Questions

**Q: How does data processing work?**  
**A:** See [Data Processing Explained](docs/DATA_PROCESSING_EXPLAINED.md)

**Q: Can I train on PDFs?**  
**A:** See [Multi-Format Data Guide](docs/MULTI_FORMAT_DATA_GUIDE.md)

**Q: Can I train on images?**  
**A:** See [Multi-Format Data Guide](docs/MULTI_FORMAT_DATA_GUIDE.md)

**Q: How do I process different file types?**  
**A:** See [Data Processing Explained](docs/DATA_PROCESSING_EXPLAINED.md)

**Q: How do I download code repositories automatically?**  
**A:** See [Repository Download Guide](docs/REPOSITORY_DOWNLOAD_GUIDE.md)

### Generation Questions

**Q: How does text generation work?**  
**A:** See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Q: What is temperature in generation?**  
**A:** See [Generation Explained](docs/GENERATION_EXPLAINED.md) - Temperature section

**Q: What is top-k and top-p sampling?**  
**A:** See [Generation Explained](docs/GENERATION_EXPLAINED.md) - Top-k and Top-p sections

### Mathematical Questions

**Q: What are the mathematical foundations?**  
**A:** See [Mathematics](docs/MATHEMATICS.md) or [Complete Guide](docs/COMPLETE_GUIDE.md) - Mathematical Foundations section

**Q: How do I understand the complete mathematical model?**  
**A:** See [Mathematics](docs/MATHEMATICS.md) for step-by-step derivations

**Q: Is there a control system perspective?**  
**A:** See [Control System Model](docs/CONTROL_SYSTEM_MODEL.md)

### Architecture Questions

**Q: How is the architecture designed?**  
**A:** See [Architecture](docs/ARCHITECTURE.md)

**Q: What is the complete system flow?**  
**A:** See [Complete Guide](docs/COMPLETE_GUIDE.md) - Architecture Explained section

### Advanced Questions

**Q: How do I optimize inference?**  
**A:** See [Optimizations](docs/OPTIMIZATIONS.md)

**Q: How do I retrain a model?**  
**A:** See [Retraining Guide](docs/RETRAINING_GUIDE.md)

**Q: How do I extract data from databases?**  
**A:** See [Database Extraction Guide](docs/DATABASE_EXTRACTION_GUIDE.md)

**Q: How do I download GitHub repositories for code training?**  
**A:** See [Repository Download Guide](docs/REPOSITORY_DOWNLOAD_GUIDE.md)

---

## Glossary

### A

**AdamW** - Advanced optimizer combining adaptive learning rates with weight decay. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Attention** - Mechanism that determines how much each word should consider other words. See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

**Autoregressive** - Generation method where the model uses its own previous outputs as inputs. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

### B

**Batch** - Small group of examples processed together during training. See [Training Explained](docs/TRAINING_EXPLAINED.md)

**Bias** - Constant added to weighted sum in neural networks. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Backpropagation** - Algorithm for computing gradients through the network. See [Training Explained](docs/TRAINING_EXPLAINED.md)

### C

**Causal Masking** - Prevents tokens from attending to future tokens. See [Complete Guide](docs/COMPLETE_GUIDE.md)

**Cosine Annealing** - Learning rate schedule that follows a cosine curve. See [Scheduling Explained](docs/SCHEDULING_EXPLAINED.md)

**Cross-Entropy Loss** - Loss function for classification tasks. See [Mathematics](docs/MATHEMATICS.md)

### D

**Data Processing** - Transformation of raw files into training-ready text. See [Data Processing Explained](docs/DATA_PROCESSING_EXPLAINED.md)

**Dropout** - Regularization technique that randomly sets activations to zero. See [Complete Guide](docs/COMPLETE_GUIDE.md)

**Decoder** - Part of transformer that generates output. See [Architecture](docs/ARCHITECTURE.md)

### E

**Embedding** - Numerical representation of words/tokens. See [Embeddings Explained](docs/EMBEDDINGS_EXPLAINED.md)

**Epoch** - One complete pass through the training data. See [Training Explained](docs/TRAINING_EXPLAINED.md)

**Evaluation** - Process of measuring model performance. See [Training Explained](docs/TRAINING_EXPLAINED.md)

### F

**Feed-Forward Network (FFN)** - Two-layer neural network that transforms features. See [Feed-Forward Explained](docs/FEED_FORWARD_EXPLAINED.md)

**Forward Pass** - Computing predictions from inputs through the model. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

### G

**GELU** - Gaussian Error Linear Unit activation function. See [Feed-Forward Explained](docs/FEED_FORWARD_EXPLAINED.md)

**Generation** - Process of creating new text from a trained model. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Gradient** - Derivative of loss with respect to parameters. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Gradient Clipping** - Technique to prevent exploding gradients. See [Complete Guide](docs/COMPLETE_GUIDE.md)

**Gradient Descent** - Basic optimization algorithm. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

### H

**Hidden State** - Intermediate representation in the model. See [Architecture](docs/ARCHITECTURE.md)

### L

**Layer Normalization** - Normalization technique applied per layer. See [Normalization Explained](docs/NORMALIZATION_EXPLAINED.md)

**Learning Rate** - Step size for weight updates. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Logits** - Raw scores before applying softmax. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Loss** - Measure of prediction error. See [Training Explained](docs/TRAINING_EXPLAINED.md)

### M

**Multi-Head Attention** - Attention mechanism with multiple parallel heads. See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

**Momentum** - Technique to accelerate gradient descent. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

### N

**Neural Network** - Computational model inspired by biological neurons. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Neuron** - Basic processing unit in neural networks. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Normalization** - Technique to standardize activations. See [Normalization Explained](docs/NORMALIZATION_EXPLAINED.md)

**Nucleus Sampling (Top-p)** - Sampling strategy keeping tokens with cumulative probability ≥ p. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

### O

**Optimization** - Process of finding optimal weights. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Optimizer** - Algorithm that updates model weights. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Overfitting** - Model memorizes training data but doesn't generalize. See [Training Explained](docs/TRAINING_EXPLAINED.md)

### P

**Perplexity** - Measure of model uncertainty (exp(loss)). See [Mathematics](docs/MATHEMATICS.md)

**Positional Encoding** - Adds position information to embeddings. See [Complete Guide](docs/COMPLETE_GUIDE.md)

**Pre-norm** - Architecture where normalization comes before sublayers. See [Architecture](docs/ARCHITECTURE.md)

**Probability Distribution** - Distribution over possible next tokens. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

### Q

**Query (Q)** - One of three representations in attention (what am I looking for?). See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

### R

**Residual Connection** - Skip connection that adds input to output. See [Architecture](docs/ARCHITECTURE.md)

### S

**Sampling** - Process of selecting a token from probability distribution. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Scheduling** - Adjusting learning rate during training. See [Scheduling Explained](docs/SCHEDULING_EXPLAINED.md)

**Self-Attention** - Attention mechanism where queries, keys, and values come from same input. See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

**Softmax** - Function that converts logits to probabilities. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

### T

**Temperature** - Parameter controlling randomness in sampling. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Token** - Basic unit of text (word or character). See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Tokenization** - Process of converting text to tokens. See [Data Processing Explained](docs/DATA_PROCESSING_EXPLAINED.md)

**Top-k Sampling** - Sampling strategy keeping only top k tokens. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Top-p Sampling** - Another name for nucleus sampling. See [Generation Explained](docs/GENERATION_EXPLAINED.md)

**Transformer** - Neural network architecture based on attention. See [Architecture](docs/ARCHITECTURE.md)

**Training** - Process of teaching model to make predictions. See [Training Explained](docs/TRAINING_EXPLAINED.md)

### V

**Value (V)** - One of three representations in attention (what information do I contain?). See [Attention Explained](docs/ATTENTION_EXPLAINED.md)

**Vocabulary** - Set of all possible tokens. See [Embeddings Explained](docs/EMBEDDINGS_EXPLAINED.md)

### W

**Weight** - Parameter in neural network that controls connection strength. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

**Weight Decay** - Regularization technique that penalizes large weights. See [Optimization Explained](docs/OPTIMIZATION_EXPLAINED.md)

**Weight Matrix** - Matrix containing all weights for a layer. See [Neural Network Explained](docs/NEURAL_NETWORK_EXPLAINED.md)

---

## Quick Links

- **Complete Documentation**: [docs/COMPLETE_GUIDE.md](docs/COMPLETE_GUIDE.md)
- **Mathematical Foundations**: [docs/MATHEMATICS.md](docs/MATHEMATICS.md)
- **System Architecture**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Control System Model**: [docs/CONTROL_SYSTEM_MODEL.md](docs/CONTROL_SYSTEM_MODEL.md)

---

## License

This project is available under a **dual license**:

### Apache 2.0 License (Research & Non-Commercial Use)

**Free for:**
- ✅ Academic research
- ✅ Educational purposes
- ✅ Personal projects
- ✅ Open source contributions
- ✅ Non-commercial use

**Terms:**
- Free to use, modify, and distribute
- Patent grant included
- Must include license and copyright notice
- Must state changes if modifying

### Commercial License (Commercial Use)

**Requires a commercial license for:**
- ⚠️ Commercial products or services
- ⚠️ SaaS applications
- ⚠️ Revenue-generating applications
- ⚠️ Internal business use (for profit-making entities)
- ⚠️ Any use that generates profit or revenue

**To obtain a commercial license:**
Contact: **carlos.gutierrez@carg.dev**  
Subject: Commercial License Inquiry - SheepOp

Please include:
- Intended use case
- Expected usage volume
- Company/Organization name
- Contact information

### Citation Requirement

**IMPORTANT:** If you use this software in academic research or publications, you **MUST cite** this work. This is a condition of use for academic purposes.

**Required Citation Format:**

BibTeX:
```bibtex
@software{sheepop2024,
  title = {SheepOp LLM: Transformer-based Language Model Implementation},
  author = {Gutierrez, Carlos},
  year = {2024},
  url = {https://github.com/[your-username]/sheepOp},
  version = {1.0}
}
```

Text format:
```
Carlos Gutierrez. (2024). SheepOp LLM: Transformer-based Language Model 
Implementation. https://github.com/[your-username]/sheepOp
```

**Note:** Citation is required for academic use. Failure to cite constitutes a violation of the terms of use.

See [LICENSE](LICENSE) or [LICENSE.txt](LICENSE.txt) for the full license text.

---

## Contact

**Carlos Gutierrez**  
Email: carlos.gutierrez@carg.dev

---

*This README serves as an index to the comprehensive documentation available in the `docs/` folder.*
