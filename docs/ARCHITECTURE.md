# SheepOp LLM - Complete Architecture Documentation

Complete documentation of the SheepOp Language Model project architecture, data flow, training pipeline, and inference system.

## Table of Contents

1. [System Overview](#system-overview)
2. [Data Ingestion Pipeline](#data-ingestion-pipeline)
3. [Training Pipeline](#training-pipeline)
4. [Model Architecture](#model-architecture)
5. [Inference Pipeline](#inference-pipeline)
6. [Complete Workflow](#complete-workflow)

---

## System Overview

```mermaid
graph TB
    subgraph "Data Sources"
        A[PDF Files] --> DataProcessor
        B[Images - PNG/JPG/etc] --> DataProcessor
        C[Code Files - .py/.js/etc] --> DataProcessor
        D[Text Files - .txt/.md/etc] --> DataProcessor
    end
    
    DataProcessor[DataProcessor<br/>Multi-Format Extractor] --> TextList[Text Lines]
    
    TextList --> Tokenizer[SimpleTokenizer<br/>Character-Level]
    Tokenizer --> DataLoader[PyTorch DataLoader<br/>Batched Sequences]
    
    DataLoader --> Trainer[Trainer<br/>Training Loop]
    
    subgraph "Training Components"
        Trainer --> Model[TransformerModel]
        Trainer --> Optimizer[AdamW Optimizer]
        Trainer --> Scheduler[CosineAnnealingLR]
        Trainer --> Loss[CrossEntropyLoss]
    end
    
    Model --> Checkpoint[Model Checkpoints<br/>checkpoints/*.pt]
    
    Checkpoint --> Inference[Inference Script]
    Inference --> GeneratedText[Generated Text]
    
    style DataProcessor fill:#e1f5ff
    style Model fill:#fff4e1
    style Trainer fill:#ffe1f5
    style Checkpoint fill:#e1ffe1
```

---

## Data Ingestion Pipeline

### Multi-Format Data Processing Flow

```mermaid
flowchart TD
    Start([Start:<br/>train.py --data path]) --> CheckPath{Path Type?}
    
    CheckPath -->|File| SingleFile[Process Single File]
    CheckPath -->|Directory| Directory[Process Directory]
    
    SingleFile --> DataProcessor[DataProcessor.process_file]
    Directory --> RecursiveScan[Recursive Directory Scan<br/>Find all files]
    
    RecursiveScan --> FileType{File Extension?}
    
    FileType -->|.txt/.md/.json/etc| TextExtract[Read as Text File<br/>Line by line]
    FileType -->|.py/.js/.java/etc| CodeExtract[Read as Code File<br/>Line by line]
    FileType -->|.pdf| PDFExtract[PDF Extraction<br/>PyPDF2/pdfplumber]
    FileType -->|.png/.jpg/.tiff/etc| ImageExtract[OCR Extraction<br/>pytesseract]
    FileType -->|Unknown| Fallback[Try Text Fallback]
    
    PDFExtract --> PDFPages[Extract Each Page]
    PDFPages --> PDFLines[Split into Lines]
    
    ImageExtract --> OCR[Perform OCR<br/>pytesseract]
    OCR --> OCRLines[Split OCR Text into Lines]
    
    TextExtract --> FilterLines[Filter Lines<br/>min_length=10]
    CodeExtract --> FilterLines
    PDFLines --> FilterLines
    OCRLines --> FilterLines
    Fallback --> FilterLines
    
    FilterLines --> Combine[List of Text Lines]
    Combine --> Validate{Texts Empty?}
    
    Validate -->|Yes| Error[Raise Error:<br/>No data extracted]
    Validate -->|No| Success[✅ Success<br/>N text samples loaded]
    
    Success --> TokenizerStep[Next: Tokenization]
    
    style DataProcessor fill:#e1f5ff
    style PDFExtract fill:#ffe1f5
    style ImageExtract fill:#fff4e1
    style Success fill:#e1ffe1
    style Error fill:#ffe1e1
```

### Data Processing Components

```mermaid
classDiagram
    class DataProcessor {
        +process_file(file_path) Iterator[str]
        +process_directory(directory) Iterator[str]
        +process_to_list(...) List[str]
        -_process_text_file() Iterator[str]
        -_process_code_file() Iterator[str]
        -_process_pdf() Iterator[str]
        -_process_image() Iterator[str]
        -_check_dependencies()
    }
    
    class SimpleTokenizer {
        +vocab: Dict[str, int]
        +inv_vocab: Dict[int, str]
        +vocab_size: int
        +encode(text: str) List[int]
        +decode(token_ids: List[int]) str
        +save_vocab(path: str)
    }
    
    class TextDataset {
        +texts: List[str]
        +tokenizer: SimpleTokenizer
        +max_length: int
        +sequences: List[torch.Tensor]
        +__getitem__(idx) Dict
        +_prepare_sequences() List[Tensor]
    }
    
    class DataLoader {
        +batch_size: int
        +shuffle: bool
        +num_workers: int
        +collate_fn: Callable
    }
    
    DataProcessor --> TextDataset : extracts text
    SimpleTokenizer --> TextDataset : tokenizes
    TextDataset --> DataLoader : creates dataset
```

---

## Training Pipeline

### Complete Training Flow

```mermaid
flowchart TD
    Start([python train.py<br/>--data path]) --> Args[Parse Arguments<br/>--data, --config, --resume, --device]
    
    Args --> ConfigLoad{Config File<br/>Provided?}
    ConfigLoad -->|Yes| LoadConfig[Load config.json]
    ConfigLoad -->|No| DefaultConfig[Use Default Config]
    
    LoadConfig --> Config[Config Object<br/>ModelConfig<br/>TrainingConfig<br/>DataConfig<br/>seed=42]
    DefaultConfig --> Config
    
    Config --> SetSeed[Set Random Seed<br/>torch.manual_seed<br/>torch.cuda.manual_seed_all<br/>CUDNN deterministic]
    
    SetSeed --> Device[Detect Device<br/>CUDA/MPS/CPU]
    
    Device --> DataIngestion[Data Ingestion Pipeline<br/>Extract text from all files]
    
    DataIngestion --> TextList[List of Text Lines<br/>N samples]
    
    TextList --> CreateTokenizer[Create SimpleTokenizer<br/>Character-level vocab]
    
    CreateTokenizer --> Tokenizer[Tokenizer Ready<br/>vocab_size calculated]
    
    Tokenizer --> CreateDataLoader[Create DataLoader<br/>Batch size<br/>Max length<br/>Shuffle]
    
    CreateDataLoader --> TrainLoader[PyTorch DataLoader<br/>Batched sequences]
    
    TrainLoader --> CheckResume{Resume<br/>Checkpoint?}
    
    CheckResume -->|Yes| LoadCheckpoint[Load Checkpoint<br/>Model state<br/>Optimizer state<br/>Scheduler state<br/>Epoch/Step]
    CheckResume -->|No| CreateModel[Create New Model<br/>TransformerModel]
    
    LoadCheckpoint --> CreateModel
    
    CreateModel --> Model[Model Ready<br/>N parameters]
    
    Model --> CreateOptimizer[Create Optimizer<br/>AdamW<br/>lr, weight_decay]
    
    CreateOptimizer --> CreateScheduler[Create Scheduler<br/>CosineAnnealingLR<br/>T_max=total_steps]
    
    CreateScheduler --> CreateTrainer[Create Trainer<br/>Model<br/>DataLoader<br/>Optimizer<br/>Scheduler<br/>Device]
    
    CreateTrainer --> Trainer[Trainer Ready]
    
    Trainer --> TrainingLoop[Training Loop<br/>For each epoch]
    
    TrainingLoop --> EpochLoop[For each batch]
    
    EpochLoop --> Forward[Forward Pass<br/>Model prediction]
    
    Forward --> Loss[Compute Loss<br/>CrossEntropyLoss]
    
    Loss --> Backward[Backward Pass<br/>Compute gradients]
    
    Backward --> GradientAccum{Gradient<br/>Accumulation?}
    
    GradientAccum -->|Not yet| Accumulate[Accumulate gradients]
    Accumulate --> EpochLoop
    
    GradientAccum -->|Ready| ClipGrad[Gradient Clipping<br/>max_grad_norm]
    
    ClipGrad --> Update[Update Weights<br/>Optimizer.step]
    
    Update --> UpdateLR[Update Learning Rate<br/>Scheduler.step]
    
    UpdateLR --> ZeroGrad[Zero Gradients]
    
    ZeroGrad --> Log{Log Interval?}
    
    Log -->|Yes| LogMetrics[Log Metrics<br/>Loss, LR<br/>Save to metrics.json]
    Log -->|No| EvalCheck{Evaluation<br/>Interval?}
    
    LogMetrics --> EvalCheck
    
    EvalCheck -->|Yes| Evaluate[Evaluate on<br/>Validation Set]
    EvalCheck -->|No| SaveCheck{End of<br/>Epoch?}
    
    Evaluate --> SaveCheck
    
    SaveCheck -->|No| EpochLoop
    SaveCheck -->|Yes| SaveCheckpoint[Save Checkpoint<br/>Model state<br/>Optimizer state<br/>Scheduler state<br/>Epoch/Step]
    
    SaveCheckpoint --> MoreEpochs{More<br/>Epochs?}
    
    MoreEpochs -->|Yes| TrainingLoop
    MoreEpochs -->|No| GeneratePlots[Generate Training Plots<br/>loss_by_epoch.png<br/>training_curve.png]
    
    GeneratePlots --> End([Training Complete!<br/>Checkpoints saved])
    
    style SetSeed fill:#ffe1f5
    style DataIngestion fill:#e1f5ff
    style Model fill:#fff4e1
    style TrainingLoop fill:#ffe1f5
    style End fill:#e1ffe1
```

### Seed Initialization Details

```mermaid
sequenceDiagram
    participant TrainScript as train.py
    participant Config as Config
    participant PyTorch as PyTorch
    participant CUDA as CUDA Backend
    
    TrainScript->>Config: Load config (seed=42)
    TrainScript->>PyTorch: torch.manual_seed(42)
    TrainScript->>CUDA: torch.cuda.manual_seed_all(42)
    TrainScript->>PyTorch: torch.backends.cudnn.deterministic = True
    TrainScript->>PyTorch: torch.backends.cudnn.benchmark = False
    
    Note over TrainScript,CUDA: Seed ensures reproducibility<br/>across runs and devices
```

### Training Loop Details

```mermaid
graph LR
    subgraph "Single Training Step"
        A[Batch Input<br/>input_ids, labels] --> B[Forward Pass<br/>Model forward]
        B --> C[Logits<br/>batch_size × seq_len × vocab_size]
        C --> D[Compute Loss<br/>CrossEntropyLoss]
        D --> E[Backward Pass<br/>Compute gradients]
        E --> F{Gradient<br/>Accumulation<br/>Steps reached?}
        F -->|No| G[Accumulate Gradients]
        F -->|Yes| H[Gradient Clipping]
        H --> I[Optimizer Step<br/>Update weights]
        I --> J[Scheduler Step<br/>Update LR]
        J --> K[Zero Gradients]
        K --> L[Log Metrics]
    end
    
    G --> A
    
    style B fill:#e1f5ff
    style D fill:#ffe1f5
    style I fill:#fff4e1
```

---

## Model Architecture

### Transformer Model Structure

```mermaid
graph TB
    Input[Input Tokens<br/>Token IDs] --> Embed[Token Embedding<br/>vocab_size → d_model]
    
    Embed --> PosEnc[Positional Encoding<br/>Sinusoidal/Cosine]
    
    PosEnc --> Dropout1[Dropout]
    
    Dropout1 --> Layer1[Transformer Block 1]
    Layer1 --> Layer2[Transformer Block 2]
    Layer2 --> Layer3[Transformer Block 3]
    Layer3 --> LayerN[Transformer Block N<br/>num_layers]
    
    LayerN --> LayerNorm[Final Layer Norm]
    
    LayerNorm --> OutputProj[Output Projection<br/>d_model → vocab_size]
    
    OutputProj --> Logits[Logits<br/>batch × seq_len × vocab_size]
    
    subgraph "Transformer Block Details"
        TBInput[Input x] --> Attention[Multi-Head<br/>Self-Attention]
        Attention --> AddNorm1[Add & Norm<br/>Residual + LayerNorm]
        AddNorm1 --> FFN[Feed-Forward<br/>Network]
        FFN --> AddNorm2[Add & Norm<br/>Residual + LayerNorm]
        AddNorm2 --> TBOutput[Output]
    end
    
    style Embed fill:#e1f5ff
    style Attention fill:#ffe1f5
    style FFN fill:#fff4e1
    style Logits fill:#e1ffe1
```

### Multi-Head Attention Mechanism

```mermaid
graph LR
    Input[Input<br/>batch × seq_len × d_model] --> Q[Query<br/>Linear Layer]
    Input --> K[Key<br/>Linear Layer]
    Input --> V[Value<br/>Linear Layer]
    
    Q --> SplitQ[Split into<br/>num_heads heads]
    K --> SplitK[Split into<br/>num_heads heads]
    V --> SplitV[Split into<br/>num_heads heads]
    
    SplitQ --> ScaledDot[Scaled Dot-Product<br/>Attention]
    SplitK --> ScaledDot
    SplitV --> ScaledDot
    
    ScaledDot --> Mask[Causal Mask<br/>Lower triangular]
    
    Mask --> Softmax[Softmax]
    
    Softmax --> AttentionOutput[Attention Output<br/>per head]
    
    AttentionOutput --> Concat[Concat Heads]
    
    Concat --> OutputProj[Output Projection<br/>Linear Layer]
    
    OutputProj --> Output[Output<br/>batch × seq_len × d_model]
    
    style ScaledDot fill:#ffe1f5
    style Mask fill:#fff4e1
    style Output fill:#e1ffe1
```

### Complete Model Component Diagram

```mermaid
classDiagram
    class TransformerModel {
        +vocab_size: int
        +d_model: int
        +num_layers: int
        +num_heads: int
        +token_embedding: Embedding
        +pos_encoding: PositionalEncoding
        +layers: ModuleList[TransformerBlock]
        +final_norm: LayerNorm
        +output_proj: Linear
        +forward(input_ids) Tuple[Tensor, Tensor]
        +generate(...) Tensor
        +get_num_params() int
    }
    
    class TransformerBlock {
        +attention: MultiHeadAttention
        +ffn: FeedForward
        +norm1: LayerNorm
        +norm2: LayerNorm
        +dropout: Dropout
        +forward(x, mask) Tensor
    }
    
    class MultiHeadAttention {
        +num_heads: int
        +d_model: int
        +d_k: int
        +q_proj: Linear
        +k_proj: Linear
        +v_proj: Linear
        +out_proj: Linear
        +forward(q, k, v, mask) Tensor
    }
    
    class FeedForward {
        +linear1: Linear
        +linear2: Linear
        +activation: GELU/ReLU
        +dropout: Dropout
        +forward(x) Tensor
    }
    
    class PositionalEncoding {
        +d_model: int
        +max_len: int
        +pe: Tensor
        +forward(x) Tensor
    }
    
    TransformerModel --> TransformerBlock : contains N layers
    TransformerModel --> PositionalEncoding : adds positional info
    TransformerBlock --> MultiHeadAttention : self-attention
    TransformerBlock --> FeedForward : feed-forward network
```

---

## Inference Pipeline

### Text Generation Flow

```mermaid
flowchart TD
    Start([python inference.py<br/>--checkpoint path<br/>--prompt text]) --> LoadModel[Load Model from Checkpoint<br/>Load state dict<br/>Set to eval mode]
    
    LoadModel --> CreateTokenizer[Create Tokenizer<br/>SimpleTokenizer]
    
    CreateTokenizer --> EncodePrompt[Encode Prompt<br/>Text → Token IDs]
    
    EncodePrompt --> CheckOptimized{Use Optimized<br/>Inference?}
    
    CheckOptimized -->|Yes| OptimizedGen[OptimizedInference<br/>with KV Caching]
    CheckOptimized -->|No| StandardGen[Standard Generation]
    
    StandardGen --> InitGen[Initialize Generation<br/>generated = input_ids]
    
    InitGen --> LoopStart[Generation Loop<br/>For max_length steps]
    
    LoopStart --> Forward[Forward Pass<br/>Model prediction]
    
    Forward --> NextToken[Get Next Token Logits<br/>Last position]
    
    NextToken --> Temperature[Apply Temperature<br/>Scale logits]
    
    Temperature --> TopK{Top-K<br/>Filtering?}
    
    TopK -->|Yes| FilterK[Filter Top-K Tokens]
    TopK -->|No| TopP{Top-P<br/>Nucleus Sampling?}
    
    FilterK --> TopP
    
    TopP -->|Yes| FilterP[Filter by Cumulative Prob]
    TopP -->|No| Sample[Sample Token<br/>Multinomial]
    
    FilterP --> Sample
    
    Sample --> Append[Append Token<br/>to Generated]
    
    Append --> CheckStop{Stop<br/>Condition?}
    
    CheckStop -->|No| LoopStart
    CheckStop -->|Yes| Decode[Decode Tokens<br/>Token IDs → Text]
    
    OptimizedGen --> KVCache[Use KV Cache<br/>Cache previous KV]
    KVCache --> LoopStart
    
    Decode --> Output[Generated Text<br/>Output]
    
    Output --> End([End])
    
    style OptimizedGen fill:#e1f5ff
    style Forward fill:#ffe1f5
    style Sample fill:#fff4e1
    style Output fill:#e1ffe1
```

### Optimized Inference with KV Caching

```mermaid
graph TB
    subgraph "Standard Generation"
        A1[Input Token] --> B1[Forward Pass<br/>Compute Q, K, V]
        B1 --> C1[Attention<br/>Full Sequence]
        C1 --> D1[Next Token]
        D1 --> E1[Append Token]
        E1 --> A1
    end
    
    subgraph "Optimized Generation with KV Cache"
        A2[Input Token] --> B2{First<br/>Token?}
        B2 -->|Yes| C2[Forward Pass<br/>Compute Q, K, V]
        B2 -->|No| C2Cache[Use Cached K, V<br/>Only compute Q]
        C2 --> D2[Cache K, V]
        D2 --> E2[Attention<br/>Only with New Token]
        C2Cache --> E2
        E2 --> F2[Next Token]
        F2 --> G2[Append Token]
        G2 --> A2
    end
    
    style C2 fill:#e1f5ff
    style C2Cache fill:#ffe1f5
    style E2 fill:#e1ffe1
```

---

## Complete Workflow

### End-to-End System Flow

```mermaid
flowchart TB
    subgraph "Phase 1: Data Preparation"
        A1[Raw Data Files<br/>PDFs, Images, Code, Text] --> A2[DataProcessor<br/>Extract Text]
        A2 --> A3[Text Lines<br/>List of Strings]
        A3 --> A4[SimpleTokenizer<br/>Build Vocabulary]
        A4 --> A5[Tokenize & Chunk<br/>Create Sequences]
        A5 --> A6[DataLoader<br/>Batched Data]
    end
    
    subgraph "Phase 2: Model Initialization"
        B1[Load Config<br/>ModelConfig] --> B2[Set Random Seed<br/>seed=42]
        B2 --> B3[Create Model<br/>TransformerModel]
        B3 --> B4[Initialize Weights<br/>Normal Distribution]
        B4 --> B5[Create Optimizer<br/>AdamW]
        B5 --> B6[Create Scheduler<br/>CosineAnnealingLR]
    end
    
    subgraph "Phase 3: Training"
        C1[Trainer Setup] --> C2[Training Loop<br/>Epochs]
        C2 --> C3[Batch Loop]
        C3 --> C4[Forward Pass]
        C4 --> C5[Compute Loss]
        C5 --> C6[Backward Pass]
        C6 --> C7[Gradient Clipping]
        C7 --> C8[Update Weights]
        C8 --> C9[Save Checkpoint]
        C9 --> C10{More Epochs?}
        C10 -->|Yes| C2
        C10 -->|No| C11[Generate Plots<br/>Training Metrics]
    end
    
    subgraph "Phase 4: Inference"
        D1[Load Checkpoint] --> D2[Load Model State]
        D2 --> D3[Encode Prompt]
        D3 --> D4[Generate Text<br/>Autoregressive]
        D4 --> D5[Decode Tokens]
        D5 --> D6[Output Text]
    end
    
    A6 --> B1
    B6 --> C1
    C11 --> D1
    
    style A2 fill:#e1f5ff
    style B3 fill:#fff4e1
    style C4 fill:#ffe1f5
    style D4 fill:#e1ffe1
```

### Checkpoint Structure

```mermaid
graph TB
    Checkpoint[Checkpoint File<br/>checkpoint_epoch_N.pt] --> ModelState[model_state_dict<br/>Model weights]
    Checkpoint --> OptimizerState[optimizer_state_dict<br/>AdamW state]
    Checkpoint --> SchedulerState[scheduler_state_dict<br/>LR scheduler state]
    Checkpoint --> ModelConfig[model_config<br/>Model hyperparameters]
    Checkpoint --> Epoch[epoch<br/>Current epoch number]
    Checkpoint --> GlobalStep[global_step<br/>Training step count]
    Checkpoint --> BestValLoss[best_val_loss<br/>Best validation loss]
    
    ModelState --> Resume[Resume Training<br/>Restore model state]
    OptimizerState --> Resume
    SchedulerState --> Resume
    ModelConfig --> Resume
    Epoch --> Resume
    GlobalStep --> Resume
    
    style Checkpoint fill:#e1f5ff
    style Resume fill:#e1ffe1
```

### Configuration Hierarchy

```mermaid
graph TB
    Config[Config<br/>Root Configuration] --> ModelConfig[ModelConfig<br/>vocab_size<br/>d_model<br/>num_layers<br/>num_heads<br/>d_ff<br/>max_seq_len<br/>dropout<br/>activation]
    
    Config --> TrainingConfig[TrainingConfig<br/>batch_size<br/>max_epochs<br/>learning_rate<br/>weight_decay<br/>warmup_steps<br/>max_grad_norm<br/>gradient_accumulation_steps<br/>use_amp]
    
    Config --> DataConfig[DataConfig<br/>data_dir<br/>max_length<br/>stride<br/>num_workers]
    
    Config --> Global[Global Settings<br/>device<br/>seed]
    
    ModelConfig --> Model[TransformerModel<br/>Model Architecture]
    TrainingConfig --> Trainer[Trainer<br/>Training Parameters]
    DataConfig --> DataLoader[DataLoader<br/>Data Parameters]
    
    style Config fill:#e1f5ff
    style Model fill:#fff4e1
    style Trainer fill:#ffe1f5
    style DataLoader fill:#e1ffe1
```

---

## Key Components Summary

### 1. **Data Processing**
- **DataProcessor**: Multi-format text extraction (PDFs, images, code, text)
- **SimpleTokenizer**: Character-level tokenization
- **TextDataset**: PyTorch dataset for training
- **DataLoader**: Batched data loading

### 2. **Model Architecture**
- **TransformerModel**: Complete transformer language model
- **TransformerBlock**: Multi-head attention + feed-forward
- **MultiHeadAttention**: Scaled dot-product attention with causal masking
- **FeedForward**: Position-wise feed-forward network
- **PositionalEncoding**: Sinusoidal position embeddings

### 3. **Training**
- **Trainer**: Complete training loop with:
  - Gradient accumulation
  - Mixed precision training (AMP)
  - Gradient clipping
  - Learning rate scheduling
  - Checkpointing
  - Metrics tracking

### 4. **Inference**
- **Standard Generation**: Autoregressive text generation
- **OptimizedInference**: KV caching for faster generation
- **RetrievalCache**: Caching for RAG systems

### 5. **Configuration**
- **Config System**: Hierarchical configuration (Model, Training, Data)
- **JSON Support**: Save/load configurations
- **Default Values**: Sensible defaults for all parameters

---

## Usage Examples

### Training
```bash
# Basic training
python train.py --data /path/to/data

# With custom config
python train.py --data /path/to/data --config config.json

# Resume from checkpoint
python train.py --data /path/to/data --resume checkpoints/checkpoint_epoch_5.pt

# Specify device
python train.py --data /path/to/data --device cuda
```

### Inference
```bash
# Basic inference
python inference.py --checkpoint checkpoints/best_checkpoint.pt --prompt "Hello world"

# With sampling parameters
python inference.py \
    --checkpoint checkpoints/best_checkpoint.pt \
    --prompt "The future of AI" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.95

# Optimized inference
python inference.py \
    --checkpoint checkpoints/best_checkpoint.pt \
    --prompt "Hello" \
    --optimized
```

---

## File Structure

```
sheepOp/
├── train.py              # Main training script
├── inference.py          # Inference script
├── config.py            # Configuration management
├── config.json          # Configuration file
├── data/                # Data module (symlink)
│   └── __init__.py      # Tokenizer, DataLoader, DataProcessor
├── models/              # Model definitions
│   ├── transformer.py  # Main transformer model
│   ├── blocks.py        # Transformer blocks
│   ├── attention.py     # Attention mechanisms
│   └── optimized_attention.py  # Optimized inference
├── training/            # Training utilities
│   ├── __init__.py      # Trainer class
│   └── metrics.py       # Training metrics
├── checkpoints/         # Saved model checkpoints
└── requirements.txt     # Dependencies
```

---

## Flow Summary

1. **Data Ingestion**: Raw files → Text extraction → Text lines
2. **Tokenization**: Text lines → Token sequences → Batched data
3. **Model Setup**: Config → Model → Optimizer → Scheduler
4. **Training**: Batches → Forward → Loss → Backward → Update → Checkpoint
5. **Inference**: Checkpoint → Model → Prompt → Generate → Output

---

*This documentation provides a complete view of the SheepOp LLM project architecture and workflow.*

