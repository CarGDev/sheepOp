"""
Main training script
"""
import torch
import argparse
from pathlib import Path
import sys
import os
import importlib.util

# Ensure current directory is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Explicitly import from local data module to avoid conflicts with stdlib 'data' module
# Python 3.12 has a standard library 'data' module that conflicts with our local data/
data_module_path = project_root / "data" / "__init__.py"
if not data_module_path.exists():
    # Try alternative paths
    alt_paths = [
        project_root / "data" / "__init__.py",
        Path("data") / "__init__.py",
        Path.cwd() / "data" / "__init__.py",
    ]
    
    found = False
    for alt_path in alt_paths:
        if alt_path.exists():
            data_module_path = alt_path
            found = True
            break
    
    if not found:
        error_msg = f"Could not find data module!\n"
        error_msg += f"  Searched:\n"
        error_msg += f"    - {project_root / 'data' / '__init__.py'}\n"
        error_msg += f"    - {Path('data') / '__init__.py'}\n"
        error_msg += f"    - {Path.cwd() / 'data' / '__init__.py'}\n"
        error_msg += f"  Current directory: {Path.cwd()}\n"
        error_msg += f"  Project root: {project_root}\n"
        error_msg += f"  Does data/ directory exist? {Path(project_root / 'data').exists()}\n"
        error_msg += f"\n  Please ensure you're running from the project root directory.\n"
        error_msg += f"  Try: cd <project_root> && python3 train.py ..."
        raise ImportError(error_msg)

spec = importlib.util.spec_from_file_location("sheepop_data", data_module_path)
sheepop_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheepop_data)

# Import from the explicitly loaded module
SimpleTokenizer = sheepop_data.SimpleTokenizer
create_dataloader = sheepop_data.create_dataloader
DataProcessor = sheepop_data.DataProcessor
extract_text_from_directory = sheepop_data.extract_text_from_directory

from models import TransformerModel
from training import Trainer
from config import Config, get_default_config
from dataclasses import asdict


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description='Train SheepOp LLM')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output', type=str, default='./checkpoints', help='Output directory')
    parser.add_argument('--resume', type=str, help='Path to checkpoint to resume from')
    parser.add_argument('--max-files', type=int, default=None, help='Maximum number of files to process (None = all)')
    parser.add_argument('--data-workers', type=int, default=0, help='Number of parallel workers for data processing (0 = sequential, -1 = auto)')
    parser.add_argument('--skip-images', action='store_true', help='Skip image files (faster processing)')
    parser.add_argument('--skip-pdfs', action='store_true', help='Skip PDF files (faster processing)')
    parser.add_argument('--no-ocr', action='store_true', help='Disable OCR for images')
    parser.add_argument('--no-pdf-extraction', action='store_true', help='Disable PDF text extraction')
    
    # Auto-detect best device
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        default_device = 'mps'
    elif torch.cuda.is_available():
        default_device = 'cuda'
    else:
        default_device = 'cpu'
    
    parser.add_argument('--device', type=str, default=default_device, 
                       help=f'Device to use (default: {default_device})')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config:
        config = Config.from_json(args.config)
    else:
        config = get_default_config()
    
    config.device = args.device
    config.training.save_dir = args.output
    
    # Set seed
    set_seed(config.seed)
    
    # Setup device with smart detection
    if config.device == 'cuda' and torch.cuda.is_available():
        device = torch.device('cuda')
    elif config.device == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device('mps')
    elif config.device == 'auto':
        # Auto-detect best available device
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = torch.device('mps')
            print("Auto-detected MPS (Apple Silicon GPU)")
        else:
            device = torch.device('cpu')
            print("Auto-detected CPU")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Load data - supports multiple file types (PDFs, images, code, text, etc.)
    data_path = Path(args.data)
    texts = []
    
    if data_path.is_file():
        # Single file - try to process it
        print(f"Processing single file: {data_path}")
        processor = DataProcessor()
        texts = list(processor.process_file(data_path))
    elif data_path.is_dir():
        # Directory - process all supported file types
        print(f"Processing directory: {data_path}")
        print("Supported file types:")
        print("  - Text files: .txt, .md, .rst, .log, .csv, .json, .jsonl, .xml, .html, .htm")
        print("  - Code files: .py, .js, .ts, .java, .cpp, .c, .go, .rs, .rb, .php, .swift, etc.")
        print("  - PDF files: .pdf (requires PyPDF2 or pdfplumber)")
        print("  - Images: .png, .jpg, .jpeg, .gif, .bmp, .tiff (requires pytesseract for OCR)")
        print()
        
        # Process directory with all file types
        try:
            texts = extract_text_from_directory(
                directory=data_path,
                recursive=True,
                use_ocr=not args.no_ocr,  # Enable OCR for images unless disabled
                use_pdf_extraction=not args.no_pdf_extraction,  # Enable PDF extraction unless disabled
                min_length=10,  # Minimum length for text lines
                max_files=args.max_files,  # Limit number of files if specified
                num_workers=args.data_workers,  # Parallel processing workers
                skip_images=args.skip_images,  # Skip images entirely
                skip_pdfs=args.skip_pdfs,  # Skip PDFs entirely
            )
        except KeyboardInterrupt:
            print("\n\n⚠️  Data processing interrupted by user (Ctrl+C).")
            print("   Note: No checkpoint is saved because training hasn't started yet.")
            print("   Checkpoints are only saved during training, not during data extraction.")
            print("   Please run the training command again to retry.")
            raise
    else:
        raise ValueError(f"Data path {args.data} does not exist")
    
    if not texts:
        raise ValueError(f"No text data extracted from {args.data}. Please check that the directory contains supported file types.")
    
    print(f"\n✅ Successfully loaded {len(texts):,} text samples from {data_path}")
    print(f"   Sample preview (first 3 lines):")
    for i, text in enumerate(texts[:3]):
        preview = text[:80] + "..." if len(text) > 80 else text
        print(f"   {i+1}. {preview}")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create data loaders
    train_loader = create_dataloader(
        texts=texts,
        tokenizer=tokenizer,
        batch_size=config.training.batch_size,
        max_length=config.data.max_length,
        shuffle=True,
        num_workers=config.data.num_workers,
    )
    
    # Create model
    model_config = config.model
    model_config.vocab_size = tokenizer.vocab_size
    
    # Resume from checkpoint if provided
    start_epoch = 0
    checkpoint = None
    if args.resume:
        checkpoint_path = Path(args.resume)
        if not checkpoint_path.exists():
            print(f"⚠️  Warning: Checkpoint file '{args.resume}' not found!")
            print(f"   Starting fresh training instead...")
            args.resume = None  # Disable resume flag
        else:
            print(f"Resuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            
            # Load model config from checkpoint if available
            if 'model_config' in checkpoint:
                checkpoint_config = checkpoint['model_config']
                model_config.vocab_size = checkpoint_config.get('vocab_size', model_config.vocab_size)
                print(f"Loaded model config from checkpoint")
            
            model = TransformerModel(**asdict(model_config))
            model.load_state_dict(checkpoint['model_state_dict'])
            start_epoch = checkpoint.get('epoch', 0) + 1  # Start from next epoch
            print(f"Resuming from epoch {start_epoch}")
    
    if not args.resume:
        model = TransformerModel(**asdict(model_config))
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Load optimizer state if resuming
    if args.resume:
        if 'optimizer_state_dict' in checkpoint:
            # Move optimizer state to correct device
            optimizer_state = checkpoint['optimizer_state_dict']
            for state in optimizer_state['state'].values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
            optimizer.load_state_dict(optimizer_state)
            print("Loaded optimizer state from checkpoint")
    
    # Setup scheduler
    total_steps = len(train_loader) * config.training.max_epochs
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
    )
    
    # Load scheduler state if resuming
    if args.resume:
        if 'scheduler_state_dict' in checkpoint and scheduler is not None:
            # Scheduler state usually doesn't need device transfer, but let's be safe
            scheduler_state = checkpoint['scheduler_state_dict']
            scheduler.load_state_dict(scheduler_state)
            print("Loaded scheduler state from checkpoint")
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=None,  # Can add validation loader
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        max_epochs=config.training.max_epochs,
        gradient_accumulation_steps=config.training.gradient_accumulation_steps,
        max_grad_norm=config.training.max_grad_norm,
        use_amp=config.training.use_amp,
        save_dir=config.training.save_dir,
        log_interval=config.training.log_interval,
        eval_interval=config.training.eval_interval,
    )
    
    # Set trainer state if resuming
    if args.resume:
        trainer.current_epoch = start_epoch - 1
        trainer.global_step = checkpoint.get('global_step', 0)
        trainer.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from global step {trainer.global_step}")
    
    # Store model config for checkpoint saving
    model_config_dict = asdict(model_config)
    
    # Override save_checkpoint to include model config
    original_save_checkpoint = trainer.save_checkpoint
    def save_checkpoint_with_config(is_best=False):
        original_save_checkpoint(is_best=is_best, model_config=model_config_dict)
    trainer.save_checkpoint = save_checkpoint_with_config
    
    # Train
    trainer.train()
    
    print("Training completed!")


if __name__ == '__main__':
    from dataclasses import asdict
    main()

