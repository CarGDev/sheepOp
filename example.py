"""
Example script demonstrating basic usage
"""
import torch
import sys
import importlib.util
from pathlib import Path

# Ensure current directory is in path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Explicitly import from local data module to avoid conflicts with stdlib 'data' module
data_module_path = project_root / "data" / "__init__.py"
spec = importlib.util.spec_from_file_location("sheepop_data", data_module_path)
sheepop_data = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sheepop_data)
SimpleTokenizer = sheepop_data.SimpleTokenizer

from models import TransformerModel


def example_model_creation():
    """Example of creating a model."""
    print("Creating model...")
    
    # Create tokenizer
    tokenizer = SimpleTokenizer()
    print(f"Vocabulary size: {tokenizer.vocab_size}")
    
    # Create model
    model = TransformerModel(
        vocab_size=tokenizer.vocab_size,
        d_model=256,
        num_layers=4,
        num_heads=4,
        d_ff=1024,
        max_seq_len=128,
    )
    
    print(f"Model created with {model.get_num_params():,} parameters")
    
    # Test forward pass
    input_ids = torch.randint(0, tokenizer.vocab_size, (2, 32))
    logits, _ = model(input_ids)
    print(f"Input shape: {input_ids.shape}")
    print(f"Output shape: {logits.shape}")
    
    return model, tokenizer


def example_generation(model, tokenizer):
    """Example of text generation."""
    print("\nGenerating text...")
    
    prompt = "Hello, world"
    print(f"Prompt: {prompt}")
    
    # Encode prompt
    input_ids = tokenizer.encode(prompt)
    input_ids = torch.tensor([input_ids])
    
    # Generate
    generated = model.generate(
        input_ids=input_ids,
        max_length=50,
        temperature=0.8,
        top_k=50,
        top_p=0.95,
        do_sample=True,
    )
    
    # Decode
    generated_text = tokenizer.decode(generated[0].tolist())
    print(f"Generated: {generated_text}")


if __name__ == '__main__':
    # Set random seed
    torch.manual_seed(42)
    
    # Create model
    model, tokenizer = example_model_creation()
    
    # Test generation
    example_generation(model, tokenizer)
    
    print("\nExample completed successfully!")


