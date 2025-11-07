"""
Data download utilities for training the language model
"""
import urllib.request
import gzip
import os
from pathlib import Path
from typing import Optional


def download_text_file(url: str, output_path: str, decompress: bool = False):
    """
    Download a text file from a URL.
    
    Args:
        url: URL to download from
        output_path: Path to save the file
        decompress: Whether to decompress gzip files
    """
    print(f"Downloading from {url}...")
    
    if decompress:
        # Download and decompress gzip file
        with urllib.request.urlopen(url) as response:
            with gzip.open(response, 'rt', encoding='utf-8') as f:
                content = f.read()
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(content)
    else:
        # Download regular file
        urllib.request.urlretrieve(url, output_path)
    
    print(f"Downloaded to {output_path}")


def download_wiki_text():
    """Download a small Wikipedia text dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = "data/wikitext_sample.txt"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        download_text_file(url, output_path)
        print(f"Successfully downloaded Wikipedia text sample to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def download_shakespeare():
    """Download Shakespeare text dataset."""
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = "data/shakespeare.txt"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        download_text_file(url, output_path)
        print(f"Successfully downloaded Shakespeare text to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def download_openwebtext_sample():
    """Download a sample from OpenWebText corpus."""
    # Using a smaller sample URL
    url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
    output_path = "data/openwebtext_sample.txt"
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    try:
        download_text_file(url, output_path)
        print(f"Successfully downloaded sample text to {output_path}")
        return output_path
    except Exception as e:
        print(f"Error downloading: {e}")
        return None


def create_sample_data(output_path: str = "data/sample_data.txt", num_samples: int = 100):
    """
    Create a sample data.txt file with generated text.
    
    Args:
        output_path: Path to save the file
        num_samples: Number of text samples to generate
    """
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming artificial intelligence.",
        "Natural language processing enables computers to understand text.",
        "Deep learning models can learn complex patterns from data.",
        "Transformers have revolutionized the field of NLP.",
        "Attention mechanisms allow models to focus on relevant information.",
        "Language models can generate coherent text.",
        "Neural networks are inspired by the human brain.",
        "Training a model requires large amounts of data.",
        "Gradient descent is used to optimize neural networks.",
        "Python is a popular programming language for machine learning.",
        "PyTorch is a flexible deep learning framework.",
        "The transformer architecture uses self-attention mechanisms.",
        "Tokenization converts text into numerical representations.",
        "Embeddings capture semantic meaning of words.",
        "The model learns to predict the next word in a sequence.",
        "Backpropagation computes gradients for training.",
        "Regularization techniques prevent overfitting.",
        "Cross-validation helps evaluate model performance.",
        "Hyperparameter tuning improves model accuracy.",
        "The training process iterates over multiple epochs.",
        "Batch processing speeds up training.",
        "GPU acceleration makes training faster.",
        "Checkpoints save model state during training.",
        "Evaluation metrics measure model quality.",
        "Perplexity measures how well a model predicts text.",
        "BLEU score evaluates translation quality.",
        "F1 score combines precision and recall.",
        "Accuracy measures correct predictions.",
        "Loss function quantifies prediction errors.",
        "Optimizers update model parameters.",
        "Learning rate controls training speed.",
        "Dropout prevents overfitting.",
        "Layer normalization stabilizes training.",
        "Residual connections help train deep networks.",
        "Multi-head attention captures different relationships.",
        "Positional encoding adds sequence information.",
        "Causal masking ensures autoregressive generation.",
        "Sampling strategies control text generation.",
        "Temperature scaling adjusts randomness.",
        "Top-k sampling limits vocabulary choices.",
        "Nucleus sampling uses cumulative probability.",
        "Beam search finds high-probability sequences.",
        "Greedy decoding selects highest probability tokens.",
        "The model architecture determines capabilities.",
        "Data quality affects model performance.",
        "Preprocessing cleans and formats data.",
        "Data augmentation increases training examples.",
        "Transfer learning uses pretrained models.",
        "Fine-tuning adapts models to specific tasks.",
        "Zero-shot learning requires no training examples.",
        "Few-shot learning uses few examples.",
        "In-context learning adapts during inference.",
        "Prompt engineering improves model outputs.",
        "Chain-of-thought reasoning breaks down problems.",
        "Self-consistency improves reliability.",
        "Ensemble methods combine multiple models.",
        "Model compression reduces size.",
        "Quantization reduces precision.",
        "Pruning removes unnecessary connections.",
        "Distillation transfers knowledge between models.",
        "The field of AI continues to evolve rapidly.",
        "Research pushes boundaries of what's possible.",
        "Open source enables collaboration.",
        "Reproducibility ensures scientific validity.",
        "Ethics guides responsible AI development.",
        "Bias detection identifies unfairness.",
        "Fairness metrics measure equity.",
        "Transparency enables understanding.",
        "Interpretability reveals model reasoning.",
        "Adversarial examples test robustness.",
        "Security protects against attacks.",
        "Privacy preserves user data.",
        "Federated learning protects privacy.",
        "Differential privacy adds noise.",
        "Homomorphic encryption enables computation.",
        "Blockchain provides decentralization.",
        "Cryptography ensures security.",
        "The future of AI is exciting.",
        "Technology empowers human potential.",
        "Innovation drives progress.",
        "Collaboration accelerates discovery.",
        "Education spreads knowledge.",
        "Understanding deepens appreciation.",
        "Curiosity fuels exploration.",
        "Experimentation leads to breakthroughs.",
        "Persistence overcomes challenges.",
        "Creativity inspires solutions.",
        "The journey of learning never ends.",
        "Every dataset tells a story.",
        "Patterns emerge from complexity.",
        "Simplicity reveals elegance.",
        "Understanding requires patience.",
        "Mastery comes from practice.",
        "Progress happens incrementally.",
        "Success builds on failures.",
        "Wisdom comes from experience.",
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for i in range(num_samples):
            # Cycle through sample texts
            text = sample_texts[i % len(sample_texts)]
            f.write(text + '\n')
    
    print(f"Created sample data file: {output_path} with {num_samples} samples")
    return output_path


def scrape_wikipedia_article(title: str, output_path: str = "data/wikipedia_article.txt"):
    """
    Download a Wikipedia article (requires wikipedia library).
    
    Args:
        title: Wikipedia article title
        output_path: Path to save the file
    """
    try:
        import wikipedia
        
        print(f"Downloading Wikipedia article: {title}")
        page = wikipedia.page(title)
        
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(page.content)
        
        print(f"Downloaded to {output_path}")
        return output_path
    except ImportError:
        print("Wikipedia library not installed. Install with: pip install wikipedia")
        return None
    except Exception as e:
        print(f"Error downloading Wikipedia article: {e}")
        return None


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Download or create training data')
    parser.add_argument('--type', type=str, choices=['shakespeare', 'sample', 'wiki', 'wikipedia'],
                        default='sample', help='Type of data to get')
    parser.add_argument('--output', type=str, help='Output file path')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for generated data')
    parser.add_argument('--title', type=str, help='Wikipedia article title')
    
    args = parser.parse_args()
    
    if args.type == 'shakespeare':
        output = download_shakespeare()
    elif args.type == 'sample':
        output_path = args.output or 'data/sample_data.txt'
        output = create_sample_data(output_path, args.samples)
    elif args.type == 'wiki':
        output = download_wiki_text()
    elif args.type == 'wikipedia':
        if not args.title:
            print("Error: --title required for Wikipedia download")
        else:
            output_path = args.output or f'data/wikipedia_{args.title.replace(" ", "_")}.txt'
            output = scrape_wikipedia_article(args.title, output_path)
    
    if output:
        print(f"\nData ready at: {output}")
        print(f"You can now train with: python train.py --data {output}")

