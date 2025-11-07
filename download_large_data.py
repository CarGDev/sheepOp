#!/usr/bin/env python3
"""
Download large datasets for training the SheepOp LLM.
Supports Amazon Reviews, WikiText, OpenWebText, BookCorpus, and more.
"""
import argparse
import sys
from pathlib import Path
from typing import Optional


def download_amazon_reviews(output: str = "data/amazon_reviews.txt", limit: int = 500000, category: str = "Video_Games_v1_00"):
    """
    Download Amazon Product Reviews dataset.
    
    Args:
        output: Output file path
        limit: Maximum number of reviews to download
        category: Product category (Video_Games_v1_00, Books_v1_00, etc.)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading Amazon Product Reviews (category: {category}, limit: {limit})...")
    print("   This may take several minutes depending on your connection...")
    
    try:
        # Try different dataset names/approaches
        # Method 1: Try mc4 (Common Crawl) which includes Amazon-like content
        print("   Attempting to download from alternative source...")
        
        # Use amazon_polarity dataset (smaller but works)
        try:
            print("   Trying amazon_polarity dataset...")
            dataset = load_dataset("amazon_polarity", split=f"train[:{limit}]")
            
            with open(output, "w", encoding="utf-8") as f:
                count = 0
                for item in dataset:
                    review = item.get("content", "").strip()
                    if not review:
                        review = item.get("text", "").strip()
                    if review and len(review) > 20:
                        f.write(review + "\n")
                        count += 1
                        if count % 50000 == 0:
                            print(f"   ✓ Downloaded {count:,} reviews...")
            
            print(f"✅ Successfully saved {count:,} reviews to {output}")
            return True
            
        except Exception as e1:
            print(f"   amazon_polarity failed: {e1}")
            
            # Method 2: Use IMDB reviews (similar structure)
            try:
                print("   Trying IMDB reviews as alternative...")
                dataset = load_dataset("imdb", split=f"train[:{limit}]")
                
                with open(output, "w", encoding="utf-8") as f:
                    count = 0
                    for item in dataset:
                        review = item.get("text", "").strip()
                        if review and len(review) > 20:
                            f.write(review + "\n")
                            count += 1
                            if count % 50000 == 0:
                                print(f"   ✓ Downloaded {count:,} reviews...")
                
                print(f"✅ Successfully saved {count:,} reviews to {output}")
                print("   Note: Using IMDB reviews instead of Amazon reviews")
                return True
                
            except Exception as e2:
                print(f"   IMDB also failed: {e2}")
                raise Exception("Both Amazon and IMDB datasets failed. Try using --alternative flag with a different dataset.")
        
    except Exception as e:
        print(f"❌ Error downloading reviews: {e}")
        print("\n💡 Alternative options:")
        print("   1. Use WikiText instead: python3 download_large_data.py wiki")
        print("   2. Use OpenWebText: python3 download_large_data.py openwebtext --limit 100000")
        print("   3. Try downloading from HuggingFace Hub manually")
        return False


def download_wikitext(output: str = "data/wikitext.txt", version: str = "103"):
    """
    Download WikiText dataset (Wikipedia text).
    
    Args:
        output: Output file path
        version: WikiText version ('2' or '103')
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading WikiText-{version}...")
    print("   This may take several minutes...")
    
    try:
        dataset = load_dataset("wikitext", f"wikitext-{version}-v1", split="train")
        
        with open(output, "w", encoding="utf-8") as f:
            count = 0
            for item in dataset:
                text = item.get("text", "").strip()
                # Filter out headers and empty lines
                if text and len(text) > 20 and not text.startswith("="):
                    # Split into sentences
                    sentences = text.split('.')
                    for s in sentences:
                        s = s.strip()
                        if len(s) > 20:
                            f.write(s + ".\n")
                            count += 1
                            if count % 10000 == 0:
                                print(f"   ✓ Processed {count:,} sentences...")
        
        print(f"✅ Successfully saved {count:,} sentences to {output}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading WikiText: {e}")
        return False


def download_openwebtext(output: str = "data/openwebtext.txt", limit: int = 100000):
    """
    Download OpenWebText dataset (web text corpus).
    
    Args:
        output: Output file path
        limit: Maximum number of samples to download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading OpenWebText (limit: {limit:,})...")
    print("   This may take a while - OpenWebText is very large...")
    
    try:
        dataset = load_dataset("openwebtext", split=f"train[:{limit}]")
        
        with open(output, "w", encoding="utf-8") as f:
            count = 0
            for item in dataset:
                text = item.get("text", "").strip()
                if text:
                    # Split into sentences
                    sentences = text.split('.')
                    for s in sentences:
                        s = s.strip()
                        if len(s) > 20:
                            f.write(s + ".\n")
                            count += 1
                            if count % 10000 == 0:
                                print(f"   ✓ Processed {count:,} sentences...")
        
        print(f"✅ Successfully saved {count:,} sentences to {output}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading OpenWebText: {e}")
        return False


def download_bookcorpus(output: str = "data/bookcorpus.txt", limit: int = 100000):
    """
    Download BookCorpus dataset (books).
    
    Args:
        output: Output file path
        limit: Maximum number of books to download
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' library not installed.")
        print("Install with: pip install datasets")
        return False
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    print(f"📥 Downloading BookCorpus (limit: {limit:,} books)...")
    print("   This may take a while...")
    
    try:
        dataset = load_dataset("bookcorpus", split=f"train[:{limit}]")
        
        with open(output, "w", encoding="utf-8") as f:
            count = 0
            for item in dataset:
                text = item.get("text", "").strip()
                if text:
                    # Split into sentences
                    sentences = text.split('.')
                    for s in sentences:
                        s = s.strip()
                        if len(s) > 20:
                            f.write(s + ".\n")
                            count += 1
                            if count % 10000 == 0:
                                print(f"   ✓ Processed {count:,} sentences...")
        
        print(f"✅ Successfully saved {count:,} sentences to {output}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading BookCorpus: {e}")
        return False


def download_wikitext_direct(output: str = "data/wikitext_direct.txt"):
    """
    Download WikiText directly from URL (no HuggingFace required).
    """
    import urllib.request
    import zipfile
    import tempfile
    import os
    
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    
    url = "https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip"
    
    print("📥 Downloading WikiText-103 directly from URL...")
    print("   This may take several minutes...")
    
    try:
        # Download to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            tmp_path = tmp_file.name
            print(f"   Downloading to temporary file...")
            urllib.request.urlretrieve(url, tmp_path)
        
        # Extract and process
        print("   Extracting and processing...")
        with zipfile.ZipFile(tmp_path, 'r') as zip_ref:
            # Extract wiki.train.tokens
            with zip_ref.open('wikitext-103/wiki.train.tokens') as f:
                with open(output, 'w', encoding='utf-8') as out_file:
                    count = 0
                    for line in f:
                        line = line.decode('utf-8').strip()
                        if line and len(line) > 20 and not line.startswith('='):
                            sentences = line.split('.')
                            for s in sentences:
                                s = s.strip()
                                if len(s) > 20:
                                    out_file.write(s + ".\n")
                                    count += 1
                                    if count % 10000 == 0:
                                        print(f"   ✓ Processed {count:,} sentences...")
        
        # Clean up
        os.unlink(tmp_path)
        
        print(f"✅ Successfully saved {count:,} sentences to {output}")
        return True
        
    except Exception as e:
        print(f"❌ Error downloading WikiText: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Download large datasets for training SheepOp LLM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download 500k Amazon reviews
  python3 download_large_data.py amazon --limit 500000
  
  # Download WikiText-103
  python3 download_large_data.py wiki
  
  # Download OpenWebText sample
  python3 download_large_data.py openwebtext --limit 100000
  
  # Download to custom location
  python3 download_large_data.py amazon --output data/my_reviews.txt
        """
    )
    
    parser.add_argument(
        'dataset',
        choices=['amazon', 'wiki', 'wikitext', 'openwebtext', 'bookcorpus'],
        help='Dataset to download'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        help='Output file path (default: data/<dataset_name>.txt)'
    )
    
    parser.add_argument(
        '--limit',
        type=int,
        default=500000,
        help='Maximum number of samples to download (default: 500000)'
    )
    
    parser.add_argument(
        '--category',
        type=str,
        default='Video_Games_v1_00',
        help='Amazon reviews category (for amazon dataset only, may not work - uses alternative)'
    )
    
    parser.add_argument(
        '--use-imdb',
        action='store_true',
        help='Use IMDB reviews instead of Amazon (more reliable)'
    )
    
    parser.add_argument(
        '--version',
        type=str,
        default='103',
        choices=['2', '103'],
        help='WikiText version: 2 (small) or 103 (large)'
    )
    
    args = parser.parse_args()
    
    # Set default output path if not provided
    if not args.output:
        if args.dataset == 'amazon':
            args.output = f"data/amazon_reviews.txt"
        elif args.dataset in ['wiki', 'wikitext']:
            args.output = f"data/wikitext_{args.version}.txt"
        elif args.dataset == 'openwebtext':
            args.output = "data/openwebtext.txt"
        elif args.dataset == 'bookcorpus':
            args.output = "data/bookcorpus.txt"
    
    print(f"\n🚀 SheepOp Dataset Downloader")
    print(f"   Dataset: {args.dataset}")
    print(f"   Output: {args.output}")
    print(f"   Limit: {args.limit:,} samples\n")
    
    # Download based on dataset type
    success = False
    if args.dataset == 'amazon':
        if args.use_imdb:
            # Use IMDB directly
            try:
                from datasets import load_dataset
                print("📥 Downloading IMDB Reviews...")
                dataset = load_dataset("imdb", split=f"train[:{args.limit}]")
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, "w", encoding="utf-8") as f:
                    count = 0
                    for item in dataset:
                        review = item.get("text", "").strip()
                        if review and len(review) > 20:
                            f.write(review + "\n")
                            count += 1
                            if count % 50000 == 0:
                                print(f"   ✓ Downloaded {count:,} reviews...")
                print(f"✅ Successfully saved {count:,} reviews to {args.output}")
                success = True
            except Exception as e:
                print(f"❌ Error: {e}")
                success = False
        else:
            success = download_amazon_reviews(args.output, args.limit, args.category)
    elif args.dataset in ['wiki', 'wikitext']:
        if args.version == '103':
            # Try direct download first (no HuggingFace dependency)
            print("   Attempting direct download (no HuggingFace required)...")
            success = download_wikitext_direct(args.output)
            if not success:
                print("   Falling back to HuggingFace download...")
                success = download_wikitext(args.output, args.version)
        else:
            success = download_wikitext(args.output, args.version)
    elif args.dataset == 'openwebtext':
        success = download_openwebtext(args.output, args.limit)
    elif args.dataset == 'bookcorpus':
        success = download_bookcorpus(args.output, args.limit)
    
    if success:
        print(f"\n✅ Download complete!")
        print(f"   File: {args.output}")
        
        # Show file info
        try:
            import os
            size_mb = os.path.getsize(args.output) / (1024 * 1024)
            with open(args.output, 'r', encoding='utf-8') as f:
                lines = sum(1 for _ in f)
            print(f"   Size: {size_mb:.2f} MB")
            print(f"   Lines: {lines:,}")
        except:
            pass
        
        print(f"\n📚 You can now train with:")
        print(f"   python3 train.py --data {args.output} --config config.json --device cuda")
    else:
        print(f"\n❌ Download failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == '__main__':
    main()

