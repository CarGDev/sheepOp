"""
Data loading and preprocessing utilities
Includes comprehensive data processor for multiple file types (PDFs, images, code, text, etc.)
"""
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List, Dict, Optional, Iterator
import json
from pathlib import Path
import logging
from tqdm import tqdm
import hashlib
import pickle
import os
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Ensure logging output is unbuffered
for handler in logger.handlers:
    handler.flush()
# Also ensure root logger handlers are unbuffered
for handler in logging.root.handlers:
    handler.flush()


class TextDataset(Dataset):
    """
    Dataset for text data.
    """
    
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        stride: Optional[int] = None,
    ):
        """
        Args:
            texts: List of text strings
            tokenizer: Tokenizer instance
            max_length: Maximum sequence length
            stride: Stride for sliding window (if None, no overlap)
        """
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride if stride is not None else max_length
        
        # Tokenize all texts
        self.sequences = self._prepare_sequences()
        
    def _prepare_sequences(self) -> List[torch.Tensor]:
        """Tokenize and chunk sequences."""
        sequences = []
        
        for text in self.texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text)
            
            # Chunk into sequences of max_length
            for i in range(0, len(tokens), self.stride):
                chunk = tokens[i:i + self.max_length]
                
                # Pad if necessary
                if len(chunk) < self.max_length:
                    chunk = chunk + [self.tokenizer.pad_token_id] * (self.max_length - len(chunk))
                
                sequences.append(torch.tensor(chunk, dtype=torch.long))
                
                # Stop if we've covered the entire sequence
                if i + self.max_length >= len(tokens):
                    break
        
        return sequences
    
    def __len__(self) -> int:
        return len(self.sequences)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sequence = self.sequences[idx]
        
        # Input is all tokens except the last one
        input_ids = sequence[:-1]
        # Labels are all tokens except the first one (shifted by 1)
        labels = sequence[1:]
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }


class SimpleTokenizer:
    """
    Simple character-level tokenizer (for backward compatibility).
    Uses BPE tokenizer by default if available, falls back to character-level.
    """
    
    def __init__(
        self,
        vocab_file: Optional[str] = None,
        use_bpe: bool = True,
        vocab_size: int = 50257,
    ):
        """
        Args:
            vocab_file: Optional path to vocabulary file
            use_bpe: Whether to use BPE tokenizer (default: True)
            vocab_size: Vocabulary size for BPE tokenizer (default: 50257)
        """
        self.use_bpe = use_bpe
        
        # Try to use BPE tokenizer if available
        if use_bpe:
            try:
                from .bpe_tokenizer import BPETokenizer
                self.bpe_tokenizer = BPETokenizer(vocab_size=vocab_size)
                self._use_bpe = True
                
                # Map BPE tokenizer attributes
                self.pad_token_id = self.bpe_tokenizer.pad_token_id
                self.unk_token_id = self.bpe_tokenizer.unk_token_id
                self.bos_token_id = self.bpe_tokenizer.bos_token_id
                self.eos_token_id = self.bpe_tokenizer.eos_token_id
                self.vocab_size = self.bpe_tokenizer.vocab_size
                self.vocab = {i: self.bpe_tokenizer.vocab.get(i, bytes([i])).decode('utf-8', errors='replace') 
                             for i in range(256)}  # Limited vocab view
                self.inv_vocab = {v: k for k, v in self.vocab.items()}
                return
            except ImportError:
                logger.warning("BPE tokenizer not available, falling back to character-level")
                self._use_bpe = False
        
        # Fallback to character-level tokenizer
        self._use_bpe = False
        
        if vocab_file and Path(vocab_file).exists():
            with open(vocab_file, 'r') as f:
                vocab = json.load(f)
            self.vocab = vocab
            self.inv_vocab = {v: k for k, v in vocab.items()}
        else:
            # Default: character-level vocabulary
            self.vocab = {
                '<pad>': 0,
                '<unk>': 1,
                '<bos>': 2,
                '<eos>': 3,
            }
            # Add printable ASCII characters
            for i in range(32, 127):
                self.vocab[chr(i)] = len(self.vocab)
            
            self.inv_vocab = {v: k for k, v in self.vocab.items()}
        
        self.pad_token_id = self.vocab.get('<pad>', 0)
        self.unk_token_id = self.vocab.get('<unk>', 1)
        self.bos_token_id = self.vocab.get('<bos>', 2)
        self.eos_token_id = self.vocab.get('<eos>', 3)
        
        self.vocab_size = len(self.vocab)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs."""
        if self._use_bpe:
            return self.bpe_tokenizer.encode(text)
        
        # Character-level encoding
        tokens = []
        for char in text:
            tokens.append(self.vocab.get(char, self.unk_token_id))
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        if self._use_bpe:
            return self.bpe_tokenizer.decode(token_ids)
        
        # Character-level decoding
        chars = []
        for tid in token_ids:
            if tid in self.inv_vocab:
                char = self.inv_vocab[tid]
                if char not in ['<pad>', '<bos>', '<eos>']:
                    chars.append(char)
        return ''.join(chars)
    
    def save_vocab(self, vocab_file: str):
        """Save vocabulary to file."""
        if self._use_bpe:
            # Save BPE tokenizer
            merges_file = str(vocab_file).replace('.json', '_merges.json')
            self.bpe_tokenizer.save(merges_file, vocab_file)
        else:
            # Save character-level vocab
            with open(vocab_file, 'w') as f:
                json.dump(self.vocab, f, indent=2)
    
    def train(self, texts: List[str], num_merges: Optional[int] = None, verbose: bool = False):
        """Train the tokenizer on texts (BPE only)."""
        if self._use_bpe:
            self.bpe_tokenizer.train(texts, num_merges=num_merges, verbose=verbose)
            # Update vocab size
            self.vocab_size = self.bpe_tokenizer.vocab_size
        else:
            logger.warning("Training not supported for character-level tokenizer")


def create_dataloader(
    texts: List[str],
    tokenizer,
    batch_size: int = 32,
    max_length: int = 512,
    shuffle: bool = True,
    num_workers: int = 0,
) -> DataLoader:
    """
    Create a DataLoader for text data.
    
    Args:
        texts: List of text strings
        tokenizer: Tokenizer instance
        batch_size: Batch size
        max_length: Maximum sequence length
        shuffle: Whether to shuffle data
        num_workers: Number of data loading workers
        
    Returns:
        DataLoader instance
    """
    dataset = TextDataset(
        texts=texts,
        tokenizer=tokenizer,
        max_length=max_length,
    )
    
    def collate_fn(batch):
        """Collate function for batching."""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'labels': labels,
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
    )


# ============================================================================
# Data Processor for Multiple File Types
# ============================================================================

class DataProcessor:
    """
    Process various file types and extract text for training.
    Supports: PDFs, images (OCR), code files, text files, and more.
    """
    
    # Supported file extensions
    TEXT_EXTENSIONS = {'.txt', '.md', '.rst', '.log', '.csv', '.json', '.jsonl', '.xml', '.html', '.htm'}
    CODE_EXTENSIONS = {
        '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cpp', '.c', '.h', '.hpp',
        '.cs', '.go', '.rs', '.rb', '.php', '.swift', '.kt', '.scala', '.r',
        '.sql', '.sh', '.bash', '.zsh', '.fish', '.yaml', '.yml', '.toml',
        '.ini', '.cfg', '.conf', '.vue', '.svelte', '.dart', '.lua', '.pl',
        '.hs', '.ml', '.mli', '.elm', '.ex', '.exs', '.jl', '.clj', '.cljs'
    }
    PDF_EXTENSIONS = {'.pdf'}
    IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif', '.webp'}
    
    def __init__(self, use_ocr: bool = True, use_pdf_extraction: bool = True, cache_dir: Optional[Path] = None):
        """
        Initialize data processor.
        
        Args:
            use_ocr: Whether to use OCR for images (requires pytesseract)
            use_pdf_extraction: Whether to extract text from PDFs (requires PyPDF2 or pdfplumber)
            cache_dir: Directory to store cache files (default: .cache in data directory)
        """
        self.use_ocr = use_ocr
        self.use_pdf_extraction = use_pdf_extraction
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._check_dependencies()
    
    def _get_cache_dir(self, directory: Path) -> Path:
        """Get cache directory for a given data directory."""
        if self.cache_dir:
            return self.cache_dir
        # Default: .cache in the data directory
        cache_dir = directory / '.cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir
    
    def _compute_directory_hash(self, directory: Path, recursive: bool = True) -> str:
        """
        Compute a hash of directory contents to detect changes.
        Uses file paths and modification times.
        """
        directory = Path(directory)
        file_info = []
        
        pattern = '**/*' if recursive else '*'
        scanned_count = 0
        
        try:
            for file_path in directory.glob(pattern):
                scanned_count += 1
                
                # Progress feedback every 5000 files (hash computation can be slow)
                if scanned_count % 5000 == 0:
                    logger.info(f"Computing directory hash: scanned {scanned_count:,} paths...")
                    sys.stderr.flush()
                
                try:
                    if file_path.is_file():
                        stat = file_path.stat()
                        file_info.append(f"{file_path.relative_to(directory)}:{stat.st_mtime}:{stat.st_size}")
                except (OSError, PermissionError):
                    continue
                except KeyboardInterrupt:
                    logger.warning(f"Directory hash computation interrupted after scanning {scanned_count:,} paths")
                    raise
        except KeyboardInterrupt:
            # Re-raise to allow graceful handling upstream
            logger.warning("Directory hash computation interrupted. Will skip cache and do fresh scan.")
            raise
        
        # Sort for consistent hashing
        file_info.sort()
        content = '\n'.join(file_info)
        return hashlib.md5(content.encode()).hexdigest()
    
    def _get_cache_path(self, directory: Path, cache_type: str = 'files') -> Path:
        """Get cache file path for a directory."""
        cache_dir = self._get_cache_dir(directory)
        # Create a safe filename from directory path
        dir_hash = hashlib.md5(str(directory.absolute()).encode()).hexdigest()[:8]
        return cache_dir / f"{cache_type}_{dir_hash}.pkl"
    
    def _load_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load cache from file."""
        if not cache_path.exists():
            return None
        try:
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache from {cache_path}: {e}")
            return None
    
    def _save_cache(self, cache_path: Path, data: Dict):
        """Save cache to file."""
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            logger.warning(f"Failed to save cache to {cache_path}: {e}")
    
    def clear_cache(self, directory: Path):
        """
        Clear cache for a directory.
        
        Args:
            directory: Directory path to clear cache for
        """
        cache_path = self._get_cache_path(directory, 'files')
        if cache_path.exists():
            try:
                cache_path.unlink()
                logger.info(f"✅ Cleared cache for {directory}")
            except Exception as e:
                logger.warning(f"Failed to clear cache: {e}")
        else:
            logger.info(f"No cache found for {directory}")
    
    def _check_dependencies(self):
        """Check if required dependencies are available."""
        if self.use_ocr:
            try:
                import pytesseract
                from PIL import Image
                self._ocr_available = True
            except ImportError:
                logger.warning("pytesseract or PIL not available. OCR disabled.")
                self._ocr_available = False
                self.use_ocr = False
        
        if self.use_pdf_extraction:
            try:
                import PyPDF2
                self._pypdf2_available = True
            except ImportError:
                try:
                    import pdfplumber
                    self._pdfplumber_available = True
                    self._pypdf2_available = False
                except ImportError:
                    logger.warning("PyPDF2 or pdfplumber not available. PDF extraction disabled.")
                    self._pdfplumber_available = False
                    self.use_pdf_extraction = False
    
    def process_file(self, file_path: Path) -> Iterator[str]:
        """
        Process a single file and yield text lines.
        
        Args:
            file_path: Path to the file
            
        Yields:
            Text lines extracted from the file
        """
        file_path = Path(file_path)
        
        if not file_path.exists():
            logger.warning(f"File not found: {file_path}")
            return
        
        suffix = file_path.suffix.lower()
        
        try:
            if suffix in self.TEXT_EXTENSIONS:
                yield from self._process_text_file(file_path)
            elif suffix in self.CODE_EXTENSIONS:
                yield from self._process_code_file(file_path)
            elif suffix in self.PDF_EXTENSIONS:
                yield from self._process_pdf(file_path)
            elif suffix in self.IMAGE_EXTENSIONS:
                yield from self._process_image(file_path)
            else:
                # Try to process as text file as fallback (many file types can be read as text)
                # Only log at debug level to avoid spam
                logger.debug(f"Unsupported file type: {file_path} (extension: {suffix}), attempting as text...")
                try:
                    yield from self._process_text_file(file_path)
                except KeyboardInterrupt:
                    raise
                except Exception as e:
                    logger.debug(f"Failed to process {file_path} as text: {e}")
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow graceful shutdown
            logger.warning(f"Interrupted while processing {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
    
    def _process_text_file(self, file_path: Path) -> Iterator[str]:
        """Process a text file."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield line
        except KeyboardInterrupt:
            # Re-raise KeyboardInterrupt to allow graceful shutdown
            logger.warning(f"Interrupted while reading {file_path}")
            raise
        except UnicodeDecodeError:
            # Try with different encoding
            try:
                with open(file_path, 'r', encoding='latin-1', errors='ignore') as f:
                    for line in f:
                        line = line.strip()
                        if line:
                            yield line
            except KeyboardInterrupt:
                logger.warning(f"Interrupted while reading {file_path}")
                raise
            except Exception as e:
                logger.error(f"Failed to read {file_path}: {e}")
    
    def _process_code_file(self, file_path: Path) -> Iterator[str]:
        """Process a code file."""
        # Code files are processed as text, but we can add syntax-aware processing
        # For now, just extract text lines
        yield from self._process_text_file(file_path)
    
    def _process_pdf(self, file_path: Path) -> Iterator[str]:
        """Extract text from PDF file."""
        if not self.use_pdf_extraction:
            logger.warning(f"PDF extraction disabled. Skipping {file_path}")
            return
        
        try:
            if self._pypdf2_available:
                import PyPDF2
                with open(file_path, 'rb') as f:
                    pdf_reader = PyPDF2.PdfReader(f)
                    for page_num, page in enumerate(pdf_reader.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                # Split into sentences/lines
                                for line in text.split('\n'):
                                    line = line.strip()
                                    if line and len(line) > 5:  # Filter very short lines
                                        yield line
                        except KeyboardInterrupt:
                            logger.warning(f"Interrupted while processing PDF page {page_num} from {file_path}")
                            raise
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
            
            elif self._pdfplumber_available:
                import pdfplumber
                with pdfplumber.open(file_path) as pdf:
                    for page_num, page in enumerate(pdf.pages):
                        try:
                            text = page.extract_text()
                            if text:
                                for line in text.split('\n'):
                                    line = line.strip()
                                    if line and len(line) > 5:
                                        yield line
                        except KeyboardInterrupt:
                            logger.warning(f"Interrupted while processing PDF page {page_num} from {file_path}")
                            raise
                        except Exception as e:
                            logger.warning(f"Error extracting page {page_num} from {file_path}: {e}")
        
        except KeyboardInterrupt:
            logger.warning(f"Interrupted while processing PDF {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract text from PDF {file_path}: {e}")
    
    def _process_image(self, file_path: Path) -> Iterator[str]:
        """Extract text from image using OCR."""
        if not self.use_ocr or not self._ocr_available:
            logger.warning(f"OCR disabled or unavailable. Skipping {file_path}")
            return
        
        try:
            import pytesseract
            from PIL import Image
            
            # Open and process image
            img = Image.open(file_path)
            
            # Perform OCR
            text = pytesseract.image_to_string(img)
            
            if text:
                # Split into lines
                for line in text.split('\n'):
                    line = line.strip()
                    if line and len(line) > 3:  # Filter very short lines
                        yield line
        
        except KeyboardInterrupt:
            logger.warning(f"Interrupted while processing image {file_path}")
            raise
        except Exception as e:
            logger.error(f"Failed to extract text from image {file_path}: {e}")
    
    def process_directory(
        self,
        directory: Path,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        min_length: int = 10,
    ) -> Iterator[str]:
        """
        Process all files in a directory.
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            include_patterns: Optional list of glob patterns to include
            exclude_patterns: Optional list of glob patterns to exclude
            min_length: Minimum length for extracted text lines
            
        Yields:
            Text lines from all processed files
        """
        directory = Path(directory)
        
        if not directory.exists():
            logger.error(f"Directory not found: {directory}")
            return
        
        # Try to load cached file list
        cache_path = self._get_cache_path(directory, 'files')
        
        # Compute directory hash (may be interrupted)
        try:
            logger.info("Computing directory hash for cache validation...")
            logger.info("(This may take a while for large directories. Press Ctrl+C to skip cache and do fresh scan)")
            sys.stderr.flush()
            current_hash = self._compute_directory_hash(directory, recursive)
            cached_data = self._load_cache(cache_path)
        except KeyboardInterrupt:
            logger.warning("\n⚠️  Directory hash computation interrupted.")
            logger.warning("   Skipping cache validation and doing fresh directory scan...")
            logger.warning("   (Press Ctrl+C again to stop completely)")
            sys.stderr.flush()
            current_hash = None  # Force cache miss
            cached_data = None
            # Don't re-raise - allow user to continue with fresh scan
            # If they want to stop completely, they can press Ctrl+C again during scanning
        
        files_to_process = []
        scanned_count = 0
        skipped_count = 0
        
        # Check if cache is valid
        if current_hash and cached_data and cached_data.get('hash') == current_hash:
            files_to_process = [Path(f) for f in cached_data.get('files', [])]
            logger.info(f"✅ Loaded {len(files_to_process):,} files from cache (skipping directory scan)")
        else:
            # Cache miss or invalid - scan directory
            logger.info("Scanning directory (cache miss or invalid)...")
            
            # Collect all supported file extensions
            all_supported_extensions = (
                self.TEXT_EXTENSIONS | 
                self.CODE_EXTENSIONS | 
                self.PDF_EXTENSIONS | 
                self.IMAGE_EXTENSIONS
            )
            
            if recursive:
                pattern = '**/*'
            else:
                pattern = '*'
            
            # Default exclude patterns for common directories that don't contain training data
            default_exclude_patterns = [
                '**/.git/**',
                '**/__pycache__/**',
                '**/node_modules/**',
                '**/.venv/**',
                '**/venv/**',
                '**/.env/**',
                '**/.pytest_cache/**',
                '**/.mypy_cache/**',
                '**/.tox/**',
                '**/.coverage/**',
                '**/dist/**',
                '**/build/**',
                '**/*.pyc',
                '**/.DS_Store',
            ]
            
            # Merge user exclude patterns with defaults
            all_exclude_patterns = default_exclude_patterns.copy()
            if exclude_patterns:
                # Convert any Path objects to strings
                all_exclude_patterns.extend(str(p) if isinstance(p, Path) else p for p in exclude_patterns)
            
            # Ensure all patterns are strings (not Path objects)
            all_exclude_patterns = [str(p) for p in all_exclude_patterns]
            
            # Convert include_patterns to strings as well
            if include_patterns:
                include_patterns = [str(p) if isinstance(p, Path) else p for p in include_patterns]
            
            logger.info(f"Scanning directory: {directory} (recursive={recursive})...")
            logger.info("This may take several minutes for large directories. Please wait...")
            sys.stderr.flush()  # Force flush to show message immediately
            
            try:
                for file_path in directory.glob(pattern):
                    scanned_count += 1
                    
                    # Progress reporting every 1000 files scanned
                    if scanned_count % 1000 == 0:
                        logger.info(f"Scanned {scanned_count:,} paths, found {len(files_to_process):,} files to process...")
                        sys.stderr.flush()  # Force flush to show progress immediately
                    
                    # Skip if not a file (handles symlinks, directories, etc. gracefully)
                    try:
                        if not file_path.is_file():
                            continue
                    except (OSError, PermissionError) as e:
                        # Skip inaccessible files (broken symlinks, permission denied, etc.)
                        skipped_count += 1
                        if skipped_count <= 10:  # Only log first 10 to avoid spam
                            logger.debug(f"Skipping inaccessible path: {file_path} ({e})")
                        continue
                    
                    # Early filtering by extension to avoid checking unsupported files
                    suffix = file_path.suffix.lower()
                    if suffix not in all_supported_extensions:
                        continue
                    
                    # Check include/exclude patterns
                    if include_patterns:
                        if not any(file_path.match(pattern) for pattern in include_patterns):
                            continue
                    
                    if all_exclude_patterns:
                        if any(file_path.match(pattern) for pattern in all_exclude_patterns):
                            continue
                    
                    files_to_process.append(file_path)
            
            except KeyboardInterrupt:
                logger.warning(f"Directory scanning interrupted. Found {len(files_to_process)} files so far.")
                raise
            except Exception as e:
                logger.error(f"Error during directory scanning: {e}")
                logger.info(f"Continuing with {len(files_to_process)} files found so far...")
            
            if skipped_count > 10:
                logger.info(f"Skipped {skipped_count} inaccessible paths")
            
            logger.info(f"Found {len(files_to_process):,} files to process (scanned {scanned_count:,} paths)")
            sys.stderr.flush()  # Force flush
            
            # Save file list to cache
            cache_data = {
                'hash': current_hash,
                'files': [str(f.absolute()) for f in files_to_process],
                'recursive': recursive,
            }
            self._save_cache(cache_path, cache_data)
            logger.info(f"💾 Cached file list ({len(files_to_process):,} files) for future use")
            sys.stderr.flush()  # Force flush
        
        # Process each file
        processed_count = 0
        skipped_count = 0
        error_count = 0
        total_lines = 0
        total_files = len(files_to_process)
        
        if total_files == 0:
            logger.warning("No files found to process!")
            return
        
        logger.info(f"Starting to process {total_files} files with progress bar...")
        
        # Create progress bar
        pbar = tqdm(
            total=total_files,
            desc="Processing files",
            unit="file",
            ncols=120,
            mininterval=0.1,  # Update at least every 0.1 seconds
            maxinterval=1.0,  # Force update at least once per second
            file=sys.stderr,  # Write to stderr to avoid buffering issues
            dynamic_ncols=True,  # Auto-adjust to terminal width
            disable=False,  # Explicitly enable
        )
        
        try:
            for idx, file_path in enumerate(files_to_process, 1):
                try:
                    file_lines = list(self.process_file(file_path))
                    if file_lines:
                        processed_count += 1
                        for line in file_lines:
                            if len(line) >= min_length:
                                yield line
                                total_lines += 1
                    else:
                        skipped_count += 1
                    
                    # Update progress bar with statistics
                    pbar.set_postfix({
                        'Processed': processed_count,
                        'Skipped': skipped_count,
                        'Errors': error_count,
                        'Lines': f"{total_lines:,}"
                    })
                    pbar.update(1)  # Advance progress bar
                    pbar.refresh()  # Force immediate refresh
                    sys.stderr.flush()  # Force flush stderr to ensure progress bar displays
                
                except KeyboardInterrupt:
                    pbar.close()
                    logger.warning(
                        f"Processing interrupted. "
                        f"Files: {idx}/{total_files}, Processed: {processed_count}, "
                        f"Skipped: {skipped_count}, Errors: {error_count}, "
                        f"Lines extracted: {total_lines:,}"
                    )
                    raise
                except Exception as e:
                    error_count += 1
                    logger.error(f"Error processing {file_path}: {e}")
                    # Update progress bar even on errors
                    pbar.set_postfix({
                        'Processed': processed_count,
                        'Skipped': skipped_count,
                        'Errors': error_count,
                        'Lines': f"{total_lines:,}"
                    })
                    pbar.update(1)  # Advance progress bar even on error
                    pbar.refresh()  # Force immediate refresh
                    sys.stderr.flush()  # Force flush stderr to ensure progress bar displays
        finally:
            pbar.close()
        
        logger.info(
            f"Processing complete: {processed_count}/{total_files} files processed successfully, "
            f"{skipped_count} skipped, {error_count} errors, {total_lines:,} lines extracted"
        )
    
    def process_to_list(
        self,
        directory: Path,
        recursive: bool = True,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
        min_length: int = 10,
        max_samples: Optional[int] = None,
    ) -> List[str]:
        """
        Process directory and return list of text lines.
        
        Args:
            directory: Directory path
            recursive: Whether to process subdirectories
            include_patterns: Optional list of glob patterns to include
            exclude_patterns: Optional list of glob patterns to exclude
            min_length: Minimum length for extracted text lines
            max_samples: Maximum number of samples to return (None = all)
            
        Returns:
            List of text lines
        """
        logger.info(f"Starting data extraction from {directory}...")
        logger.info("This may take a while for large directories. Progress will be shown below.")
        sys.stderr.flush()  # Force flush to show message immediately
        
        texts = []
        try:
            for text in self.process_directory(
                directory=directory,
                recursive=recursive,
                include_patterns=include_patterns,
                exclude_patterns=exclude_patterns,
                min_length=min_length,
            ):
                texts.append(text)
                if max_samples and len(texts) >= max_samples:
                    logger.info(f"Reached max_samples limit ({max_samples}). Stopping extraction.")
                    break
        except KeyboardInterrupt:
            # Return partial results if interrupted
            logger.warning(
                f"Data processing interrupted. Returning {len(texts):,} text samples collected so far."
            )
            # Re-raise to allow caller to handle if needed
            raise
        
        logger.info(f"✅ Extracted {len(texts):,} text samples from {directory}")
        return texts


def extract_text_from_directory(
    directory: Path,
    recursive: bool = True,
    use_ocr: bool = True,
    use_pdf_extraction: bool = True,
    min_length: int = 10,
    max_samples: Optional[int] = None,
) -> List[str]:
    """
    Convenience function to extract text from a directory.
    
    Args:
        directory: Directory path
        recursive: Whether to process subdirectories
        use_ocr: Whether to use OCR for images
        use_pdf_extraction: Whether to extract text from PDFs
        min_length: Minimum length for extracted text lines
        max_samples: Maximum number of samples to return (None = all)
        
    Returns:
        List of text lines
    """
    processor = DataProcessor(use_ocr=use_ocr, use_pdf_extraction=use_pdf_extraction)
    try:
        return processor.process_to_list(
            directory=directory,
            recursive=recursive,
            min_length=min_length,
            max_samples=max_samples,
        )
    except KeyboardInterrupt:
        logger.error(
            "\n⚠️  Data processing interrupted by user (Ctrl+C).\n"
            "   No data was loaded. Please run the training command again to retry."
        )
        # Re-raise to stop training
        raise


# Try to import BPE tokenizer for direct access
try:
    from .bpe_tokenizer import BPETokenizer
    __all__ = [
        'TextDataset', 
        'SimpleTokenizer',
        'BPETokenizer',
        'create_dataloader',
        'DataProcessor',
        'extract_text_from_directory',
    ]
except ImportError:
    __all__ = [
        'TextDataset', 
        'SimpleTokenizer', 
        'create_dataloader',
        'DataProcessor',
        'extract_text_from_directory',
    ]

