"""
Improved BPE Tokenizer based on GPT-4 tokenization approach
Addresses common tokenization challenges:
- UTF-8 byte-level encoding
- Better Python code handling
- Case-insensitive contraction matching
- Limited number merging (1-3 digits)
- Proper special token handling
- Trailing whitespace warnings
"""
import re
import json
from typing import List, Dict, Tuple, Optional, Set
from collections import defaultdict
from pathlib import Path


class BPETokenizer:
    """
    Byte Pair Encoding Tokenizer with GPT-4-inspired improvements.
    
    Key features:
    - UTF-8 byte-level encoding
    - BPE merging algorithm
    - GPT-4 style regex pattern for text splitting
    - Better whitespace handling for Python code
    - Case-insensitive matching for contractions
    - Limited number merging (1-3 digits)
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        special_tokens: Optional[Dict[str, int]] = None,
        merges_file: Optional[str] = None,
        vocab_file: Optional[str] = None,
    ):
        """
        Initialize BPE tokenizer.
        
        Args:
            vocab_size: Target vocabulary size (default 50257 for GPT-2 style)
            special_tokens: Dictionary of special token names to IDs
            merges_file: Path to saved merges file
            vocab_file: Path to saved vocab file
        """
        # Special tokens
        self.special_tokens = special_tokens or {
            '<pad>': 0,
            '<unk>': 1,
            '<bos>': 2,
            '<eos>': 3,
        }
        
        # Initialize byte vocabulary (0-255)
        self.byte_to_token = {i: i for i in range(256)}
        self.token_to_byte = {i: bytes([i]) for i in range(256)}
        self.next_token_id = 256
        
        # BPE merges: (left, right) -> merged_token_id
        self.merges: Dict[Tuple[int, int], int] = {}
        
        # Vocabulary: token_id -> bytes
        self.vocab: Dict[int, bytes] = {}
        self.inv_vocab: Dict[bytes, int] = {}
        
        # Initialize vocab with bytes
        for i in range(256):
            self.vocab[i] = bytes([i])
            self.inv_vocab[bytes([i])] = i
        
        # GPT-4 style regex pattern for splitting text
        # Improvements over GPT-2:
        # - Case-insensitive matching (flag)
        # - Better whitespace handling
        # - Limit number merging to 1-3 digits
        self.pattern = self._create_gpt4_pattern()
        
        # Load pre-trained tokenizer if files provided
        if merges_file and vocab_file:
            self.load(merges_file, vocab_file)
        else:
            self.target_vocab_size = vocab_size
        
        # Token IDs for special tokens
        self.pad_token_id = self.special_tokens.get('<pad>', 0)
        self.unk_token_id = self.special_tokens.get('<unk>', 1)
        self.bos_token_id = self.special_tokens.get('<bos>', 2)
        self.eos_token_id = self.special_tokens.get('<eos>', 3)
    
    def _create_gpt4_pattern(self) -> re.Pattern:
        """
        Create GPT-4 style regex pattern for splitting text.
        
        Improvements over GPT-2:
        - Case-insensitive matching for contractions
        - Better whitespace handling (groups multiple spaces)
        - Limit number merging (1-3 digits)
        """
        # GPT-4 style pattern with improvements
        # Pattern breakdown:
        # 1. Contractions: '(?i:[sdmt]|ll|ve|re) - case-insensitive
        # 2. Letters: [^\r\n\p{L}\p{N}]?+\p{L}+ - optional space + letters
        # 3. Numbers: \p{N}{1,3} - 1-3 digits only
        # 4. Punctuation: ?[^\s\p{L}\p{N}]++ - optional space + non-letter/number
        # 5. Whitespace: \r?\n - newlines
        # 6. Trailing whitespace: \s+ - multiple spaces
        pattern = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++|\r?\n|\s+"""
        
        # Compile with case-insensitive flag for contractions
        return re.compile(pattern, re.IGNORECASE | re.UNICODE)
    
    def _get_stats(self, tokens: List[int]) -> Dict[Tuple[int, int], int]:
        """
        Get statistics of consecutive token pairs.
        
        Args:
            tokens: List of token IDs
            
        Returns:
            Dictionary mapping pair tuples to counts
        """
        stats = defaultdict(int)
        for i in range(len(tokens) - 1):
            pair = (tokens[i], tokens[i + 1])
            stats[pair] += 1
        return dict(stats)
    
    def _merge(self, tokens: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
        """
        Merge consecutive occurrences of a pair into a new token.
        
        Args:
            tokens: List of token IDs
            pair: Tuple of (left, right) tokens to merge
            new_id: New token ID to replace the pair
            
        Returns:
            New list with merged tokens
        """
        if len(tokens) < 2:
            return tokens
        
        new_tokens = []
        i = 0
        while i < len(tokens):
            # Check if we can merge at position i
            if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                new_tokens.append(new_id)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        
        return new_tokens
    
    def train(
        self,
        texts: List[str],
        num_merges: Optional[int] = None,
        verbose: bool = False,
    ):
        """
        Train BPE tokenizer on a corpus of texts.
        
        Args:
            texts: List of training texts
            num_merges: Number of merges to perform (default: vocab_size - 256)
            verbose: Whether to print progress
        """
        if num_merges is None:
            num_merges = self.target_vocab_size - 256
        
        # Convert all texts to byte sequences
        all_tokens = []
        for text in texts:
            # Split text using regex pattern
            chunks = self.pattern.findall(text)
            
            # Convert each chunk to bytes and tokenize
            for chunk in chunks:
                bytes_seq = chunk.encode('utf-8')
                tokens = list(bytes_seq)
                all_tokens.extend(tokens)
                # Add separator between chunks (optional)
                # all_tokens.append(256)  # separator token
        
        # Perform BPE merges
        for merge_num in range(num_merges):
            # Get statistics
            stats = self._get_stats(all_tokens)
            
            if not stats:
                break
            
            # Find most frequent pair
            pair = max(stats, key=stats.get)
            
            # Create new token
            new_id = self.next_token_id
            self.next_token_id += 1
            
            # Merge
            all_tokens = self._merge(all_tokens, pair, new_id)
            
            # Store merge
            self.merges[pair] = new_id
            
            # Update vocabulary
            left_bytes = self.vocab.get(pair[0], bytes([pair[0]]))
            right_bytes = self.vocab.get(pair[1], bytes([pair[1]]))
            merged_bytes = left_bytes + right_bytes
            self.vocab[new_id] = merged_bytes
            self.inv_vocab[merged_bytes] = new_id
            
            if verbose and (merge_num + 1) % 1000 == 0:
                print(f"Merged {merge_num + 1}/{num_merges} pairs")
        
        if verbose:
            print(f"Training complete. Vocabulary size: {len(self.vocab)}")
    
    def _encode_chunk(self, text: str) -> List[int]:
        """
        Encode a single text chunk using BPE.
        
        Args:
            text: Text chunk to encode
            
        Returns:
            List of token IDs
        """
        # Convert to bytes
        bytes_seq = text.encode('utf-8')
        tokens = list(bytes_seq)
        
        # If no merges trained yet, return byte tokens
        if not self.merges:
            return tokens
        
        # Apply merges in order
        # Sort merges by their token ID (merge order)
        sorted_merges = sorted(self.merges.items(), key=lambda x: x[1])
        
        # Keep merging until no more merges are possible
        changed = True
        while changed:
            changed = False
            best_pair = None
            best_idx = float('inf')
            
            # Find the earliest merge we can apply
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i + 1])
                if pair in self.merges:
                    merge_idx = self.merges[pair]
                    if merge_idx < best_idx:
                        best_idx = merge_idx
                        best_pair = pair
            
            # Apply the best merge
            if best_pair is not None:
                merged_id = self.merges[best_pair]
                tokens = self._merge(tokens, best_pair, merged_id)
                changed = True
        
        return tokens
    
    def encode(self, text: str, allowed_special: Optional[Set[str]] = None) -> List[int]:
        """
        Encode text into token IDs.
        
        Args:
            text: Input text
            allowed_special: Set of special tokens to allow in text
            
        Returns:
            List of token IDs
        """
        # Check for trailing whitespace (warn if present)
        if text and text[-1] == ' ':
            import warnings
            warnings.warn(
                "Text ends with trailing whitespace. This may cause worse performance "
                "due to how the tokenizer splits text into tokens.",
                UserWarning
            )
        
        # Handle special tokens
        if allowed_special:
            for special_name, special_id in self.special_tokens.items():
                if special_name in allowed_special and special_name in text:
                    # Simple special token replacement (can be improved)
                    if text == special_name:
                        return [special_id]
        
        # Split text using regex pattern
        chunks = self.pattern.findall(text)
        
        # Encode each chunk
        tokens = []
        for chunk in chunks:
            chunk_tokens = self._encode_chunk(chunk)
            tokens.extend(chunk_tokens)
        
        return tokens
    
    def decode(self, token_ids: List[int], errors: str = 'replace') -> str:
        """
        Decode token IDs back to text.
        
        Args:
            token_ids: List of token IDs
            errors: Error handling for invalid UTF-8 ('strict', 'replace', 'ignore')
            
        Returns:
            Decoded text string
        """
        # Handle special tokens
        if self.eos_token_id in token_ids:
            # Stop at EOS token
            eos_idx = token_ids.index(self.eos_token_id)
            token_ids = token_ids[:eos_idx]
        
        # Convert tokens to bytes
        bytes_parts = []
        for token_id in token_ids:
            if token_id in self.special_tokens.values():
                # Skip special tokens (except maybe keep them for debugging)
                continue
            
            if token_id in self.vocab:
                bytes_parts.append(self.vocab[token_id])
            else:
                # Unknown token - try to use byte representation
                if token_id < 256:
                    bytes_parts.append(bytes([token_id]))
                else:
                    # Unknown token - use replacement character
                    bytes_parts.append(b'\ufffd')
        
        # Concatenate bytes
        if not bytes_parts:
            return ''
        
        try:
            combined_bytes = b''.join(bytes_parts)
            return combined_bytes.decode('utf-8', errors=errors)
        except UnicodeDecodeError:
            # Fallback with replacement
            return combined_bytes.decode('utf-8', errors='replace')
    
    def save(self, merges_file: str, vocab_file: str):
        """
        Save tokenizer to files.
        
        Args:
            merges_file: Path to save merges
            vocab_file: Path to save vocabulary
        """
        # Save merges
        merges_list = [
            (left, right, merged_id)
            for (left, right), merged_id in sorted(self.merges.items(), key=lambda x: x[1])
        ]
        
        with open(merges_file, 'w') as f:
            json.dump(merges_list, f, indent=2)
        
        # Save vocabulary (convert bytes to base64 or hex)
        vocab_dict = {
            str(token_id): token_bytes.hex()
            for token_id, token_bytes in self.vocab.items()
        }
        
        with open(vocab_file, 'w') as f:
            json.dump({
                'vocab': vocab_dict,
                'special_tokens': self.special_tokens,
                'next_token_id': self.next_token_id,
            }, f, indent=2)
    
    def load(self, merges_file: str, vocab_file: str):
        """
        Load tokenizer from files.
        
        Args:
            merges_file: Path to merges file
            vocab_file: Path to vocabulary file
        """
        # Load merges
        with open(merges_file, 'r') as f:
            merges_list = json.load(f)
        
        for left, right, merged_id in merges_list:
            self.merges[(left, right)] = merged_id
            self.next_token_id = max(self.next_token_id, merged_id + 1)
        
        # Load vocabulary
        with open(vocab_file, 'r') as f:
            vocab_data = json.load(f)
        
        vocab_dict = vocab_data['vocab']
        for token_id_str, token_bytes_hex in vocab_dict.items():
            token_id = int(token_id_str)
            token_bytes = bytes.fromhex(token_bytes_hex)
            self.vocab[token_id] = token_bytes
            self.inv_vocab[token_bytes] = token_id
        
        if 'special_tokens' in vocab_data:
            self.special_tokens.update(vocab_data['special_tokens'])
        
        if 'next_token_id' in vocab_data:
            self.next_token_id = vocab_data['next_token_id']
    
    @property
    def vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.vocab) + len(self.special_tokens)


# Backward compatibility alias
SimpleTokenizer = BPETokenizer

