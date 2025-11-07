# Multi-Format Data Processing Guide

## Overview

The training script now supports processing multiple file types from your `data/` directory:

- **Text files**: `.txt`, `.md`, `.rst`, `.log`, `.csv`, `.json`, `.jsonl`, `.xml`, `.html`, `.htm`
- **Code files**: `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, and many more
- **PDF files**: `.pdf` (requires PyPDF2 or pdfplumber)
- **Images**: `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp` (requires pytesseract for OCR)

## Basic Usage

Simply point the training script to your data directory:

```bash
python train.py --data /path/to/your/data/directory
```

The script will automatically:
1. Scan the directory (recursively by default)
2. Extract text from all supported file types
3. Process and tokenize the text
4. Train the model on all extracted content

## Installation

### Core Dependencies

The core dependencies are already in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

### Optional Dependencies for PDF and Image Processing

If you want to process PDFs or images, install the optional dependencies:

```bash
# For PDF processing (choose one):
pip install PyPDF2
# OR
pip install pdfplumber  # Alternative, often better for complex PDFs

# For image OCR:
pip install pytesseract Pillow

# Also install Tesseract OCR engine:
# macOS: brew install tesseract
# Ubuntu/Debian: sudo apt-get install tesseract-ocr
# Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
```

## How It Works

### 1. Text Files

Text files are read line by line. Each non-empty line becomes a training sample.

### 2. Code Files

Code files are processed as text. Each line of code becomes a training sample. This allows the model to learn code patterns and syntax.

### 3. PDF Files

PDFs are processed page by page:
- Text is extracted from each page
- Split into lines
- Filtered to remove very short lines
- Each line becomes a training sample

**Note**: PDF extraction works best with text-based PDFs. Scanned PDFs (images) should use OCR instead.

### 4. Image Files

Images are processed using OCR (Optical Character Recognition):
- Images are opened using PIL/Pillow
- pytesseract extracts text from the image
- Text is split into lines
- Each line becomes a training sample

**Note**: OCR quality depends on image quality. For best results:
- Use high-resolution images
- Ensure good contrast between text and background
- Avoid images with complex layouts

## Configuration Options

You can customize the data processing behavior:

```python
from pathlib import Path
from data import DataProcessor

processor = DataProcessor(
    use_ocr=True,           # Enable OCR for images
    use_pdf_extraction=True # Enable PDF extraction
)

# Process directory
texts = processor.process_to_list(
    directory=Path("data/"),
    recursive=True,         # Process subdirectories
    min_length=10,          # Minimum line length
    max_samples=None,       # Limit number of samples (None = all)
)
```

## Examples

### Example 1: Process all files in directory

```bash
python train.py --data /mnt/storage/sheepOp/data
```

### Example 2: Process single file

```bash
python train.py --data /mnt/storage/sheepOp/data/document.pdf
```

### Example 3: Using Python API

```python
from pathlib import Path
from data import extract_text_from_directory

# Extract text from all supported files
texts = extract_text_from_directory(
    directory=Path("data/"),
    recursive=True,
    use_ocr=True,
    use_pdf_extraction=True,
    min_length=10,
)

print(f"Extracted {len(texts)} text samples")
```

## Supported File Types Summary

| Category | Extensions | Requirements |
|----------|-----------|--------------|
| Text | `.txt`, `.md`, `.rst`, `.log`, `.csv`, `.json`, `.jsonl`, `.xml`, `.html`, `.htm` | None |
| Code | `.py`, `.js`, `.ts`, `.java`, `.cpp`, `.c`, `.go`, `.rs`, `.rb`, `.php`, `.swift`, and 30+ more | None |
| PDF | `.pdf` | PyPDF2 or pdfplumber |
| Images | `.png`, `.jpg`, `.jpeg`, `.gif`, `.bmp`, `.tiff`, `.webp` | pytesseract + Pillow + Tesseract OCR |

## Troubleshooting

### PDF extraction not working

- Install PyPDF2: `pip install PyPDF2`
- Or install pdfplumber (better for complex PDFs): `pip install pdfplumber`
- If PDFs are scanned images, use OCR instead

### OCR not working

1. Install pytesseract: `pip install pytesseract Pillow`
2. Install Tesseract OCR engine (see installation instructions above)
3. On some systems, you may need to set the tesseract path:
   ```python
   import pytesseract
   pytesseract.pytesseract.tesseract_cmd = '/usr/local/bin/tesseract'  # macOS example
   ```

### No text extracted

- Check that files are in supported formats
- Verify file permissions
- Check logs for error messages
- Try processing a single file first to debug

## Performance Tips

1. **Large directories**: Processing can take time for large directories. Progress is logged every 100 files.

2. **Parallel processing**: Consider processing files in parallel if you have many large files.

3. **Filtering**: Use `min_length` to filter out very short lines that may not be useful for training.

4. **Caching**: For repeated processing, consider saving extracted text to a file first.

## Next Steps

Once your data is processed:

1. The training script will automatically tokenize the text
2. Create training batches
3. Train your model

For more information on training, see `RETRAINING_GUIDE.md`.

