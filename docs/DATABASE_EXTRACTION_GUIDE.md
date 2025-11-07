# Database Extraction Guide

This guide shows you how to extract text from your 1TB database for training.

## Quick Start

### SQLite Database

```bash
# Extract from SQLite database
python3 extract_from_database.py \
    --type sqlite \
    --db-path /path/to/your/database.db \
    --table your_table_name \
    --column text_column_name \
    --output data/database_training.txt \
    --limit 1000000  # Limit to 1M samples (or omit for all)
```

### PostgreSQL Database

```bash
# Install PostgreSQL driver first
pip install psycopg2-binary

# Extract with SQL query
python3 extract_from_database.py \
    --type sql \
    --connection "host=localhost dbname=mydb user=myuser password=mypass" \
    --query "SELECT text_column FROM your_table WHERE length(text_column) > 50" \
    --output data/database_training.txt \
    --limit 1000000
```

### MySQL Database

```bash
# Install MySQL driver first
pip install pymysql

# Extract with SQL query
python3 extract_from_database.py \
    --type sql \
    --connection "mysql+pymysql://user:pass@localhost/dbname" \
    --query "SELECT text_column FROM your_table" \
    --output data/database_training.txt
```

### JSON/JSONL Files

```bash
# Extract from JSON Lines file
python3 extract_from_database.py \
    --type json \
    --json-path /path/to/data.jsonl \
    --text-field content \
    --output data/database_training.txt \
    --limit 1000000
```

## Examples

### Example 1: Extract All Text from SQLite Table

```bash
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table articles \
    --column body_text \
    --output data/training_data.txt
```

### Example 2: Extract Filtered Data (Longer Texts Only)

```bash
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table articles \
    --column body_text \
    --where "WHERE length(body_text) > 200" \
    --output data/training_data.txt \
    --min-length 50
```

### Example 3: Extract from Multiple Tables

```bash
# Extract from table 1
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table articles \
    --column content \
    --output data/articles.txt

# Extract from table 2
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table comments \
    --column text \
    --output data/comments.txt

# Combine files
cat data/articles.txt data/comments.txt > data/combined_training.txt
```

### Example 4: PostgreSQL with Complex Query

```bash
python3 extract_from_database.py \
    --type sql \
    --connection "host=localhost dbname=mydb user=myuser password=mypass" \
    --query "SELECT description FROM products WHERE description IS NOT NULL AND length(description) > 100 UNION SELECT review_text FROM reviews WHERE review_text IS NOT NULL" \
    --output data/products_and_reviews.txt
```

## Options

### Filtering Options

```bash
# Only extract texts longer than 100 characters
--min-length 100

# Limit total samples
--limit 1000000

# Add WHERE clause (SQLite)
--where "WHERE created_at > '2024-01-01' AND length(text) > 200"
```

### Output Options

```bash
# Custom output path
--output data/my_training_data.txt

# Don't clean/split text (preserve original format)
--no-clean
```

## Performance Tips

1. **Use LIMIT for Testing**: Start with `--limit 10000` to test
2. **Filter in Database**: Use `--where` clause to filter at database level (faster)
3. **Batch Processing**: The script processes in batches automatically
4. **Monitor Progress**: Progress updates every 1000 texts

## Data Format

The output file will have:
- One text sample per line
- Cleaned and split into sentences
- Minimum length filtering applied
- UTF-8 encoding

## Next Steps

After extraction:

```bash
# Check how much data you extracted
wc -l data/database_training.txt

# Train with the extracted data
python3 train.py --data data/database_training.txt --config config.json --device mps
```

## Troubleshooting

### SQLite Database Locked
- Close any applications using the database
- Copy database to a local location first

### Large Database (1TB)
- Use `--limit` to extract in batches
- Use `--where` to filter at database level
- Consider extracting to multiple files and combining

### Memory Issues
- The script processes in batches (streaming)
- Use `--limit` to control size
- Process in chunks if needed

## Example Workflow

```bash
# 1. Extract 1M samples for testing
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table your_table \
    --column text_column \
    --output data/test_extraction.txt \
    --limit 1000000

# 2. Check the data
head -20 data/test_extraction.txt
wc -l data/test_extraction.txt

# 3. If good, extract more (or all)
python3 extract_from_database.py \
    --type sqlite \
    --db-path /Volumes/YourDisk/database.db \
    --table your_table \
    --column text_column \
    --output data/full_training.txt

# 4. Train with the data
python3 train.py --data data/full_training.txt --config config.json --device mps
```

Good luck extracting your 1TB database! 🚀

