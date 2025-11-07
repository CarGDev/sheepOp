"""
Database extraction utility for training data
Extracts text from various database types and formats for LLM training
"""
import sqlite3
import argparse
from pathlib import Path
from typing import List, Optional, Iterator
import json


def extract_from_sqlite(
    db_path: str,
    table: str,
    text_column: str,
    limit: Optional[int] = None,
    where_clause: Optional[str] = None,
) -> Iterator[str]:
    """
    Extract text from SQLite database.
    
    Args:
        db_path: Path to SQLite database file
        table: Table name to extract from
        text_column: Column name containing text data
        limit: Maximum number of rows to extract (None = all)
        where_clause: Optional WHERE clause (e.g., "WHERE length(text) > 100")
        
    Yields:
        Text strings from the database
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    query = f"SELECT {text_column} FROM {table}"
    if where_clause:
        query += f" {where_clause}"
    if limit:
        query += f" LIMIT {limit}"
    
    cursor.execute(query)
    
    for row in cursor:
        text = row[0]
        if text and isinstance(text, str) and len(text.strip()) > 0:
            # Clean and split text into sentences/lines
            cleaned_text = text.strip()
            yield cleaned_text
    
    conn.close()


def extract_from_sql(
    connection_string: str,
    query: str,
    text_column: int = 0,
    batch_size: int = 1000,
) -> Iterator[str]:
    """
    Extract text using a raw SQL query.
    Works with any database that supports the connection string format.
    
    Args:
        connection_string: Database connection string
        query: SQL query to execute
        text_column: Column index containing text (0-based)
        batch_size: Number of rows to fetch at once
        
    Yields:
        Text strings from the database
    """
    try:
        import psycopg2  # PostgreSQL
        conn = psycopg2.connect(connection_string)
    except ImportError:
        try:
            import pymysql  # MySQL
            conn = pymysql.connect(connection_string)
        except ImportError:
            raise ImportError("Install psycopg2 for PostgreSQL or pymysql for MySQL")
    
    cursor = conn.cursor()
    cursor.execute(query)
    
    while True:
        rows = cursor.fetchmany(batch_size)
        if not rows:
            break
        
        for row in rows:
            text = row[text_column]
            if text and isinstance(text, str) and len(text.strip()) > 0:
                yield text.strip()
    
    conn.close()


def extract_from_json_file(
    json_path: str,
    text_field: str,
    limit: Optional[int] = None,
) -> Iterator[str]:
    """
    Extract text from JSON file (e.g., JSONL format).
    
    Args:
        json_path: Path to JSON file
        text_field: Field name containing text (use dot notation for nested: "data.text")
        limit: Maximum number of records to extract
        
    Yields:
        Text strings from the JSON file
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if limit and count >= limit:
                break
            
            try:
                data = json.loads(line)
                
                # Handle nested fields with dot notation
                fields = text_field.split('.')
                value = data
                for field in fields:
                    value = value.get(field)
                    if value is None:
                        break
                
                if value and isinstance(value, str) and len(value.strip()) > 0:
                    yield value.strip()
                    count += 1
            except json.JSONDecodeError:
                continue


def clean_and_split_text(text: str, min_length: int = 10) -> List[str]:
    """
    Clean text and split into sentences/lines.
    
    Args:
        text: Raw text string
        min_length: Minimum length for a text sample
        
    Returns:
        List of cleaned text samples
    """
    import re
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Split by sentences (periods, exclamation, question marks)
    sentences = re.split(r'[.!?]+\s+', text)
    
    # Also split by newlines
    lines = []
    for sentence in sentences:
        lines.extend(sentence.split('\n'))
    
    # Clean and filter
    cleaned = []
    for line in lines:
        line = line.strip()
        if len(line) >= min_length:
            cleaned.append(line)
    
    return cleaned


def save_to_training_file(
    texts: Iterator[str],
    output_path: str,
    min_length: int = 10,
    max_samples: Optional[int] = None,
    clean_text: bool = True,
):
    """
    Save extracted texts to training file.
    
    Args:
        texts: Iterator of text strings
        output_path: Path to save training data
        min_length: Minimum length for text samples
        max_samples: Maximum number of samples to save
        clean_text: Whether to clean and split text
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    count = 0
    total_texts = 0
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for text in texts:
            if max_samples and count >= max_samples:
                break
            
            if clean_text:
                # Clean and split into sentences
                cleaned_texts = clean_and_split_text(text, min_length)
                for cleaned in cleaned_texts:
                    if max_samples and count >= max_samples:
                        break
                    f.write(cleaned + '\n')
                    count += 1
            else:
                # Write as-is
                if len(text.strip()) >= min_length:
                    f.write(text.strip() + '\n')
                    count += 1
            
            total_texts += 1
            
            # Progress update every 1000 texts
            if total_texts % 1000 == 0:
                print(f"Processed {total_texts} texts, saved {count} samples...")
    
    print(f"\n✅ Extraction complete!")
    print(f"   Total texts processed: {total_texts}")
    print(f"   Samples saved: {count}")
    print(f"   Output file: {output_path}")
    print(f"   File size: {output_path.stat().st_size / (1024*1024):.2f} MB")


def main():
    parser = argparse.ArgumentParser(description='Extract text from database for training')
    parser.add_argument('--type', type=str, choices=['sqlite', 'sql', 'json'],
                        required=True, help='Database type')
    parser.add_argument('--output', type=str, default='data/database_extracted.txt',
                        help='Output file path')
    parser.add_argument('--limit', type=int, help='Maximum number of samples to extract')
    parser.add_argument('--min-length', type=int, default=10,
                        help='Minimum text length')
    
    # SQLite options
    parser.add_argument('--db-path', type=str, help='SQLite database path')
    parser.add_argument('--table', type=str, help='Table name')
    parser.add_argument('--column', type=str, help='Text column name')
    parser.add_argument('--where', type=str, help='WHERE clause (e.g., "WHERE length(text) > 100")')
    
    # SQL query options
    parser.add_argument('--connection', type=str, help='Database connection string')
    parser.add_argument('--query', type=str, help='SQL query')
    parser.add_argument('--text-column', type=int, default=0, help='Text column index (0-based)')
    
    # JSON options
    parser.add_argument('--json-path', type=str, help='JSON/JSONL file path')
    parser.add_argument('--text-field', type=str, help='JSON field name containing text')
    
    parser.add_argument('--no-clean', action='store_true', help='Do not clean/split text')
    
    args = parser.parse_args()
    
    # Extract based on type
    if args.type == 'sqlite':
        if not all([args.db_path, args.table, args.column]):
            print("Error: --db-path, --table, and --column required for SQLite")
            return
        
        texts = extract_from_sqlite(
            db_path=args.db_path,
            table=args.table,
            text_column=args.column,
            limit=args.limit,
            where_clause=args.where,
        )
    
    elif args.type == 'sql':
        if not all([args.connection, args.query]):
            print("Error: --connection and --query required for SQL")
            return
        
        texts = extract_from_sql(
            connection_string=args.connection,
            query=args.query,
            text_column=args.text_column,
        )
    
    elif args.type == 'json':
        if not all([args.json_path, args.text_field]):
            print("Error: --json-path and --text-field required for JSON")
            return
        
        texts = extract_from_json_file(
            json_path=args.json_path,
            text_field=args.text_field,
            limit=args.limit,
        )
    
    # Save to training file
    save_to_training_file(
        texts=texts,
        output_path=args.output,
        min_length=args.min_length,
        max_samples=args.limit,
        clean_text=not args.no_clean,
    )


if __name__ == '__main__':
    main()

