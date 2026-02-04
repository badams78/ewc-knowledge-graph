#!/usr/bin/env python3
"""
Generate embeddings from full PDF text
Uses OpenAI's text-embedding-3-small (same as database)
"""

import os
import json
import numpy as np
from pathlib import Path
from datetime import datetime
import time
import sys
print("DEBUG: Script started", flush=True)

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
    print("DEBUG: OpenAI imported", flush=True)
except ImportError:
    OPENAI_AVAILABLE = False
    print("WARNING: openai not installed. Install with: pip install openai")

from neo4j_connection import Neo4jConnection

OUTPUT_DIR = Path(r"C:\Users\Brandon Adams\EWC_Database_v2")
EXTRACTED_FILE = OUTPUT_DIR / "extracted_texts.json"
MATCHED_FILE = OUTPUT_DIR / "pdf_article_matches.json"
EMBEDDINGS_FILE = OUTPUT_DIR / "fulltext_embeddings.json"
EMBEDDINGS_NPY = OUTPUT_DIR / "fulltext_embeddings.npy"

# OpenAI settings
EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536
MAX_TOKENS = 8000  # Leave headroom from 8192 limit
BATCH_SIZE = 10

def truncate_text(text, max_chars=30000):
    """
    Truncate text to fit within token limits
    ~4 chars per token on average for English
    """
    if len(text) <= max_chars:
        return text
    
    # Take from beginning and end (important context often at both)
    half = max_chars // 2
    return text[:half] + "\n\n[...truncated...]\n\n" + text[-half:]

def get_embedding(client, text, model=EMBEDDING_MODEL):
    """Get embedding for a single text"""
    text = truncate_text(text)
    response = client.embeddings.create(input=text, model=model)
    return response.data[0].embedding

def get_embeddings_batch(client, texts, model=EMBEDDING_MODEL):
    """Get embeddings for a batch of texts"""
    texts = [truncate_text(t) for t in texts]
    response = client.embeddings.create(input=texts, model=model)
    return [item.embedding for item in response.data]

def load_matches_and_texts():
    """Load matched PDFs and their extracted text"""
    
    # Load matches
    if not MATCHED_FILE.exists():
        raise FileNotFoundError(f"Run pdf_extraction.py first! Missing: {MATCHED_FILE}")
    
    with open(MATCHED_FILE, 'r', encoding='utf-8') as f:
        match_data = json.load(f)
    
    matches = {m['pdf_filename']: m for m in match_data['matches']}
    print(f"Loaded {len(matches)} PDF-article matches")
    
    # Load extracted texts
    if not EXTRACTED_FILE.exists():
        raise FileNotFoundError(f"Run pdf_extraction.py first! Missing: {EXTRACTED_FILE}")
    
    with open(EXTRACTED_FILE, 'r', encoding='utf-8') as f:
        extraction_data = json.load(f)
    
    extractions = {e['filename']: e for e in extraction_data['extractions'] if e.get('extraction_success')}
    print(f"Loaded {len(extractions)} extracted texts")
    
    # Combine: only matched PDFs with successful extraction
    combined = []
    for filename, match in matches.items():
        if filename in extractions:
            extraction = extractions[filename]
            combined.append({
                'post_id': match['post_id'],
                'article_title': match['article_title'],
                'pdf_filename': filename,
                'text': extraction['text'],
                'word_count': extraction['word_count']
            })
    
    print(f"Combined: {len(combined)} articles with full text")
    return combined

def generate_embeddings(articles, resume=True):
    """Generate embeddings for all articles"""
    
    if not OPENAI_AVAILABLE:
        print("ERROR: OpenAI library not available")
        return None
    
    print("DEBUG: Checking API Key", flush=True)
    # Check for API key
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("ERROR: OPENAI_API_KEY environment variable not set")
        print("Set it with: $env:OPENAI_API_KEY = 'your-key-here'")
        return None
    
    print("DEBUG: Creating OpenAI Client", flush=True)
    try:
        client = OpenAI(api_key=api_key)
        print("DEBUG: OpenAI Client Created", flush=True)
    except Exception as e:
        print(f"ERROR: Failed to create client: {e}", flush=True)
        return None
    
    # Load existing embeddings if resuming
    existing = {}
    print("DEBUG: Checking existing embeddings", flush=True)
    if resume and EMBEDDINGS_FILE.exists():
        with open(EMBEDDINGS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
            existing = {e['post_id']: e for e in data.get('embeddings', [])}
        print(f"Loaded {len(existing)} existing embeddings")
    
    # Filter to articles needing embeddings
    to_process = [a for a in articles if a['post_id'] not in existing]
    print(f"Need to generate embeddings for {len(to_process)} articles")
    
    if not to_process:
        print("All embeddings already generated!")
        return list(existing.values())
    
    results = list(existing.values())
    start_time = datetime.now()
    processed = 0
    errors = 0
    
    # Process in batches
    for i in range(0, len(to_process), BATCH_SIZE):
        batch = to_process[i:i+BATCH_SIZE]
        texts = [a['text'] for a in batch]
        
        try:
            embeddings = get_embeddings_batch(client, texts)
            
            for article, embedding in zip(batch, embeddings):
                results.append({
                    'post_id': article['post_id'],
                    'article_title': article['article_title'],
                    'embedding': embedding,
                    'word_count': article['word_count']
                })
            
            processed += len(batch)
            
        except Exception as e:
            print(f"  Batch error: {e}")
            errors += len(batch)
            
            # Try individual processing for failed batch
            for article in batch:
                try:
                    embedding = get_embedding(client, article['text'])
                    results.append({
                        'post_id': article['post_id'],
                        'article_title': article['article_title'],
                        'embedding': embedding,
                        'word_count': article['word_count']
                    })
                    processed += 1
                    errors -= 1
                except Exception as e2:
                    print(f"    Individual error for {article['post_id']}: {e2}")
            
            time.sleep(1)  # Rate limit backoff
        
        # Progress update
        total_done = len(existing) + processed
        pct = total_done / len(articles) * 100
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = processed / elapsed if elapsed > 0 else 0
        
        print(f"  Progress: {total_done}/{len(articles)} ({pct:.1f}%) - {rate:.1f} articles/sec", flush=True)
        
        # Save checkpoint every 50 articles
        if processed % 50 < BATCH_SIZE:
            save_embeddings(results)
        
        # Rate limiting
        time.sleep(0.1)
    
    # Final save
    save_embeddings(results)
    
    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nEmbedding generation complete!")
    print(f"  Total time: {elapsed/60:.1f} minutes")
    print(f"  Generated: {processed}")
    print(f"  Errors: {errors}")
    
    return results

def save_embeddings(results):
    """Save embeddings to JSON and NPY"""
    
    # JSON format (with embeddings as lists)
    output = {
        'generated_at': datetime.now().isoformat(),
        'model': EMBEDDING_MODEL,
        'total': len(results),
        'embeddings': results
    }
    
    with open(EMBEDDINGS_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)
    
    # NPY format (for fast numpy loading)
    if results and results[0].get('embedding'):
        post_ids = [r['post_id'] for r in results]
        embeddings_array = np.array([r['embedding'] for r in results])
        
        np.savez(
            str(EMBEDDINGS_NPY).replace('.npy', '.npz'),
            post_ids=post_ids,
            embeddings=embeddings_array
        )
    
    print(f"  Saved checkpoint: {len(results)} embeddings")

def main():
    print("=" * 60)
    print("FULL-TEXT EMBEDDING GENERATION")
    print("=" * 60)
    
    # Load data
    print("DEBUG: Calling load_matches_and_texts", flush=True)
    articles = load_matches_and_texts()
    print(f"DEBUG: Loaded {len(articles)} articles", flush=True)
    
    # Generate embeddings
    print("\n" + "=" * 60)
    print("GENERATING EMBEDDINGS")
    print("=" * 60)
    
    results = generate_embeddings(articles, resume=True)
    
    if results:
        print("\n" + "=" * 60)
        print("EMBEDDING GENERATION COMPLETE")
        print("=" * 60)
        print(f"""
Summary:
- Articles with full-text embeddings: {len(results)}
- Embedding model: {EMBEDDING_MODEL}
- Embedding dimensions: {EMBEDDING_DIM}

Next step:
  python taxonomy_development_v2.py
""")

if __name__ == "__main__":
    main()
