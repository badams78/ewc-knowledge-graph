
print("Start Debug Script")
import sys
print(f"Python: {sys.version}")

try:
    import openai
    print("OpenAI imported")
except ImportError:
    print("OpenAI NOT installed")

try:
    import neo4j
    print("Neo4j imported")
except ImportError:
    print("Neo4j NOT installed")

import json
from pathlib import Path
import time

p = Path("extracted_texts.json")
print(f"File exists: {p.exists()}")
print(f"File size: {p.stat().st_size}")

print("Loading JSON...")
start = time.time()
with open(p, 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f"JSON loaded in {time.time() - start:.2f}s")
print(f"Keys: {list(data.keys())}")
