#!/usr/bin/env python3
"""
Rename clusters based on TF-IDF keyphrase extraction from titles.
Updates derived_taxonomy_v2.json with meaningful names.
"""

import json
import re
from pathlib import Path
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

OUTPUT_DIR = Path(r"C:\Users\Brandon Adams\EWC_Database_v2")
TAXONOMY_FILE = OUTPUT_DIR / "derived_taxonomy_v2.json"
ASSIGNMENTS_FILE = OUTPUT_DIR / "cluster_assignments_v2.json"

CUSTOM_STOP_WORDS = list(ENGLISH_STOP_WORDS) + [
    'new', 'york', 'times', 'wsj', 'report', 'study', 'finds', 'argues', 'suggests',
    'notes', 'citing', 'data', 'analysis', 'percent', 'year', 'years', '2023', '2024',
    '2025', 'high', 'low', 'rate', 'rates', 'increase', 'decline', 'growth', 'level',
    'countries', 'global', 'market', 'markets', 'economic', 'economy', 'policy',
    'vs', 'say', 'says', 'likely', 'amid', 'term', 'change', 'average', 'recent',
    'just', 'like', 'significant', 'share', 'rose', 'fell', 'higher', 'lower',
    'potential', 'impact', 'estimated', 'estimate', 'estimates', 'total', 'using',
    'including', 'associated', 'levels', 'continue', 'continued', 'remains', 'remain',
    'based', 'according', 'compared', 'relative', 'evidence', 'suggest', 'argue',
    'billion', 'million', 'trillion', 'points', 'point', 'quarter', 'month'
]

def generate_name(titles, n_gram_range=(2, 3)):
    if not titles:
        return "Unknown Cluster"
    
    # Preprocess
    clean_titles = []
    for t in titles:
        # Remove handles, urls, special chars
        t = re.sub(r'@[A-Za-z0-9_]+', '', t)
        t = re.sub(r'http\S+', '', t)
        t = re.sub(r'[^\w\s]', '', t)
        clean_titles.append(t)
        
    try:
        tfidf = TfidfVectorizer(
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=n_gram_range,
            max_features=10,
            token_pattern=r'(?u)\b[a-zA-Z][a-zA-Z]+\b'  # Ignore numbers
        )
        matrix = tfidf.fit_transform(clean_titles)
        sum_scores = matrix.sum(axis=0)
        
        # Get top phrase
        scores = [(word, sum_scores[0, idx]) for word, idx in tfidf.vocabulary_.items()]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if scores:
            top_phrase = scores[0][0].title()
            return top_phrase
    except ValueError:
        pass
        
    # Fallback to single words if bigrams fail
    try:
        tfidf = TfidfVectorizer(
            stop_words=CUSTOM_STOP_WORDS,
            ngram_range=(1, 1),
            max_features=5
        )
        matrix = tfidf.fit_transform(clean_titles)
        sum_scores = matrix.sum(axis=0)
        scores = [(word, sum_scores[0, idx]) for word, idx in tfidf.vocabulary_.items()]
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        
        if scores:
            return scores[0][0].title()
    except:
        pass
        
    return "Miscellaneous Topics"

def main():
    print("Loading data...")
    with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
        
    with open(ASSIGNMENTS_FILE, 'r', encoding='utf-8') as f:
        assignments = json.load(f)
        
    # Group titles
    # Cluster -> [titles]
    # Cluster_Subcluster -> [titles]
    cluster_titles = {}
    subcluster_titles = {}
    
    for a in assignments:
        c_id = a['cluster']
        s_id = a['subcluster']
        title = a['title']
        
        # Primary key
        if c_id not in cluster_titles: cluster_titles[c_id] = []
        cluster_titles[c_id].append(title)
        
        # Subcluster key: T1-0_T2-0
        s_key = f"T1-{c_id}_T2-{s_id}"
        if s_key not in subcluster_titles: subcluster_titles[s_key] = []
        subcluster_titles[s_key].append(title)
        
    # Rename Primaries
    print("\nRenaming Primary Clusters...")
    for c_key, c_data in taxonomy['clusters'].items():
        c_id = int(c_key.split('_')[1])
        titles = cluster_titles.get(c_id, [])
        
        new_name = generate_name(titles, n_gram_range=(2, 3))
        old_name = c_data.get('top_existing_topics', [['Unknown']])[0][0] if c_data.get('top_existing_topics') else "Unknown"
        
        print(f"Cluster {c_id}: {old_name} -> {new_name}")
        
        # Inject into top_existing_topics so downstream scripts pick it up
        # We replace the first item or insert it
        if 'top_existing_topics' not in c_data:
            c_data['top_existing_topics'] = []
            
        # Insert as fake topic with high count
        c_data['top_existing_topics'].insert(0, [new_name, 999])
        
        # Rename Subclusters
        for s_key, s_data in c_data.get('tier2_clusters', {}).items():
            s_titles = subcluster_titles.get(s_key, [])
            s_name = generate_name(s_titles, n_gram_range=(2, 3))
            
            # If subcluster name is same as primary, try 3-gram or fallback
            if s_name == new_name:
                 s_name = generate_name(s_titles, n_gram_range=(1, 2))
            
            # Inject
            if 'top_existing_topics' not in s_data:
                s_data['top_existing_topics'] = []
            s_data['top_existing_topics'].insert(0, [s_name, 999])

    # Save
    print("\nSaving updated taxonomy...")
    with open(TAXONOMY_FILE, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
