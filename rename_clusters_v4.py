#!/usr/bin/env python3
"""
Rename clusters based on Simplified Editorial Overrides.
v4: Removes compound names ("& B") in favor of dominant topics.
"""

import json
from pathlib import Path

OUTPUT_DIR = Path(r"C:\Users\Brandon Adams\EWC_Database_v2")
TAXONOMY_FILE = OUTPUT_DIR / "derived_taxonomy_v2.json"

# V4 SIMPLIFIED MAP
CLUSTER_OVERRIDES = {
    0: "Productivity",
    1: "Fiscal Policy",
    2: "Global Macro",
    3: "Trade Policy",
    4: "Inequality",
    5: "Private Markets",
    6: "Labor Economics",
    7: "Society",
    8: "Technology"
}

def clean_subcluster_name(name, primary_name):
    """
    Cleans subcluster names by removing generic terms and suffixes.
    """
    # Remove "& Database" or "& News article" - though viz script handles this too now
    name = name.split(" & ")[0]
    
    # If the subcluster name is just the Primary name, genericize it
    if name.lower() == primary_name.lower():
        return "General"
        
    return name

def main():
    print("Loading taxonomy...")
    with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f:
        taxonomy = json.load(f)
        
    print("\nApplying Semantic Renaming (V4)...")
    
    for c_key, c_data in taxonomy['clusters'].items():
        c_id = int(c_key.split('_')[1])
        
        # 1. Apply Primary Override
        if c_id in CLUSTER_OVERRIDES:
            new_name = CLUSTER_OVERRIDES[c_id]
            old_name = c_data.get('top_existing_topics', [['Unknown']])[0][0]
            print(f"Cluster {c_id}: {old_name} -> {new_name}")
            
            # Inject as top topic
            if 'top_existing_topics' not in c_data: c_data['top_existing_topics'] = []
            c_data['top_existing_topics'].insert(0, [new_name, 1000])
            
        # 2. Clean Subclusters
        primary_name = c_data['top_existing_topics'][0][0]
        
        for s_key, s_data in c_data.get('tier2_clusters', {}).items():
            current_s_name = s_data.get('top_existing_topics', [['Unknown']])[0][0]
            
            # Clean it 
            clean_s = clean_subcluster_name(current_s_name, primary_name)
            
            # Update if changed
            if clean_s != current_s_name:
                s_data['top_existing_topics'].insert(0, [clean_s, 1000])

    print("\nSaving updated taxonomy...")
    with open(TAXONOMY_FILE, 'w', encoding='utf-8') as f:
        json.dump(taxonomy, f, indent=2)
        
    print("Done!")

if __name__ == "__main__":
    main()
