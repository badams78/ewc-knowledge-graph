#!/usr/bin/env python3
"""
Phase 4 v2: Taxonomy Development with Full-Text Embeddings
Uses embeddings generated from complete PDF content
"""

import numpy as np
import json
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from neo4j_connection import Neo4jConnection

OUTPUT_DIR = Path(r"C:\Users\Brandon Adams\EWC_Database_v2")
EMBEDDINGS_NPZ = OUTPUT_DIR / "fulltext_embeddings.npz"
EMBEDDINGS_JSON = OUTPUT_DIR / "fulltext_embeddings.json"

def load_fulltext_embeddings():
    """Load embeddings generated from full PDF text"""
    
    # Try NPZ first (faster)
    if EMBEDDINGS_NPZ.exists():
        print("Loading embeddings from NPZ file...")
        data = np.load(EMBEDDINGS_NPZ, allow_pickle=True)
        post_ids = data['post_ids'].tolist()
        embeddings = data['embeddings']
        print(f"Loaded {len(post_ids)} embeddings ({embeddings.shape[1]} dimensions)")
        return post_ids, embeddings
    
    # Fall back to JSON
    if EMBEDDINGS_JSON.exists():
        print("Loading embeddings from JSON file...")
        with open(EMBEDDINGS_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        post_ids = [e['post_id'] for e in data['embeddings']]
        embeddings = np.array([e['embedding'] for e in data['embeddings']])
        print(f"Loaded {len(post_ids)} embeddings ({embeddings.shape[1]} dimensions)")
        return post_ids, embeddings
    
    raise FileNotFoundError(
        f"No embedding files found! Run generate_embeddings.py first.\n"
        f"Expected: {EMBEDDINGS_NPZ} or {EMBEDDINGS_JSON}"
    )

def fetch_article_metadata(post_ids):
    """Fetch article metadata from database for the given post_ids"""
    print("\nFetching article metadata from database...")
    
    with Neo4jConnection() as conn:
        results = conn.query("""
            MATCH (a:Article)
            WHERE a.post_id IN $post_ids
            OPTIONAL MATCH (a)-[:HAS_PRIMARY_TOPIC_V2]->(pt:PrimaryTopicV2)
            OPTIONAL MATCH (a)-[:HAS_CATEGORY]->(c:Category)
            RETURN a.post_id as post_id,
                   a.title as title,
                   pt.name as existing_primary_topic,
                   collect(DISTINCT c.name)[0] as existing_category
        """, {'post_ids': post_ids})
    
    # Index by post_id
    metadata = {r['post_id']: r for r in results}
    print(f"Fetched metadata for {len(metadata)} articles")
    return metadata

def find_optimal_clusters(embeddings, min_k=7, max_k=15):
    """Find optimal number of clusters using silhouette score"""
    print(f"\nFinding optimal number of Tier 1 clusters ({min_k}-{max_k})...")
    
    scores = {}
    for k in range(min_k, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(embeddings)
        score = silhouette_score(embeddings, labels)
        scores[k] = score
        print(f"  k={k}: silhouette={score:.4f}")
    
    best_k = max(scores, key=scores.get)
    print(f"\nOptimal k={best_k} (silhouette={scores[best_k]:.4f})")
    return best_k, scores

def do_kmeans_clustering(embeddings, n_clusters):
    """Cluster embeddings using K-means"""
    # Handle edge case: need at least n_clusters samples
    if len(embeddings) < n_clusters:
        n_clusters = max(1, len(embeddings))
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings)
    
    # Silhouette score requires at least 2 clusters with samples
    n_unique = len(set(labels))
    if n_unique < 2 or len(embeddings) < 3:
        score = 0.0
    else:
        score = silhouette_score(embeddings, labels)
    
    return labels, kmeans.cluster_centers_, score

def build_hierarchical_taxonomy(embeddings, post_ids, metadata, n_tier1=12, n_tier2_per_cluster=8):
    """
    Build hierarchical taxonomy from full-text embeddings
    """
    print(f"\n" + "=" * 60)
    print("BUILDING HIERARCHICAL TAXONOMY (Full-Text Embeddings)")
    print("=" * 60)
    
    # Tier 1 clustering
    print(f"\nTier 1: Clustering into {n_tier1} primary categories...")
    tier1_labels, tier1_centers, tier1_score = do_kmeans_clustering(embeddings, n_tier1)
    
    taxonomy = {
        "version": "v2_fulltext",
        "tier1_count": n_tier1,
        "tier1_silhouette": tier1_score,
        "total_articles": len(post_ids),
        "clusters": {}
    }
    
    tier2_assignments = {}
    
    # For each Tier 1 cluster, analyze and create Tier 2
    for cluster_id in range(n_tier1):
        mask = tier1_labels == cluster_id
        cluster_indices = np.where(mask)[0]
        cluster_embeddings = embeddings[mask]
        cluster_post_ids = [post_ids[i] for i in cluster_indices]
        
        n_articles = len(cluster_post_ids)
        
        # Get metadata for cluster articles
        cluster_meta = [metadata.get(pid, {}) for pid in cluster_post_ids]
        
        # Determine number of Tier 2 clusters
        n_tier2 = min(max(3, n_articles // 50), n_tier2_per_cluster)
        
        if n_articles < 10:
            tier2_labels = np.zeros(n_articles, dtype=int)
            tier2_score = 0
        else:
            tier2_labels, _, tier2_score = do_kmeans_clustering(cluster_embeddings, n_tier2)
        
        # Analyze cluster content from existing categories
        existing_topics = [m.get('existing_primary_topic') for m in cluster_meta if m.get('existing_primary_topic')]
        existing_cats = [m.get('existing_category') for m in cluster_meta if m.get('existing_category')]
        
        top_topics = Counter(existing_topics).most_common(5)
        top_cats = Counter(existing_cats).most_common(5)
        
        # Sample titles
        sample_titles = [m.get('title', '')[:100] for m in cluster_meta[:15] if m.get('title')]
        
        # Build Tier 2 structure
        tier2_clusters = {}
        for t2_id in range(max(tier2_labels) + 1):
            t2_mask = tier2_labels == t2_id
            t2_indices = np.where(t2_mask)[0]
            t2_meta = [cluster_meta[i] for i in t2_indices]
            
            t2_topics = Counter([m.get('existing_primary_topic') for m in t2_meta if m.get('existing_primary_topic')]).most_common(3)
            t2_cats = Counter([m.get('existing_category') for m in t2_meta if m.get('existing_category')]).most_common(3)
            t2_sample = [m.get('title', '')[:80] for m in t2_meta[:5] if m.get('title')]
            
            tier2_clusters[f"T1-{cluster_id}_T2-{t2_id}"] = {
                "article_count": len(t2_indices),
                "top_existing_topics": t2_topics,
                "top_existing_categories": t2_cats,
                "sample_titles": t2_sample
            }
        
        for idx, pid in enumerate(cluster_post_ids):
            tier2_assignments[pid] = int(tier2_labels[idx])

        taxonomy["clusters"][f"Cluster_{cluster_id}"] = {
            "article_count": n_articles,
            "tier2_count": len(tier2_clusters),
            "tier2_silhouette": tier2_score,
            "top_existing_topics": top_topics,
            "top_existing_categories": top_cats,
            "sample_titles": sample_titles,
            "tier2_clusters": tier2_clusters
        }
    
    return taxonomy, tier1_labels, tier1_centers, tier2_assignments

def compute_loading_matrix(embeddings, tier1_centers, tier1_labels):
    """
    Compute article-tag loading matrix
    Loading = 1 - normalized_distance to each cluster center
    """
    print("\n" + "=" * 60)
    print("COMPUTING ARTICLE-TAG LOADING MATRIX")
    print("=" * 60)
    
    n_articles = len(embeddings)
    n_clusters = len(tier1_centers)
    
    # Compute distances to all cluster centers
    distances = np.zeros((n_articles, n_clusters))
    for i, emb in enumerate(embeddings):
        for j, center in enumerate(tier1_centers):
            distances[i, j] = np.linalg.norm(emb - center)
    
    # Convert to loadings
    max_dist = distances.max(axis=1, keepdims=True)
    normalized_dist = distances / (max_dist + 1e-10)
    loadings = 1 - normalized_dist
    
    # Scale to 0-1 range per article
    loadings = (loadings - loadings.min(axis=1, keepdims=True)) / \
               (loadings.max(axis=1, keepdims=True) - loadings.min(axis=1, keepdims=True) + 1e-10)
    
    print(f"Loading matrix shape: {loadings.shape} (articles x clusters)")
    print(f"Loading range: {loadings.min():.3f} to {loadings.max():.3f}")
    
    # Summary statistics
    print(f"\n{'Cluster':<12} {'Mean Load':>10} {'Articles >0.5':>15} {'Articles >0.7':>15}")
    print("-" * 55)
    
    for j in range(n_clusters):
        mean_load = loadings[:, j].mean()
        above_05 = (loadings[:, j] > 0.5).sum()
        above_07 = (loadings[:, j] > 0.7).sum()
        print(f"Cluster_{j:<3} {mean_load:>10.3f} {above_05:>15} {above_07:>15}")
    
    # Cross-loading analysis
    multi_loading = (loadings > 0.5).sum(axis=1)
    print(f"\nCross-loading analysis:")
    print(f"  Articles loading >0.5 on 1 cluster: {(multi_loading == 1).sum()}")
    print(f"  Articles loading >0.5 on 2 clusters: {(multi_loading == 2).sum()}")
    print(f"  Articles loading >0.5 on 3+ clusters: {(multi_loading >= 3).sum()}")
    
    return loadings

def print_taxonomy_summary(taxonomy, metadata, post_ids, tier1_labels):
    """Print readable summary of derived taxonomy"""
    print("\n" + "=" * 60)
    print("TAXONOMY SUMMARY (Full-Text Analysis)")
    print("=" * 60)
    
    # Sort clusters by size
    clusters_by_size = sorted(
        taxonomy["clusters"].items(),
        key=lambda x: x[1]["article_count"],
        reverse=True
    )
    
    print(f"\n{'Cluster':<12} {'Articles':>8} {'Dominant Theme':<45}")
    print("-" * 70)
    
    for cluster_name, cluster_data in clusters_by_size:
        count = cluster_data["article_count"]
        if cluster_data["top_existing_topics"]:
            theme = cluster_data["top_existing_topics"][0][0]
        elif cluster_data["top_existing_categories"]:
            theme = cluster_data["top_existing_categories"][0][0]
        else:
            theme = "Unknown"
        
        print(f"{cluster_name:<12} {count:>8} {theme[:45]:<45}")
    
    # Detailed view
    print("\n" + "=" * 60)
    print("DETAILED CLUSTER ANALYSIS")
    print("=" * 60)
    
    for cluster_name, cluster_data in clusters_by_size[:6]:  # Top 6 clusters
        print(f"\n{'='*60}")
        print(f"{cluster_name} ({cluster_data['article_count']} articles)")
        print(f"{'='*60}")
        
        print(f"\nTop Existing Topics:")
        for topic, count in cluster_data["top_existing_topics"][:3]:
            print(f"  - {topic}: {count}")
        
        print(f"\nSample Titles:")
        for title in cluster_data["sample_titles"][:5]:
            print(f"  - {title[:80]}...")

def save_results(taxonomy, post_ids, tier1_labels, loadings, metadata, tier2_assignments):
    """Save all results"""
    print("\n" + "=" * 60)
    print("SAVING RESULTS")
    print("=" * 60)
    
    # Save taxonomy
    with open(OUTPUT_DIR / "derived_taxonomy_v2.json", "w") as f:
        json.dump(taxonomy, f, indent=2, default=str)
    print("Saved: derived_taxonomy_v2.json")
    
    # Save loading matrix (numpy)
    np.save(OUTPUT_DIR / "loading_matrix_v2.npy", loadings)
    print("Saved: loading_matrix_v2.npy")
    
    # Save loading sample (JSON)
    loading_sample = []
    for i in range(min(100, len(post_ids))):
        meta = metadata.get(post_ids[i], {})
        row = {
            "post_id": post_ids[i],
            "title": meta.get('title', '')[:60],
            "primary_cluster": int(tier1_labels[i]),
            "subcluster": tier2_assignments.get(post_ids[i], 0)
        }
        for j in range(loadings.shape[1]):
            row[f"Cluster_{j}"] = round(float(loadings[i, j]), 3)
        loading_sample.append(row)
    
    with open(OUTPUT_DIR / "loading_matrix_v2_sample.json", "w") as f:
        json.dump(loading_sample, f, indent=2)
    print("Saved: loading_matrix_v2_sample.json")
    
    # Save cluster assignments
    assignments = []
    for i, pid in enumerate(post_ids):
        meta = metadata.get(pid, {})
        assignments.append({
            "post_id": pid,
            "title": meta.get('title', ''),
            "cluster": int(tier1_labels[i]),
            "subcluster": tier2_assignments.get(pid, 0),
            "existing_primary_topic": meta.get('existing_primary_topic'),
            "existing_category": meta.get('existing_category')
        })
    
    with open(OUTPUT_DIR / "cluster_assignments_v2.json", "w") as f:
        json.dump(assignments, f, indent=2)
    print("Saved: cluster_assignments_v2.json")

def main():
    print("=" * 60)
    print("TAXONOMY DEVELOPMENT v2 (Full-Text Embeddings)")
    print("=" * 60)
    
    # Load embeddings
    post_ids, embeddings = load_fulltext_embeddings()
    
    # Fetch metadata
    metadata = fetch_article_metadata(post_ids)
    
    # Find optimal clusters
    best_k, scores = find_optimal_clusters(embeddings, min_k=7, max_k=15)
    n_tier1 = best_k if 8 <= best_k <= 14 else 12
    
    # Build taxonomy
    taxonomy, tier1_labels, tier1_centers, tier2_assignments = build_hierarchical_taxonomy(
        embeddings, post_ids, metadata, n_tier1=n_tier1
    )
    
    # Compute loading matrix
    loadings = compute_loading_matrix(embeddings, tier1_centers, tier1_labels)
    
    # Print summary
    print_taxonomy_summary(taxonomy, metadata, post_ids, tier1_labels)
    
    # Save results
    save_results(taxonomy, post_ids, tier1_labels, loadings, metadata, tier2_assignments)
    
    print("\n" + "=" * 60)
    print("TAXONOMY DEVELOPMENT v2 COMPLETE")
    print("=" * 60)
    print(f"""
Summary:
- Articles analyzed: {len(post_ids)} (with full PDF text)
- Tier 1 Clusters: {n_tier1}
- Silhouette Score: {taxonomy['tier1_silhouette']:.4f}

Output files:
- derived_taxonomy_v2.json
- loading_matrix_v2.npy
- cluster_assignments_v2.json

Next: Run visualizations_v2.py to generate updated charts
""")

if __name__ == "__main__":
    main()
