import json
import numpy as np
import pandas as pd
import umap
import plotly.express as px
import argparse
from tqdm.auto import tqdm
from sklearn.cluster import KMeans

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSONL embeddings file')
    parser.add_argument('--titles', required=False, help='Optional CSV file containing paper titles for hover info')
    parser.add_argument('--output', default='umap_clusters.html', help='Output HTML file for visualization')
    parser.add_argument('--n-clusters', type=int, default=5, help='Number of clusters for topic grouping')
    parser.add_argument('--neighbors', type=int, default=15, help='UMAP n_neighbors parameter (lower = more local structure)')
    args = parser.parse_args()

    # -------------------- Load embeddings --------------------
    embeddings = []
    paper_ids = []

    print(f"Loading embeddings from {args.input} ...")
    with open(args.input, 'r') as f:
        for line in tqdm(f):
            item = json.loads(line)
            paper_ids.append(item['paper_id'])
            embeddings.append(item['embedding'])

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} papers with {embeddings.shape[1]}-dim embeddings")

    # -------------------- Load titles --------------------
    titles = [f"Paper {i}" for i in range(len(embeddings))]
    if args.titles:
        df_titles = pd.read_csv(args.titles)
        if 'title' in df_titles.columns:
            titles = df_titles['title'].tolist()[:len(embeddings)]

    # -------------------- Run UMAP --------------------
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=min(args.neighbors, len(embeddings)-1),
        min_dist=0.1,
        n_components=2,
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # -------------------- Clustering --------------------
    print(f"Running KMeans clustering with {args.n_clusters} clusters...")
    kmeans = KMeans(n_clusters=args.n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)

    # -------------------- Prepare DataFrame --------------------
    df_vis = pd.DataFrame(embedding_2d, columns=['x', 'y'])
    df_vis['paper_id'] = paper_ids
    df_vis['title'] = titles
    df_vis['cluster'] = cluster_labels

    # -------------------- Visualization --------------------
    fig = px.scatter(
        df_vis,
        x='x', y='y',
        color='cluster',
        hover_data=['paper_id', 'title'],
        title=f'SPECTER Embeddings (UMAP 2D) — {args.n_clusters} Clusters',
        color_continuous_scale='Viridis',
        width=950, height=650
    )

    fig.write_html(args.output)
    print(f"Saved interactive clustered plot to {args.output}")
    fig.show()

if __name__ == "__main__":
    main()