#python src\demo1.py --input output/embeddings.jsonl --titles data/biorxiv_sciedu.csv --output output/umap_hdbscan_clusters_fast.html
import json
import numpy as np
import pandas as pd
import umap
import hdbscan
import plotly.express as px
from tqdm.auto import tqdm
from sklearn.metrics import silhouette_score
import argparse
import pathlib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True, help='Path to JSONL embeddings file')
    parser.add_argument('--titles', required=False, help='Optional CSV file containing paper titles for hover info')
    parser.add_argument('--output', default='umap_hdbscan_clusters_fast.html', help='Output HTML file for visualization')
    parser.add_argument('--neighbors', type=int, default=10, help='UMAP n_neighbors parameter (lower = more local structure)')
    parser.add_argument('--min-dist', type=float, default=0.1, help='UMAP min_dist parameter')
    parser.add_argument('--min-cluster-size', type=int, default=30, help='HDBSCAN min_cluster_size parameter')
    args = parser.parse_args()

    # -------------------- Load embeddings --------------------
    print(f"Loading embeddings from {args.input} ...")
    embeddings, paper_ids = [], []

    with open(args.input, "r", encoding="utf-8") as f:
        for line in tqdm(f):
            item = json.loads(line)
            paper_ids.append(item["paper_id"])
            embeddings.append(item["embedding"])

    embeddings = np.array(embeddings)
    print(f"Loaded {len(embeddings)} papers with {embeddings.shape[1]}-dim embeddings")

    # -------------------- Load titles --------------------
    titles = [f"Paper {i}" for i in range(len(embeddings))]
    if args.titles:
        df_titles = pd.read_csv(args.titles)
        if "title" in df_titles.columns:
            titles = df_titles["title"].astype(str).fillna("(No title)").tolist()[:len(embeddings)]

    # -------------------- Run UMAP --------------------
    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(
        n_neighbors=min(args.neighbors, len(embeddings) - 1),
        min_dist=args.min_dist,
        n_components=2,
        metric="cosine",
        random_state=42
    )
    embedding_2d = reducer.fit_transform(embeddings)

    # -------------------- Run HDBSCAN --------------------
    print("Running HDBSCAN clustering...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=args.min_cluster_size,
        min_samples=5,
        metric="euclidean",
        cluster_selection_method="eom"
    )
    cluster_labels = clusterer.fit_predict(embedding_2d)

    n_clusters_found = len(np.unique(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    print(f"Found {n_clusters_found} clusters and {np.sum(cluster_labels == -1)} noise points")

    # -------------------- Prepare DataFrame --------------------
    df_vis = pd.DataFrame(embedding_2d, columns=["x", "y"])
    df_vis["paper_id"] = paper_ids
    df_vis["title"] = titles
    df_vis["cluster"] = cluster_labels.astype(str)

    # -------------------- Compute Silhouette Score --------------------
    mask = cluster_labels != -1
    if np.sum(mask) > 1 and len(np.unique(cluster_labels[mask])) > 1:
        silhouette = silhouette_score(embedding_2d[mask], cluster_labels[mask], metric="euclidean")
        print(f"Silhouette Score (not include interference ): {silhouette:.4f}")
    else:
        print("Not enough valid clusters to calculate Silhouette Score.")

    # -------------------- Visualization --------------------
    print(f"Creating interactive scatter plot... Saving to: {args.output}")
    cluster_order = sorted(df_vis["cluster"].unique(), key=lambda x: int(x) if x != "-1" else 999)
    fig = px.scatter(
        df_vis,
        x="x",
        y="y",
        color="cluster",
        hover_data=["paper_id", "title"],
        title=f"UMAP + HDBSCAN Clustering ({n_clusters_found} clusters)",
        category_orders={"cluster": cluster_order},
        color_discrete_sequence=px.colors.qualitative.Safe,
        width=950,
        height=650
    )

    fig.update_layout(
        legend_title_text="Cluster ID",
        xaxis_title="UMAP 1",
        yaxis_title="UMAP 2",
        template="plotly_white"
    )

    fig.write_html(args.output)
    print(f"🎉 Done! Open in browser:\n{args.output}")


if __name__ == "__main__":
    main()
