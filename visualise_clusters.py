#!/usr/bin/env python3
"""Generate static 2D and 3D HTML plots of the clustered songs."""

import os
from pathlib import Path

import chromadb
import numpy as np
import pandas as pd
import plotly.express as px
import umap
from chromadb.config import Settings

CHROMA_DIR = Path(os.environ.get("MUSIC_CHROMA_DIR", Path.home() / "music_effnet" / "chroma"))
CHROMA_COLLECTION = "songs_effnet"
FETCH_BATCH = 1000

PROJECT_DIR = Path(os.environ.get("MUSIC_PROJECT_DIR", "/mnt/c/Users/baldi/Desktop/music_clustering_test"))
CLUSTERING_RESULTS_DIR = Path(
    os.environ.get("MUSIC_CLUSTERING_RESULTS_DIR", PROJECT_DIR / "clustering_results")
)
OUTPUT_DIR = Path(
    os.environ.get("MUSIC_VISUALISATION_RESULTS_DIR", PROJECT_DIR / "visualisation_results")
)
CLUSTERS_CSV = CLUSTERING_RESULTS_DIR / "song_clusters.csv"

STANDARDIZE = True
UMAP_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
UMAP_METRIC = "cosine"
MARKER_SIZE_2D = 6
MARKER_SIZE_3D = 3

CLUSTER_PALETTE = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
OUTLIERS = "Outliers"
UNCLUSTERED = "Not clustered yet"
GREY = {OUTLIERS: "#808080", UNCLUSTERED: "#5a5a5a"}


def load_collection():
    """Load song IDs, embeddings, and metadata from Chroma in stable ID order."""
    if not CHROMA_DIR.is_dir():
        raise SystemExit(f"Chroma folder not found: {CHROMA_DIR}. Run embed_music_effnet.py first.")

    client = chromadb.PersistentClient(
        path=str(CHROMA_DIR),
        settings=Settings(anonymized_telemetry=False),
    )
    try:
        collection = client.get_collection(CHROMA_COLLECTION)
    except Exception as exc:
        raise SystemExit(
            f"Could not open collection '{CHROMA_COLLECTION}' in {CHROMA_DIR}/: {exc}"
        ) from exc

    total = collection.count()
    if total == 0:
        raise SystemExit(f"Collection '{CHROMA_COLLECTION}' is empty.")

    ids, vectors, metadata = [], [], []
    for offset in range(0, total, FETCH_BATCH):
        batch = collection.get(
            include=["embeddings", "metadatas"],
            limit=FETCH_BATCH,
            offset=offset,
        )
        ids.extend(batch["ids"])
        vectors.extend(batch["embeddings"])
        metadata.extend(batch["metadatas"])

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise SystemExit("Unexpected embedding data returned by Chroma.")

    order = sorted(range(len(ids)), key=lambda index: ids[index])
    return (
        [ids[index] for index in order],
        embeddings[order],
        [metadata[index] or {} for index in order],
    )


def load_cluster_labels(ids, metadata):
    """Load clustering output by song ID, falling back to labels saved in Chroma."""
    if CLUSTERS_CSV.is_file():
        table = pd.read_csv(CLUSTERS_CSV, dtype={"id": str})
        missing = {"id", "cluster", "cluster_label"} - set(table.columns)
        if missing:
            raise SystemExit(
                f"{CLUSTERS_CSV} lacks columns {sorted(missing)}. Run cluster_songs.py again."
            )
        if "probability" not in table.columns:
            table["probability"] = np.nan
        table = table.drop_duplicates("id").set_index("id")
        source = str(CLUSTERS_CSV)
    elif any("cluster" in song for song in metadata):
        table = pd.DataFrame(
            {
                "cluster": [song.get("cluster", np.nan) for song in metadata],
                "cluster_label": [song.get("cluster_label", "") for song in metadata],
                "probability": [song.get("cluster_probability", np.nan) for song in metadata],
            },
            index=ids,
        )
        source = "cluster metadata in Chroma"
    else:
        raise SystemExit(
            f"No cluster labels found. Run cluster_songs.py first (expected {CLUSTERS_CSV})."
        )

    return table.reindex(ids)[["cluster", "cluster_label", "probability"]], source


def prepare_embeddings(embeddings):
    if not STANDARDIZE:
        return embeddings
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    return ((embeddings - mean) / (std + 1e-8)).astype(np.float32)


def run_umap(embeddings, dimensions):
    neighbors = max(2, min(UMAP_NEIGHBORS, len(embeddings) - 1))
    print(f"UMAP -> {dimensions}D ({len(embeddings)} songs, n_neighbors={neighbors})...")
    reducer = umap.UMAP(
        n_components=dimensions,
        n_neighbors=neighbors,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
    )
    return reducer.fit_transform(embeddings)


def make_plot_data(ids, metadata, clusters, coordinates):
    raw_clusters = clusters["cluster"].to_numpy(dtype=float)
    cluster_ids = np.where(np.isnan(raw_clusters), -2, raw_clusters).astype(int)
    names = clusters["cluster_label"].fillna("").astype(str).to_numpy()
    labels = [
        OUTLIERS if cluster == -1 else
        UNCLUSTERED if cluster == -2 else
        name or f"Cluster {cluster}"
        for cluster, name in zip(cluster_ids, names)
    ]
    artists = [str(song.get("artists") or "") for song in metadata]
    titles = [str(song.get("title") or "") for song in metadata]
    albums = [str(song.get("album") or "") for song in metadata]

    return pd.DataFrame(
        {
            "id": ids,
            "label": [f"{title} — {artist}" for title, artist in zip(titles, artists)],
            "album": albums,
            "cluster": cluster_ids,
            "cluster_label": labels,
            "probability": clusters["probability"].to_numpy(),
            **{
                f"axis_{axis + 1}": coordinates[:, axis]
                for axis in range(coordinates.shape[1])
            },
        }
    )


def color_settings(data):
    sizes = data[data["cluster"] >= 0]["cluster_label"].value_counts()
    order = list(sizes.index)
    colors = {
        label: CLUSTER_PALETTE[index % len(CLUSTER_PALETTE)]
        for index, label in enumerate(order)
    }
    for label in (OUTLIERS, UNCLUSTERED):
        if label in set(data["cluster_label"]):
            order.append(label)
            colors[label] = GREY[label]
    return {"color_discrete_map": colors, "category_orders": {"cluster_label": order}}


def make_figure(data, dimensions):
    axes = [f"axis_{index}" for index in range(1, dimensions + 1)]
    color_options = color_settings(data)
    hover = {"cluster": True, "probability": ":.2f", "album": True}
    cluster_count = data.loc[data["cluster"] >= 0, "cluster"].nunique()
    outlier_count = int((data["cluster"] == -1).sum())

    if dimensions == 2:
        figure = px.scatter(
            data,
            x=axes[0],
            y=axes[1],
            color="cluster_label",
            hover_name="label",
            hover_data={**hover, axes[0]: False, axes[1]: False},
            title=f"UMAP 2D: {len(data)} songs, {cluster_count} clusters, {outlier_count} outliers",
            render_mode="webgl",
            template="plotly_dark",
            labels={"cluster_label": "cluster", axes[0]: "UMAP 1", axes[1]: "UMAP 2"},
            **color_options,
        )
        figure.update_traces(marker=dict(size=MARKER_SIZE_2D, opacity=0.85))
        figure.update_yaxes(scaleanchor="x")
    else:
        figure = px.scatter_3d(
            data,
            x=axes[0],
            y=axes[1],
            z=axes[2],
            color="cluster_label",
            hover_name="label",
            hover_data={**hover, axes[0]: False, axes[1]: False, axes[2]: False},
            template="plotly_dark",
            labels={
                "cluster_label": "cluster",
                axes[0]: "UMAP 1",
                axes[1]: "UMAP 2",
                axes[2]: "UMAP 3",
            },
            **color_options,
        )
        figure.update_traces(marker=dict(size=MARKER_SIZE_3D, opacity=0.85))

    figure.update_traces(marker=dict(opacity=0.35), selector=dict(name=OUTLIERS))
    return figure


def main():
    ids, embeddings, metadata = load_collection()
    print(f"Loaded {len(ids)} embeddings of dimension {embeddings.shape[1]} from Chroma.")
    if len(ids) < 5:
        raise SystemExit("Need at least 5 songs for UMAP.")

    clusters, source = load_cluster_labels(ids, metadata)
    print(f"Cluster labels from {source}.")
    embeddings = prepare_embeddings(embeddings)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dimensions, filename in ((2, "clusters_2d.html"), (3, "clusters_3d.html")):
        data = make_plot_data(
            ids,
            metadata,
            clusters,
            run_umap(embeddings, dimensions),
        )
        path = OUTPUT_DIR / filename
        make_figure(data, dimensions).write_html(
            path,
            include_plotlyjs="cdn",
            full_html=True,
            default_width="100%",
            default_height="100vh",
            config={"responsive": True},
        )
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
