#!/usr/bin/env python3
"""
Topic-modelling-style clustering of the song embeddings, without BERTopic.

It runs the same two steps BERTopic runs internally, each used directly:
    embeddings --UMAP (UMAP_N_COMPONENTS dims)--> reduced vectors --HDBSCAN--> clusters ("topics")
Cluster -1 means "outlier", exactly like topic -1 in BERTopic.

BERTopic's other parts need text documents (CountVectorizer, stop words, MMR), which songs
do not have. Their counterpart here is a class-based TF-IDF over ARTISTS: for each cluster it
finds the artists that are most distinctive of it, plus representative songs and simple stats.

Run it in the Ubuntu venv:
    python cluster_songs.py

The number of clusters is never set: HDBSCAN finds the dense groups by itself, and the
HDBSCAN_* settings below only decide how small a group may be and how strict the density test is.
The CLI displays a review of the proposed groups and requires explicit approval before it writes
cluster assignments, map coordinates, or centroids to Chroma. One run also creates the map layout
and summary CSVs under `clustering_results/`, then generates visualizations under
`visualisation_results/`.

Requires: chromadb, umap-learn, hdbscan, pandas, numpy
Install: pip install hdbscan
"""

import os
import time
from pathlib import Path

import chromadb
import joblib
import numpy as np
import pandas as pd
import umap
import hdbscan
from chromadb.config import Settings

import visualise_clusters

# --- Configuration ---
CHROMA_DIR = Path(os.environ.get("MUSIC_CHROMA_DIR", Path.home() / "music_effnet" / "chroma"))
CHROMA_COLLECTION = "songs_effnet"
FETCH_BATCH = 1000

# Windows project folder, seen from WSL.
OUTPUT_DIR = Path(os.environ.get("MUSIC_PROJECT_DIR", "/mnt/c/Users/baldi/Desktop/music_clustering_test"))
CLUSTERING_RESULTS_DIR = Path(
    os.environ.get("MUSIC_CLUSTERING_RESULTS_DIR", OUTPUT_DIR / "clustering_results")
)

# Standardise every dimension across the library first (the stored vectors are [mean, std]
# of the EffNet patches, which live on very different scales).
STANDARDIZE = True

# UMAP: your BERTopic settings
UMAP_N_NEIGHBORS = 15
UMAP_N_COMPONENTS = 10
UMAP_MIN_DIST = 0.0
UMAP_METRIC = "cosine"

# HDBSCAN: your BERTopic settings
HDBSCAN_MIN_CLUSTER_SIZE = 8
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_EPSILON = 0.0
HDBSCAN_METRIC = "euclidean"
HDBSCAN_METHOD = "eom"

# Cluster description (class-based TF-IDF over artists)
REDUCE_FREQUENT_ARTISTS = True
MIN_ARTIST_SONGS = 2         # like CountVectorizer(min_df=2): ignore artists with fewer songs
TOP_ARTISTS = 5              # distinctive artists listed per cluster
TOP_SONGS = 5                # representative songs listed per cluster

# Outputs
SAVE_MODELS = True           # Save the fitted standardisation + UMAP + HDBSCAN to reuse later
                             # (compressed; the file also contains the training vectors, so it grows with the library)
CLUSTER_COLLECTION = "clusters_effnet"
INCLUDE_OUTLIERS_IN_CENTROIDS = False
MAP_UMAP_NEIGHBORS = 15
MAP_UMAP_MIN_DIST = 0.1
# ---------------------


# --- Loading ---

def load_collection():
    """Fetch ids, embeddings and metadata for every song, sorted by id."""
    if not CHROMA_DIR.is_dir():
        raise SystemExit(f"Chroma folder not found: {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR), settings=Settings(anonymized_telemetry=False))
    try:
        col = client.get_collection(CHROMA_COLLECTION)
    except Exception as exc:
        raise SystemExit(f"Could not open collection '{CHROMA_COLLECTION}' in {CHROMA_DIR}/: {exc}")

    total = col.count()
    if total == 0:
        raise SystemExit(f"Collection '{CHROMA_COLLECTION}' is empty.")

    ids, vectors, metas = [], [], []
    for offset in range(0, total, FETCH_BATCH):
        batch = col.get(include=["embeddings", "metadatas"], limit=FETCH_BATCH, offset=offset)
        ids.extend(batch["ids"])
        vectors.extend(batch["embeddings"])
        metas.extend(batch["metadatas"])

    embeddings = np.asarray(vectors, dtype=np.float32)
    if embeddings.ndim != 2 or len(embeddings) != len(ids):
        raise SystemExit("Unexpected embedding data returned by Chroma.")

    order = np.argsort(np.asarray(ids), kind="stable")
    ids = [ids[i] for i in order]
    embeddings = embeddings[order]
    metas = [metas[i] or {} for i in order]
    return client, col, ids, embeddings, metas


# --- Step 1 + 2: UMAP and HDBSCAN, used directly ---

def standardize(embeddings):
    mean = embeddings.mean(axis=0)
    std = embeddings.std(axis=0)
    return ((embeddings - mean) / (std + 1e-8)).astype(np.float32), mean, std


def reduce_with_umap(embeddings):
    neighbors = max(2, min(UMAP_N_NEIGHBORS, len(embeddings) - 1))
    print(f"UMAP: {embeddings.shape[1]}D -> {UMAP_N_COMPONENTS}D (n_neighbors={neighbors}, "
          f"min_dist={UMAP_MIN_DIST}, metric={UMAP_METRIC})...")
    reducer = umap.UMAP(
        n_neighbors=neighbors,
        n_components=UMAP_N_COMPONENTS,
        min_dist=UMAP_MIN_DIST,
        metric=UMAP_METRIC,
    )
    return reducer, reducer.fit_transform(embeddings)


def make_clusterer():
    """Return HDBSCAN instance with your BERTopic settings (no number of clusters is given)."""
    return hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_MIN_CLUSTER_SIZE,
        min_samples=HDBSCAN_MIN_SAMPLES,
        cluster_selection_epsilon=HDBSCAN_EPSILON,
        metric=HDBSCAN_METRIC,
        cluster_selection_method=HDBSCAN_METHOD,
        prediction_data=True,
    )


def soft_membership(clusterer):
    """Best cluster and its strength for every song, outliers included (hdbscan package only)."""
    try:
        soft = np.nan_to_num(hdbscan.all_points_membership_vectors(clusterer))
        return soft.argmax(axis=1), soft.max(axis=1)
    except Exception as exc:
        print(f"  (soft membership skipped: {exc})")
        return None, None


# --- Describing the clusters ---

def artist_list(meta):
    """Read the exact pipe-delimited artist names stored in song metadata."""
    artists = str(meta.get("artists") or "")
    names = [name.strip() for name in artists.split("|") if name.strip()]
    return list(dict.fromkeys(names))


def class_tfidf_artists(labels, artist_lists):
    """BERTopic's c-TF-IDF with artists as the terms and clusters as the 'documents'.

    Returns (class labels, artist names, score matrix, count matrix), rows = clusters.
    """
    rows = [(label, artist) for label, artists in zip(labels, artist_lists) for artist in set(artists)]
    if not rows:                                  # no artist information at all
        return np.array([]), np.array([]), np.zeros((0, 0)), np.zeros((0, 0))
    counts = pd.crosstab(pd.Series([r[0] for r in rows], name="cluster"),
                         pd.Series([r[1] for r in rows], name="artist"))
    counts = counts.loc[:, counts.sum(axis=0) >= MIN_ARTIST_SONGS]       # like min_df
    if counts.shape[1] == 0:
        return counts.index.to_numpy(), counts.columns.to_numpy(), np.zeros(counts.shape), counts.to_numpy(float)

    X = counts.to_numpy(float)
    df = X.sum(axis=0)                           # artist frequency across all clusters
    avg_per_class = int(X.sum(axis=1).mean())    # average "words" per cluster, as in BERTopic
    idf = np.log(avg_per_class / df + 1)
    idf = np.maximum(idf, 0)

    tf = X / np.maximum(X.sum(axis=1, keepdims=True), 1)                 # L1 normalise rows
    if REDUCE_FREQUENT_ARTISTS:
        tf = np.sqrt(tf)
    return counts.index.to_numpy(), counts.columns.to_numpy(), tf * idf, X


def cluster_record_id(cluster):
    """Return a stable, human-readable ID for an HDBSCAN label (0 -> CLST0001)."""
    return f"CLST{int(cluster) + 1:04d}"


def describe_clusters(df, reduced, labels, probabilities, artist_lists):
    cluster_ids, artists, scores, counts = class_tfidf_artists(labels, artist_lists)
    row_of = {cluster: i for i, cluster in enumerate(cluster_ids)}

    summaries, names = [], {}
    for cluster in sorted(set(labels)):
        members = np.flatnonzero(labels == cluster)

        top_artists = []
        if cluster in row_of and len(artists):
            i = row_of[cluster]
            for j in np.argsort(-scores[i])[:TOP_ARTISTS]:
                if counts[i, j] > 0 and scores[i, j] > 0:
                    top_artists.append(f"{artists[j]} ({int(counts[i, j])})")

        centroid = reduced[members].mean(axis=0)
        distance = np.linalg.norm(reduced[members] - centroid, axis=1)
        best = members[np.lexsort((distance, -probabilities[members]))][:TOP_SONGS]
        examples = [f"{df.at[m, 'artists']} — {df.at[m, 'title']}" for m in best]

        if cluster == -1:
            name = "Outliers"
        else:
            lead = [a.rsplit(" (", 1)[0] for a in top_artists[:3]]
            name = f"{cluster}: " + (" / ".join(lead) if lead else "(no distinctive artists)")
        names[cluster] = name

        summaries.append({
            "cluster": cluster,
            "cluster_id": cluster_record_id(cluster),
            "name": name,
            "size": len(members),
            "top_artists": "; ".join(top_artists),
            "representative_songs": "; ".join(examples),
        })
    return pd.DataFrame(summaries), names


# --- Optional outputs ---

def primary_artist(meta):
    stored = str(meta.get("primary_artist") or "").strip()
    if stored:
        return stored
    names = artist_list(meta)
    return names[0] if names else ""


def build_map_dataframe(ids, metas, labels, probabilities, names, coords):
    return pd.DataFrame({
        "id": ids,
        "title": [meta.get("title", "") for meta in metas],
        "artists": [meta.get("artists", "") for meta in metas],
        "primary_artist": [primary_artist(meta) for meta in metas],
        "album": [meta.get("album", "") for meta in metas],
        "release_date": [meta.get("release_date", "") for meta in metas],
        "youtube_id": [meta.get("youtube_id", "") for meta in metas],
        "cluster": labels.astype(int),
        "cluster_id": [cluster_record_id(cluster) for cluster in labels],
        "cluster_label": [names[int(cluster)] for cluster in labels],
        "probability": probabilities.round(4),
        "x": coords[:, 0],
        "y": coords[:, 1],
        "z": coords[:, 2],
    })


def write_clusters_to_chroma(col, ids, labels, probabilities, names, map_df):
    print("Writing clustering labels and map coordinates to the Chroma song metadata...")
    row_of = {song_id: i for i, song_id in enumerate(ids)}
    map_rows = map_df.set_index("id")
    for start in range(0, len(ids), 500):
        chunk = ids[start:start + 500]
        current = col.get(ids=chunk, include=["metadatas"])
        merged = []
        for song_id, meta in zip(current["ids"], current["metadatas"]):
            i = row_of[song_id]
            meta = dict(meta or {})
            meta.pop("explicit", None)
            meta.update({
                "cluster": int(labels[i]),
                "cluster_id": cluster_record_id(labels[i]),
                "cluster_probability": float(probabilities[i]),
                "cluster_label": names[int(labels[i])],
                "map_x": float(map_rows.at[song_id, "x"]),
                "map_y": float(map_rows.at[song_id, "y"]),
                "map_z": float(map_rows.at[song_id, "z"]),
            })
            merged.append(meta)
        col.update(ids=current["ids"], metadatas=merged)


def _year(value):
    digits = str(value or "")[:4]
    return int(digits) if digits.isdigit() and int(digits) >= 1000 else None


def _artists_from_metadata(meta):
    artist = primary_artist(meta)
    return [artist] if artist else []


def _fallback_cluster_description(members, metas, probabilities):
    counts = {}
    for index in members:
        for artist in set(_artists_from_metadata(metas[index])):
            counts[artist] = counts.get(artist, 0) + 1
    top = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:TOP_ARTISTS]
    best = sorted(members, key=lambda index: -probabilities[index])[:TOP_SONGS]
    return {
        "top_artists": "; ".join(f"{artist} ({count})" for artist, count in top),
        "representative_songs": "; ".join(
            f"{metas[index].get('artists', '')} — {metas[index].get('title', '')}" for index in best
        ),
    }


def build_cluster_records(embeddings, metas, labels, probabilities, names, summary):
    records = []
    for cluster in sorted(set(labels.tolist())):
        if cluster == -1 and not INCLUDE_OUTLIERS_IN_CENTROIDS:
            continue
        members = np.flatnonzero(labels == cluster)
        row = summary.loc[summary["cluster"] == cluster]
        if not row.empty:
            row = row.iloc[0]
            description = {
                "top_artists": str(row["top_artists"] or ""),
                "representative_songs": str(row["representative_songs"] or ""),
            }
        else:
            description = _fallback_cluster_description(members, metas, probabilities)

        metadata = {
            "cluster": int(cluster),
            "cluster_label": str(names[cluster]),
            "size": int(len(members)),
            "avg_probability": round(float(probabilities[members].mean()), 4),
            **description,
        }
        years = [_year(metas[index].get("release_date")) for index in members]
        years = [year for year in years if year is not None]
        if years:
            metadata["median_year"] = float(np.median(years))
        centroid = embeddings[members].astype(np.float64).mean(axis=0).astype(np.float32)
        records.append({
            "id": cluster_record_id(cluster),
            "embedding": centroid.tolist(),
            "document": str(names[cluster]),
            "metadata": metadata,
        })
    return records


def write_cluster_collection(client, records, model_config):
    if not records:
        print("No non-outlier clusters to store in the centroid collection.")
        return
    metadata = {"source_collection": CHROMA_COLLECTION}
    if model_config:
        metadata["model_config"] = model_config
    col = client.get_or_create_collection(
        name=CLUSTER_COLLECTION,
        configuration={"hnsw": {"space": "cosine"}},
        metadata=metadata,
    )
    col.modify(metadata=metadata)
    old_ids = set(col.get(include=[])["ids"])
    new_ids = {record["id"] for record in records}
    col.upsert(
        ids=[record["id"] for record in records],
        embeddings=[record["embedding"] for record in records],
        metadatas=[record["metadata"] for record in records],
        documents=[record["document"] for record in records],
    )
    stale_ids = old_ids - new_ids
    if stale_ids:
        col.delete(ids=sorted(stale_ids))
    print(f"Saved {len(records)} cluster centroid(s) to '{CLUSTER_COLLECTION}'.")


def layout_3d(embeddings):
    neighbors = max(2, min(MAP_UMAP_NEIGHBORS, len(embeddings) - 1))
    print(f"UMAP -> 3D ({len(embeddings)} songs, n_neighbors={neighbors})...")
    reducer = umap.UMAP(
        n_components=3,
        n_neighbors=neighbors,
        min_dist=MAP_UMAP_MIN_DIST,
        metric="cosine",
    )
    return reducer.fit_transform(embeddings)


def confirm_chroma_write():
    print("\nReview the cluster summary above.")
    try:
        answer = input(
            "Save cluster labels, probabilities, map coordinates, and centroids to Chroma? [y/N]: "
        ).strip().casefold()
    except EOFError:
        print("\nNo interactive confirmation received; nothing saved.")
        return False
    if answer not in {"y", "yes"}:
        print("Clustering not approved; Chroma was not modified.")
        return False
    return True


def main():
    client, col, ids, embeddings, metas = load_collection()
    print(f"Loaded {len(ids)} embeddings of dimension {embeddings.shape[1]} from Chroma.")
    if len(ids) < max(30, 2 * UMAP_N_COMPONENTS):
        raise SystemExit(f"Too few songs for a {UMAP_N_COMPONENTS}-dimensional UMAP "
                         f"(need at least {max(30, 2 * UMAP_N_COMPONENTS)}). Embed more songs first.")

    raw_embeddings = embeddings          # as stored in Chroma: the cluster centroids are computed from these
    mean = std = None
    if STANDARDIZE:
        embeddings, mean, std = standardize(embeddings)

    t0 = time.perf_counter()
    reducer, reduced = reduce_with_umap(embeddings)
    print(f"  done in {time.perf_counter() - t0:.1f}s")

    clusterer = make_clusterer()
    print(f"HDBSCAN: min_cluster_size={HDBSCAN_MIN_CLUSTER_SIZE}, min_samples={HDBSCAN_MIN_SAMPLES}, "
          f"epsilon={HDBSCAN_EPSILON}, method={HDBSCAN_METHOD}...")
    clusterer.fit(reduced)
    labels = np.asarray(clusterer.labels_)
    probabilities = np.asarray(clusterer.probabilities_, dtype=float)

    n_clusters = int(labels.max()) + 1 if (labels >= 0).any() else 0
    n_outliers = int((labels == -1).sum())
    print(f"  found {n_clusters} clusters; {n_outliers} outliers ({100 * n_outliers / len(labels):.1f}%)")
    if n_clusters == 0:
        raise SystemExit("No clusters found. Try a smaller HDBSCAN_MIN_CLUSTER_SIZE / HDBSCAN_MIN_SAMPLES.")

    soft_cluster, soft_probability = soft_membership(clusterer)

    df = pd.DataFrame({
        "id": ids,
        "artists": [m.get("artists", "") for m in metas],
        "title": [m.get("title", "") for m in metas],
        "album": [m.get("album", "") for m in metas],
        "release_date": [m.get("release_date", "") for m in metas],
        "musicbrainz_recording_id": [m.get("musicbrainz_recording_id", "") for m in metas],
        "youtube_id": [m.get("youtube_id", "") for m in metas],
        "musicbrainz_recording_url": [m.get("musicbrainz_recording_url", "") for m in metas],
    })
    summary, names = describe_clusters(
        df,
        reduced,
        labels,
        probabilities,
        [_artists_from_metadata(meta) for meta in metas],
    )

    df["cluster"] = labels
    df["cluster_id"] = [cluster_record_id(cluster) for cluster in labels]
    df["cluster_label"] = [names[int(c)] for c in labels]
    df["probability"] = probabilities.round(4)
    if soft_cluster is not None:
        df["soft_cluster"] = soft_cluster
        df["soft_probability"] = soft_probability.round(4)

    print("\n=== Clusters (largest first) ===")
    for row in summary.sort_values("size", ascending=False).itertuples():
        print(f"\n[{row.name}]  {row.size} songs")
        if row.top_artists:
            print(f"  distinctive artists: {row.top_artists}")
        if row.representative_songs:
            print(f"  representative songs: {row.representative_songs}")

    if not confirm_chroma_write():
        return

    # The 3D layout is the slow part, so it only runs once the clustering is approved.
    map_embeddings = embeddings
    if not STANDARDIZE:
        map_embeddings, _, _ = standardize(raw_embeddings)
    map_df = build_map_dataframe(
        ids,
        metas,
        labels,
        probabilities,
        names,
        layout_3d(map_embeddings),
    )

    write_clusters_to_chroma(col, ids, labels, probabilities, names, map_df)

    records = build_cluster_records(
        raw_embeddings, metas, labels, probabilities, names, summary
    )
    write_cluster_collection(client, records, (col.metadata or {}).get("model_config"))

    CLUSTERING_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    songs_path = CLUSTERING_RESULTS_DIR / "song_clusters.csv"
    summary_path = CLUSTERING_RESULTS_DIR / "cluster_summary.csv"
    map_path = CLUSTERING_RESULTS_DIR / "map_data.csv"
    df.to_csv(songs_path, index=False, encoding="utf-8")
    summary.sort_values("size", ascending=False).to_csv(summary_path, index=False, encoding="utf-8")
    map_df.to_csv(map_path, index=False, encoding="utf-8")
    print(f"\nSaved {songs_path}\nSaved {summary_path}\nSaved {map_path}")

    if SAVE_MODELS:
        models_path = CLUSTERING_RESULTS_DIR / "cluster_models.joblib"
        joblib.dump(
            {
                "mean": mean,
                "std": std,
                "umap": reducer,
                "clusterer": clusterer,
                "collection": CHROMA_COLLECTION,
            },
            models_path,
            compress=3,
        )
        print(f"Saved {models_path} ({models_path.stat().st_size / 1e6:.1f} MB)")

    visualise_clusters.main()


if __name__ == "__main__":
    main()
