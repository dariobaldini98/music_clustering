# Music Clustering

Turns a folder of audio files into **one 2560-dimensional vector per song** that you can use for
similarity search and clustering. The scripts prepare metadata, embed songs, then cluster and visualise
the results:

```
audio folder ──► prepare_songs_metadata.py ──► songs_metadata.json ──► embed_music_effnet.py ──► Chroma database
                    reads tags and enriches       song IDs, paths,         creates audio               │
                    metadata with MusicBrainz     and song metadata        embeddings                  │
                                                                                                       │
                                                                                                       ▼
                                                                        clustering_results/ ◄── cluster_songs.py
                                                                           cluster CSVs/models     finds song groups,
                                                                                 │                 updates Chroma
                                                                                 ▼
                                                                        visualisation_results/
                                                                           interactive 2D/3D plots
```

| Script | Runs on | Does |
|--------|---------|------|
| `prepare_songs_metadata.py` | Windows, Linux or WSL (plain Python + `mutagen`) | Scans the audio folder, reads the embedded tags, enriches new songs through MusicBrainz and writes `songs_metadata.json`. No audio is decoded and nothing is downloaded. |
| `embed_music_effnet.py` | **Linux / WSL only** (needs `essentia-tensorflow`) | Reads `songs_metadata.json`, embeds the songs that have no vector yet and keeps the Chroma database in sync with the JSON. |
| `cluster_songs.py` | Linux / WSL | Clusters Chroma embeddings, asks you to approve before saving the assignments, then saves CSVs and model files in `clustering_results/` and automatically generates visualisations. |
| `visualise_clusters.py` | Linux / WSL | Generates two plain interactive 2D/3D plot HTML files in `visualisation_results/`. It can also be run separately to regenerate them. |

When running in WSL, set `MUSIC_PROJECT_DIR` to your project folder. This is especially important for
clustering and visualisation, which otherwise use the script's configured default project path.
Clustering outputs can be relocated with `MUSIC_CLUSTERING_RESULTS_DIR` and visualisation outputs with
`MUSIC_VISUALISATION_RESULTS_DIR`.

---

## Quick start

```bash
# Install dependencies once in each platform's venv (see "Setup" below).
# Windows venv: only prepare_songs_metadata.py is supported.
python -m pip install -r requirements.txt

# WSL/Linux venv: embedding, clustering, and visualisation.
python -m pip install -r /path/to/music_clustering/requirements.txt

# Each time you add songs:
# In the Windows venv:
python prepare_songs_metadata.py --contact-email you@example.com
# In the WSL/Linux venv:
python embed_music_effnet.py
python cluster_songs.py
```

The metadata script asks for a MusicBrainz contact email if you do not pass
`--contact-email` or set `MUSICBRAINZ_CONTACT_EMAIL`. The preparation and embedding scripts ask their
other questions on the keyboard; press Enter to accept the value in brackets. Clustering separately
asks you to approve before it writes results.

---

## Setup

### Requirements

| | |
|---|---|
| Python | 3.10+ for `prepare_songs_metadata.py`; for embedding, use a Python version supported by the `essentia-tensorflow` wheel (check its PyPI page) |
| `prepare_songs_metadata.py` | `mutagen` |
| `embed_music_effnet.py` | `essentia-tensorflow`, `chromadb`, `numpy`, and the model file `discogs-effnet-bs64-1.pb` |
| `cluster_songs.py` | `chromadb`, `joblib`, `numpy`, `pandas`, `umap-learn`, `hdbscan` |
| `visualise_clusters.py` | `chromadb`, `numpy`, `pandas`, `plotly`, `umap-learn` |
| Dependency file | `requirements.txt`; Linux-only dependencies are excluded automatically on Windows |
| Audio | `.mp3 .m4a .opus .flac .wav .ogg .aac`, ideally with title / artist / album tags |
| Internet | only for `prepare_songs_metadata.py` (MusicBrainz) and the one-off model download |

### Folder layout

```
music_project/                  <- the "project folder"
├── songs/                      <- your audio files (sub-folders are fine)
│   ├── Artist - Title.mp3
│   └── ...
├── prepare_songs_metadata.py   <- (other scripts can be elsewhere)
├── songs_metadata.json         <- created by prepare_songs_metadata.py
├── clustering_results/         <- created by cluster_songs.py
└── visualisation_results/      <- created by visualise_clusters.py
```

### Windows: why WSL, and how to set it up

The Essentia TensorFlow build only exists for Linux, so on Windows the **embedding** script runs inside
**WSL 2** with Ubuntu. Your audio and JSON can stay on the Windows side. Use a separate virtual
environment for the Windows preparation script and the Linux/WSL embedding, clustering, and visualisation
scripts; a virtual environment is not shared across Windows and WSL.

**Windows preparation environment** (PowerShell, from the project folder):

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python prepare_songs_metadata.py --contact-email you@example.com
```

The Windows requirements install `mutagen`; packages needed only by the Linux/WSL scripts are skipped.
If PowerShell blocks activation, use the venv interpreter directly:
`.venv\Scripts\python.exe -m pip install -r requirements.txt`.

**1. Install WSL + Ubuntu** (PowerShell as Administrator, then restart if asked):

```powershell
wsl --install -d Ubuntu
```

**2. Create the environment** (inside the Ubuntu terminal):

```bash
sudo apt update
sudo apt install -y python3 python3-venv python3-pip

mkdir -p ~/music_effnet && cd ~/music_effnet
python3 -m venv venv
source venv/bin/activate

pip install --upgrade pip
pip install -r /mnt/c/path/to/music_clustering/requirements.txt
```

Replace `/mnt/c/path/to/music_clustering` with the WSL path to this repository. On Linux, the
requirements file installs the embedding and clustering dependencies as well as `mutagen`.

**3. Download the model:**

```bash
cd ~/music_effnet
wget https://essentia.upf.edu/models/feature-extractors/discogs-effnet/discogs-effnet-bs64-1.pb
```

**4. Windows folders as seen from WSL** (drives are mounted under `/mnt/`):

| Windows path | Path in WSL |
|--------------|-------------|
| `C:\Users\you\Desktop\music_project` | `/mnt/c/Users/you/Desktop/music_project` |
| `D:\Music` | `/mnt/d/Music` |

**Good to know**

- Keep the **Chroma database on the Linux filesystem** (default `~/music_effnet/chroma`), not under
  `/mnt/c/...`: SQLite is slow and less reliable across the Windows/Linux boundary.
- Reading audio from `/mnt/c/...` works but is slower. For big libraries, copy the audio into WSL.
- Browse the WSL files from Explorer at `\\wsl.localhost\Ubuntu\home\<user>\music_effnet`.
- WSL 2 limits memory by default. If long runs are killed, raise `memory=` in `%UserProfile%\.wslconfig`.

---

## Step 1 — `prepare_songs_metadata.py`: build `songs_metadata.json`

```bash
python prepare_songs_metadata.py
```

It first shows its defaults and asks `Keep this configuration? [Y/n]`. Answer `n` to change them one by
one (Enter keeps the value in brackets). For a real run it also asks for the contact email used in the
MusicBrainz User-Agent, unless supplied on the command line or in `MUSICBRAINZ_CONTACT_EMAIL`.

### What it does

1. **Scans** the audio folder recursively and reads each file's embedded **title, artist(s), album and
   duration**. If a file has no title tag, the file name is used (a leading `123_` or `SNG123_` is removed
   and underscores become spaces). Several artists in the tags are joined with `|`.
2. **Matches** each file to an entry of the existing JSON by its audio path. Existing tags are not read
   again; new files are appended, and unfinished or errored MusicBrainz enrichment is retried.
3. **Gives every new song a `song_id`**, once, and never changes it afterwards. The id is
   `file_<sanitized file name without extension>` (for example `file_Artist_Title`). If two files share a name,
   the next ones get `_1`, `_2`, ... Renaming a file therefore makes it a new song.
4. **Enriches new and unfinished songs with MusicBrainz**: it searches by title + artist (the full artist credit first,
   then each `|`-separated artist), scores the candidates, and if the best one reaches the minimum match
   score it looks up the recording and stores the canonical title, artists, album, release date, ISRCs and
   links. The canonical title / artists / album replace the tag values.
5. **Reports possible duplicates** (same MusicBrainz recording, ISRC, YouTube id, audio path, or same
   title + artist). Nothing is ever removed automatically.
6. **Saves** `songs_metadata.json` (atomically), every 100 songs and at the end. `Ctrl+C` or `SIGTERM`
   saves the progress before exiting.

MusicBrainz asks clients to stay at **1 request per second**; the script waits 1.1 s between requests and
retries HTTP 503 answers with a growing delay. Expect roughly 3–4 requests (about 4 s) per song.

### Questions

| Question | Default | Meaning |
|----------|---------|---------|
| Output metadata JSON | `<script folder>/songs_metadata.json` | File that is read and extended. |
| Existing audio folder | `<script folder>/songs` | Folder scanned for audio files. |
| Base folder for relative paths | `<script folder>` | `audio_path` values are stored relative to this folder, with forward slashes. Files outside it are stored as absolute paths. |
| Minimum MusicBrainz match score (0–1) | `0.72` | Below it the song is recorded as `not-found`. |
| MusicBrainz contact email | `MUSICBRAINZ_CONTACT_EMAIL` or prompted | Required for a non-dry run; passed in the request User-Agent. |
| Dry run only? | `no` | Prints what it would do; no network access, no file written. **Use it first.** |

### Constants at the top of the script

| Constant | Default | Meaning |
|----------|---------|---------|
| `USER_AGENT_NAME` | `PlaylistMetadataEnricher/1.0` | The script adds the contact email you provide to form a MusicBrainz-compliant User-Agent. |
| `REQUEST_INTERVAL_SECONDS` | `1.1` | Minimum pause between MusicBrainz requests. |
| `MAX_503_RETRIES` | `5` | Retries when MusicBrainz answers HTTP 503. |
| `RETRY_BACKOFF_SECONDS` / `MAX_RETRY_BACKOFF_SECONDS` | `5` / `60` | Start and cap of the retry delay (doubles each time; `Retry-After` is honoured). |
| `CHECKPOINT_INTERVAL` | `100` | Save after this many enriched songs. |
| `MINIMUM_MATCH_SCORE` | `0.72` | Default for the question above; can also be set with `--min-score`. |
| `AUDIO_EXTENSIONS` | mp3, m4a, opus, flac, wav, ogg, aac | File types that are scanned. |

### How a candidate is scored

With artist tags: `score = 0.40·title + 0.40·artist + 0.12·album + 0.08·duration`.
Without artist tags, title weight becomes `0.80` and artist weight is omitted.

- **title**: 1 if equal after normalisation (accents, case and punctuation ignored; `(feat. ...)` removed),
  0.75 if one contains the other.
- **artist**: 1 if one of the wanted artists equals a credited artist, 0.8 if one contains the other.
- **album**: 1 if the tag matches one of the recording's releases; 0.5 if the file has no album tag.
- **duration**: 1 within 2 s, 0.7 within 5 s, 0 beyond; 0.5 if either duration is unknown.

The release matching the tagged album is preferred; otherwise the earliest dated official release is
used when available.

### Command-line options

| Option | Meaning |
|--------|---------|
| `--output PATH` | Output metadata JSON (default: `songs_metadata.json` beside the script). |
| `--audio-dir PATH` | Audio folder (default: `songs` beside the script). |
| `--project-dir PATH` | Base folder for relative `audio_path` values. |
| `--min-score 0..1` | Minimum MusicBrainz match score. |
| `--contact-email EMAIL` | Contact included in the MusicBrainz User-Agent; alternatively set `MUSICBRAINZ_CONTACT_EMAIL`. |
| `--dry-run` | Do not make MusicBrainz requests or write files. |
| `--no-prompt` | Use supplied values and defaults. A contact email is required for a non-dry run. |

### Format of one entry

```json
{
  "song_id": "file_Artist_Title",
  "audio_path": "songs/Artist - Title.mp3",
  "title": "Title",
  "artists": "Artist|Guest",
  "album": "Album",
  "duration": 215,
  "musicbrainz": {
    "status": "matched",
    "processing_complete": true,
    "match_score": 0.98,
    "search_method": "title+artist",
    "recording_id": "00000000-0000-0000-0000-000000000000",
    "recording_url": "https://musicbrainz.org/recording/...",
    "title": "Title",
    "artist_credit": "Artist|Guest",
    "artists": [{"name": "Artist", "id": "...", "url": "https://musicbrainz.org/artist/..."}],
    "release_id": "...",
    "release_url": "https://musicbrainz.org/release/...",
    "album": "Album",
    "release_date": "2019-05-17",
    "length_ms": 215000,
    "isrcs": ["USABC1234567"],
    "video": false
  }
}
```

`musicbrainz.status` is `matched`, `not-found`, `search-error` (network problem while searching) or
`lookup-error`. Entries that only have the first fields (no `musicbrainz` block) are valid too; the
embedding script then falls back to the plain `title` / `artists` / `album`.

### Things to know

- Completed entries are not re-processed, so changed audio tags alone do not update them. Entries
  without a MusicBrainz block or with `search-error` / `lookup-error` are retried; to refresh a completed
  `matched` or `not-found` entry, delete it from the JSON and run the script again.
- Entries whose audio file is no longer in the folder are **kept** (the script says how many).
- Entries with no MusicBrainz block, or with `search-error` / `lookup-error`, are retried automatically
  when their audio file is present. A completed `not-found` result is not retried; delete its entry to
  try it again (for example after fixing its tags).
- A file with **no artist tag** is searched and scored by title, with the artist weight reassigned to title;
  adding accurate artist tags can still improve match confidence.
- Moving or renaming an audio file makes it a new song with a new `song_id`; the old entry is kept.

---

## Step 2 — `embed_music_effnet.py`: embed and sync

```bash
cd ~/music_effnet
source venv/bin/activate
python embed_music_effnet.py
```

The script asks, one at a time (Enter accepts the value in brackets):

```
Project folder (songs_metadata.json + audio; from WSL: /mnt/c/Users/you/...): /mnt/c/Users/you/Desktop/music_project
Work folder for model + Chroma database [/home/you/music_effnet]:
Audio root (base of the audio_path values) [/mnt/c/Users/you/Desktop/music_project]:
Dry run, write nothing? (y/n) [n]: y
Song ids to re-embed, separated by spaces (empty = none):
```

Answer `y` to the dry run first, read the summary, then run again with `n`.

### What it does

For every song that has no vector yet:

1. Decode the file to **16 kHz mono**.
2. Drop the first and last `TRIM_SECONDS` (skipped if the song would be too short afterwards).
3. Run **Discogs-EffNet**: one 1280-d embedding per ~2 s patch.
4. **Pool** the patches as `[mean, std]` → one **2560-d** vector per song.
5. Store vector + metadata + a text document (`"artists — title"`) in Chroma (cosine space).

Every run does three things:

| Step | What happens |
|------|--------------|
| **Sync** | Each JSON entry is matched to its Chroma record (by `song_id` *and* song identity, then by identity alone). If title / artists / album / ISRC ... changed, metadata and document are updated; if the id changed, the record is re-keyed. Stored vectors are never recomputed; extra fields (`cluster`, `cluster_probability`, `cluster_label`, `n_patches`) are kept. |
| **Embed** | Songs with no record (or named in the re-embed question) are embedded. Missing audio files are reported as skipped and reconsidered after they are restored; other failures (such as corrupt audio) are listed at the end and retried on the next run. |
| **Report** | Chroma records that no JSON entry points to are listed. They are **never deleted**. If one holds an id that another song needs it is moved to an `orphan_...` id. |

Re-keying is crash-safe: records are first copied to temporary `__rekey__...` ids, and an interrupted run is
finished by the next one. This mainly supports databases created with older, position-based IDs; the
filename-based IDs now made by `prepare_songs_metadata.py` are stable.

The run summary reads:
`already in sync / metadata update / id change / metadata rewrite / new to embed / re-embed (forced) / not in JSON`.

### Command-line options

Every question has an option; giving it skips that question. `--no-prompt` turns all questions off
(for scripts): options and defaults are used.

| Option | Default | Meaning |
|--------|---------|---------|
| `--project-dir` | asked (default `$MUSIC_PROJECT_DIR`) | Folder with `songs_metadata.json` and the audio. |
| `--json` | `<project-dir>/songs_metadata.json` | Song list. |
| `--audio-root` | asked (default `<project-dir>`) | Base folder for relative `audio_path` values. An absolute Windows path (`C:\x.mp3`) is mapped to `/mnt/c/x.mp3`. |
| `--work-dir` | asked (default `$MUSIC_WORK_DIR`, else `~/music_effnet`) | Folder holding the model and the Chroma database. |
| `--model` | `<work-dir>/discogs-effnet-bs64-1.pb` | Model file. |
| `--chroma-dir` | `<work-dir>/chroma` | Chroma database folder. |
| `--collection` | `songs_effnet` | Collection name. |
| `--dry-run` | asked (default no) | Print what would change and write nothing. |
| `--reembed ID [ID ...]` | asked (default none) | Recompute these songs even if they have a vector (e.g. after replacing a bad file). Their extra fields (`cluster`, ...) are reset. |
| `--no-prompt` | off | Never ask questions. |

### Processing settings (constants at the top of the script)

> These change the vectors. They are stored in the collection as `model_config`; if you change one while the
> collection already holds vectors, the script **refuses to run**, so vectors made with different settings
> are never mixed. Revert the setting, or delete the Chroma folder and re-embed.

| Constant | Default | Meaning |
|----------|---------|---------|
| `MODEL_OUTPUT` | `PartitionedCall:1` | Embedding layer of the model graph. Don't change it for this model. |
| `EMBEDDING_DIM` | `1280` | Per-patch size, used to validate the output. |
| `SAMPLE_RATE` | `16000` | Required by the model. |
| `RESAMPLE_QUALITY` | `1` | Essentia resampler: `0` best … `1` default … `4` fast approximation. |
| `TRIM_SECONDS` | `15` | Seconds ignored at the start and end of each song (`0` = whole song). |
| `MIN_SECONDS_AFTER_TRIM` | `30` | Trim only if at least this much audio remains. |

### Behaviour settings (do not change the vectors)

| Constant | Default | Meaning |
|----------|---------|---------|
| `MIN_MATCH_SCORE` | `0.0` | MusicBrainz blocks with a lower `match_score` are ignored when building metadata. |
| `DURATION_WARN` | `10` | Warn if the decoded length differs from the JSON duration by more than this many seconds. |
| `SYNC_BATCH` | `200` | Records read / written per Chroma call. |

### What is stored in Chroma

| Field | Content |
|-------|---------|
| id | the song's `song_id` |
| embedding | 2560 floats = `[mean(1280), std(1280)]` |
| document | `"artists — title"` |
| metadata | `youtube_id`, `title`, `artists` (`A\|B`), `primary_artist`, `album`, `url`, `isrc`, `release_date`, `musicbrainz_recording_id`, `musicbrainz_recording_url`, `duration_seconds`, `n_patches` |

MusicBrainz data is used only when `musicbrainz.status == "matched"` and `match_score >= MIN_MATCH_SCORE`;
otherwise title / artists / album come from the plain JSON fields.

Reading the data back:

```python
import chromadb
col = chromadb.PersistentClient(path="/home/<user>/music_effnet/chroma").get_collection("songs_effnet")
song_id = "file_Artist_Title"
vec = col.get(ids=[song_id], include=["embeddings"])["embeddings"][0]
res = col.query(query_embeddings=[vec], n_results=5, include=["metadatas", "distances"])
```

---

## Typical workflow

1. Copy new audio files into `songs/`.
2. Activate the Windows venv and run `python prepare_songs_metadata.py`, with *Dry run* the first time.
   Check the duplicate report, then run it for real.
3. Open `songs_metadata.json` if you want to fix anything by hand (see the notes above).
4. Activate the WSL/Linux venv and run `python embed_music_effnet.py` with the dry run, check the summary,
   then run it for real.
5. In that same WSL/Linux venv, run `python cluster_songs.py`. Review the proposed clusters and
   approve the write when prompted; approval updates the song metadata in Chroma and saves clustering
   results, then visualisation runs automatically.

---

## Step 3 — `cluster_songs.py`: cluster and visualise

Run this in the Linux/WSL environment that contains the Chroma database:

```bash
cd ~/music_effnet
source venv/bin/activate
export MUSIC_PROJECT_DIR=/mnt/c/Users/you/Desktop/music_project
python cluster_songs.py
```

The script loads all song embeddings from the `songs_effnet` Chroma collection, optionally standardises
them, reduces them with UMAP, then uses HDBSCAN to assign clusters. The cluster count is discovered by
HDBSCAN rather than specified in advance; songs labeled `-1` are outliers. Cluster labels, probabilities,
map coordinates, and cluster centroids are written to Chroma only after you approve the proposed
summary. Declining approval leaves Chroma and result files unchanged.

After approval, clustering output is saved under `clustering_results/`:

| File | Contents |
|------|----------|
| `song_clusters.csv` | Per-song cluster, label, and membership probabilities. |
| `cluster_summary.csv` | Cluster sizes, distinctive artists, and representative songs. |
| `map_data.csv` | Per-song 3D map coordinates. |
| `cluster_models.joblib` | Fitted standardisation, UMAP, and HDBSCAN objects (when `SAVE_MODELS` is enabled). |

The script then automatically runs the visualisation pipeline and saves these files under
`visualisation_results/`:

| File | Contents |
|------|----------|
| `clusters_2d.html` | Interactive 2D plot. |
| `clusters_3d.html` | Interactive 3D plot. |

The visualisation script only writes the two HTML plot files. Open the HTML files in a browser to view them. 
Plotly is loaded from its CDN, so the browser needs internet access to render the interactive plots. To regenerate 
the plots separately, run `python visualise_clusters.py` from the WSL/Linux environment.

Set `MUSIC_CLUSTERING_RESULTS_DIR` or `MUSIC_VISUALISATION_RESULTS_DIR` to change the respective output
folders. Both default to subfolders of `MUSIC_PROJECT_DIR`.

### Clustering settings

The main tuning constants are near the top of `cluster_songs.py`:

| Constant | Default | Effect |
|----------|---------|--------|
| `STANDARDIZE` | `True` | Standardise embedding dimensions before clustering. |
| `UMAP_N_NEIGHBORS` | `15` | UMAP neighborhood size for the reduced vectors used by HDBSCAN. |
| `UMAP_N_COMPONENTS` | `10` | Dimensions passed from UMAP to HDBSCAN. |
| `UMAP_MIN_DIST` | `0.0` | UMAP minimum distance for the clustering representation. |
| `HDBSCAN_MIN_CLUSTER_SIZE` | `8` | Smallest group HDBSCAN may call a cluster. |
| `HDBSCAN_MIN_SAMPLES` | `3` | Density strictness; increase it for more conservative assignments. |
| `HDBSCAN_EPSILON` | `0.0` | HDBSCAN cluster-selection epsilon. |
| `SAVE_MODELS` | `True` | Save the fitted model objects to `cluster_models.joblib`. |

The visualisation script computes a separate 2D and 3D UMAP layout for plotting; its
`UMAP_NEIGHBORS`, `UMAP_MIN_DIST`, and `STANDARDIZE` settings control those layouts, not the cluster
assignments. UMAP layouts and resulting clusters may vary between runs because no random seed is fixed.

---

## Troubleshooting

| Symptom | Cause / fix |
|---------|-------------|
| `No module named 'mutagen'` | `python -m pip install mutagen`. |
| `Audio folder not found` | Check the folder you typed; paths are relative to where you run the script unless absolute. |
| MusicBrainz `HTTP 403` | Check that `--contact-email` or `MUSICBRAINZ_CONTACT_EMAIL` contains your contact email. |
| `HTTP 503` messages | MusicBrainz is busy; the script waits and retries by itself. |
| Many songs `not-found` | Tags are missing or differ a lot from MusicBrainz (no artist tag, "Artist - Title" in the title field, ...). Fix the tags, delete those entries and run again. |
| `Missing dependency: essentia` | You are not in WSL/Linux, or the venv is not active (`source venv/bin/activate`). |
| `Model file not found` | Download `discogs-effnet-bs64-1.pb` (the message shows the command) or pass `--model`. |
| `Collection ... holds embeddings made with different settings` | You changed a processing setting. Revert it, or delete the Chroma folder and re-embed. |
| `Audio file not found` (embedding) | `audio_path` is resolved against the audio root. Check the `/mnt/c/...` path, or whether the file was moved. |
| `Unexpected embedding shape` | File shorter than ~2 s after trimming, or wrong model / layer. |
| `WARNING Xs vs Ys in JSON` | Decoded length differs from the tag duration: likely a truncated file. |
| Very slow reading | Audio on `/mnt/c`. Copy it to the Linux filesystem. |

---

## Repository notes

- Do **not** commit audio files, the Chroma folder, the `.pb` model, or `songs_metadata.json` if it holds
  data you don't want public. The Discogs-EffNet models are distributed by the Essentia project under a
  non-commercial license (CC BY-NC-SA 4.0), Essentia itself is AGPL-3.0, and MusicBrainz data is CC0 (core
  data).
- The repository `.gitignore` excludes generated clustering and visualisation results, audio, metadata,
  model files, Chroma data, and Python environments. Review it before publishing if your outputs contain
  personal or sensitive metadata.

---

## Licence

The source code in this repository is licensed under the
PolyForm Noncommercial License 1.0.0.

See [`LICENSE`](LICENSE) for the complete licence text.

Third-party software, models and data retain their
respective licences. See Repository notes above.

Copyright © 2026 Dario Baldini. All rights reserved.
