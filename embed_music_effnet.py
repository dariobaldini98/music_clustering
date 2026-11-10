#!/usr/bin/env python3
"""
Keep the Chroma collection in sync with songs_metadata.json, and embed new songs.

Run it inside WSL/Linux, in the venv that has essentia-tensorflow and chromadb.
It is safe to run again and again: only what changed is touched.

    python embed_music_effnet.py            # asks for folders, dry run and ids to re-embed
    python embed_music_effnet.py --project-dir /mnt/c/Users/you/Desktop/music_project --dry-run --no-prompt
    python embed_music_effnet.py --project-dir ... --reembed file_song_1 file_song_2

Anything not given on the command line is asked on the keyboard (Enter keeps the default in brackets).

Every run does three things:
  1. SYNC existing records. Each JSON entry is matched to its Chroma record by `song_id`;
     legacy records can also be found through ISRC, MusicBrainz recording ID, YouTube ID,
     or audio path. If the JSON data changed (title/artists/album/release_date/ISRC...),
     the record's metadata and document are updated; if its prepared JSON ID changed,
     the record is re-keyed. The stored embedding is never recomputed
     and extra metadata (cluster, cluster_probability, cluster_label, n_patches) is preserved.
  2. EMBED songs that have no record yet (or that were named with --reembed):
       decode to 16 kHz mono -> drop first/last TRIM_SECONDS -> Discogs-EffNet (1280-d per ~2 s
       patch) -> pool [mean, std] -> 2560 values -> upsert vector + metadata + document.
  3. REPORT records that no JSON entry points to any more. They are never deleted here; if one holds an id
     that another song needs, it is moved to an "orphan_..." id.

ARTISTS: "artists" holds exact matched MusicBrainz artist names joined with "|" ("A|B"), falling back to the source artist string. "primary_artist" holds the first artist when available.

The stable, filename-based `song_id` values in songs_metadata.json are used as the Chroma record IDs.
The source video ID remains separately available as `youtube_id`. Older Chroma records with legacy
IDs are still matched by metadata identity and re-keyed to `song_id`.

The re-key and orphan handling also protects databases made with older, position-based IDs that could
shift when songs were added. It verifies song identity before reusing a record, and interrupted re-key
operations are finished by the next run.

MusicBrainz data is used only when musicbrainz.status == "matched" (and match_score >= MIN_MATCH_SCORE).
Otherwise title/artists/album come from the plain JSON fields.

Requires: essentia-tensorflow, numpy, chromadb
"""

import argparse
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path, PureWindowsPath

import numpy as np

try:
    import chromadb
    from chromadb.config import Settings
    import essentia
    from essentia.standard import MonoLoader, TensorflowPredictEffnetDiscogs
except ImportError as exc:  # pragma: no cover - depends on the environment
    raise SystemExit(
        f"Missing dependency: {exc.name}. Run this script inside WSL/Linux, in the virtual environment "
        "where essentia-tensorflow, chromadb and numpy are installed "
        "(essentia-tensorflow is not available for native Windows)."
    )

# --- Processing settings ---
# These change the vectors. They are stored in the collection (MODEL_CONFIG below), and the script
# refuses to run if they differ from the ones the existing vectors were made with.
MODEL_OUTPUT = "PartitionedCall:1"  # the embedding layer
EMBEDDING_DIM = 1280                # per-patch embedding size of this model
SAMPLE_RATE = 16000                 # the model requires 16 kHz mono
RESAMPLE_QUALITY = 1                # Essentia: 0 = best ... 1 = default ... 4 = fast linear approximation.

TRIM_SECONDS = 15                   # Ignore this much at the start and end (0 = use the whole song)
MIN_SECONDS_AFTER_TRIM = 30         # Only trim if at least this much audio would remain

# --- Behaviour settings (do not change the vectors) ---
DURATION_WARN = 10                  # Warn if real length differs from JSON duration by more than N seconds
MIN_MATCH_SCORE = 0.0               # Ignore MusicBrainz blocks whose match_score is lower than this
SYNC_BATCH = 200                    # Records read/updated per Chroma call
# ---------------------

MODEL_FILENAME = "discogs-effnet-bs64-1.pb"
MODEL_URL = "https://essentia.upf.edu/models/feature-extractors/discogs-effnet/" + MODEL_FILENAME
DEFAULT_WORK_DIR = Path(os.environ.get("MUSIC_WORK_DIR", Path.home() / "music_effnet"))

# Stored in the collection metadata so a settings change is detected instead of mixing vectors.
MODEL_CONFIG = (
    f"discogs-effnet-bs64-1|{MODEL_OUTPUT}|mean+std|trim{TRIM_SECONDS}|sr{SAMPLE_RATE}|rq{RESAMPLE_QUALITY}"
    # The default (30) adds nothing, so collections made before this setting was tracked stay valid.
    + ("" if MIN_SECONDS_AFTER_TRIM == 30 else f"|min{MIN_SECONDS_AFTER_TRIM}")
)
TEMP_PREFIX = "__rekey__"           # temporary record ids used while records are re-keyed
ORPHAN_PREFIX = "orphan_"           # records of songs no longer in the JSON that must give up their id
OBSOLETE_FIELDS = {"artist_names", "audio_path", "actual_duration_seconds"}  # old metadata keys, removed on sync

essentia.log.warningActive = False


# --- Command line ---

def ask_text(question, default, ask=True, required=False):
    """Ask on the keyboard; Enter keeps the default. Without a keyboard the default is used."""
    while True:
        answer = ""
        if ask:
            shown = f" [{default}]" if default else ""
            try:
                answer = input(f"{question}{shown}: ").strip().strip('"').strip("'")
            except EOFError:
                ask = False
        value = answer or default
        if value or not required:
            return value
        if not ask:
            raise SystemExit(f"{question} is required: pass it on the command line or answer the question.")
        print("  This one is required.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Sync a Chroma collection with songs_metadata.json and embed new songs "
                    "with Discogs-EffNet.",
    )
    parser.add_argument("--project-dir", type=Path, default=None,
                        help="Folder with songs_metadata.json and the audio files "
                             "(env MUSIC_PROJECT_DIR). Asked interactively if not given. "
                             "From WSL, C:\\Users\\you\\x is /mnt/c/Users/you/x.")
    parser.add_argument("--json", type=Path, default=None, dest="json_file",
                        help="Song list (default: <project-dir>/songs_metadata.json).")
    parser.add_argument("--audio-root", type=Path, default=None,
                        help="Base folder for the relative audio_path values (default: <project-dir>).")
    parser.add_argument("--work-dir", type=Path, default=None,
                        help="Linux-side folder for the model and the Chroma database "
                             "(env MUSIC_WORK_DIR; default: ~/music_effnet). Asked interactively if not given.")
    parser.add_argument("--model", type=Path, default=None, dest="model_file",
                        help=f"Path to {MODEL_FILENAME} (default: <work-dir>/{MODEL_FILENAME}).")
    parser.add_argument("--chroma-dir", type=Path, default=None,
                        help="Chroma database folder (default: <work-dir>/chroma).")
    parser.add_argument("--collection", default="songs_effnet", help="Chroma collection name.")
    parser.add_argument("--dry-run", action="store_const", const=True, default=None,
                        help="Print what would change and write nothing. Asked interactively if not given.")
    parser.add_argument("--reembed", nargs="+", metavar="SONG_ID", default=None,
                        help="Recompute the vector of these songs even if they already have one "
                             "(for example after replacing a bad audio file). Asked interactively if not given.")
    parser.add_argument("--no-prompt", action="store_true",
                        help="Never ask questions: use the command-line values and the defaults.")
    cfg = parser.parse_args(argv)

    # Anything not given on the command line is asked now (Enter keeps the value in brackets).
    # With --no-prompt, or when there is no keyboard input (cron, pipes), the defaults are used.
    ask = not cfg.no_prompt
    if cfg.project_dir is None:
        default = os.environ.get("MUSIC_PROJECT_DIR", "")
        cfg.project_dir = Path(
            ask_text("Project folder (songs_metadata.json + audio; from WSL: /mnt/c/Users/you/...)",
                     default, ask, required=True))
    if cfg.work_dir is None:
        cfg.work_dir = Path(
            ask_text("Work folder for model + Chroma database",
                     str(DEFAULT_WORK_DIR), ask))
    cfg.project_dir = cfg.project_dir.expanduser().resolve()
    cfg.work_dir = cfg.work_dir.expanduser().resolve()
    if cfg.audio_root is None:
        cfg.audio_root = Path(ask_text("Audio root (base of the audio_path values)",
                                       str(cfg.project_dir), ask))
    if cfg.dry_run is None:
        cfg.dry_run = ask_text("Dry run, write nothing? (y/n)", "n", ask).lower().startswith("y")
    if cfg.reembed is None:
        cfg.reembed = ask_text("Song ids to re-embed, separated by spaces (empty = none)",
                               "", ask).split()
    cfg.json_file = (cfg.json_file or cfg.project_dir / "songs_metadata.json").expanduser()
    cfg.audio_root = (cfg.audio_root or cfg.project_dir).expanduser()
    cfg.model_file = (cfg.model_file or cfg.work_dir / MODEL_FILENAME).expanduser()
    cfg.chroma_dir = (cfg.chroma_dir or cfg.work_dir / "chroma").expanduser()
    return cfg


def parse_duration(text):
    """'2:16' -> 136.0, '1:02:03' -> 3723.0, anything invalid -> None."""
    if not text:
        return None
    try:
        seconds = 0
        for part in str(text).split(":"):
            seconds = seconds * 60 + int(part)
        return float(seconds)
    except ValueError:
        return None


def resolve_audio_path(raw, audio_root):
    """Turn a path as stored in the JSON into a real path.

    'songs\\\\x.mp3' (Windows, relative) -> audio_root/songs/x.mp3
    'C:\\\\Music\\\\x.mp3' (Windows, absolute) -> /mnt/c/Music/x.mp3 (WSL mount)
    '/data/x.mp3' (absolute POSIX) is used as it is.
    """
    windows = PureWindowsPath(str(raw))
    if len(windows.drive) == 2 and windows.drive[1] == ":":
        return Path("/mnt") / windows.drive[0].lower() / Path(*windows.parts[1:])
    return Path(audio_root) / windows.as_posix()


# --- Embedding ---

def trim_edges(audio):
    """Cut TRIM_SECONDS from both ends, unless that would leave too little audio."""
    margin = int(TRIM_SECONDS * SAMPLE_RATE)
    if margin > 0 and len(audio) > 2 * margin + MIN_SECONDS_AFTER_TRIM * SAMPLE_RATE:
        audio = audio[margin:-margin]
    return np.ascontiguousarray(audio, dtype=np.float32)


def embed_song(model, path):
    """Return (song_vector, real_duration_seconds, n_patches)."""
    audio = MonoLoader(filename=str(path), sampleRate=SAMPLE_RATE, resampleQuality=RESAMPLE_QUALITY)()
    if audio.size == 0:
        raise ValueError("Audio file is empty")
    real_duration = len(audio) / SAMPLE_RATE

    patches = np.asarray(model(trim_edges(audio)), dtype=np.float32)
    if patches.ndim != 2 or patches.shape[0] == 0 or patches.shape[1] != EMBEDDING_DIM:
        raise ValueError(f"Unexpected embedding shape: {patches.shape}")
    if not np.isfinite(patches).all():
        raise ValueError("Non-finite embeddings")

    vector = np.concatenate([patches.mean(axis=0), patches.std(axis=0)]).astype(np.float32)
    return vector, real_duration, patches.shape[0]


# --- Chroma ---

def open_collection(cfg):
    """Open (or create) the persistent cosine collection, refusing to mix model settings."""
    cfg.chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(
        path=str(cfg.chroma_dir),
        settings=Settings(anonymized_telemetry=False),
    )

    def create():
        options = dict(
            name=cfg.collection,
            configuration={"hnsw": {"space": "cosine"}},
            metadata={"model_config": MODEL_CONFIG},
        )
        # Vectors are always supplied explicitly, so Chroma must never embed anything itself:
        # embedding_function=None switches its default text model off.
        try:
            return client.get_or_create_collection(embedding_function=None, **options)
        except ValueError:
            # A collection made by an older version of this script remembers Chroma's default
            # embedding function and may refuse the change. It is never used, so keep it as it is.
            return client.get_or_create_collection(**options)

    col = create()
    stored = (col.metadata or {}).get("model_config")
    if stored != MODEL_CONFIG:
        if col.count() == 0:
            # Empty, so nothing to lose: recreate it with the current settings.
            client.delete_collection(cfg.collection)
            col = create()
        else:
            raise SystemExit(
                f"Collection '{cfg.collection}' holds embeddings made with different settings "
                f"({stored} vs {MODEL_CONFIG}). Change the settings back, or delete {cfg.chroma_dir}/ "
                "to start over."
            )
    return col


# --- IDs and metadata built from the JSON entry ---

def musicbrainz_data(song):
    """The song's MusicBrainz block, or {} if it is missing, not matched, or scored too low."""
    mb = song.get("musicbrainz")
    if not isinstance(mb, dict) or mb.get("status") != "matched":
        return {}
    try:
        score = float(mb.get("match_score") or 0)
    except (TypeError, ValueError):
        score = 0.0
    return mb if score >= MIN_MATCH_SCORE else {}


def underscored(value):
    """Any run of punctuation/spaces becomes a single '_': 'a1b2-c3d4' -> 'a1b2_c3d4'."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")


def clean_isrc(value):
    """'us-abc-12-34567' -> 'USABC1234567': punctuation/spaces removed, upper-case."""
    return re.sub(r"[^A-Za-z0-9]", "", value).upper() if isinstance(value, str) else ""


def song_isrc(song):
    isrcs = musicbrainz_data(song).get("isrcs") or []
    if isinstance(isrcs, str):
        isrcs = [isrcs]
    if not isinstance(isrcs, list):
        return ""
    return next((code for code in map(clean_isrc, isrcs) if code), "")


def base_id(song):
    """ISRC, else MusicBrainz recording id, else YouTube id (before duplicate suffixes)."""
    isrc = song_isrc(song)
    if isrc:
        return isrc
    recording_id = underscored(musicbrainz_data(song).get("recording_id"))
    if recording_id:
        return f"mb_{recording_id}"
    youtube_id = underscored(song.get("youtube_id"))
    if youtube_id:
        return f"yt_{youtube_id}"
    return f"file_{PureWindowsPath(str(song['audio_path'])).stem}"


def fits(record_id, base):
    """True if record_id is `base` itself or `base` plus a duplicate suffix ('_1', '_2', ...)."""
    return record_id == base or re.fullmatch(re.escape(base) + r"_\d+", record_id) is not None


def path_key(raw):
    return PureWindowsPath(str(raw)).as_posix()


def artist_names(song):
    """The exact MusicBrainz artist names joined with '|' ('' if the song has no matched block).

    '|' is the separator because it practically never appears in a name; if one does, it becomes '/'.
    """
    names = []
    for artist in musicbrainz_data(song).get("artists") or []:
        name = artist.get("name") if isinstance(artist, dict) else artist
        name = str(name or "").replace("|", "/").strip()
        if name and name not in names:
            names.append(name)
    return "|".join(names)


def static_metadata(song):
    """Everything that comes from the JSON (not from the audio). Values are str/int/float/bool."""
    mb = musicbrainz_data(song)
    artists = artist_names(song) or str(song.get("artists") or "")
    primary_artist = next(
        (name.strip() for name in artists.split("|") if name.strip()),
        "",
    )
    meta = {
        "youtube_id": str(song.get("youtube_id") or ""),
        "title": str(mb.get("title") or song.get("title") or ""),
        "artists": artists,
        "primary_artist": primary_artist,
        "album": str(mb.get("album") or song.get("album") or ""),
        "url": str(song.get("url") or ""),
        "isrc": song_isrc(song),
        "release_date": str(mb.get("release_date") or ""),
        "musicbrainz_recording_id": str(mb.get("recording_id") or ""),
        "musicbrainz_recording_url": str(mb.get("recording_url") or ""),
    }
    duration = parse_duration(song.get("duration"))
    if duration is not None:
        meta["duration_seconds"] = duration
    return meta


def song_document(meta):
    return f"{meta['artists']} — {meta['title']}"


def song_label(song):
    return song_document(static_metadata(song))


# --- Matching JSON entries with existing Chroma records ---

def load_existing(col):
    """{id: (metadata, document)} for every record in the collection."""
    records = {}
    total = col.count()
    for offset in range(0, total, SYNC_BATCH):
        batch = col.get(include=["metadatas", "documents"], limit=SYNC_BATCH, offset=offset)
        for rid, meta, doc in zip(batch["ids"], batch["metadatas"], batch["documents"]):
            records[rid] = (dict(meta or {}), doc or "")
    return records


def same_song(song, meta, document):
    """True if a Chroma record (metadata + document) is evidently the same song as this JSON entry."""
    pairs = (
        (clean_isrc(meta.get("isrc")), song_isrc(song)),
        (str(meta.get("musicbrainz_recording_id") or "").strip().casefold(),
         str(musicbrainz_data(song).get("recording_id") or "").strip().casefold()),
        (str(meta.get("youtube_id") or "").strip().casefold(),
         str(song.get("youtube_id") or "").strip().casefold()),
    )
    if any(old and new and old == new for old, new in pairs):
        return True
    return bool(document) and document == song_document(static_metadata(song))


def match_existing(tracks, records):
    """Match JSON entries to Chroma records. Returns {track index: record id}.

    1. A record with the same id as the song's song_id is accepted only if it is evidently the same
       song (same_song). Older position-based IDs can move when the list changes, so an id can now
       belong to a different song: trusting it would attach the wrong vector.
    2. Otherwise by identity (ISRC, MusicBrainz recording id, YouTube id), then by legacy audio_path.
    """
    by_identity = defaultdict(list)
    by_path = defaultdict(list)
    for rid, (meta, _) in records.items():
        for key in ("isrc", "musicbrainz_recording_id", "youtube_id"):
            raw_value = meta.get(key)
            value = clean_isrc(raw_value) if key == "isrc" else str(raw_value or "").strip()
            value = value.casefold()
            if value:
                by_identity[(key, value)].append(rid)
        if meta.get("audio_path"):
            by_path[path_key(meta["audio_path"])].append(rid)

    match, claimed = {}, set()

    # Pass 1: same id AND same song.
    for index, song in enumerate(tracks):
        song_id = str(song.get("song_id") or "").strip()
        if song_id in records and song_id not in claimed and same_song(song, *records[song_id]):
            match[index] = song_id
            claimed.add(song_id)

    # Pass 2: everything else, by identity.
    for index, song in enumerate(tracks):
        if index in match:
            continue
        identities = [
            ("isrc", song_isrc(song)),
            ("musicbrainz_recording_id", str(musicbrainz_data(song).get("recording_id") or "")),
            ("youtube_id", str(song.get("youtube_id") or "")),
        ]
        candidates = []
        for priority, (key, value) in enumerate(identities):
            if value:
                candidates.extend(
                    (priority, rid)
                    for rid in by_identity.get((key, value.strip().casefold()), [])
                    if rid not in claimed
                )
        if not candidates:
            candidates = [
                (len(identities), rid)
                for rid in by_path.get(path_key(song["audio_path"]), [])
                if rid not in claimed
            ]
        if not candidates:
            continue
        base = base_id(song)
        candidates.sort(key=lambda item: (item[0], not fits(item[1], base), item[1]))
        match[index] = candidates[0][1]
        claimed.add(candidates[0][1])
    return match


# --- Applying changes ---

def build_sync(tracks, match, ids, records):
    """Split matched songs into rewrites (new id and/or obsolete keys to drop) and in-place updates.

    Chroma's update() merges metadata, so it cannot remove old keys. A record that still carries an
    obsolete key is therefore rewritten through the re-key path (which creates it afresh), even if its
    id does not change. Otherwise the same "change" would be reported again on every run.
    """
    renames, updates, changed_fields = [], [], Counter()
    for index, old_id in match.items():
        meta_old, doc_old = records[old_id]
        desired = static_metadata(tracks[index])
        stale = [key for key in OBSOLETE_FIELDS if key in meta_old]
        merged = {key: value for key, value in meta_old.items() if key not in OBSOLETE_FIELDS}
        merged.update(desired)                         # keeps cluster, n_patches, ... untouched
        document = song_document(desired)
        diff = [k for k, v in desired.items() if meta_old.get(k) != v] + stale
        new_id = ids[index]
        if old_id != new_id or stale:
            renames.append((old_id, new_id, merged, document))
        elif diff or document != doc_old:
            updates.append((old_id, merged, document))
        changed_fields.update(diff)
    return renames, updates, changed_fields


def stored_vectors(col, old_ids):
    # The stored vectors are always passed back explicitly: a Chroma update that changes a document
    # without an embedding would recompute it with Chroma's default text model and destroy it.
    stored = col.get(ids=old_ids, include=["embeddings"])
    vectors = {rid: np.asarray(vec, dtype=np.float32).tolist()
               for rid, vec in zip(stored["ids"], stored["embeddings"])}
    missing = [rid for rid in old_ids if rid not in vectors]
    if missing:
        raise RuntimeError(f"Records disappeared from Chroma while syncing: {missing[:5]}")
    return vectors


def finish_rekey(col):
    """Complete every re-key that left temporary records behind. Safe to repeat, in any state.

    Each temporary record holds the full data of a song under TEMP_PREFIX + its final id, and its old id
    in `_rekey_from`. We delete the old ids, write every song under its final id, then delete the
    temporary records. Writing the finals last also repairs a final id that was deleted as somebody's old id.
    """
    temp_ids = [rid for rid in col.get(include=[])["ids"] if rid.startswith(TEMP_PREFIX)]
    if not temp_ids:
        return 0
    for start in range(0, len(temp_ids), SYNC_BATCH):
        chunk = temp_ids[start:start + SYNC_BATCH]
        old_ids = [m.get("_rekey_from") for m in col.get(ids=chunk, include=["metadatas"])["metadatas"]]
        old_ids = [rid for rid in old_ids if rid and not rid.startswith(TEMP_PREFIX)]
        if old_ids:
            col.delete(ids=old_ids)
    for start in range(0, len(temp_ids), SYNC_BATCH):
        chunk = temp_ids[start:start + SYNC_BATCH]
        got = col.get(ids=chunk, include=["embeddings", "metadatas", "documents"])
        metas = []
        for meta in got["metadatas"]:
            meta = dict(meta or {})
            meta.pop("_rekey_from", None)
            metas.append(meta)
        col.upsert(
            ids=[rid[len(TEMP_PREFIX):] for rid in got["ids"]],
            embeddings=[np.asarray(v, dtype=np.float32).tolist() for v in got["embeddings"]],
            metadatas=metas,
            documents=got["documents"],
        )
    col.delete(ids=temp_ids)
    return len(temp_ids)


def apply_sync(col, renames, updates):
    for start in range(0, len(updates), SYNC_BATCH):
        chunk = updates[start:start + SYNC_BATCH]
        vectors = stored_vectors(col, [u[0] for u in chunk])
        col.update(
            ids=[u[0] for u in chunk],
            embeddings=[vectors[u[0]] for u in chunk],
            metadatas=[u[1] for u in chunk],
            documents=[u[2] for u in chunk],
        )

    # Legacy position-based IDs can move, so a new id may still belong to a record being moved.
    # too. Writing new records one by one would overwrite those. So: copy everything to temporary ids
    # first (nothing is lost if we stop here), then finish_rekey() swaps them in.
    for start in range(0, len(renames), SYNC_BATCH):
        chunk = renames[start:start + SYNC_BATCH]
        vectors = stored_vectors(col, [r[0] for r in chunk])
        col.upsert(
            ids=[TEMP_PREFIX + r[1] for r in chunk],
            embeddings=[vectors[r[0]] for r in chunk],
            metadatas=[{**r[2], "_rekey_from": r[0]} for r in chunk],
            documents=[r[3] for r in chunk],
        )
    if renames:
        finish_rekey(col)


def embed_new(col, new_items, ids, cfg):
    """Embed and insert songs without a record. new_items = [(track index, song)]."""
    if not new_items:
        return 0, [], []

    available_items, skipped = [], []
    for index, song in new_items:
        song_id = ids[index]
        label = song_label(song)
        path = resolve_audio_path(song.get("audio_path"), cfg.audio_root)
        if path is None or not path.is_file():
            skipped.append((song_id, label, str(song.get("audio_path") or "")))
        else:
            available_items.append((index, song, path))
    for song_id, label, audio_path in skipped:
        print(f"SKIPPED {song_id}: {label}: audio file is missing or unavailable ({audio_path})")
    if not available_items:
        return 0, [], skipped

    if not cfg.model_file.is_file():
        raise FileNotFoundError(
            f"Model file not found: {cfg.model_file}. Download it with:\n  wget {MODEL_URL}\n"
            "or pass its location with --model."
        )
    print(f"Loading Discogs-EffNet model ({cfg.model_file.name})...")
    t0 = time.perf_counter()
    model = TensorflowPredictEffnetDiscogs(graphFilename=str(cfg.model_file), output=MODEL_OUTPUT)
    print(f"Model loaded in {time.perf_counter() - t0:.1f}s. {len(available_items)} songs to embed.")

    failures, embedded = [], 0
    run_start = time.perf_counter()
    for n, (index, song, path) in enumerate(available_items, start=1):
        song_id = ids[index]
        label = song_label(song)
        song_start = time.perf_counter()
        try:
            vector, real_duration, n_patches = embed_song(model, path)
        except Exception as exc:
            failures.append((song_id, label, str(exc)))
            print(f"[{n}/{len(available_items)}] FAILED {song_id}: {label}: {exc}")
            continue

        # Outside the try on purpose: a Chroma error is a real problem and should stop the run.
        meta = static_metadata(song)
        meta["n_patches"] = int(n_patches)
        col.upsert(
            ids=[song_id],
            embeddings=[np.asarray(vector, dtype=np.float32).tolist()],
            metadatas=[meta],
            documents=[song_document(meta)],
        )
        embedded += 1

        note = ""
        expected = parse_duration(song.get("duration"))
        if expected is not None and abs(real_duration - expected) > DURATION_WARN:
            note = f" | WARNING: {real_duration:.0f}s vs {expected:.0f}s in JSON"
        print(f"[{n}/{len(available_items)}] OK {song_id}: {label} | {n_patches} patches | "
              f"{time.perf_counter() - song_start:.1f}s{note}")

    print(f"\nEmbedding time: {time.perf_counter() - run_start:.1f}s")
    return embedded, failures, skipped


def main(argv=None):
    cfg = parse_args(argv)
    if not cfg.json_file.is_file():
        raise FileNotFoundError(
            f"JSON file not found: {cfg.json_file}. Pass --project-dir (or --json) pointing at your data."
        )

    with cfg.json_file.open(encoding="utf-8") as f:
        songs = json.load(f)
    if not isinstance(songs, list):
        raise ValueError(f"{cfg.json_file} must contain a JSON array of songs")

    tracks = [s for s in songs if isinstance(s, dict) and s.get("audio_path")]
    if len(songs) - len(tracks):
        print(f"Skipping {len(songs) - len(tracks)} entries without an audio_path.")
    if not tracks:
        print("No tracks with audio_path found.")
        return

    col = open_collection(cfg)
    leftovers = [rid for rid in col.get(include=[])["ids"] if rid.startswith(TEMP_PREFIX)]
    if leftovers and not cfg.dry_run:
        print(f"An earlier run was interrupted while re-keying: finishing {len(leftovers)} record(s)...")
        finish_rekey(col)
    elif leftovers:
        print(f"{len(leftovers)} record(s) are waiting to be re-keyed from an interrupted run (dry run: not touched).")
    records = {rid: value for rid, value in load_existing(col).items() if not rid.startswith(TEMP_PREFIX)}
    print(f"Chroma collection '{cfg.collection}' holds {len(records)} records; JSON has {len(tracks)} tracks.")

    ids = {}
    groups = defaultdict(list)
    for index, song in enumerate(tracks):
        song_id = song.get("song_id")
        if not isinstance(song_id, str) or not song_id.strip():
            raise ValueError(
                f"Track {index + 1} has no prepared song_id. Run the metadata preparation "
                "step to populate songs_metadata.json before embedding."
            )
        ids[index] = song_id.strip()
        groups[base_id(song)].append(index)
    if len(set(ids.values())) != len(tracks):
        raise ValueError("songs_metadata.json contains duplicate song_id values for tracks with audio.")

    forced = set(cfg.reembed)
    unknown = sorted(forced - set(ids.values()))
    if unknown:
        print(f"WARNING: --reembed ids not found in the JSON: {', '.join(unknown)}")

    match = match_existing(tracks, records)
    renames, updates, changed_fields = build_sync(tracks, match, ids, records)
    n_in_sync = len(match) - len(renames) - len(updates)
    moving = {old for old, _, _, _ in renames}          # records that leave their id: it becomes free
    # A song_id can be held by a record that belongs to no song of the JSON any more (a song removed from
    # the list). That record is never deleted: it is parked under an "orphan_" id, so the id is free.
    parked = []
    for song_id in sorted({sid for sid in ids.values() if sid in records and sid not in moving
                           and sid not in match.values()}):
        meta, document = records[song_id]
        parking, number = ORPHAN_PREFIX + song_id, 1
        while parking in records or any(r[1] == parking for r in renames):
            number += 1
            parking = f"{ORPHAN_PREFIX}{song_id}_{number}"
        renames.append((song_id, parking, dict(meta), document))
        parked.append((song_id, parking))
    if parked:
        print(f"{len(parked)} record(s) of songs no longer in the JSON hold ids that are needed again; "
              "they are kept under ids starting with 'orphan_' (e.g. "
              f"{parked[0][0]} -> {parked[0][1]}).")

    new_items = [(i, s) for i, s in enumerate(tracks) if i not in match or ids[i] in forced]
    n_forced = sum(1 for i, _ in new_items if i in match)
    matched_ids = set(match.values())
    orphans = [rid for rid in records if rid not in matched_ids]
    duplicates = {b: [ids[i] for i in idx] for b, idx in groups.items() if len(idx) > 1}
    n_id_change = sum(1 for old, new, _, _ in renames if old != new)

    print(f"  already in sync : {n_in_sync}")
    print(f"  metadata update : {len(updates)}")
    print(f"  id change       : {n_id_change}")
    print(f"  metadata rewrite: {len(renames) - n_id_change}  (old keys removed)")
    print(f"  new to embed    : {len(new_items) - n_forced}")
    print(f"  re-embed (forced): {n_forced}")
    print(f"  not in JSON     : {len(orphans)}")
    if changed_fields:
        print("  changed fields  : " + ", ".join(f"{k} ({n})" for k, n in changed_fields.most_common()))
    if duplicates:
        print(f"\n{len(duplicates)} ISRC / recording id(s) are shared by several songs (duplicates in the library?):")
        for base, used in list(duplicates.items())[:20]:
            print(f"  {base}: {', '.join(used)}")
    if orphans:
        print(f"\n{len(orphans)} Chroma record(s) match no JSON entry (left untouched):")
        for rid in orphans[:20]:
            meta = records[rid][0]
            print(f"  {rid}: {meta.get('artists', '')} — {meta.get('title', '')}")

    if cfg.dry_run:
        print("\nDry run: nothing was written.")
        return

    apply_sync(col, renames, updates)
    embedded, failures, skipped = embed_new(col, new_items, ids, cfg)

    print(f"\nUpdated {len(updates)}, rewrote/re-keyed {len(renames)}, embedded {embedded}. "
          f"Collection now holds {col.count()} records ({cfg.chroma_dir}/).")
    if skipped:
        print(f"\n{len(skipped)} song(s) skipped because their audio files are missing or unavailable.")
    if failures:
        print(f"\n{len(failures)} song(s) failed:")
        for song_id, label, error in failures:
            print(f"  {song_id} {label}: {error}")


if __name__ == "__main__":
    main()
