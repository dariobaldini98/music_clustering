#!/usr/bin/env python3
"""Build or resume metadata for local audio files in songs_metadata.json.

The script scans the audio folder, reads each file's embedded title, artist, album,
and duration tags, and enriches those fields through MusicBrainz. It does not
download audio. Existing records are reused, while incomplete or failed
MusicBrainz enrichment is retried.

Requires: mutagen
Install: pip install mutagen
"""

from __future__ import annotations

import argparse
import json
import os
import re
import signal
import sys
import time
import unicodedata
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path, PureWindowsPath
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = PROJECT_DIR / "songs_metadata.json"
DEFAULT_AUDIO_DIR = PROJECT_DIR / "songs"
API_ROOT = "https://musicbrainz.org/ws/2"
REQUEST_INTERVAL_SECONDS = 1.1
MAX_503_RETRIES = 5
RETRY_BACKOFF_SECONDS = 5
MAX_RETRY_BACKOFF_SECONDS = 60
CHECKPOINT_INTERVAL = 100
MINIMUM_MATCH_SCORE = 0.72
MINIMUM_CANDIDATE_SCORE = 0.70
USER_AGENT_NAME = "PlaylistMetadataEnricher/1.0"
AUDIO_EXTENSIONS = {".mp3", ".m4a", ".opus", ".flac", ".wav", ".ogg", ".aac"}


@dataclass(frozen=True)
class RunConfig:
    output_path: Path
    audio_dir: Path
    project_dir: Path
    minimum_match_score: float
    dry_run: bool
    user_agent: str


def prompt_yes_no(prompt: str, default: bool) -> bool:
    default_label = "Y/n" if default else "y/N"
    while True:
        answer = input(f"{prompt} [{default_label}]: ").strip().casefold()
        if not answer:
            return default
        if answer in {"y", "yes"}:
            return True
        if answer in {"n", "no"}:
            return False
        print("Please answer y or n.")


def prompt_path(label: str, default: Path) -> Path:
    answer = input(f"{label} [{default}]: ").strip()
    return Path(answer).expanduser() if answer else default


def prompt_match_score(default: float) -> float:
    while True:
        answer = input(f"Minimum MusicBrainz match score (0-1) [{default}]: ").strip()
        if not answer:
            return default
        try:
            score = float(answer)
        except ValueError:
            score = -1
        if 0 <= score <= 1:
            return score
        print("Enter a number from 0 to 1.")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or resume songs_metadata.json using local audio tags and MusicBrainz."
    )
    parser.add_argument("--output", type=Path, help="Output metadata JSON file.")
    parser.add_argument("--audio-dir", type=Path, help="Folder containing the audio files.")
    parser.add_argument("--project-dir", type=Path, help="Base folder for relative audio paths.")
    parser.add_argument("--min-score", type=float, help="Minimum MusicBrainz match score (0-1).")
    parser.add_argument(
        "--contact-email",
        default=os.environ.get("MUSICBRAINZ_CONTACT_EMAIL"),
        help="Contact email for the MusicBrainz User-Agent (or MUSICBRAINZ_CONTACT_EMAIL).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Do not make requests or write files.")
    parser.add_argument("--no-prompt", action="store_true", help="Use supplied values and defaults.")
    return parser.parse_args(argv)


def build_user_agent(contact_email: str) -> str:
    return f"{USER_AGENT_NAME} ({contact_email.strip()})" if contact_email.strip() else USER_AGENT_NAME


def prompt_contact_email(default: str, required: bool = True) -> str:
    while True:
        answer = input(f"MusicBrainz contact email [{default}]: ").strip()
        email = answer or default
        if email or not required:
            return email
        print("A contact email is required for MusicBrainz requests.")


def choose_configuration(args: argparse.Namespace) -> RunConfig:
    defaults = RunConfig(
        output_path=args.output or DEFAULT_OUTPUT,
        audio_dir=args.audio_dir or DEFAULT_AUDIO_DIR,
        project_dir=args.project_dir or PROJECT_DIR,
        minimum_match_score=(
            args.min_score if args.min_score is not None else MINIMUM_MATCH_SCORE
        ),
        dry_run=args.dry_run,
        user_agent=build_user_agent(args.contact_email or ""),
    )
    if not 0 <= defaults.minimum_match_score <= 1:
        raise ValueError("--min-score must be between 0 and 1.")
    if args.no_prompt:
        return defaults

    print("Default configuration:")
    print(f"  Existing audio folder:    {defaults.audio_dir}")
    print(f"  Output metadata:          {defaults.output_path}")
    print(f"  Relative path base:       {defaults.project_dir}")
    print(f"  Minimum match score:      {defaults.minimum_match_score}")
    print(f"  Dry run (no network/write): {'yes' if defaults.dry_run else 'no'}")

    if prompt_yes_no("Keep this configuration?", default=True):
        if not args.contact_email and not defaults.dry_run:
            defaults = replace(
                defaults,
                user_agent=build_user_agent(prompt_contact_email("", required=True)),
            )
        return defaults

    print("\nEnter new values, or press Enter to keep each displayed default.")
    dry_run = prompt_yes_no("Dry run only?", default=defaults.dry_run)
    contact_email = args.contact_email or prompt_contact_email(
        "", required=not dry_run
    )
    return RunConfig(
        output_path=prompt_path("Output metadata JSON", defaults.output_path),
        audio_dir=prompt_path("Existing audio folder", defaults.audio_dir),
        project_dir=prompt_path("Base folder for relative paths", defaults.project_dir),
        minimum_match_score=prompt_match_score(defaults.minimum_match_score),
        dry_run=dry_run,
        user_agent=build_user_agent(contact_email),
    )


def clean_title(value: str | None) -> str:
    """Remove parenthesized featuring credits from a track title."""
    if not value:
        return ""
    value = re.sub(
        r"\s*\((?:feat\.?|ft\.?|featuring)\s+[^)]*\)",
        "",
        value,
        flags=re.IGNORECASE,
    )
    return value.strip()


def split_artist_names(value: str | None) -> list[str]:
    """Split only the explicit pipe separator; commas may be part of an artist's name."""
    if not value:
        return []
    return [artist.strip() for artist in value.split("|") if artist.strip()]


def artist_search_terms(value: str | None) -> list[str]:
    """Try the full source credit, then explicitly pipe-separated artist names."""
    if not value or not value.strip():
        return []
    full_credit = value.strip()
    terms = [full_credit]
    if "|" in full_credit:
        terms.extend(artist.strip() for artist in full_credit.split("|") if artist.strip())
    return list(dict.fromkeys(terms))


def normalize(value: str | None) -> str:
    """Normalize text for forgiving comparisons."""
    if not value:
        return ""
    value = unicodedata.normalize("NFKD", value)
    value = "".join(char for char in value if not unicodedata.combining(char))
    value = value.casefold()
    value = re.sub(r"[^\w\s]", " ", value)
    return " ".join(value.split())


def duplicate_song_groups(songs: list) -> list[tuple[str, list[dict]]]:
    """Find shared identifiers and possible same-title/artist duplicates without removing entries."""
    grouped: dict[tuple[str, str], list[dict]] = {}
    for index, song in enumerate(songs, start=1):
        if not isinstance(song, dict):
            continue

        identities: set[tuple[str, str]] = set()
        musicbrainz = song.get("musicbrainz")
        if isinstance(musicbrainz, dict):
            recording_id = str(musicbrainz.get("recording_id") or "").strip().casefold()
            if recording_id:
                identities.add(("MusicBrainz recording", recording_id))
            isrcs = musicbrainz.get("isrcs") or []
            if isinstance(isrcs, str):
                isrcs = [isrcs]
            if isinstance(isrcs, list):
                for isrc in isrcs:
                    value = str(isrc or "").strip().casefold()
                    if value:
                        identities.add(("ISRC", value))

        for field in ("source_youtube_id", "youtube_id"):
            value = str(song.get(field) or "").strip().casefold()
            if value:
                identities.add(("YouTube ID", value))

        audio_path = str(song.get("audio_path") or "").strip()
        if audio_path:
            identities.add(("audio path", audio_path_key(audio_path)))

        title = normalize(clean_title(str(song.get("title") or "")))
        artists = str(song.get("artists") or "").strip()
        artist_parts = artists.split("|") if "|" in artists else [artists]
        artist_key = "|".join(sorted(filter(None, (normalize(part) for part in artist_parts))))
        if title and artist_key:
            identities.add(("same title and artist", f"{title}|{artist_key}"))

        for identity in identities:
            grouped.setdefault(identity, []).append(
                {"index": index, "song": song}
            )

    unique_groups: dict[tuple[int, ...], tuple[set[str], list[dict]]] = {}
    for (kind, _), entries in grouped.items():
        if len(entries) < 2:
            continue
        indices = tuple(entry["index"] for entry in entries)
        kinds, _ = unique_groups.setdefault(indices, (set(), entries))
        kinds.add(kind)

    return [
        (", ".join(sorted(kinds)), entries)
        for _, (kinds, entries) in sorted(
            unique_groups.items(),
            key=lambda item: item[0],
        )
    ]


def report_duplicate_songs(songs: list) -> int:
    groups = duplicate_song_groups(songs)
    if not groups:
        print("No likely duplicate records found.")
        return 0

    print(f"Possible duplicate records: {len(groups)} group(s); no entries were changed.")
    for kind, entries in groups:
        print(f"  [{kind}]")
        for entry in entries:
            song = entry["song"]
            song_id = song.get("song_id") or song.get("youtube_id") or f"entry {entry['index']}"
            title = str(song.get("title") or "Unknown title")
            artists = str(song.get("artists") or "Unknown artist")
            print(f"    {song_id}: {artists} - {title}")
    return len(groups)


def parse_duration(value: str | int | float | None) -> int | None:
    """Convert tagged durations, expressed in seconds, to integer seconds."""
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return round(value) if value >= 0 else None
    value = value.strip()
    if not value or not re.fullmatch(r"\d+:\d{1,2}(:\d{1,2})?", value):
        return None
    parts = [int(part) for part in value.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def parse_musicbrainz_duration(value: str | int | float | None) -> int | None:
    """Convert the MusicBrainz recording length, expressed in milliseconds, to seconds."""
    if value is None:
        return None
    try:
        milliseconds = float(value)
    except (TypeError, ValueError):
        return None
    return round(milliseconds / 1000) if milliseconds >= 0 else None


def artist_names(recording: dict) -> list[str]:
    names = []
    for credit in recording.get("artist-credit", []):
        if isinstance(credit, dict) and credit.get("artist"):
            names.append(credit["artist"].get("name") or credit.get("name", ""))
        elif isinstance(credit, str):
            names.append(credit)
    return [name for name in names if name]


def score_candidate(song: dict, candidate: dict) -> float:
    """Score a MusicBrainz recording using title, artists, album, and duration."""
    wanted_title = normalize(clean_title(song.get("title")))
    found_title = normalize(candidate.get("title"))
    title_score = 1.0 if wanted_title == found_title else (
        0.75 if wanted_title and (wanted_title in found_title or found_title in wanted_title) else 0.0
    )

    wanted_artists = [
        normalize(name) for name in artist_search_terms(song.get("artists"))
    ]
    wanted_artists = [name for name in wanted_artists if name]
    found_artists = [normalize(name) for name in artist_names(candidate)]
    artist_score = 0.0
    if any(wanted and wanted == found for wanted in wanted_artists for found in found_artists):
        artist_score = 1.0
    elif any(
        wanted and found and (wanted in found or found in wanted)
        for wanted in wanted_artists
        for found in found_artists
    ):
        artist_score = 0.8

    wanted_album = normalize(song.get("album"))
    found_albums = [normalize(release.get("title")) for release in candidate.get("releases") or []]
    album_score = 1.0 if wanted_album and wanted_album in found_albums else (
        0.5 if not wanted_album else 0.0
    )

    wanted_duration = parse_duration(song.get("duration"))
    found_duration = parse_musicbrainz_duration(candidate.get("length"))
    if wanted_duration is None or found_duration is None:
        duration_score = 0.5
    else:
        difference = abs(wanted_duration - found_duration)
        duration_score = 1.0 if difference <= 2 else (
            0.7 if difference <= 5 else 0.0
        )

    title_weight, artist_weight = (0.40, 0.40) if wanted_artists else (0.80, 0.0)
    return round(
        title_score * title_weight
        + artist_score * artist_weight
        + album_score * 0.12
        + duration_score * 0.08,
        3,
    )


class MusicBrainzClient:
    def __init__(self, user_agent: str):
        self.user_agent = user_agent
        self.last_request_at = 0.0

    @staticmethod
    def _retry_after_seconds(value: str | None) -> float | None:
        if not value:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            try:
                retry_at = parsedate_to_datetime(value)
                if retry_at.tzinfo is None:
                    retry_at = retry_at.replace(tzinfo=timezone.utc)
                return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
            except (TypeError, ValueError, OverflowError):
                return None

    def get_json(self, endpoint: str, params: dict) -> dict:
        url = f"{API_ROOT}/{endpoint}?{urlencode(params)}"
        request = Request(
            url,
            headers={"User-Agent": self.user_agent, "Accept": "application/json"},
        )

        for attempt in range(MAX_503_RETRIES + 1):
            elapsed = time.monotonic() - self.last_request_at
            if elapsed < REQUEST_INTERVAL_SECONDS:
                time.sleep(REQUEST_INTERVAL_SECONDS - elapsed)
            self.last_request_at = time.monotonic()
            try:
                print(f"  MusicBrainz request: {endpoint}")
                with urlopen(request, timeout=30) as response:
                    return json.loads(response.read().decode("utf-8"))
            except HTTPError as exc:
                if exc.code != 503:
                    raise RuntimeError(
                        f"MusicBrainz returned HTTP {exc.code} for {url}"
                    ) from exc
                if attempt == MAX_503_RETRIES:
                    raise RuntimeError(
                        f"MusicBrainz remained busy (HTTP 503) after "
                        f"{MAX_503_RETRIES + 1} attempts for {url}"
                    ) from exc
                retry_after = self._retry_after_seconds(
                    exc.headers.get("Retry-After") if exc.headers else None
                )
                backoff = min(
                    RETRY_BACKOFF_SECONDS * (2 ** attempt),
                    MAX_RETRY_BACKOFF_SECONDS,
                )
                delay = max(
                    retry_after if retry_after is not None else backoff,
                    REQUEST_INTERVAL_SECONDS,
                )
                print(
                    f"  MusicBrainz returned HTTP 503; retrying in {delay:.1f}s "
                    f"({attempt + 1}/{MAX_503_RETRIES})",
                    file=sys.stderr,
                )
                time.sleep(delay)
            except URLError as exc:
                raise RuntimeError(f"Network error calling MusicBrainz: {exc}") from exc
        raise RuntimeError("MusicBrainz request failed after retries")

    def search_recordings(self, query: str, limit: int = 15) -> list[dict]:
        data = self.get_json(
            "recording",
            {"query": query, "fmt": "json", "limit": limit},
        )
        return data.get("recordings", [])

    def lookup_recording(self, recording_id: str) -> dict:
        return self.get_json(
            f"recording/{recording_id}",
            {"inc": "artists+releases+isrcs+url-rels", "fmt": "json"},
        )


def build_query(song: dict, artist: str | None = None) -> str:
    title = clean_title(song.get("title"))
    artists = artist if artist is not None else (song.get("artists") or "").strip()

    def quote(value: str) -> str:
        value = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{value}"'

    if title and artists:
        return f"recording:{quote(title)} AND artist:{quote(artists)}"
    return f"recording:{quote(title)}" if title else quote(artists)


def search_best_match(client: MusicBrainzClient, song: dict) -> tuple[dict | None, float, str]:
    title = clean_title(song.get("title"))
    artists = artist_search_terms(song.get("artists"))
    if not title:
        return None, 0.0, "missing-title"

    queries = [(build_query(song, artist), "title+artist") for artist in artists]
    queries.append((f'recording:"{title.replace(chr(34), "")}"', "title-only"))
    seen_ids = set()
    candidates = []
    search_method = "none"
    search_had_error = False

    for query, method in queries:
        try:
            results = client.search_recordings(query, limit=15)
        except Exception as exc:
            search_had_error = True
            print(f"  Search failed ({method}): {exc}", file=sys.stderr)
            continue
        for result in results:
            recording_id = result.get("id")
            if recording_id and recording_id not in seen_ids:
                seen_ids.add(recording_id)
                candidates.append(result)
        if candidates:
            search_method = method
            if method == "title+artist":
                best = max(candidates, key=lambda candidate: score_candidate(song, candidate))
                if score_candidate(song, best) >= MINIMUM_CANDIDATE_SCORE:
                    break

    if not candidates:
        return None, 0.0, "search-error" if search_had_error else "not-found"

    best = max(candidates, key=lambda candidate: score_candidate(song, candidate))
    best_score = score_candidate(song, best)
    return best, best_score, search_method


def choose_release(recording: dict, preferred_album: str | None) -> dict | None:
    releases = recording.get("releases") or []
    if not releases:
        return None
    preferred = normalize(preferred_album)
    if preferred:
        exact = [release for release in releases if normalize(release.get("title")) == preferred]
        if exact:
            return min(exact, key=lambda release: release.get("date") or "9999")
    dated_official = [
        release for release in releases
        if release.get("date") and str(release.get("status") or "").casefold() == "official"
    ]
    if dated_official:
        return min(dated_official, key=lambda release: release["date"])
    dated = [release for release in releases if release.get("date")]
    return min(dated, key=lambda release: release["date"]) if dated else releases[0]


def enrich_song(
    client: MusicBrainzClient,
    song: dict,
    min_score: float,
) -> dict:
    song = dict(song)
    song["title"] = clean_title(song.get("title"))
    existing_mb = song.get("musicbrainz") or {}
    recording_id = existing_mb.get("recording_id")
    match_score = 1.0 if recording_id else 0.0
    method = "existing-recording-id" if recording_id else "search"

    if recording_id:
        try:
            recording = client.lookup_recording(recording_id)
        except Exception as exc:
            print(f"  Lookup failed for existing ID {recording_id}: {exc}", file=sys.stderr)
            recording = None
    else:
        result, match_score, method = search_best_match(client, song)
        recording = None
        if result and match_score >= min_score:
            recording_id = result.get("id")
            try:
                recording = client.lookup_recording(recording_id)
            except Exception as exc:
                print(f"  Recording lookup failed: {exc}", file=sys.stderr)

    if not recording:
        song["musicbrainz"] = {
            **existing_mb,
            "status": (
                "search-error"
                if method == "search-error"
                else "not-found" if not recording_id else "lookup-error"
            ),
            "processing_complete": method != "search-error" and not recording_id,
            "match_score": match_score,
            "search_method": method,
        }
        return song

    release = choose_release(recording, song.get("album"))
    canonical_artists = "|".join(artist_names(recording))
    canonical_title = recording.get("title")
    canonical_album = release.get("title") if release else None
    recording_id = recording.get("id")
    song["musicbrainz"] = {
        **existing_mb,
        "status": "matched",
        "processing_complete": True,
        "match_score": match_score,
        "search_method": method,
        "recording_id": recording_id,
        "recording_url": f"https://musicbrainz.org/recording/{recording_id}",
        "title": canonical_title,
        "artist_credit": canonical_artists or None,
        "artists": [
            {
                "name": credit["artist"].get("name"),
                "id": credit["artist"].get("id"),
                "url": f"https://musicbrainz.org/artist/{credit['artist'].get('id')}",
            }
            for credit in recording.get("artist-credit") or []
            if isinstance(credit, dict) and credit.get("artist")
        ],
        "release_id": release.get("id") if release else None,
        "release_url": f"https://musicbrainz.org/release/{release.get('id')}" if release else None,
        "album": canonical_album,
        "release_date": (release or {}).get("date") or recording.get("first-release-date"),
        "length_ms": recording.get("length"),
        "isrcs": recording.get("isrcs", []),
        "video": recording.get("video"),
    }

    if canonical_title:
        song["title"] = canonical_title
    if canonical_artists:
        song["artists"] = canonical_artists
    if canonical_album:
        song["album"] = canonical_album
    return song


def is_completed(song: object) -> bool:
    if not isinstance(song, dict):
        return False
    musicbrainz = song.get("musicbrainz")
    if not isinstance(musicbrainz, dict):
        return False
    return musicbrainz.get("status") == "matched" or (
        musicbrainz.get("status") == "not-found"
        and musicbrainz.get("processing_complete") is True
    )


def audio_path_key(raw: str) -> str:
    return PureWindowsPath(str(raw)).as_posix().casefold()


def resolve_audio_path(raw: object, project_dir: Path) -> Path | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(PureWindowsPath(raw).as_posix())
    if not path.is_absolute():
        path = project_dir / path
    return path if path.is_file() and path.suffix.casefold() in AUDIO_EXTENSIONS else None


def recorded_audio_path(path: Path, project_dir: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(project_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def read_audio_metadata(path: Path) -> dict:
    """Read searchable fields from embedded tags; use the filename as a title fallback."""
    from mutagen import File as MutagenFile
    from mutagen import MutagenError

    try:
        audio = MutagenFile(path, easy=True)
    except (MutagenError, OSError, ValueError) as exc:
        print(f"  Could not read tags from {path.name}: {exc}", file=sys.stderr)
        audio = None

    tags = audio or {}

    def tag_value(*keys: str) -> str:
        for key in keys:
            values = tags.get(key)
            if isinstance(values, str):
                values = [values]
            if values:
                joined = "|".join(str(value).strip() for value in values if str(value).strip())
                if joined:
                    return joined
        return ""

    title = tag_value("title")
    artists = tag_value("artist", "artists")
    album = tag_value("album")
    duration = parse_duration(getattr(getattr(audio, "info", None), "length", None))
    if not title:
        title = re.sub(r"^(?:SNG)?\d+_", "", path.stem).replace("_", " ").strip()
    if not artists:
        print(f"  No artist tag in {path.name}; searching and scoring by title only.", file=sys.stderr)
    if not title:
        print(f"  No title tag or usable filename in {path.name}; skipping.", file=sys.stderr)
        return {}
    song = {
        "title": title,
        "artists": artists,
        "album": album,
    }
    if duration is not None:
        song["duration"] = duration
    return song


def load_audio_songs(
    audio_dir: Path,
    project_dir: Path,
    previous: list | None = None,
) -> list[dict]:
    if not audio_dir.is_dir():
        raise FileNotFoundError(f"Audio folder not found: {audio_dir}")
    existing_paths = {
        resolved_audio_path_key(song.get("audio_path"), project_dir)
        for song in previous or []
        if isinstance(song, dict) and song.get("audio_path")
    }
    songs = []
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.casefold() not in AUDIO_EXTENSIONS:
            continue
        audio_path = recorded_audio_path(path, project_dir)
        if resolved_audio_path_key(audio_path, project_dir) in existing_paths:
            songs.append({"audio_path": audio_path})
            continue
        song = read_audio_metadata(path)
        if not song:
            continue
        song["audio_path"] = audio_path
        songs.append(song)
    return songs


def underscored(value: object) -> str:
    """Make an identifier safe for Chroma by replacing punctuation with underscores."""
    return re.sub(r"[^A-Za-z0-9]+", "_", str(value or "").strip()).strip("_")


def base_song_id(song: dict) -> str:
    """Return a stable, Chroma-safe ID based on the source filename."""
    audio_path = song.get("audio_path")
    if audio_path:
        stem = underscored(PureWindowsPath(str(audio_path)).stem)
        return f"file_{stem or 'unknown'}"
    return "file_unknown"


def assign_song_ids(songs: list, new_songs: list[dict] | None = None) -> None:
    """Assign IDs only to new songs, reserving IDs already present in the output."""
    new_song_ids = {id(song) for song in new_songs} if new_songs is not None else None
    used = {
        str(song.get("song_id")).strip()
        for song in songs
        if isinstance(song, dict) and str(song.get("song_id") or "").strip()
    }
    next_suffix: dict[str, int] = {}
    for song in songs:
        if not isinstance(song, dict) or (new_song_ids is not None and id(song) not in new_song_ids):
            continue
        if str(song.get("song_id") or "").strip():
            continue
        base = base_song_id(song)
        suffix = next_suffix.get(base, 0)
        candidate = base if suffix == 0 else f"{base}_{suffix}"
        while candidate in used:
            suffix += 1
            candidate = f"{base}_{suffix}"
        song["song_id"] = candidate
        used.add(candidate)
        next_suffix[base] = suffix + 1


def resolved_audio_path_key(raw: object, project_dir: Path) -> str:
    path = resolve_audio_path(raw, project_dir)
    if path:
        return str(path.resolve()).casefold()
    return audio_path_key(str(raw or ""))


def merge_audio_songs(
    audio_songs: list[dict],
    previous: list,
    project_dir: Path,
) -> tuple[list[dict], int, int, list[dict]]:
    by_audio_path: dict[str, list[int]] = {}
    for index, old_song in enumerate(previous):
        if not isinstance(old_song, dict):
            continue
        raw_path = old_song.get("audio_path")
        if isinstance(raw_path, str) and raw_path.strip():
            key = resolved_audio_path_key(raw_path, project_dir)
            by_audio_path.setdefault(key, []).append(index)

    merged = []
    active_songs = []
    reused_metadata = 0
    unused_previous = set(range(len(previous)))

    for song in audio_songs:
        key = resolved_audio_path_key(song["audio_path"], project_dir)
        previous_index = next(
            (index for index in by_audio_path.get(key, []) if index in unused_previous),
            None,
        )
        previous_song = previous[previous_index] if previous_index is not None else None
        if previous_index is not None:
            unused_previous.remove(previous_index)

        if isinstance(previous_song, dict):
            old_musicbrainz = previous_song.get("musicbrainz")
            if isinstance(old_musicbrainz, dict):
                reused_metadata += 1
            merged.append(previous_song)
            active_songs.append(previous_song)
            continue

        merged.append(song)
        active_songs.append(song)

    preserved_previous = 0
    for index in sorted(unused_previous):
        merged.append(previous[index])
        preserved_previous += 1

    return merged, reused_metadata, preserved_previous, active_songs


def save_json(path: Path, data: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f"{path.name}.tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(path)


def read_json(path: Path) -> list:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, list):
        raise ValueError(f"Expected {path} to contain a JSON array.")
    return data


def handle_termination_signal(signum, frame) -> None:
    signal_name = signal.Signals(signum).name
    raise KeyboardInterrupt(f"received {signal_name} at line {frame.f_lineno}")


def main() -> int:
    try:
        args = parse_args()
        config = choose_configuration(args)
    except EOFError:
        print("\nConfiguration cancelled.")
        return 130
    except ValueError as exc:
        print(f"Configuration error: {exc}", file=sys.stderr)
        return 2

    if not config.dry_run and config.user_agent == USER_AGENT_NAME:
        print(
            "MusicBrainz requests require a contact email. Pass --contact-email "
            "or set MUSICBRAINZ_CONTACT_EMAIL.",
            file=sys.stderr,
        )
        return 2

    output_path = config.output_path.resolve()
    audio_dir = config.audio_dir.resolve()
    project_dir = config.project_dir.resolve()
    print("\n[1/4] Loading existing metadata and scanning local audio...")
    try:
        previous = read_json(output_path) if output_path.is_file() else []
        if output_path.is_file():
            print(f"  Loaded {len(previous)} entries from {output_path}")
        else:
            print(f"  No existing output at {output_path}; starting with an empty metadata set.")
        audio_songs = load_audio_songs(audio_dir, project_dir, previous)
    except (OSError, ValueError, ImportError) as exc:
        if isinstance(exc, ImportError):
            print(
                "The 'mutagen' package is required to read audio tags. "
                "Install it with: python -m pip install mutagen",
                file=sys.stderr,
            )
        else:
            print(f"Could not load metadata or audio files: {exc}", file=sys.stderr)
        return 2

    print(f"  Found {len(audio_songs)} supported audio file(s) in {audio_dir}.")
    print("[2/4] Matching files to existing metadata and reusing MusicBrainz results...")
    data, reused_metadata, preserved_previous, active_songs = merge_audio_songs(
        audio_songs,
        previous,
        project_dir,
    )
    assign_song_ids(data, active_songs)
    pending = sum(
        not is_completed(song)
        for song in active_songs
    )
    ready = sum(
        isinstance(song, dict) and bool(song.get("audio_path"))
        for song in data
    )
    print(
        f"Prepared {len(data)} entries: {ready} with local audio; "
        f"reused MusicBrainz metadata for {reused_metadata} file(s)."
    )
    report_duplicate_songs(data)
    if preserved_previous:
        print(f"Preserved {preserved_previous} existing output entries whose files are not in the audio folder.")

    if config.dry_run:
        print(f"Would make up to {pending} MusicBrainz request(s). No network or files changed.")
        return 0

    print(f"[3/4] Enriching {pending} song(s) with MusicBrainz data...")
    if pending == 0:
        print("  All songs already have completed MusicBrainz processing.")
    client = MusicBrainzClient(config.user_agent)
    processed_since_save = 0
    processed = 0
    active_song_ids = {id(song) for song in active_songs}
    try:
        signal.signal(signal.SIGTERM, handle_termination_signal)
        for index, song in enumerate(data, start=1):
            if not isinstance(song, dict) or id(song) not in active_song_ids or is_completed(song):
                continue
            processed += 1
            label = f"{song.get('artists', 'Unknown artist')} — {song.get('title', 'Unknown title')}"
            print(f"  MusicBrainz [{processed}/{pending}] (entry {index}/{len(data)}): {label}")
            data[index - 1] = enrich_song(
                client,
                song,
                config.minimum_match_score,
            )
            result = data[index - 1].get("musicbrainz", {})
            if result.get("status") == "matched":
                print(f"  Matched: {result.get('recording_id')} (score={result.get('match_score')})")
            else:
                print(f"  No confident match (score={result.get('match_score')})")

            processed_since_save += 1
            if processed_since_save >= CHECKPOINT_INTERVAL:
                save_json(output_path, data)
                processed_since_save = 0
                print(f"  Checkpoint saved: {output_path}")
    except KeyboardInterrupt as exc:
        print("\nSaving progress after interruption...")
        save_json(output_path, data)
        detail = f" ({exc})" if exc.args else ""
        print(f"\nInterrupted{detail}; progress saved to {output_path}. Run again to resume.")
        return 130

    print("[4/4] Saving final metadata...")
    save_json(output_path, data)
    matched = sum(
        isinstance(song, dict)
        and isinstance(song.get("musicbrainz"), dict)
        and song["musicbrainz"].get("status") == "matched"
        for song in data
    )
    print(f"\nFinished: {matched}/{len(data)} songs matched.")
    print(f"Output written to: {output_path}")
    print("Run embed_music_effnet.py to sync metadata and embed songs with available local audio.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
