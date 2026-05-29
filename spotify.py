"""Spotify playlist creation helpers for AniPlaylist sync."""

from __future__ import annotations

import asyncio
import json
import os
import re
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


def _read_spotify_env(name: str) -> str:
    value = os.getenv(name)
    if value is not None and value.strip():
        return value.strip()

    if name.startswith("SPOTIPY_"):
        fallback_name = name.replace("SPOTIPY_", "SPOTIFY_", 1)
        fallback_value = os.getenv(fallback_name)
        if fallback_value is not None and fallback_value.strip():
            return fallback_value.strip()

    if name.startswith("SPOTIFY_"):
        fallback_name = name.replace("SPOTIFY_", "SPOTIPY_", 1)
        fallback_value = os.getenv(fallback_name)
        if fallback_value is not None and fallback_value.strip():
            return fallback_value.strip()

    raise RuntimeError(f"Missing required environment variable: {name}")


def spotify_link_kind_and_id(spotify_link: str) -> tuple[str | None, str | None]:
    cleaned_link = spotify_link.strip()
    if not cleaned_link:
        return None, None

    if cleaned_link.startswith("spotify:track:"):
        return "track", cleaned_link.rsplit(":", 1)[-1]

    if cleaned_link.startswith("spotify:album:"):
        return "album", cleaned_link.rsplit(":", 1)[-1]

    track_match = re.search(r"/track/([A-Za-z0-9]+)", cleaned_link)
    if track_match:
        return "track", track_match.group(1)

    album_match = re.search(r"/album/([A-Za-z0-9]+)", cleaned_link)
    if album_match:
        return "album", album_match.group(1)

    return None, None


def spotify_link_to_track_uri(spotify_link: str) -> str | None:
    kind, resource_id = spotify_link_kind_and_id(spotify_link)
    if kind != "track" or not resource_id:
        return None
    return f"spotify:track:{resource_id}"


def _unique_preserving_order(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    unique_values: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        unique_values.append(value)
    return unique_values


async def fetch_result_links(db_path: Path) -> list[str]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("""
            SELECT spotify_link
            FROM results
            WHERE spotify_link IS NOT NULL AND TRIM(spotify_link) <> ''
            ORDER BY id
            """) as cursor:
            rows = await cursor.fetchall()

    links: list[str] = []
    for row in rows:
        link = str(row[0]).strip()
        if link:
            links.append(link)

    return links


async def fetch_series_playlist_sources(
    db_path: Path,
) -> list[tuple[str, list[int]]]:
    async with aiosqlite.connect(db_path) as db:
        async with db.execute("""
            SELECT series_name, member_ids_json
            FROM series
            ORDER BY series_name COLLATE NOCASE, representative_mal_id
            """) as cursor:
            rows = await cursor.fetchall()

    sources: list[tuple[str, list[int]]] = []
    for series_name, member_ids_json in rows:
        try:
            member_ids = [int(anime_id) for anime_id in json.loads(member_ids_json)]
        except (TypeError, ValueError, json.JSONDecodeError):
            continue

        cleaned_name = str(series_name).strip()
        if cleaned_name and member_ids:
            sources.append((cleaned_name, member_ids))

    return sources


async def fetch_playlist_links_for_mal_ids(
    db_path: Path, mal_ids: Iterable[int]
) -> list[str]:
    unique_ids = sorted({int(mal_id) for mal_id in mal_ids})
    if not unique_ids:
        return []

    placeholders = ",".join("?" for _ in unique_ids)
    query = f"""
        SELECT spotify_link
        FROM results
        WHERE mal_id IN ({placeholders})
          AND spotify_link IS NOT NULL
          AND TRIM(spotify_link) <> ''
        ORDER BY id
    """

    async with aiosqlite.connect(db_path) as db:
        async with db.execute(query, unique_ids) as cursor:
            rows = await cursor.fetchall()

    links: list[str] = []
    for row in rows:
        link = str(row[0]).strip()
        if link:
            links.append(link)

    return links


def _chunked(values: list[str], size: int) -> Iterable[list[str]]:
    for start in range(0, len(values), size):
        yield values[start : start + size]


async def _resolve_album_track_uris(client: Any, album_id: str) -> list[str]:
    track_uris: list[str] = []
    offset = 0
    limit = 50

    while True:
        page = await asyncio.to_thread(
            client.album_tracks,
            album_id,
            limit=limit,
            offset=offset,
        )

        for item in page.get("items", []):
            uri = item.get("uri")
            if isinstance(uri, str) and uri.startswith("spotify:track:"):
                track_uris.append(uri)

        if not page.get("next"):
            break

        offset += limit

    return track_uris


async def resolve_spotify_link_to_track_uris(
    client: Any, spotify_link: str
) -> list[str]:
    kind, resource_id = spotify_link_kind_and_id(spotify_link)
    if not kind or not resource_id:
        return []

    if kind == "track":
        uri = spotify_link_to_track_uri(spotify_link)
        return [uri] if uri else []

    if kind == "album":
        try:
            from spotipy.exceptions import SpotifyException

            return await _resolve_album_track_uris(client, resource_id)
        except (SpotifyException, RuntimeError, ValueError, KeyError, TypeError):
            return []

    return []


async def create_spotify_playlist(
    client: Any, user_id: str, name: str, links: list[str]
) -> None:
    resolved_uris: list[str] = []
    for link in links:
        resolved_uris.extend(await resolve_spotify_link_to_track_uris(client, link))

    uris = _unique_preserving_order(resolved_uris)

    if not uris:
        return

    playlist = await asyncio.to_thread(
        client.user_playlist_create,
        user=user_id,
        name=name,
        public=False,
        collaborative=False,
        description="Created by aniplaylist_sync",
    )
    playlist_id = playlist["id"]

    for uri_batch in _chunked(uris, 100):
        await asyncio.to_thread(client.playlist_add_items, playlist_id, uri_batch)


async def run_spotify_stage(
    db_path: Path,
    *,
    megaplaylist: bool,
    progress: Progress | None = None,
) -> None:
    try:
        import spotipy
        from spotipy.oauth2 import SpotifyOAuth
    except ImportError as exc:
        raise RuntimeError(
            "spotipy is required for Spotify playlist creation. Install it before running without --dry-run."
        ) from exc

    auth_manager = SpotifyOAuth(
        client_id=_read_spotify_env("SPOTIPY_CLIENT_ID"),
        client_secret=_read_spotify_env("SPOTIPY_CLIENT_SECRET"),
        redirect_uri=_read_spotify_env("SPOTIPY_REDIRECT_URI"),
        scope="playlist-modify-private playlist-modify-public",
    )
    client = spotipy.Spotify(auth_manager=auth_manager)
    user_id = (await asyncio.to_thread(client.current_user))["id"]

    if megaplaylist:
        playlist_sources: list[tuple[str, list[str]]] = [
            ("AniPlaylist Megaplaylist", await fetch_result_links(db_path))
        ]
    else:
        playlist_sources = []
        for series_name, member_ids in await fetch_series_playlist_sources(db_path):
            links = await fetch_playlist_links_for_mal_ids(db_path, member_ids)
            playlist_sources.append((series_name, links))

    progress_context = (
        Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            transient=True,
        )
        if progress is None
        else nullcontext(progress)
    )

    with progress_context as progress:
        assert progress is not None

        task_id = progress.add_task("Spotify", total=len(playlist_sources))

        for playlist_name, links in playlist_sources:
            await create_spotify_playlist(client, user_id, playlist_name, links)
            progress.advance(task_id)
