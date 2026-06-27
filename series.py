"""MAL series discovery helpers."""

from __future__ import annotations

import asyncio
import json
import logging
import re
import unicodedata
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Protocol

import aiohttp
import aiosqlite
import networkx as nx
from rich.progress import Progress

from mal import MALClient


class HasAnimeDetails(Protocol):
    async def get_anime_details(self, anime_id: int) -> dict[str, Any]: ...

logger = logging.getLogger(__name__)

ALLOWED_SERIES_RELATIONS = {"sequel", "prequel"}


@dataclass(slots=True)
class SeriesCluster:
    series_name: str
    member_ids: list[int]
    representative_id: int


@dataclass(slots=True)
class SeriesDiscoveryResult:
    graph: nx.Graph
    clusters: list[SeriesCluster]
    details_by_id: dict[int, dict[str, Any]]


def _clean_title(value: object | None) -> str | None:
    if isinstance(value, str):
        cleaned = value.strip()
        if cleaned:
            return cleaned
    return None


def _normalize_relation_type(value: str | None) -> str | None:
    if not value:
        return None
    value = value.strip().casefold()
    if value.startswith("prequel"):
        return "prequel"
    if value.startswith("sequel"):
        return "sequel"
    return None


def _extract_related_ids(details: dict[str, Any]) -> list[tuple[int, str | None]]:
    related_ids: list[tuple[int, str | None]] = []
    related_anime = details.get("related_anime")
    if not isinstance(related_anime, list):
        return related_ids
    for relation in related_anime:
        if not isinstance(relation, dict):
            continue
        node = relation.get("node")
        if not isinstance(node, dict):
            continue
        try:
            related_id = int(node.get("id"))
        except (TypeError, ValueError):
            continue
        relation_type = relation.get("relation_type")
        if isinstance(relation_type, str):
            relation_type = _normalize_relation_type(relation_type)
        else:
            relation_type = None
        if relation_type is None:
            continue
        related_ids.append((related_id, relation_type))
    return related_ids


_SERIES_ALIASES = {
    "Bakemonogatari": "Monogatari",
    "Fate/stay night": "Fate",
}

_TITLE_CLEANUP_PATTERNS = [
    r"\(\d{4}\)",
    r"\b\d+(?:st|nd|rd|th)\s+season\b",
    r"\bseason\s+\d+\b",
    r"\bpart\s+\d+\b",
    r"\bfinal\s+season\b",
]


def _canonical_series_title(title: str) -> str:
    title = unicodedata.normalize("NFKC", title).strip()
    alias = _SERIES_ALIASES.get(title)
    if alias:
        return alias
    for pattern in _TITLE_CLEANUP_PATTERNS:
        title = re.sub(pattern, "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title)
    return title.strip(" :-")


def _series_name_for_component(
    member_ids: Iterable[int],
    details_by_id: dict[int, dict[str, Any]],
) -> tuple[str, int]:
    known_titles: list[tuple[int, str]] = []
    for anime_id in sorted(set(member_ids)):
        details = details_by_id.get(anime_id) or {}
        alt = details.get("alternative_titles") or {}
        title = (
            _clean_title(alt.get("en"))
            or _clean_title(details.get("title"))
            or _clean_title(alt.get("jp"))
        )
        if title:
            known_titles.append((anime_id, title))
    if not known_titles:
        representative_id = min(member_ids)
        return f"Unknown Series ({representative_id})", representative_id
    representative_id, representative_title = min(
        known_titles, key=lambda item: item[0]
    )
    return _canonical_series_title(representative_title), representative_id


async def discover_series(
    client: HasAnimeDetails,
    seed_ids: Iterable[int],
    *,
    progress: Progress,
) -> SeriesDiscoveryResult:
    """Discover series clusters via BFS over MAL related-anime edges.

    Args:
        client: Authenticated MAL client.
        seed_ids: Starting MAL IDs (typically the user's list).
        progress: A *started* Rich Progress instance owned by the caller.
                  Tasks will be added and advanced; the caller retains
                  ownership and must not stop the Progress here.
    """
    queue = deque(int(anime_id) for anime_id in seed_ids)
    if not queue:
        logger.info("Series discovery skipped — no seed IDs provided")
        return SeriesDiscoveryResult(graph=nx.Graph(), clusters=[], details_by_id={})

    logger.info("Series discovery started — %d seed ID(s)", len(queue))

    visited_ids: set[int] = set()
    queued_ids: set[int] = set(queue)
    graph = nx.Graph()
    details_by_id: dict[int, dict[str, Any]] = {}
    fetch_failures = 0

    task_id = progress.add_task("Series", total=len(queue))
    total = len(queue)

    while queue:
        current_id = queue.popleft()
        if current_id in visited_ids:
            continue

        visited_ids.add(current_id)
        graph.add_node(current_id)

        try:
            details = await client.get_anime_details(current_id)
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            RuntimeError,
            ValueError,
        ) as exc:
            fetch_failures += 1
            logger.warning(
                "Series discovery: failed to fetch details for MAL ID %d: %s",
                current_id,
                exc,
            )
            progress.advance(task_id)
            continue

        if not isinstance(details, dict):
            logger.debug(
                "Series discovery: unexpected details payload for MAL ID %d (type=%s)",
                current_id,
                type(details).__name__,
            )
            progress.advance(task_id)
            continue

        details_by_id[current_id] = details

        related = _extract_related_ids(details)
        if related:
            logger.debug(
                "Series discovery: MAL ID %d has %d related entry(ies)",
                current_id,
                len(related),
            )

        for related_id, relation_type in related:
            graph.add_edge(current_id, related_id, relation_type=relation_type)
            if related_id not in visited_ids and related_id not in queued_ids:
                queue.append(related_id)
                queued_ids.add(related_id)
                graph.add_node(related_id)
                total += 1
                progress.update(task_id, total=total)

        progress.advance(task_id)

    # Fetch details for nodes discovered only as relations (not in seed)
    remaining = [node for node in graph.nodes if node not in details_by_id]
    if remaining:
        logger.debug(
            "Series discovery: fetching details for %d related-only node(s)",
            len(remaining),
        )
    progress.update(task_id, total=total + len(graph.nodes))
    for node in graph.nodes:
        if node in details_by_id:
            progress.advance(task_id)
            continue
        try:
            details = await client.get_anime_details(node)
            if isinstance(details, dict):
                details_by_id[node] = details
        except (
            aiohttp.ClientError,
            asyncio.TimeoutError,
            RuntimeError,
            ValueError,
        ) as exc:
            fetch_failures += 1
            logger.warning(
                "Series discovery: failed to fetch details for related MAL ID %d: %s",
                node,
                exc,
            )
        finally:
            progress.advance(task_id)

    clusters: list[SeriesCluster] = []
    for component in nx.connected_components(graph):
        member_ids = sorted(int(anime_id) for anime_id in component)
        series_name, representative_id = _series_name_for_component(
            member_ids, details_by_id
        )
        clusters.append(
            SeriesCluster(
                series_name=series_name,
                member_ids=member_ids,
                representative_id=representative_id,
            )
        )

    clusters.sort(
        key=lambda cluster: (cluster.series_name.casefold(), cluster.representative_id)
    )

    logger.info(
        "Series discovery complete — %d node(s), %d cluster(s), %d fetch failure(s)",
        len(graph.nodes),
        len(clusters),
        fetch_failures,
    )

    return SeriesDiscoveryResult(
        graph=graph, clusters=clusters, details_by_id=details_by_id
    )


async def save_series_clusters(
    db_path: Path, clusters: Iterable[SeriesCluster]
) -> None:
    cluster_list = list(clusters)
    logger.debug("Saving %d series cluster(s) to %s", len(cluster_list), db_path)
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM series")
        for cluster in cluster_list:
            await db.execute(
                """
                INSERT INTO series(series_name, member_ids_json, member_count, representative_mal_id)
                VALUES (?, ?, ?, ?)
                """,
                (
                    cluster.series_name,
                    json.dumps(cluster.member_ids),
                    len(cluster.member_ids),
                    cluster.representative_id,
                ),
            )
        await db.commit()
    logger.info("Saved %d series cluster(s)", len(cluster_list))
