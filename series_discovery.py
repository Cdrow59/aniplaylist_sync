from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import aiosqlite
import networkx as nx
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from mal import MALClient

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
            relation_type = relation_type.strip().casefold() or None
        else:
            relation_type = None

        if relation_type not in ALLOWED_SERIES_RELATIONS:
            continue

        related_ids.append((related_id, relation_type))

    return related_ids


def _series_name_for_component(
    member_ids: Iterable[int], details_by_id: dict[int, dict[str, Any]]
) -> tuple[str, int]:
    known_titles: list[tuple[int, str]] = []

    for anime_id in sorted(set(member_ids)):
        title = _clean_title(details_by_id.get(anime_id, {}).get("title"))
        if title:
            known_titles.append((anime_id, title))

    if known_titles:
        representative_id, series_name = min(known_titles, key=lambda item: item[0])
        return series_name, representative_id

    member_list = sorted(set(member_ids))
    representative_id = member_list[0]
    return f"Series {representative_id}", representative_id


async def discover_series(
    client: MALClient, seed_ids: Iterable[int]
) -> SeriesDiscoveryResult:
    queue = deque(int(anime_id) for anime_id in seed_ids)
    visited_ids: set[int] = set()
    queued_ids: set[int] = set(queue)
    graph = nx.Graph()
    details_by_id: dict[int, dict[str, Any]] = {}

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task("Series", total=len(queue))

        while queue:
            current_id = queue.popleft()
            if current_id in visited_ids:
                continue

            visited_ids.add(current_id)
            graph.add_node(current_id)

            try:
                details = await client.get_anime_details(current_id)
            except Exception:
                progress.advance(task)
                continue

            if not isinstance(details, dict):
                progress.advance(task)
                continue

            details_by_id[current_id] = details

            for related_id, relation_type in _extract_related_ids(details):
                graph.add_edge(current_id, related_id, relation_type=relation_type)
                if related_id not in visited_ids and related_id not in queued_ids:
                    queue.append(related_id)
                    queued_ids.add(related_id)
                    progress.update(task, total=progress.tasks[task].total + 1)

            progress.advance(task)

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
    return SeriesDiscoveryResult(
        graph=graph,
        clusters=clusters,
        details_by_id=details_by_id,
    )


async def save_series_clusters(
    db_path: Path, clusters: Iterable[SeriesCluster]
) -> None:
    async with aiosqlite.connect(db_path) as db:
        await db.execute("DELETE FROM series")

        for cluster in clusters:
            await db.execute(
                """
                INSERT INTO series(
                    series_name,
                    member_ids_json,
                    member_count,
                    representative_mal_id
                ) VALUES (?, ?, ?, ?)
                """,
                (
                    cluster.series_name,
                    json.dumps(cluster.member_ids),
                    len(cluster.member_ids),
                    cluster.representative_id,
                ),
            )

        await db.commit()
