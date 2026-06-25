"""Data loading utilities for the first G-SITM recommendation prototype."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


@dataclass(frozen=True)
class Space:
    space_id: str
    label: str
    space_type: str
    floor: int
    capacity: int


@dataclass(frozen=True)
class Connection:
    source: str
    target: str
    distance: float
    connection_type: str
    accessibility_cost: float = 0.0


@dataclass(frozen=True)
class POI:
    poi_id: str
    label: str
    theme: str
    tags: List[str]
    space_id: str
    importance: float


@dataclass(frozen=True)
class TrajectoryEvent:
    visitor_id: str
    time: str
    space_id: str
    poi_id: Optional[str]
    dwell_time_seconds: float
    event_type: str


@dataclass(frozen=True)
class ContextState:
    time: str
    entity_id: str
    entity_type: str
    crowd_level: str
    status: str
    note: str = ""


@dataclass
class MuseumDataset:
    spaces: Dict[str, Space]
    connections: List[Connection]
    pois: Dict[str, POI]
    trajectory: List[TrajectoryEvent]
    context: Dict[str, ContextState]


def _read_csv(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def load_dataset(data_dir: str | Path) -> MuseumDataset:
    """Load the CSV-based museum scenario.

    The prototype deliberately uses CSV files to remain easy to inspect,
    modify, and replace with real museum data later.
    """
    data_dir = Path(data_dir)

    spaces = {
        row["space_id"]: Space(
            space_id=row["space_id"],
            label=row["label"],
            space_type=row["space_type"],
            floor=int(row["floor"]),
            capacity=int(row["capacity"]),
        )
        for row in _read_csv(data_dir / "spaces.csv")
    }

    connections = [
        Connection(
            source=row["source"],
            target=row["target"],
            distance=float(row["distance"]),
            connection_type=row["connection_type"],
            accessibility_cost=float(row.get("accessibility_cost") or 0.0),
        )
        for row in _read_csv(data_dir / "connections.csv")
    ]

    pois = {
        row["poi_id"]: POI(
            poi_id=row["poi_id"],
            label=row["label"],
            theme=row["theme"],
            tags=[tag.strip().lower() for tag in row["tags"].split(";") if tag.strip()],
            space_id=row["space_id"],
            importance=float(row["importance"]),
        )
        for row in _read_csv(data_dir / "pois.csv")
    }

    trajectory = [
        TrajectoryEvent(
            visitor_id=row["visitor_id"],
            time=row["time"],
            space_id=row["space_id"],
            poi_id=row["poi_id"] or None,
            dwell_time_seconds=float(row.get("dwell_time_seconds") or 0.0),
            event_type=row["event_type"],
        )
        for row in _read_csv(data_dir / "visitor_trajectory.csv")
    ]

    # The first prototype stores only one context state per entity, assumed to be
    # the most recent valid state. Later this can become a temporal table.
    context = {
        row["entity_id"]: ContextState(
            time=row["time"],
            entity_id=row["entity_id"],
            entity_type=row["entity_type"],
            crowd_level=row["crowd_level"].lower(),
            status=row["status"].lower(),
            note=row.get("note", ""),
        )
        for row in _read_csv(data_dir / "context.csv")
    }

    return MuseumDataset(
        spaces=spaces,
        connections=connections,
        pois=pois,
        trajectory=trajectory,
        context=context,
    )
