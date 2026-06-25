"""Explainable G-SITM-based POI and path recommendation.

The first version is intentionally simple and transparent:
- G-SITM dimensions are represented by node and edge attributes.
- Candidate POIs are generated from the graph.
- Recommendations are ranked using semantic relevance, importance, distance,
  and contextual crowd cost.
- Paths are computed with Dijkstra over static or dynamic edge weights.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import networkx as nx

from .data_loader import MuseumDataset, POI


CROWD_VALUE = {
    "empty": 0.0,
    "low": 0.0,
    "medium": 0.5,
    "high": 1.0,
    "very_high": 1.5,
    "closed": float("inf"),
}


@dataclass
class RecommendationConfig:
    semantic_weight: float = 0.45
    importance_weight: float = 0.25
    distance_weight: float = 0.15
    crowd_weight: float = 0.15
    distance_normalizer: float = 60.0
    path_crowd_penalty: float = 12.0
    path_accessibility_penalty: float = 5.0
    avoid_closed_spaces: bool = True


@dataclass
class PathResult:
    path: List[str]
    cost: float
    distance: float
    crowd_cost: float
    path_type: str


@dataclass
class POIRecommendation:
    poi_id: str
    poi_label: str
    poi_theme: str
    poi_space_id: str
    poi_space_label: str
    score: float
    semantic_relevance: float
    importance: float
    distance_cost: float
    crowd_cost: float
    shortest_path: Optional[PathResult]
    crowd_aware_path: Optional[PathResult]
    explanation: List[str] = field(default_factory=list)


class GSITMRecommender:
    """First implementable recommender based on G-SITM.

    The prototype represents G=(V,E,D,T) as a graph whose nodes and edges are
    annotated by dimensions:
        D_space: rooms, galleries, corridors, doors, stairs
        D_POI: artworks, exhibitions, showcases
        D_MO: visitors and their semantic trajectories
        D_context: crowd, closure, accessibility state
        T: time is currently represented through timestamps in contextual states
    """

    def __init__(self, dataset: MuseumDataset, config: Optional[RecommendationConfig] = None):
        self.dataset = dataset
        self.config = config or RecommendationConfig()
        self.graph = self._build_graph(dataset)

    @staticmethod
    def canonical_edge_id(a: str, b: str) -> str:
        """Canonical identifier used to attach context to undirected edges."""
        return "--".join(sorted([a, b]))

    def _build_graph(self, dataset: MuseumDataset) -> nx.Graph:
        g = nx.Graph()

        # D_space nodes
        for space in dataset.spaces.values():
            context = dataset.context.get(space.space_id)
            g.add_node(
                space.space_id,
                dimension="D_space",
                label=space.label,
                type=space.space_type,
                floor=space.floor,
                capacity=space.capacity,
                crowd_level=context.crowd_level if context else "low",
                status=context.status if context else "open",
            )

        # Spatial edges in D_space with contextual attributes
        for connection in dataset.connections:
            edge_id = self.canonical_edge_id(connection.source, connection.target)
            context = dataset.context.get(edge_id)
            g.add_edge(
                connection.source,
                connection.target,
                edge_id=edge_id,
                dimension="D_space",
                distance=connection.distance,
                connection_type=connection.connection_type,
                accessibility_cost=connection.accessibility_cost,
                crowd_level=context.crowd_level if context else "low",
                status=context.status if context else "open",
            )

        # D_POI nodes and POI-space cross-dimensional relations
        for poi in dataset.pois.values():
            g.add_node(
                poi.poi_id,
                dimension="D_POI",
                label=poi.label,
                theme=poi.theme,
                tags=poi.tags,
                importance=poi.importance,
            )
            g.add_edge(
                poi.poi_id,
                poi.space_id,
                dimension="E_cross",
                relation="LOCATED_IN",
                distance=0.0,
                crowd_level="low",
                status="open",
                accessibility_cost=0.0,
            )

        # D_MO visitor node and trajectory relations
        visitors = sorted({event.visitor_id for event in dataset.trajectory})
        for visitor_id in visitors:
            g.add_node(visitor_id, dimension="D_MO", label=f"Visitor {visitor_id}", type="visitor")

        for event in dataset.trajectory:
            if event.poi_id:
                g.add_edge(
                    event.visitor_id,
                    event.poi_id,
                    dimension="D_MO",
                    relation="VISITED",
                    time=event.time,
                    dwell_time_seconds=event.dwell_time_seconds,
                    distance=0.0,
                    crowd_level="low",
                    status="open",
                    accessibility_cost=0.0,
                )
            g.add_edge(
                event.visitor_id,
                event.space_id,
                dimension="D_MO",
                relation="WAS_IN",
                time=event.time,
                distance=0.0,
                crowd_level="low",
                status="open",
                accessibility_cost=0.0,
            )

        return g

    def visitor_events(self, visitor_id: str):
        return [e for e in self.dataset.trajectory if e.visitor_id == visitor_id]

    def current_space(self, visitor_id: str) -> str:
        events = self.visitor_events(visitor_id)
        if not events:
            raise ValueError(f"No trajectory events found for visitor {visitor_id}")
        return events[-1].space_id

    def visited_pois(self, visitor_id: str) -> set[str]:
        return {e.poi_id for e in self.visitor_events(visitor_id) if e.poi_id}

    def infer_interests(self, visitor_id: str) -> Dict[str, float]:
        """Infer visitor interests from dwell time on visited POI tags.

        This is deliberately explainable: a tag becomes important if the visitor
        spent time near POIs containing this tag.
        """
        raw: Dict[str, float] = {}
        for event in self.visitor_events(visitor_id):
            if not event.poi_id:
                continue
            poi = self.dataset.pois.get(event.poi_id)
            if not poi:
                continue
            weight = max(event.dwell_time_seconds, 1.0)
            raw[poi.theme.lower()] = raw.get(poi.theme.lower(), 0.0) + weight
            for tag in poi.tags:
                raw[tag] = raw.get(tag, 0.0) + weight

        total = sum(raw.values())
        if total <= 0:
            return {}
        return {tag: value / total for tag, value in sorted(raw.items(), key=lambda kv: kv[1], reverse=True)}

    def semantic_relevance(self, interests: Dict[str, float], poi: POI) -> float:
        if not interests:
            return 0.0
        tags = {poi.theme.lower(), *poi.tags}
        return min(sum(interests.get(tag, 0.0) for tag in tags), 1.0)

    def entity_crowd(self, entity_id: str) -> float:
        context = self.dataset.context.get(entity_id)
        if not context:
            return 0.0
        return CROWD_VALUE.get(context.crowd_level.lower(), 0.0)

    def is_open_space(self, space_id: str) -> bool:
        context = self.dataset.context.get(space_id)
        if not context:
            return True
        return context.status != "closed"

    def _edge_distance_cost(self, a: str, b: str, attrs: Dict[str, Any]) -> float:
        if attrs.get("status") == "closed":
            return float("inf")
        return float(attrs.get("distance", 1.0))

    def _edge_crowd_aware_cost(self, a: str, b: str, attrs: Dict[str, Any]) -> float:
        if attrs.get("status") == "closed":
            return float("inf")

        if self.config.avoid_closed_spaces:
            if self.graph.nodes[a].get("status") == "closed" or self.graph.nodes[b].get("status") == "closed":
                return float("inf")

        distance = float(attrs.get("distance", 1.0))
        edge_crowd = CROWD_VALUE.get(str(attrs.get("crowd_level", "low")).lower(), 0.0)
        node_crowd = max(
            CROWD_VALUE.get(str(self.graph.nodes[a].get("crowd_level", "low")).lower(), 0.0),
            CROWD_VALUE.get(str(self.graph.nodes[b].get("crowd_level", "low")).lower(), 0.0),
        )
        access = float(attrs.get("accessibility_cost", 0.0))
        return (
            distance
            + self.config.path_crowd_penalty * max(edge_crowd, node_crowd)
            + self.config.path_accessibility_penalty * access
        )

    def _space_subgraph(self) -> nx.Graph:
        """Return a graph containing only indoor spaces and spatial transitions."""
        nodes = [n for n, attrs in self.graph.nodes(data=True) if attrs.get("dimension") == "D_space"]
        return self.graph.subgraph(nodes).copy()

    def compute_path(self, source_space: str, target_space: str, path_type: str) -> Optional[PathResult]:
        space_graph = self._space_subgraph()
        if source_space not in space_graph or target_space not in space_graph:
            return None

        weight_fn = self._edge_distance_cost if path_type == "shortest" else self._edge_crowd_aware_cost

        try:
            path = nx.shortest_path(space_graph, source_space, target_space, weight=weight_fn)
            cost = nx.shortest_path_length(space_graph, source_space, target_space, weight=weight_fn)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

        distance = 0.0
        crowd_cost = 0.0
        for a, b in zip(path[:-1], path[1:]):
            attrs = space_graph[a][b]
            distance += float(attrs.get("distance", 0.0))
            edge_crowd = CROWD_VALUE.get(str(attrs.get("crowd_level", "low")).lower(), 0.0)
            node_crowd = max(self.entity_crowd(a), self.entity_crowd(b))
            crowd_cost += max(edge_crowd, node_crowd)

        return PathResult(
            path=path,
            cost=float(cost),
            distance=distance,
            crowd_cost=crowd_cost,
            path_type=path_type,
        )

    def candidate_pois(self, visitor_id: str) -> Iterable[POI]:
        visited = self.visited_pois(visitor_id)
        current = self.current_space(visitor_id)
        for poi in self.dataset.pois.values():
            if poi.poi_id in visited:
                continue
            if not self.is_open_space(poi.space_id):
                continue
            # Keep only reachable POIs.
            if self.compute_path(current, poi.space_id, "shortest") is None:
                continue
            yield poi

    def recommend(self, visitor_id: str, top_k: int = 5) -> List[POIRecommendation]:
        current = self.current_space(visitor_id)
        interests = self.infer_interests(visitor_id)
        recommendations: List[POIRecommendation] = []

        for poi in self.candidate_pois(visitor_id):
            shortest = self.compute_path(current, poi.space_id, "shortest")
            crowd_aware = self.compute_path(current, poi.space_id, "crowd_aware")

            if not shortest:
                continue

            semantic = self.semantic_relevance(interests, poi)
            importance = poi.importance
            distance_norm = min(shortest.distance / self.config.distance_normalizer, 1.0)
            crowd_norm = min(self.entity_crowd(poi.space_id), 1.0)

            score = (
                self.config.semantic_weight * semantic
                + self.config.importance_weight * importance
                - self.config.distance_weight * distance_norm
                - self.config.crowd_weight * crowd_norm
            )

            poi_space = self.dataset.spaces[poi.space_id]
            explanation = self._build_explanation(
                poi=poi,
                semantic=semantic,
                importance=importance,
                shortest=shortest,
                crowd_aware=crowd_aware,
                interests=interests,
            )

            recommendations.append(
                POIRecommendation(
                    poi_id=poi.poi_id,
                    poi_label=poi.label,
                    poi_theme=poi.theme,
                    poi_space_id=poi.space_id,
                    poi_space_label=poi_space.label,
                    score=score,
                    semantic_relevance=semantic,
                    importance=importance,
                    distance_cost=distance_norm,
                    crowd_cost=crowd_norm,
                    shortest_path=shortest,
                    crowd_aware_path=crowd_aware,
                    explanation=explanation,
                )
            )

        recommendations.sort(key=lambda rec: rec.score, reverse=True)
        return recommendations[:top_k]

    def _build_explanation(
        self,
        poi: POI,
        semantic: float,
        importance: float,
        shortest: Optional[PathResult],
        crowd_aware: Optional[PathResult],
        interests: Dict[str, float],
    ) -> List[str]:
        explanation: List[str] = []
        top_interests = [tag for tag, _ in list(interests.items())[:3]]
        if top_interests:
            explanation.append(
                f"The visitor's dominant inferred interests are {', '.join(top_interests)}."
            )
        if semantic > 0:
            explanation.append(
                f"The POI shares semantic tags with the visitor trajectory: {', '.join(sorted(set([poi.theme, *poi.tags])))}."
            )
        explanation.append(f"The POI has curatorial importance {importance:.2f}.")
        space_context = self.dataset.context.get(poi.space_id)
        if space_context:
            explanation.append(
                f"The destination space is {space_context.status} with {space_context.crowd_level} crowd level."
            )
        if shortest and crowd_aware:
            if shortest.path != crowd_aware.path:
                explanation.append(
                    "The crowd-aware path differs from the shortest path because it penalizes congested passages."
                )
            else:
                explanation.append("The shortest path is also acceptable under the current crowd state.")
        return explanation

    def as_dict(self, rec: POIRecommendation) -> Dict[str, Any]:
        def path_to_dict(path: Optional[PathResult]) -> Optional[Dict[str, Any]]:
            if not path:
                return None
            return {
                "path_type": path.path_type,
                "path": path.path,
                "cost": round(path.cost, 3),
                "distance": round(path.distance, 3),
                "crowd_cost": round(path.crowd_cost, 3),
            }

        return {
            "poi_id": rec.poi_id,
            "poi_label": rec.poi_label,
            "poi_theme": rec.poi_theme,
            "poi_space": {"id": rec.poi_space_id, "label": rec.poi_space_label},
            "score": round(rec.score, 4),
            "score_components": {
                "semantic_relevance": round(rec.semantic_relevance, 4),
                "importance": round(rec.importance, 4),
                "distance_cost": round(rec.distance_cost, 4),
                "crowd_cost": round(rec.crowd_cost, 4),
            },
            "shortest_path": path_to_dict(rec.shortest_path),
            "crowd_aware_path": path_to_dict(rec.crowd_aware_path),
            "explanation": rec.explanation,
        }
