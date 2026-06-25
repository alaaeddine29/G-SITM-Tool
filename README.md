# G-SITM-Tool - First Prototype

This repository contains a first implementable prototype for a **G-SITM-based museum recommendation system**.

The prototype is intentionally simple and explainable. It does not use deep learning yet. Its objective is to validate the first part of the PhD idea: using the G-SITM dynamic graph to recommend both:

1. the next relevant POI;
2. the path to reach it;
3. an explanation based on semantic trajectory, spatial distance, crowd state, and curatorial importance.

## 1. Conceptual basis

The prototype follows the G-SITM definition:

```text
G = (V, E, D, T)
```

where:

```text
D = {D_space, D_POI, D_MO, D_context}
```

The recommender uses these dimensions as follows:

| Dimension | Role |
|---|---|
| `D_space` | rooms, galleries, corridors, stairs, spatial connections |
| `D_POI` | artworks, showcases, exhibitions, collections |
| `D_MO` | visitor position, visited POIs, dwell time, semantic trajectory |
| `D_context` | crowd level, open/closed spaces, contextual state |
| `T` | temporal validity of context and movement observations |

## 2. Recommendation logic

For each candidate POI, the tool computes:

```text
Score(v,p,t) =
  α SemanticRelevance(v,p)
+ β CuratorialImportance(p)
- γ Distance(v,p)
- δ Crowd(p,t)
```

Then it computes two paths:

1. **shortest path**, based only on distance;
2. **crowd-aware path**, based on distance, crowd level, and accessibility cost.

Crowd values are encoded as:

| Crowd level | Value |
|---|---:|
| empty / low | 0 |
| medium | 0.5 |
| high | 1 |
| very_high | 1.5 |
| closed | infinity |

## 3. Project structure

```text
gsitm_rec_tool/
├── data/
│   ├── spaces.csv
│   ├── connections.csv
│   ├── pois.csv
│   ├── visitor_trajectory.csv
│   └── context.csv
├── gsitm_rec/
│   ├── data_loader.py
│   ├── recommender.py
│   └── visualization.py
├── examples/
├── run_demo.py
├── requirements.txt
└── README.md
```

## 4. Run the prototype

From the project folder:

```bash
pip install -r requirements.txt
python run_demo.py --visitor V1 --top-k 5
```

The tool prints recommendations and saves:

```text
examples/recommendations.json
examples/recommended_path.png
```

## 5. Example output

The output has this structure:

```json
{
  "poi_id": "P4",
  "poi_label": "Greek Coin Collection",
  "poi_theme": "archaeology",
  "poi_space": {
    "id": "R4",
    "label": "Coins and Trade Gallery"
  },
  "score": 0.427,
  "shortest_path": {
    "path": ["C1", "R2", "C2", "R4"],
    "distance": 34.0
  },
  "crowd_aware_path": {
    "path": ["C1", "R2", "C2", "R4"],
    "distance": 34.0
  },
  "explanation": [
    "The visitor's dominant inferred interests are archaeology, ceramics, ancient.",
    "The destination space is open with low crowd level."
  ]
}
```

## 6. How to extend it

The next realistic extensions are:

1. replace the synthetic CSV files with a real museum plan;
2. add Neo4j storage and Cypher queries;
3. add a small dashboard interface;
4. add dynamic context snapshots over time;
5. later, add deep learning for next-POI prediction or crowd prediction.

## 7. Scientific contribution of this first version

This prototype validates the first publishable brick:

> A G-SITM-based explainable recommendation model that jointly recommends POIs and adaptive paths by combining visitor semantic trajectories, museum topology, POI semantics, contextual information, and temporal validity.
