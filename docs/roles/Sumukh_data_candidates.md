# P1: Data and Candidates

**You are the critical path.** Everyone else's work depends on your candidate table and labels. Release a small sample early, then iterate.

## Mission
Turn raw `.track/.nodes/.arcs/.route` files into a clean, correct `candidates.parquet` with trustworthy labels, plus `folds.json`, network files and the EDA evidence for the report.

## Inputs / Outputs
| Inputs | Outputs |
|---|---|
| Raw data package (per trip: `.track`, `.nodes`, `.arcs`, `.route` where supplied) | `candidates.parquet`, `folds.json`, `splits.json`, `arcs.parquet`, `nodes.parquet`, adjacency, EDA plots/tables, Folium EDA maps, `docs/decisions.md` entries |

## You own (files)
`src/io.py`, `src/geo.py`, `src/candidates.py`, `src/labels.py`, `src/features.py`, `notebooks/eda.ipynb`, `results/eda/`, `data/processed/`

## Timeline
| When (12h example) | Task | Done when |
|---|---|---|
| 0:00-0:40 | Read all 4 PDFs with team. Load a few raw files. Check file naming and counts | You can state what each file type contains |
| 0:40-1:30 | Write readers in `src/io.py`. Verify lon/lat order, dtypes, row counts, missing values | One trip loads cleanly into DataFrames |
| 1:30-2:30 | Full inventory + EDA (checklist below). Pick UTM zone, write `src/geo.py` | Inventory table saved to `results/eda/` |
| 2:30-3:30 | Spatial index, candidate generation, first label alignment, base features | **10-trip sample released (checkpoint C2)** |
| 3:30-4:20 | Run on all 100 trips, create `folds.json` + `splits.json`, release full table | Everyone can load it |
| 4:20-6:30 | Candidate miss rate at several radii, label-alignment QA, features v2 (with P2's feature-importance feedback) | Miss-rate table done |
| 6:30-9:10 | Folium EDA maps, anomalies, support teammates, report figures | Plots in `results/eda/` |
| 9:10-10:20 | Compare API output with local results on a few trips | Matches |
| 10:20-12:00 | Write report section 2.2 | Draft reviewed by teammate |

## Detailed task list

### A. Loading and inventory
- [ ] Read `.track` (lon, lat, timestamp), `.nodes` (lon, lat, row index = node id), `.arcs` (from_node, to_node, row index = arc_id), `.route` (arc_id sequence).
- [ ] Verify **column order** by plotting a trip. Coordinates swapped means points land in the wrong place.
- [ ] Inventory per partition: trips, files per trip, rows per trip, file sizes. Are all trips identical in structure? Check.
- [ ] Is `.nodes`/`.arcs` per trip or shared? Do not assume. Check.
- [ ] Are `.route` files present for all trips, or only for training/student-test? (Evaluator trips' labels are server-side only.)

### B. EDA checklist (required by the dataset doc; label the partition on every output)
- [ ] Missing / invalid coordinates, NaNs, out-of-range lon/lat
- [ ] Timestamp ordering, duplicates, gap distribution (about 3% of steps are not exactly 1 s)
- [ ] Trips with large gaps or other anomalies
- [ ] Extent (bounding box) per trip
- [ ] Network: node count, arc count, connected components, isolated nodes, degree distribution
- [ ] Route structure: length, are arc IDs valid, is the route connected in the network?
- [ ] Plots: coordinate distributions, trajectory overlays, gap histogram, trip lengths
- [ ] At least one Folium map: GPS (blue dots) + road network + reference route (green)
- [ ] Log every issue and decision in `docs/decisions.md`

### C. Projection
- Determine the location from the coordinates, pick the matching UTM EPSG with `pyproj`, justify it in the report.
- Project nodes and GPS points once. Cache them.

### D. Candidate generation
- Each arc = segment between two node coordinates.
- Build an STRtree (shapely) or KD-tree over arc midpoints plus segment checks.
- For each GPS point, take all arcs within radius `R` (start 50 m, evaluate 30/50/75/100/150 m).
- Compute perpendicular distance and fractional position along the arc.
- Handle points with no candidates (record them, never silently drop).
- **Report the miss rate per radius.** This caps downstream accuracy and is a required metric.
- Tradeoff: larger R improves recall but increases rows and class imbalance. Decide with evidence.

### E. Label alignment (highest risk)
`.route` gives an ordered arc list for the whole trip, not an arc per GPS point. Derive per-point labels:
1. For each GPS point, consider only arcs in the route.
2. Walk monotonically along the route: assign each point to the nearest route arc whose index is >= the previous assignment (allow small backward tolerance).
3. Enforce a maximum distance; mark points too far from any route arc as `label = NaN` and report them.
4. Set `label = 1` for the candidate row matching the assigned arc, else 0.
5. Validate by plotting several trips (green route, blue points coloured by assigned arc).
- Document this as an assumption and a limitation: the reference route is not perfect ground truth.
- **Review the method with the team before releasing the full table.**

### F. Features (v1, see the schema in the common README)
Perpendicular distance, fraction along arc, heading difference, speed, time gap, arc length, distance rank, candidate count. v2 ideas: distance of previous/next point to the same arc, whether the arc connects to the previous best candidate, local trajectory curvature, smoothed speed.
- All feature code lives in `src/features.py`, **one source of truth** that P4's API also imports. No feature logic in notebooks only.
- For every feature, write one line on *why* (needed for report section 2.2).

### G. Folds and splits
- `splits.json`: the partition assigned in the dataset (80 / 10 / 10). Do not invent your own.
- `folds.json`: GroupKFold by trip on the 80 training trips (5 folds), seed 42. Release once.

## Definition of done
- [ ] Candidate table follows the schema exactly
- [ ] Label alignment reviewed and visually checked on at least 5 trips
- [ ] Miss-rate table exists at several radii
- [ ] `folds.json`, `splits.json`, network files released
- [ ] `src/features.py` runs identically on a single trip for the API
- [ ] EDA artefacts saved in `results/eda/` with partition labels

## Pitfalls
- Swapped lon/lat
- Using degrees for distance
- Dropping points with no candidates silently
- Using `.route` information for anything other than labels
- Letting evaluator-test labels be loaded in a notebook
- Memory blowups: process trip by trip, write parquet in chunks

## Report section 2.2: what to answer
1. How was the data preprocessed?
2. What patterns were observed, which features were extracted and why?
Include: missing-value analysis, trajectory plots, network visualisations, a Folium map, and the partition used for each analysis.
