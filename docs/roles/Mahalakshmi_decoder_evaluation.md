# P3: Decoder and Evaluation (HMM + Viterbi)

**You turn independent per-point predictions into a coherent route**, and prove with numbers that it helps (and where it doesn't).

## Mission
Implement HMM/Viterbi route reconstruction on top of P2's probabilities and P1's road network, tune decoder parameters, and produce the classifier-only vs classifier+HMM comparison and the prediction Folium maps.

## Inputs / Outputs
| Inputs | Outputs |
|---|---|
| `candidates.parquet`, `arcs/nodes/adjacency` from P1; `preds_<model>.parquet` from P2 | `src/hmm.py`, `decode()` function, `artifacts/hmm_config.json`, `src/metrics.py` (co-owned), `results/hmm/` (tables, plots), Folium prediction maps |

## You own (files)
`src/hmm.py`, `notebooks/hmm_analysis.ipynb`, `artifacts/hmm_config.json`, `results/hmm/`, `results/maps/` (prediction maps), report sections 5 and 6.

## Timeline
| When (12h example) | Task | Done when |
|---|---|---|
| 0:00-0:40 | Read PDFs (especially Models guideline sections 8-9) | You can state the 6 report questions |
| 0:40-2:30 | Design state space, define sequence-level and continuity metrics with P2 in `src/metrics.py` | Metric functions written and unit-tested on toy data |
| 2:30-4:20 | Prototype Viterbi on P1's 10-trip sample using **distance-only emissions** | Runs end to end, returns a route |
| 4:20-5:30 | Swap in P2's probabilities as emissions (checkpoint C4) | Route + continuity score on student trips |
| 5:30-7:20 | Handle gaps, no-candidate points, disconnected transitions. Add hop-limited transition model | Edge cases covered by tests |
| 7:20-9:10 | Tune decoder parameters on training (out-of-fold probs) and student-test. Classifier-only vs HMM | Comparison table done |
| 9:10-10:20 | Final prediction Folium maps. Failure-case analysis. Hand `decode()` + config to P4 | P4 integrated |
| 10:20-12:00 | Report sections 5 and 6 | Reviewed by teammate |

## Design

### Hidden states
At time `t`, the state is a **candidate arc** for GPS point `t`. The state set varies per step: `S_t = {candidate arcs for obs t}`. Keep the candidate-to-arc mapping so you can return `arc_id` per observation.

### Emission probability
`log e_t(s) = log P(arc s correct | features at t)` taken from the classifier's probability (clip to `[eps, 1-eps]`). Options to test:
- classifier probability only
- classifier probability combined with a Gaussian distance term `exp(-d^2 / 2*sigma^2)`
- temperature scaling of probabilities (`prob ** alpha`) so the decoder does not over-trust emissions

### Transition probability
For states `a` (at t-1) and `b` (at t):
| Relationship | Probability (before normalisation) |
|---|---|
| `a == b` | high (`p_stay`) |
| `to_node(a) == from_node(b)` (directly connected) | moderate (`p_next`) |
| reachable within k hops (k = 2 or 3) | decays with hops: `exp(-hops / beta)` |
| not reachable | tiny floor (`p_floor`), not zero, so Viterbi never dead-ends |
Optionally compare the route distance between the projected points with the straight-line distance (Newson-Krumm style) and penalise big differences.
Remember arcs are **directed**: check whether the network has both directions as separate arcs.

### Viterbi (log-space)
```
delta_0(s) = log pi(s) + log e_0(s)
delta_t(b) = max_a [ delta_{t-1}(a) + log T(a, b) ] + log e_t(b)
psi_t(b)   = argmax_a [ ... ]
backtrack from argmax delta_T
```
Always work in log-space. Use sparse/precomputed adjacency (dict of arc -> set of neighbours, or hop distances via BFS limited to k) so decoding a trip is fast on Laptop 1.

### Edge cases (the report asks about each)
| Case | Handling to implement and justify |
|---|---|
| No candidates at a point | Option A: expand radius for that point. Option B: skip the point and bridge across it. Record which points were affected |
| Large time gap / GPS gap | Relax the transition (flatten it) or reset the chain and restart, then stitch segments |
| Consecutive candidates not connected | Use `p_floor`, or allow k-hop reachability. If every transition is near-floor, break the sequence and restart |
| Reference arc missing from candidates | Can't be fixed by the decoder, so report it as part of the miss rate |
| Start of trip | Uniform prior or emission-only initialisation |

### Decoder parameters to tune
`sigma` (distance), `alpha`/temperature, `p_stay`, `p_next`, `beta`, `p_floor`, max hops, gap threshold, candidate radius expansion. **Tune only on training (out-of-fold probabilities) and student-test trips.** Never on evaluator-test. Use a small grid or Optuna, and log everything to `results/hmm/tuning.csv`.

## Evaluation you must produce
For the same trips and metrics, compare:
| Method | Pointwise Top-1 | Sequence-level | Route continuity |
|---|---|---|---|
| Classifier-only (argmax per point) | | | |
| Classifier + HMM/Viterbi | | | |
Add: per-trip breakdown, improvement distribution, and **the trips/segments where HMM did not help or made things worse**, with a hypothesis (e.g. candidate miss, dense parallel roads, gaps, wrong labels).
Also run the decoder with a **distance-only HMM** (no classifier) as a baseline, which shows how much the classifier contributes.

## Folium maps (report section 6; mandatory)
Per selected trip, layers:
- Raw GPS points (blue dots)
- Road-network geometry near the trip (grey)
- Reference route (green)
- Classifier-only route (orange, optional layer)
- HMM predicted route (red)
Include legend, `LayerControl`, appropriate zoom (`fit_bounds`), tooltips with `obs_idx` and `arc_id`. Save HTML in `results/maps/` and commit the generating script. Take at least one screenshot for the PDF. Use real outputs only.

## Handoff to P4
```python
# src/hmm.py
def decode(cands_df, probs, network, config) -> list[int]:
    """Returns predicted arc_id for each observation (and optionally the stitched route)."""
```
plus `artifacts/hmm_config.json`. Include a 3-line usage example and a note on expected runtime per trip.

## Definition of done
- [ ] `decode()` works on all train, student trips without crashing
- [ ] Edge cases documented and tested (no candidates, gaps, disconnected)
- [ ] Comparison table: classifier-only vs HMM, plus distance-only HMM baseline
- [ ] Failure cases analysed with examples
- [ ] Decoder parameters tuned on permitted data only; logs saved
- [ ] Folium maps meeting the 5 requirements
- [ ] P4 has integrated and run `decode()` through the API

## Pitfalls
- Tuning decoder on in-sample (overconfident) probabilities: **use out-of-fold ones from P2**
- Zero transition probabilities causing -inf paths
- Forgetting arc direction
- Slow decoding from computing path distances naively. Precompute or limit hops
- Evaluating HMM only by pointwise accuracy. Also report route continuity and sequence metrics

## Report sections 5 and 6: what to answer
1. Hidden states
2. Emission calculation
3. Transition calculation
4. Missing candidates and large gaps
5. Disconnected consecutive observations
6. Did HMM improve the final route? (with the comparison table; discuss failures)
Section 6: Folium map requirements and submission items (screenshot in PDF, HTML or code in GitHub).
