# tests/test_hmm.py
import pandas as pd
from hmm import build_network, decode

arcs = pd.DataFrame({"from_node": [0, 1, 2, 10], "to_node": [1, 2, 3, 11]})
net = build_network(arcs)

def mk(rows):
    return pd.DataFrame(rows, columns=["obs_idx", "arc_id", "dist_m", "dt_prev_s"])

def test_connected_route_beats_closer_disconnected_arc():
    df = mk([(0, 0, 5, 1), (1, 1, 12, 1), (1, 3, 9, 1), (2, 2, 5, 1)])
    assert decode(df, None, net) == [0, 1, 2]

def test_missing_candidates_returns_minus_one():
    df = mk([(0, 0, 5, 1), (2, 1, 5, 1)])
    out = decode(df, None, net)
    assert out[0] == 0 and out[1] == -1 and out[2] == 1

def test_large_gap_does_not_crash():
    df = mk([(0, 0, 5, 1), (1, 2, 5, 60)])
    assert decode(df, None, net) == [0, 2]