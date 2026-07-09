"""in-FOV filter + length-bucketed candidate pool (pure geometry, no Habitat)."""

import numpy as np

from pipeline import action_gen
from pipeline.actions import Turn, Forward


# --- path_stays_in_fov ---------------------------------------------------

def test_forward_stays_in_fov():
    assert action_gen.path_stays_in_fov([Forward(1.0)]) is True


def test_small_turn_then_forward_stays():
    assert action_gen.path_stays_in_fov([Turn(30), Forward(1.5)]) is True   # 30 < 39.5


def test_big_turn_then_forward_exits():
    assert action_gen.path_stays_in_fov([Turn(90), Forward(1.0)]) is False


def test_pure_turn_trivially_in_fov():
    assert action_gen.path_stays_in_fov([Turn(90)]) is True                 # no translation


def test_trailing_turn_after_forward_stays():
    assert action_gen.path_stays_in_fov([Forward(1.0), Turn(90)]) is True   # turn emits no samples


# --- fov_bucketed_pool ---------------------------------------------------

def _keys(seqs):
    return [action_gen._seq_key(s) for s in seqs]


def test_pool_structure_and_invariants():
    rng = np.random.default_rng(0)
    lengths = (1, 2, 3, 4, 5, 6)
    pool = action_gen.fov_bucketed_pool(rng, lengths, pool_per_length=6)

    for L in lengths:
        seqs = pool[L]
        assert len(seqs) >= 1
        assert len(seqs) <= 6
        keys = _keys(seqs)
        assert len(set(keys)) == len(keys)                  # de-duplicated
        for s in seqs:
            assert len(s) == L                              # exact primitive count
            assert action_gen.path_stays_in_fov(s)          # every one is in-FOV
            assert action_gen._valid_combo(s)               # no adj forward / same-sign turn
            # no two adjacent forwards; no two adjacent same-direction turns
            for a, b in zip(s, s[1:]):
                assert not (isinstance(a, Forward) and isinstance(b, Forward))
                if isinstance(a, Turn) and isinstance(b, Turn):
                    assert (a.deg > 0) != (b.deg > 0)
            n_fwd = sum(isinstance(a, Forward) for a in s)
            n_turn = sum(isinstance(a, Turn) for a in s)
            if L == 1:
                assert n_fwd + n_turn == 1                  # a single primitive
            else:
                assert n_fwd >= 1 and n_turn >= 1           # mixed, non-degenerate


def test_pool_length1_covers_turns_and_forwards():
    rng = np.random.default_rng(1)
    pool = action_gen.fov_bucketed_pool(rng, (1,), pool_per_length=99)
    # length-1 distinct set = len(GEN_TURNS_DEG) + len(GEN_FORWARDS_M)
    from pipeline import config
    assert len(pool[1]) == len(config.GEN_TURNS_DEG) + len(config.GEN_FORWARDS_M)
