"""reasoning_trace: step-by-step derivation text (pure, no Habitat)."""

from pipeline import viz, consequence
from pipeline.body import Cylinder
from pipeline.actions import Turn, Forward
from tests._synthetic import make_frame


def _trace(acts, radius=0.25):
    fr = make_frame()
    oc = consequence.judge(fr, Cylinder(radius), acts, nav=None)
    return viz.reasoning_trace(oc, fr.objects), oc


def test_trace_forward_collision_mentions_class_and_arc():
    # synthetic wall at z=1 -> a forward collides
    lines, oc = _trace([Forward(5.0)])
    assert oc["collided"] is True
    text = "\n".join(lines)
    assert "COLLISION" in text
    assert "wall" in text                          # contact class surfaced
    assert f"{oc['first_contact_arc_m']:.2f}" in text
    assert "during action" in text                 # names which action collided
    assert lines[0].startswith("Body: cylinder r=0.25")


def test_trace_left_right_wording():
    left, _ = _trace([Turn(-30), Forward(0.2)])
    right, _ = _trace([Turn(30), Forward(0.2)])
    assert any("turn left 30" in s for s in left)
    assert any("turn right 30" in s for s in right)


def test_trace_pure_turn_no_contact():
    lines, oc = _trace([Turn(90)])
    assert oc["collided"] is False
    text = "\n".join(lines)
    assert "no contact" in text
    assert "reaches the end freely" in text


def test_trace_step_count_matches_actions():
    acts = [Turn(15), Forward(0.5), Forward(0.5)]
    lines, _ = _trace(acts)
    # one step line per primitive action (circled markers ①②③)
    assert sum(s[0] in viz._CIRCLED for s in lines) == len(acts)


def test_trace_lists_all_object_distances():
    # synthetic frame has one non-structural object (chair) -> it must be listed
    lines, oc = _trace([Turn(10)])
    assert oc["collided"] is False
    assert any("Distances to objects after move" in s for s in lines)
    assert any(s.startswith("· chair:") and " m " in s for s in lines)


def test_trace_marks_contact_object():
    # a non-structural collision would tag '← contact'; synthetic wall is structural,
    # so no tag, but the object list still renders and COLLISION line names the wall
    lines, oc = _trace([Forward(5.0)])
    assert oc["collided"] is True
    assert any("· chair:" in s for s in lines)          # every object still listed
    assert any('class "wall"' in s for s in lines)
