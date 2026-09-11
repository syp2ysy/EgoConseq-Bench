"""A3 uses semantic LLM judgments; other tasks use extracted answers and fixed rules."""

import importlib.util
import json
from pathlib import Path
import sys

import pytest


def evaluator():
    path = Path(__file__).parents[1] / "data/benchmark/evaluation/eval.py"
    spec = importlib.util.spec_from_file_location("answer_evaluation", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def item(task="B1", gt="6.65 m", question="<image> At 50% of forward 2 m, what is the distance?"):
    return {"task_id": task, "dataset": "r2r", "images": ["missing.png"], "messages": [
        {"role": "system", "content": "You are the robot."},
        {"role": "user", "content": question}, {"role": "assistant", "content": gt}]}


@pytest.mark.parametrize("task", ["A1", "A2", "A4", "B1", "B2", "B3", "C1"])
def test_extractor_receives_question_and_verbatim_response_but_no_gt(task):
    m = evaluator()
    response = "Travel 2 m; my final answer is 3.2 m. Literal <image>."
    request = m.extraction_item(item(task, "SECRET_REFERENCE"), response)
    assert request["images"] == []
    assert "SECRET_REFERENCE" not in json.dumps(request)
    assert "You are the robot." not in json.dumps(request)
    payload = json.loads(request["messages"][1]["content"])
    assert payload["response"] == response
    assert "50%" in payload["question"]
    assert "<image>" not in payload["question"]


@pytest.mark.parametrize("response,c25,c50", [
    ("6.90 m", True, True), ("6.91 m", False, True),
    ("7.15 m", False, True), ("7.16 m", False, False),
    ("6.15 m", False, True), ("6.14 m", False, False),
    ("640 cm", True, True), ("6.4e0 m", True, True),
    ("The distance is about 6.90 meters.", True, True),
    ("Travel: 1 m.\nFinal answer: 6.90 m", True, True),
    ("-6.65", False, False), ("−6.65", False, False),
    ("NaN", False, False), ("Infinity", False, False), ("", False, False),
])
def test_regex_distance_and_both_inclusive_tolerances(response, c25, c50):
    m = evaluator(); i = item()
    value = m.extract_with_rules(i, response)
    assert value is not m.UNPARSED
    result = m.score_extracted(i, value)
    assert result["correct"] is c25
    assert result["correct_at_0.5m"] is c50


@pytest.mark.parametrize("response", [
    "Travel is 2 m, but the target distance is 6.9 m.",
    "Between 6 and 7 meters.", "Camera height is 6.65 m.",
    "Distance is not 6.65 m.", "6.65 degrees",
])
def test_ambiguous_numeric_text_goes_to_extraction_instead_of_picking_first_number(response):
    m = evaluator()
    assert m.extract_with_rules(item(), response) is m.UNPARSED


def test_long_distance_answer_is_extracted_then_scored_by_both_thresholds():
    m = evaluator(); i = item()
    value = m.read_extraction(i, {"finish_reason": "stop", "answer": '{"answer":"6.91 m","correct":true}'}, "My final answer is 6.91 m.")
    result = m.score_extracted(i, value)
    assert result["correct"] is False
    assert result["correct_at_0.5m"] is True


@pytest.mark.parametrize("task,gt,response,want", [
    ("A1", "Collision.", "No collision", False),
    ("A2", "Action 4.", "Step 4", True),
    ("A2", "Action 2.", "Action (1)", False),
    ("A2", "Action 3.", "Action 3.5", False),
    ("C1", "C", "Option C.", True),
    ("C1", "C", "A", False),
])
def test_explicit_answers_use_the_same_rule_scorer(task, gt, response, want):
    m = evaluator(); i = item(task, gt)
    value = m.extract_with_rules(i, response)
    assert value is not m.UNPARSED
    assert m.score_extracted(i, value)["correct"] is want


def test_forward_ordinal_is_mapped_in_code_and_explicit_index_is_not_repaired():
    m = evaluator()
    i = item("A2", "Action 4.", "(1) turn left 30 degree\n(2) forward 1 m\n(3) turn right 30 degree\n(4) forward 2 m")
    assert m.extract_with_rules(i, "The second Forward action.") == 4
    value = m.read_extraction(i, {"finish_reason": "stop", "answer": '{"answer":"Action 1"}'}, "Action 1")
    assert value == 1
    assert m.score_extracted(i, value)["correct"] is False


@pytest.mark.parametrize("response,want", [
    ("Action <3>.", 3), ("Action <1>", 1), ("Step <5>", 5),
    ("Action <3.5>", None), ("Action <number>.", None),
    ("Action <ref>", None), ("Action <x>", None), ("Action <index>", None),
])
def test_a2_distinguishes_bracketed_numbers_from_unfilled_placeholders(response, want):
    m = evaluator()
    assert m.extract_with_rules(item("A2", "Action 3."), response) == want


@pytest.mark.parametrize("response", ["Action <3> or <4>", "Not Action <3>", "Action <3).", "Action <3)>"])
def test_a2_brackets_do_not_hide_alternatives_negation_or_broken_delimiters(response):
    m = evaluator()
    assert m.extract_with_rules(item("A2", "Action 3."), response) is m.UNPARSED


@pytest.mark.parametrize("gt,response,axes", [
    ("Front.", "Back", [False, True, True]),
    ("Left and above.", "Left and above", [True, True, True]),
    ("Front and below.", "Below", [False, False, True]),
    ("Rear-right and above.", "Behind and to the right and up", [True, True, True]),
    ("Front and below.", "Center and below", [False, False, True]),
])
def test_standard_directions_get_deterministic_axis_credit(gt, response, axes):
    m = evaluator(); i = item("B2", gt)
    value = m.extract_with_rules(i, response)
    result = m.score_extracted(i, value)
    assert list(result["axis_correct"].values()) == axes
    assert result["correct"] is all(axes)


@pytest.mark.parametrize("response,horizontal,vertical", [
    ("The marked point is in front of the camera.", "front", "level"),
    ("The dot is behind and to the right of the robot.", "rear-right", "level"),
    ("The marked point is below and to the left of the camera.", "left", "below"),
    ("The direction from the camera to the marked point is forward and to the left.", "front-left", "level"),
    ("The marked point is in front of the robot at this moment.", "front", "level"),
    ("In front of me", "front", "level"), ("On the right.", "right", "level"),
    ("From behind.", "rear", "level"), ("Far ahead and left", "front-left", "level"),
    ("Top-left", "left", "above"),
    ("The marked point is in front of the robot after the sequence of actions.", "front", "level"),
    ("The marked point is to the right and above the robot at the end of the sequence.", "right", "above"),
    ("The marked point is to the right and above the camera.", "right", "above"),
])
def test_unambiguous_direction_sentences_do_not_need_llm(response, horizontal, vertical):
    m = evaluator()
    assert m.extract_with_rules(item("B2"), response) == {"horizontal": horizontal, "vertical": vertical}


@pytest.mark.parametrize("response", [
    "The point is not to the left.", "The point is above the ground.",
    "The camera is above the point.", "The point is left of the table and above the camera.",
    "Initially left, finally right.",
])
def test_direction_sentence_regex_leaves_negation_reference_and_time_to_llm(response):
    m = evaluator()
    assert m.extract_with_rules(item("B2"), response) is m.UNPARSED


@pytest.mark.parametrize("response,horizontal,vertical", [
    ("Bottom left and above", "left", None),
    ("Bottom right and above", "right", None),
    ("Ahead and left of target.", None, None),
    ("Below and to the right of the marked point.", None, None),
    ("The marked point is in front of the car.", None, None),
    ("The marked point is to the right of the initial position.", None, None),
    ("The marked point is to the left of the robot at the beginning.", None, None),
    ("The marked point is 1.35 meters to the right of the center of the ground.", None, None),
    ("To the right and below the marked point.", None, None),
    ("forward 0.5 m", None, None),
    ("In the left background", "left", "level"),
    ("In the left background, above the floor.", "left", None),
    ("Background and right", "right", "level"),
])
def test_short_directions_preserve_reference_and_conflicting_components(response, horizontal, vertical):
    m = evaluator()
    assert m.extract_with_rules(item("B2"), response) == {"horizontal": horizontal, "vertical": vertical}


@pytest.mark.parametrize("response,horizontal,vertical", [
    ("7.5 degrees left of forward", "front", "level"),
    ("7.5001 degrees left of forward", "front-left", "level"),
    ("22.65 degrees right of front", "front-right", "level"),
    ("82.5 degrees right of front", "right", "level"),
    ("97.5 degrees right of front and below", "right", "below"),
    ("180 degrees left of front", "rear", "level"),
    ("194.66 degrees.", None, None), ("8.32 degrees and below", None, "below"),
    ("-263.24° and below", None, "below"),
    ("After the sequence of movements, the direction from the camera to the marked point is approximately 180.47 degrees, and the answer is in air.", None, None),
])
def test_angles_use_explicit_reference_and_inclusive_cardinal_bins(response, horizontal, vertical):
    m = evaluator()
    assert m.extract_with_rules(item("B2"), response) == {"horizontal": horizontal, "vertical": vertical}


@pytest.mark.parametrize("response", [
    "Initially 7.5 degrees left of forward.",
    "Not 7.5 degrees left of forward.",
    "Left or 7.5 degrees right of forward.",
    "At the start, the point is 7.5 degrees left of forward.",
    "At action 1, the point is 10 degrees left of forward.",
    "Centered on the table, 7.5 degrees left of forward",
    "The 3d point is roughly at 22.65 degrees right of front.",
    "The marked point is 8.33 meters in the distance, in front, and 1.41 degrees to the right.",
])
def test_angle_regex_does_not_choose_an_initial_negated_or_alternative_answer(response):
    m = evaluator()
    assert m.extract_with_rules(item("B2"), response) is m.UNPARSED


def test_llm_extracts_direction_values_not_axis_grades_and_keeps_unknowns():
    m = evaluator(); i = item("A4", "Front and below.")
    value = m.read_extraction(i, {"finish_reason": "stop", "answer": '{"answer":{"horizontal":null,"vertical":"below"}}'}, "Below")
    result = m.score_extracted(i, value)
    assert list(result["axis_correct"].values()) == [False, False, True]
    assert result["score"] == pytest.approx(1 / 3)


@pytest.mark.parametrize("task,answer", [
    ("B1", '{"correct":true}'), ("B1", '{"answer":true}'),
    ("B1", '{"answer":"2 or 3 m"}'), ("B1", 'not JSON'),
    ("A4", '{"answer":{"horizontal":"unrecognized bearing","vertical":"below"}}'),
    ("A4", '{"answer":{"horizontal":"left"}}'),
    ("A4", '{"answer":{"horizontal":true,"vertical":false}}'),
])
def test_bad_extraction_format_is_not_a_student_error(task, answer):
    with pytest.raises((ValueError, KeyError, TypeError)):
        evaluator().read_extraction(item(task), {"answer": answer, "finish_reason": "stop"}, "Unparsed response")


@pytest.mark.parametrize("horizontal,vertical,expected", [
    ("forward", "and below", {"horizontal": "front", "vertical": "below"}),
    ("center", "above", {"horizontal": None, "vertical": "above"}),
    ("null", "below", {"horizontal": None, "vertical": "below"}),
    ("between", "unknown", {"horizontal": None, "vertical": None}),
])
def test_extractor_label_variants_normalize_without_inventing_information(horizontal, vertical, expected):
    output = {"answer": json.dumps({"answer": {"horizontal": horizontal, "vertical": vertical}}), "finish_reason": "stop"}
    assert evaluator().read_extraction(item("A4"), output, "Direction or height unknown") == expected


def test_truncated_extraction_is_an_error_but_explicit_no_answer_is_scored_zero():
    m = evaluator(); i = item()
    with pytest.raises(ValueError):
        m.read_extraction(i, {"answer": '{"answer":"6.65 m"}', "finish_reason": "length"}, "6.65 m")
    value = m.read_extraction(i, {"answer": '{"answer":null}', "finish_reason": "stop"}, "I cannot tell.")
    result = m.score_extracted(i, value)
    assert result["correct"] is False and result["correct_at_0.5m"] is False


def test_retry_does_not_expose_gt_or_edit_response():
    m = evaluator()
    request = m.extraction_item(item(gt="SECRET_REFERENCE"), "Long unchanged answer", "Invalid JSON")
    assert "SECRET_REFERENCE" not in json.dumps(request)
    assert json.loads(request["messages"][1]["content"])["response"] == "Long unchanged answer"


@pytest.mark.parametrize("reasoning", [
    '<think>Intermediate candidate: {"answer":"4 m"}</think>',
    'Intermediate candidate: {"answer":"4 m"}</think>',
])
def test_extractor_reasoning_is_not_used_as_the_final_answer(reasoning):
    m = evaluator()
    value = m.read_extraction(item(), {"answer": reasoning + '\n{"answer":"6.91 m"}',
                                      "finish_reason": "stop"}, "The final distance is 6.91 m.")
    assert value == "6.91 m"
    assert m.score_extracted(item(), value)["correct"] is False


def test_extraction_context_keeps_query_moment_but_excludes_geometry_and_category_choices():
    m = evaluator()
    question = ("Configuration:\nCamera height: SECRET_HEIGHT\nActions:\n(1) turn left 45 degree\n"
                "Query moment: At 50% of action 2.\n\nQuestion: Where is the target?\nAnswer briefly.")
    request = m.extraction_item(item("A4", "SECRET_GT", question), "Front and below")
    payload = json.loads(request["messages"][1]["content"])
    assert "50% of action 2" in payload["question"]
    assert "Where is the target?" in payload["question"]
    assert "SECRET_HEIGHT" not in json.dumps(request)
    assert "turn left" not in json.dumps(request)
    assert "category_aliases" not in payload


@pytest.mark.parametrize("response", ["wall", "couch", "wooden wall", "", "It hits the couch first."])
def test_a3_always_goes_to_semantic_judging(response):
    m = evaluator()
    assert m.extract_with_rules(item("A3", "wall"), response) is m.UNPARSED


def test_a3_judge_receives_gt_and_verbatim_response_without_scene_geometry():
    question = "Camera height: SECRET_HEIGHT\nActions: SECRET_ACTION\nQuestion: What do you hit first?"
    request = evaluator().extraction_item(item("A3", "sofa", question), "It hits the couch first.")
    payload = json.loads(request["messages"][1]["content"])
    assert payload["ground_truth"] == "sofa"
    assert payload["response"] == "It hits the couch first."
    assert request["images"] == []
    assert "SECRET_HEIGHT" not in json.dumps(request) and "SECRET_ACTION" not in json.dumps(request)


@pytest.mark.parametrize("correct", [True, False])
def test_a3_uses_boolean_judgment_without_category_string_matching(correct):
    m = evaluator(); i = item("A3", "sofa")
    value = m.read_extraction(i, {"answer": '<think>{"correct":false}</think>' + json.dumps({"correct": correct}),
                                  "finish_reason": "stop"}, "It hits the couch first.")
    assert value is correct
    assert m.score_extracted(i, value) == {"correct": correct, "score": float(correct)}


@pytest.mark.parametrize("document", [{"correct": "true"}, {"correct": 1}, {"correct": None}, {"answer": "sofa"}, {}])
def test_a3_malformed_judgments_are_errors_instead_of_incorrect_answers(document):
    with pytest.raises(ValueError, match="correct"):
        evaluator().read_extraction(item("A3", "sofa"),
            {"answer": json.dumps(document), "finish_reason": "stop"}, "couch")


@pytest.mark.parametrize("value", [None, "sofa", "true", 1])
def test_a3_scorer_rejects_a_non_boolean_decision(value):
    with pytest.raises(ValueError, match="boolean"):
        evaluator().score_extracted(item("A3", "sofa"), value)


def test_a3_reports_overall_accuracy_not_a_mean_of_category_accuracies(tmp_path):
    m = evaluator()
    rows = [dict(split="seen", dataset="r2r", task_id="A3", status="scored", ground_truth=gt,
                 **m.score_extracted(item("A3", gt), decision))
            for gt, decision in [("wall", True)] * 4 + [("chair", False)]]
    result = {"config": {}, "results": rows}
    m.write_result(tmp_path / "scores.json", result)
    assert result["summary"]["all"]["tasks"]["A3"] == {
        "total": 5, "scored": 5, "pending": 0, "correct": 4, "accuracy": 0.8,
        "truncated_responses": 0}
    assert "by_category" not in result["summary"]


@pytest.mark.parametrize("retry", [False, True])
def test_a3_cli_routes_only_a3_to_judge_and_resumes_rules_only(tmp_path, monkeypatch, retry):
    m = evaluator()
    answers = {"A1": "No collision", "A2": "Action 2", "A3": "sofa", "A4": "Front and below",
               "B1": "6.65 m", "B2": "Left and above", "B3": "5.00 m", "C1": "D"}
    questions = []
    predictions = []
    sources = {"seen": {"qa_path": "test.json", "sha256": "test"}}
    for task, gt in answers.items():
        i = item(task, gt, "(1) turn left 30 degree\n(2) forward 1 m\nQuestion: What is the answer?")
        i["id"] = task
        questions.append(("seen", tmp_path, i))
        predictions.append(dict(split="seen", id=task, dataset="r2r", task_id=task,
                                answer="couch" if task == "A3" else gt, finish_reason="stop"))
    source = tmp_path / "responses.json"
    source.write_text(json.dumps({"config": {"variant": "full", "sources": sources}, "predictions": predictions}))
    out = tmp_path / "eval.json"
    monkeypatch.setattr(m, "load_questions", lambda *args, **kwargs: (questions, sources))
    calls = []
    thinking_modes = []
    runtime = []
    def predict(batch):
        calls.extend(batch)
        thinking_modes.append(runtime[0].chat_template_kwargs["enable_thinking"])
        payload = json.loads(batch[0][2]["messages"][1]["content"])
        assert len(batch) == 1 and payload["task"] == "A3"
        assert payload["ground_truth"] == "sofa" and payload["response"] == "couch"
        value = "true" if retry and len(calls) == 1 else True
        return [{"answer": json.dumps({"correct": value, "evidence": "couch"}), "finish_reason": "stop"}]
    def factory(args):
        runtime.append(args)
        return predict, {}
    monkeypatch.setattr(m, "api_predictor", factory)
    argv = ["eval.py", "--responses", str(source), "--output", str(out), "--base-url", "http://unused.test"]
    monkeypatch.setattr(sys, "argv", argv + ["--rules-only"])
    assert m.main() == 3 and not calls
    before = json.loads(out.read_text())
    assert before["summary"]["all"]["tasks"]["A3"]["accuracy"] is None
    assert before["summary"]["all"]["macro_joint_accuracy"] is None
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    after = json.loads(out.read_text())
    assert len(calls) == (2 if retry else 1)
    assert thinking_modes == ([True, True] if retry else [True])
    for old, new in zip(before["results"], after["results"]):
        if new["task_id"] != "A3":
            assert old == new
        else:
            assert new["correct"] and new["scoring_method"] == "llm"
            assert json.loads(new["judge_output"]) == {"correct": True, "evidence": "couch"}
            assert "extraction_error" not in new and "extracted_answer" not in new
    assert after["summary"]["all"]["tasks"]["A3"]["accuracy"] == 1.0


@pytest.mark.parametrize("mismatch", ["protocol_sha256", "extractor_model", "sources", "extraction_prompt"])
def test_resume_rejects_changed_protocol_model_sources_or_prompts(tmp_path, monkeypatch, mismatch):
    m = evaluator()
    sources = {"seen": {"qa_path": "QA.json", "sha256": "source"}}
    questions, predictions = [], []
    for task, gt, response in [("A2", "Action 2", "Action <number>"), ("A3", "sofa", "couch")]:
        q = item(task, gt, "(1) forward 1 m\n(2) forward 2 m\nQuestion: Answer.")
        q["id"] = task
        questions.append(("seen", tmp_path, q))
        predictions.append(dict(split="seen", id=task, dataset="r2r", task_id=task,
                                answer=response, finish_reason="stop"))
    source, out = tmp_path / "responses.json", tmp_path / "eval.json"
    source.write_text(json.dumps({"config": {"variant": "full", "sources": sources}, "predictions": predictions}))
    monkeypatch.setattr(m, "load_questions", lambda *a, **kw: (questions, sources))
    monkeypatch.setattr(m, "api_predictor", lambda args: (
        lambda batch: [{"answer": '{"correct":true,"evidence":"couch"}', "finish_reason": "stop"} for _ in batch], {}))
    argv = ["eval.py", "--responses", str(source), "--output", str(out), "--base-url", "http://unused.test"]
    monkeypatch.setattr(sys, "argv", argv)
    assert m.main() == 0
    saved = json.loads(out.read_text())
    if mismatch == "extraction_prompt":
        saved["extraction_prompt"]["system"] = "different instructions"
    else:
        saved["config"][mismatch] = "different"
    out.write_text(json.dumps(saved))
    monkeypatch.setattr(m, "api_predictor", lambda args: pytest.fail("Mismatched runs must stop before model calls"))
    with pytest.raises(SystemExit):
        m.main()


@pytest.mark.parametrize("response,vertical", [
    ("The point is right and further away.", "level"),
    ("The point is right, but its height relative to the camera is unknown.", None),
    ("The point is right and above the ground.", None),
])
def test_omitted_height_uses_the_same_convention_for_regex_and_llm(response, vertical):
    value = evaluator().read_extraction(item("B2"),
        {"answer": '{"answer":{"horizontal":"right","vertical":null}}', "finish_reason": "stop"}, response)
    assert value == {"horizontal": "right", "vertical": vertical}


@pytest.mark.parametrize("response,vertical", [
    ("Either left or right, I cannot tell.", None),
    ("Horizontal direction unknown; at the same height as the camera.", "level"),
])
def test_unknown_horizontal_cannot_gain_implicit_height_credit(response, vertical):
    value = evaluator().read_extraction(item("B2"),
        {"answer": '{"answer":{"horizontal":null,"vertical":"level"}}', "finish_reason": "stop"}, response)
    assert value == {"horizontal": None, "vertical": vertical}


def test_scores_include_both_distance_thresholds_and_keep_pending_out(tmp_path):
    m = evaluator(); i = item()
    rows = [dict(split="seen", dataset="r2r", task_id="B1", status="scored", **m.score_extracted(i, value))
            for value in ("6.90 m", "7.15 m", None)]
    path = tmp_path / "scores.json"
    result = {"config": {}, "results": rows}
    m.write_result(path, result)
    metrics = json.loads(path.read_text())["summary"]["all"]["tasks"]["B1"]
    assert metrics["accuracy_at_0.25m"] == pytest.approx(1 / 3)
    assert metrics["accuracy_at_0.5m"] == pytest.approx(2 / 3)
    rows.append(dict(split="seen", dataset="r2r", task_id="B1", status="extraction_error"))
    m.write_result(path, result)
    metrics = json.loads(path.read_text())["summary"]["all"]["tasks"]["B1"]
    assert metrics["accuracy_at_0.25m"] is None
    assert metrics["accuracy_at_0.5m"] is None


def test_outputs_keep_variants_and_extractors_separate():
    m = evaluator()
    a = m.default_output(Path("student_response.json"), "Qwen/model", "full")
    b = m.default_output(Path("student_no_height_response.json"), "Qwen/model", "no_height")
    c = m.default_output(Path("student_response.json"), "Qwen/other", "full")
    assert a.parent == b.parent == c.parent
    assert len({a, b, c}) == 3


@pytest.mark.parametrize("task,response", [
    ("B2", "After the first action, the point is left."),
    ("B2", "After moving 1000 m, the point is front."),
    ("B2", "After turning right, the marked point is left and below."),
    ("A4", "After completing all actions, the point is front."),
    ("A4", "After action 1, the bearing is 10 degrees left of forward."),
])
def test_direction_parser_preserves_unverified_time_conditions(task, response):
    m = evaluator()
    assert m.extract_with_rules(item(task), response) is m.UNPARSED


def test_extractor_keeps_measurement_definitions_and_sequence_length():
    m = evaluator()
    from post_QA.templates import DIRECTION_CONVENTION
    measure = "Measure the 3D straight-line distance from your camera's optical center to the marked point."
    for task, definition in [("B2", DIRECTION_CONVENTION), ("B3", measure)]:
        question = ("Configuration:\n- Radius: SECRET\nActions:\n(1) forward 1 m\n"
                    "(2) turn left 30 degree\n\n" + definition +
                    "\n\nQuestion: Where is the point?")
        request = m.extraction_item(item(task, "SECRET_GT", question), "A long answer")
        payload = json.loads(request["messages"][1]["content"])
        assert definition in payload["question"]
        assert payload["action_count"] == 2
        assert "SECRET" not in json.dumps(request)
        assert "turn left" not in payload["question"]


@pytest.mark.parametrize("bearing,want", [("7.5 degrees left of forward", "front"),
                                        ("10 degrees left of forward", "front-left")])
def test_llm_extracted_bearing_is_binned_in_python(bearing, want):
    m = evaluator()
    output = {"finish_reason": "stop", "answer": json.dumps(
        {"answer": {"horizontal": bearing, "vertical": "level"}})}
    value = m.read_extraction(item("B2"), output, "The target has that bearing.")
    assert value == {"horizontal": want, "vertical": "level"}


def test_student_truncation_reaches_extractor_without_changing_answer():
    m = evaluator()
    request = m.extraction_item(item(), "Intermediate estimate: 3 m", response_finish_reason="length")
    payload = json.loads(request["messages"][1]["content"])
    assert payload["response_finish_reason"] == "length"
    assert payload["response"] == "Intermediate estimate: 3 m"
    assert "truncated" in request["messages"][0]["content"]


@pytest.mark.parametrize("response,horizontal,vertical", [
    ("left 45 degree", "front-left", "level"),
    ("left 60 degrees and above", "front-left", "above"),
    ("right 90 degrees and below", "right", "below"),
    ("Horizontal: 102.5° and above", None, "above"),
    ("Turn left 45 degrees", None, None),
    ("turn left 45 degrees and above", None, "above"),
    ("Forward 5.5 m, right 22.5 degree", None, None),
    ("Forward 3.25 m, turn left 22.5 degree, Forward 3.25 m, turn right 12.5 degree.", None, None),
])
@pytest.mark.parametrize("task", ["A4", "B2"])
def test_numeric_bearings_and_motion_only_answers_have_one_interpretation(task, response, horizontal, vertical):
    m = evaluator()
    assert m.extract_with_rules(item(task), response) == {"horizontal": horizontal, "vertical": vertical}


def test_llm_evidence_cannot_introduce_answer_words_absent_from_response():
    m = evaluator()
    output = {"answer": '{"answer":{"horizontal":"left","vertical":"above"},"evidence":"left and above"}',
              "finish_reason": "stop"}
    with pytest.raises(ValueError, match="evidence"):
        m.read_extraction(item("B2"), output, "left 45 degree", require_evidence=True)


def test_direction_normalization_uses_quoted_answer_not_invented_height():
    m = evaluator()
    output = {"answer": '{"answer":{"horizontal":"left","vertical":"above"},"evidence":"left 45 degree"}',
              "finish_reason": "stop"}
    result = m.read_extraction(item("B2"), output, "Final answer: left 45 degree", require_evidence=True)
    assert result == {"horizontal": "front-left", "vertical": "level"}


def test_distance_normalization_does_not_repair_the_quoted_number():
    m = evaluator()
    output = {"answer": '{"answer":"6.65 m","evidence":"6.91 m"}', "finish_reason": "stop"}
    result = m.read_extraction(item(), output, "Final answer: 6.91 m", require_evidence=True)
    assert result == "6.91 m"


def test_positive_a3_judgment_requires_grounded_evidence():
    m = evaluator()
    with pytest.raises(ValueError, match="evidence"):
        m.read_extraction(item("A3", "sofa"), {"answer": '{"correct":true,"evidence":null}',
                         "finish_reason": "stop"}, "couch", require_evidence=True)


def test_judge_transport_preserves_literal_image_tokens_and_uses_fixed_sampling(monkeypatch):
    from io import BytesIO
    from types import SimpleNamespace
    m = evaluator()
    requests = []
    def urlopen(request, timeout):
        requests.append(request)
        document = ({"data": [{"id": "judge"}]} if request.full_url.endswith("/models") else
                    {"choices": [{"message": {"content": '{"answer":null,"evidence":null}'},
                                  "finish_reason": "stop"}]})
        return BytesIO(json.dumps(document).encode())
    monkeypatch.setattr(m, "urlopen", urlopen)
    args = SimpleNamespace(model="judge", base_url="http://unused.test/v1", api_workers=1,
                           max_new_tokens=4096, chat_template_kwargs={"enable_thinking": True})
    predict, _ = m.api_predictor(args)
    response = "Literal <image>; I cannot tell."
    predict([("seen", m.BENCHMARK, m.extraction_item(item(), response))])
    body = json.loads(requests[-1].data)
    assert json.loads(body["messages"][1]["content"])["response"] == response
    assert all(body[k] == v for k, v in m.GENERATION.items())
    assert body["chat_template_kwargs"] == {"enable_thinking": True}
    assert body["max_tokens"] == 4096


@pytest.mark.parametrize("value", [None, "null", "unknown"])
def test_explicit_no_answer_is_not_replaced_by_a_quoted_intermediate_estimate(value):
    m = evaluator()
    output = {"finish_reason": "stop", "answer": json.dumps({"answer": value, "evidence": "3.0 m"})}
    assert m.read_extraction(item(), output, "Intermediate estimate: 3.0 m", require_evidence=True) is None
