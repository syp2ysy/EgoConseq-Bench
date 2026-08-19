"""Collection-time publication of private terminal image assets."""

from __future__ import annotations

from pipeline import future_view_selection, outcome, record


def _completed_outcome(accepted_outcomes, *, frame_id: str, tag: str):
    by_frame = (accepted_outcomes.get(str(tag)) or {}).get(frame_id) or []
    return next((value for value in by_frame
                 if outcome.is_completed_clear(value)), None)


def counterfactual_terminal_outcome_ids(
        accepted_outcomes, *, frame_id: str, families) -> set[str]:
    """Return exactly the clear outcomes authorized for one C1 transaction.

    A family without its clear query cannot form a C1 item, so none of its
    otherwise-clear neighbours spend terminal-rendering work. Incidental clear
    outcomes outside the frozen family list are never authorized.
    """
    outcome_ids: set[str] = set()
    for family in families or ():
        query = _completed_outcome(
            accepted_outcomes, frame_id=frame_id, tag=family.query_tag)
        if query is None:
            continue
        members = [query]
        members.extend(filter(None, (
            _completed_outcome(
                accepted_outcomes, frame_id=frame_id, tag=tag)
            for tag in family.neighbor_tags
        )))
        for value in members:
            outcome_id = str(value.get("outcome_id") or "")
            if not outcome_id:
                raise RuntimeError("counterfactual terminal member lacks an ID")
            outcome_ids.add(outcome_id)
    return outcome_ids


def preload_counterfactual_terminal_rgb(
        accepted_outcomes, *, frame_id: str, families,
        render_cache, terminal_batch_renderer,
        authorized_outcome_ids: set[str]) -> int:
    """Render all surviving C1 families in one backend transaction.

    B1K RGB is a real renderer output but is not pose-pure under sequential
    temporal history.  The query and its surviving neighbours therefore enter
    one temporary multi-camera transaction before ordinary terminal-asset
    publication.  A rejected neighbour is simply absent; a rejected query
    means there is no C1 item to render for this frame.
    """
    if not families or terminal_batch_renderer is None:
        return 0

    candidates = []
    for family in families:
        query = _completed_outcome(
            accepted_outcomes, frame_id=frame_id, tag=family.query_tag)
        if query is None:
            continue
        candidates.append(query)
        for tag in family.neighbor_tags:
            value = _completed_outcome(
                accepted_outcomes, frame_id=frame_id, tag=tag)
            if value is not None:
                candidates.append(value)
    poses = []
    seen = set()
    rendered_candidates = []
    for value in candidates:
        try:
            pose = future_view_selection._terminal_checkpoint(value)
        except future_view_selection.TerminalRGBAssetError:
            continue
        rendered_candidates.append(value)
        values = tuple(float(pose[key]) for key in (
            "x", "z", "heading_deg"))
        key = tuple(round(item, 6) for item in values)
        if key in seen:
            continue
        seen.add(key)
        poses.append(values)
    if not poses:
        return 0
    # A geometry-stage cache entry may have been rendered sequentially.  It is
    # not B1K C1 authority: force every selected pose through the simultaneous
    # temporary-camera transaction before authorizing any outcome.
    for key in seen:
        render_cache.pop(key, None)
    terminal_batch_renderer(poses)
    missing = [
        tuple(round(float(value), 6) for value in pose)
        for pose in poses
        if tuple(round(float(value), 6) for value in pose) not in render_cache
    ]
    if missing:
        raise RuntimeError(
            "counterfactual terminal batch omitted rendered poses")
    outcome_ids = {
        str(value.get("outcome_id") or "") for value in rendered_candidates
    }
    if "" in outcome_ids:
        raise RuntimeError("counterfactual terminal batch member lacks an ID")
    authorized_outcome_ids.update(outcome_ids)
    return len(outcome_ids)


def attach_terminal_rgb_assets(
        root, frame, outcomes, render_cache, *, source,
        collection_contract, terminal_renderer=None,
        eligible_outcome_ids=None, render_transaction=None) -> dict:
    """Attach native C1 endpoint PNGs from the evaluation cache."""
    is_b1k = (source or {}).get("source_dataset") == "b1k"
    if is_b1k and (eligible_outcome_ids is None or
                   render_transaction != record.B1K_C1_RENDER_MODE):
        raise ValueError(
            "B1K terminal publication requires an authorized C1 batch")
    eligible_ids = (None if eligible_outcome_ids is None else
                    {str(value) for value in eligible_outcome_ids})
    materialized = withheld = 0
    renderable = []
    for result in outcomes:
        try:
            clear = outcome.is_completed_clear(result)
        except (KeyError, TypeError, ValueError):
            clear = False
        if clear:
            renderable.append(result)
    for result in renderable:
        if (eligible_ids is not None and
                str(result.get("outcome_id") or "") not in eligible_ids):
            result.pop("terminal_rgb_asset", None)
            result["terminal_rgb_asset_withhold"] = \
                "terminal_render_transaction_unavailable"
            result["terminal_rgb_asset_withhold_authority"] = \
                future_view_selection.TERMINAL_ASSET_RUNTIME_AUTHORITY
            withheld += 1
            continue
        try:
            key = future_view_selection.terminal_render_cache_key(result)
            if key not in render_cache and terminal_renderer is not None:
                pose = future_view_selection._terminal_checkpoint(result)
                terminal_renderer(tuple(
                    pose[field] for field in ("x", "z", "heading_deg")))
            result["terminal_rgb_asset"] = \
                future_view_selection.materialize_terminal_rgb_asset(
                    root, base_frame=frame, outcome=result,
                    render_cache=render_cache, source=source,
                    collection_contract=collection_contract,
                    render_transaction=render_transaction)
        except future_view_selection.TerminalRGBAssetError as error:
            result.pop("terminal_rgb_asset", None)
            result["terminal_rgb_asset_withhold"] = error.reason
            result["terminal_rgb_asset_withhold_authority"] = error.authority
            withheld += 1
        else:
            result.pop("terminal_rgb_asset_withhold", None)
            result.pop("terminal_rgb_asset_withhold_authority", None)
            materialized += 1
    return {"materialized": materialized, "withheld": withheld}
