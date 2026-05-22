"""Tests for the multi-state helpers in :mod:`aure.state`."""

from aure.state import (
    create_initial_state,
    flatten_data_files,
    is_multi_state,
    iter_states,
)


# ---------------------------------------------------------------------------
# iter_states
# ---------------------------------------------------------------------------


def test_iter_states_returns_explicit_states_verbatim():
    states = [
        {"name": "D2O", "data_files": [{"file": "a.dat", "label": "a"}]},
        {"name": "H2O", "data_files": [{"file": "b.dat", "label": "b"}]},
    ]
    definition = {"states": states}

    result = iter_states(definition)

    assert result == states
    # Returns a list, but the inner state dicts are the same objects
    assert result[0] is states[0]
    assert result[1] is states[1]


def test_iter_states_synthesises_single_state_from_legacy_fields():
    definition = {
        "data_file": "/abs/primary.dat",
        "data_files": [
            {"file": "/abs/primary.dat", "label": "primary"},
            {"file": "/abs/extra.dat", "label": "extra"},
        ],
        "intensity": {"value": 1.0, "min": 0.9, "max": 1.1},
        "ambient": {"name": "D2O", "sld": 6.4},
        "back_reflection": True,
    }

    result = iter_states(definition)

    assert len(result) == 1
    state = result[0]
    assert state["name"] == "state0"
    assert state["data_files"] == definition["data_files"]
    assert state["intensity"] == definition["intensity"]
    assert state["ambient"] == definition["ambient"]
    assert state["back_reflection"] is True


def test_iter_states_falls_back_to_data_file_only():
    definition = {"data_file": "/abs/only.dat"}

    result = iter_states(definition)

    assert len(result) == 1
    assert result[0]["data_files"] == [{"file": "/abs/only.dat", "label": "primary"}]


def test_iter_states_empty_states_treated_as_legacy():
    definition = {"states": [], "data_file": "/abs/x.dat"}

    result = iter_states(definition)

    assert len(result) == 1
    assert result[0]["data_files"][0]["file"] == "/abs/x.dat"


# ---------------------------------------------------------------------------
# flatten_data_files
# ---------------------------------------------------------------------------


def test_flatten_data_files_preserves_order():
    states = [
        {
            "name": "s1",
            "data_files": [
                {"file": "a.dat", "label": "a"},
                {"file": "b.dat", "label": "b"},
            ],
        },
        {
            "name": "s2",
            "data_files": [{"file": "c.dat", "label": "c"}],
        },
    ]

    flat = flatten_data_files(states)

    assert [ds["label"] for ds in flat] == ["a", "b", "c"]


def test_flatten_data_files_handles_missing_or_empty_files():
    states = [
        {"name": "s1"},  # no data_files key
        {"name": "s2", "data_files": []},
        {"name": "s3", "data_files": [{"file": "x.dat", "label": "x"}]},
    ]

    flat = flatten_data_files(states)

    assert len(flat) == 1
    assert flat[0]["file"] == "x.dat"


# ---------------------------------------------------------------------------
# is_multi_state
# ---------------------------------------------------------------------------


def test_is_multi_state_false_for_legacy_or_single_state():
    assert is_multi_state({"data_file": "x.dat"}) is False
    assert is_multi_state({"states": []}) is False
    assert is_multi_state({"states": [{"name": "only"}]}) is False


def test_is_multi_state_true_for_two_or_more():
    assert is_multi_state({"states": [{"name": "a"}, {"name": "b"}]}) is True


# ---------------------------------------------------------------------------
# create_initial_state — states wiring
# ---------------------------------------------------------------------------


def test_create_initial_state_default_no_states():
    state = create_initial_state("dummy.dat", "test sample")
    assert state["states"] == []
    assert state["data_files"] == []


def test_create_initial_state_with_states_populates_data_files():
    states = [
        {
            "name": "D2O",
            "data_files": [
                {"file": "a.dat", "label": "a"},
                {"file": "b.dat", "label": "b"},
            ],
        },
        {
            "name": "H2O",
            "data_files": [{"file": "c.dat", "label": "c"}],
        },
    ]

    state = create_initial_state("a.dat", "desc", states=states)

    assert state["states"] == states
    assert [ds["label"] for ds in state["data_files"]] == ["a", "b", "c"]


def test_create_initial_state_explicit_data_files_wins_over_states_flatten():
    """If the caller passes both, ``data_files`` is not overwritten."""
    states = [
        {"name": "s", "data_files": [{"file": "a.dat", "label": "a"}]},
    ]
    explicit = [{"file": "z.dat", "label": "z"}]

    state = create_initial_state("a.dat", "desc", states=states, data_files=explicit)

    assert state["data_files"] == explicit
    assert state["states"] == states
