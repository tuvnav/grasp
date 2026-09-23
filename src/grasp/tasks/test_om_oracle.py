"""
Tests for grasp.tasks.om_oracle.

These tests avoid touching real KgManager/SPARQL machinery: `find_manager`,
`prepare_entity`, `ensure_known` and `get_entity_shape_text` are monkeypatched
(see `stub_kg_access` below) with lightweight fakes, so everything here
exercises only om_oracle's own logic (OracleState bookkeeping, verdict
messages, function dispatch, prompt building).

A few tests are written to assert the *intended* behavior discussed in
review and therefore currently FAIL against the code as of this writing.
Each is marked "EXPECTED TO CURRENTLY FAIL" with a short reason, so a red
result there is a known, already-diagnosed issue, not a surprise:

  - TestOracleState.test_get_verdict_after_add
      -> OracleState.get_verdict() looks up `(entity_source, entity_source)`
         instead of `(entity_source, entity_target)`.
  - TestUndoVerdict.test_undo_after_confirm
      -> depends on the same get_verdict bug (undo_verdict() can never find
         an existing verdict, so it always raises).
  - TestCallFunction.test_undo_verdict_dispatch
      -> `functions()` declares "undo_verdict" as a callable tool, but
         `call_function()` has no matching branch for it.
  - TestAddVerdict.test_conflict_notice_on_reused_target
      -> the "existing_with_target" message branch still has the inverted
         "not equivalent" / "equivalent" wording (the "existing_with_source"
         branch above it was already fixed, this one was not).
"""

import pytest

from grasp.configs import GraspConfig
from grasp.tasks.entities import Entity
from grasp.tasks import om_oracle as oo


def make_entity(iri: str, label: str | None = None) -> Entity:
    return Entity(identifier=iri, entity=iri, label=label)


def make_task_input(n: int = 1, source_kg: str = "src-kg", target_kg: str = "tgt-kg") -> oo.OracleTaskInput:
    candidates = [
        oo.CandidateEquivalence(
            entity_source=make_entity(f"s{i}"),
            entity_target=make_entity(f"t{i}"),
        )
        for i in range(1, n + 1)
    ]
    return oo.OracleTaskInput(
        candidate_equivalences=candidates,
        source_kg=source_kg,
        target_kg=target_kg,
    )


@pytest.fixture(autouse=True)
def stub_kg_access(monkeypatch):
    """
    Replace every function om_oracle.py uses to talk to a real KgManager
    with a trivial fake, so tests never need a real ontology/index.
    """

    def fake_find_manager(managers, kg):
        # om.py's find_manager returns (manager, others); the manager
        # object itself is never inspected by om_oracle's own logic once
        # prepare_entity/ensure_known/get_entity_shape_text are stubbed too
        return kg, []

    def fake_prepare_entity(manager, iri):
        return make_entity(iri)

    def fake_ensure_known(manager, iri, known):
        known.add(iri)

    def fake_get_entity_shape_text(manager, iri, known=None):
        return None

    monkeypatch.setattr(oo, "find_manager", fake_find_manager)
    monkeypatch.setattr(oo, "prepare_entity", fake_prepare_entity)
    monkeypatch.setattr(oo, "ensure_known", fake_ensure_known)
    monkeypatch.setattr(oo, "get_entity_shape_text", fake_get_entity_shape_text)


# ---------------------------------------------------------------------------
# OracleState
# ---------------------------------------------------------------------------


class TestOracleState:
    def test_get_verdict_missing_returns_none(self):
        state = oo.OracleState()
        assert state.get_verdict("s1", "t1") is None

    def test_get_verdict_after_add(self):
        """EXPECTED TO CURRENTLY FAIL - see module docstring (get_verdict bug)."""
        state = oo.OracleState()
        v = oo.Verdict(entity_source=make_entity("s1"), entity_target=make_entity("t1"), equivalent=True)
        state.add_verdict(v)
        assert state.get_verdict("s1", "t1") is v

    def test_same_source_multiple_targets_both_kept(self):
        # this is the scenario from your matcher->GRASP split: one source
        # entity, several candidate target entities, each its own verdict
        state = oo.OracleState()
        v1 = oo.Verdict(entity_source=make_entity("s1"), entity_target=make_entity("t1"), equivalent=True)
        v2 = oo.Verdict(entity_source=make_entity("s1"), entity_target=make_entity("t2"), equivalent=False)
        state.add_verdict(v1)
        state.add_verdict(v2)
        assert len(state.verdicts) == 2
        assert state.verdicts[("s1", "t1")] is v1
        assert state.verdicts[("s1", "t2")] is v2

    def test_undo_verdict_removes_entry(self):
        state = oo.OracleState()
        v = oo.Verdict(entity_source=make_entity("s1"), entity_target=make_entity("t1"), equivalent=True)
        state.add_verdict(v)
        state.undo_verdict("s1", "t1")
        assert ("s1", "t1") not in state.verdicts

    def test_undo_verdict_on_missing_pair_is_a_noop(self):
        state = oo.OracleState()
        # should not raise
        state.undo_verdict("s1", "t1")

    def test_format_and_to_dict_when_empty(self):
        state = oo.OracleState()
        assert state.format() == "No verdicts yet."
        assert state.to_dict()["verdicts"] == []

    def test_to_dict_lists_every_verdict_exactly_once(self):
        state = oo.OracleState()
        state.add_verdict(oo.Verdict(entity_source=make_entity("s1"), entity_target=make_entity("t1"), equivalent=True))
        state.add_verdict(oo.Verdict(entity_source=make_entity("s2"), entity_target=make_entity("t2"), equivalent=False))
        d = state.to_dict()
        assert len(d["verdicts"]) == 2
        pairs = {(v["entity_source"]["identifier"], v["entity_target"]["identifier"]) for v in d["verdicts"]}
        assert pairs == {("s1", "t1"), ("s2", "t2")}


# ---------------------------------------------------------------------------
# add_verdict (confirm_equivalence / reject_equivalence share this)
# ---------------------------------------------------------------------------


class TestAddVerdict:
    def test_confirm_records_true_verdict_and_says_so(self):
        state = oo.OracleState()
        msg = oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=set(),
        )
        assert state.verdicts[("s1", "t1")].equivalent is True
        assert "confirmed" in msg

    def test_reject_records_false_verdict_and_says_so(self):
        state = oo.OracleState()
        msg = oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=False, state=state, known=set(),
        )
        assert state.verdicts[("s1", "t1")].equivalent is False
        assert "rejected" in msg

    def test_no_conflict_notice_for_first_verdict(self):
        state = oo.OracleState()
        msg = oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=set(),
        )
        assert "already involved" not in msg

    def test_conflict_notice_on_reused_source(self):
        # same source entity (s1) already judged against a different target (t1)
        state = oo.OracleState()
        oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=set(),
        )
        msg = oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t2",
            equivalent=False, state=state, known=set(),
        )
        assert "already involved" in msg
        assert "t1" in msg

    def test_conflict_notice_on_reused_target(self):
        """EXPECTED TO CURRENTLY FAIL - see module docstring (inverted wording
        in the 'existing_with_target' message branch: it currently renders
        e.g. "has been jugded as not equivalent" for a verdict that was
        actually equivalent=True, or with a stray leading space otherwise).
        This test only checks the *content* (which entity is mentioned), not
        the exact wording, and should already pass once that block mirrors
        the fixed 'existing_with_source' one - if it still fails, compare the
        two message-building blocks in add_verdict() line by line.
        """
        # same target entity (t1) already judged against a different source (s1)
        state = oo.OracleState()
        oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=set(),
        )
        msg = oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s2", entity_target="t1",
            equivalent=False, state=state, known=set(),
        )
        assert "already involved" in msg
        assert "s1" in msg

    def test_known_set_is_updated(self):
        state = oo.OracleState()
        known: set[str] = set()
        oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=known,
        )
        assert "s1" in known and "t1" in known


# ---------------------------------------------------------------------------
# undo_verdict (free function)
# ---------------------------------------------------------------------------


class TestUndoVerdict:
    def test_undo_after_confirm(self):
        """EXPECTED TO CURRENTLY FAIL - see module docstring (get_verdict bug
        makes undo_verdict() always believe no verdict exists)."""
        state = oo.OracleState()
        oo.add_verdict(
            managers=[], kg_source="src-kg", kg_target="tgt-kg",
            entity_source="s1", entity_target="t1",
            equivalent=True, state=state, known=set(),
        )
        msg = oo.undo_verdict("s1", "t1", state)
        assert ("s1", "t1") not in state.verdicts
        assert "s1" in msg and "t1" in msg

    def test_undo_nonexistent_raises(self):
        state = oo.OracleState()
        with pytest.raises(oo.FunctionCallException):
            oo.undo_verdict("s1", "t1", state)


# ---------------------------------------------------------------------------
# call_function dispatch
# ---------------------------------------------------------------------------


def state_with_input(n: int = 1) -> oo.OracleState:
    state = oo.OracleState()
    state.task_input = make_task_input(n)
    return state


class TestCallFunction:
    def test_confirm_equivalence_dispatch(self):
        state = state_with_input()
        oo.call_function(
            GraspConfig(), [], "confirm_equivalence",
            {"source_entity": "s1", "target_entity": "t1"},
            set(), state, None,
        )
        assert state.verdicts[("s1", "t1")].equivalent is True

    def test_reject_equivalence_dispatch(self):
        state = state_with_input()
        oo.call_function(
            GraspConfig(), [], "reject_equivalence",
            {"source_entity": "s1", "target_entity": "t1"},
            set(), state, None,
        )
        assert state.verdicts[("s1", "t1")].equivalent is False

    def test_show_decisions_dispatch(self):
        state = state_with_input()
        oo.call_function(
            GraspConfig(), [], "confirm_equivalence",
            {"source_entity": "s1", "target_entity": "t1"},
            set(), state, None,
        )
        out = oo.call_function(GraspConfig(), [], "show_decisions", {}, set(), state, None)
        assert "s1" in out and "t1" in out

    def test_stop_dispatch(self):
        state = state_with_input()
        assert oo.call_function(GraspConfig(), [], "stop", {}, set(), state, None) == "Stopping"

    def test_unknown_function_raises(self):
        state = state_with_input()
        with pytest.raises(ValueError):
            oo.call_function(GraspConfig(), [], "not_a_real_function", {}, set(), state, None)

    def test_undo_verdict_dispatch(self):
        """EXPECTED TO CURRENTLY FAIL - see module docstring: `functions()`
        declares "undo_verdict" as a tool the model can call, but
        `call_function()` has no branch for that name and falls through to
        `raise ValueError(f"Unknown function {fn_name}")`."""
        state = state_with_input()
        oo.call_function(
            GraspConfig(), [], "confirm_equivalence",
            {"source_entity": "s1", "target_entity": "t1"},
            set(), state, None,
        )
        oo.call_function(
            GraspConfig(), [], "undo_verdict",
            {"source_entity": "s1", "target_entity": "t1"},
            set(), state, None,
        )
        assert ("s1", "t1") not in state.verdicts


# ---------------------------------------------------------------------------
# input_instructions / OracleTaskInput / input_and_state
# ---------------------------------------------------------------------------


class TestOracleTaskInput:
    def test_valid_input_parses(self):
        ti = make_task_input(1)
        assert ti.source_kg == "src-kg"
        assert len(ti.candidate_equivalences) == 1

    def test_missing_required_field_raises(self):
        with pytest.raises(Exception):
            oo.OracleTaskInput(candidate_equivalences=[], source_kg="src-kg")  # target_kg missing


class TestInputInstructions:
    def test_agentic_single_candidate_mentions_function_name(self):
        ti = make_task_input(1)
        text = oo.input_instructions(ti, oo.OracleState(), managers=[], known=set(), agentic=True)
        assert "confirm_equivalence" in text
        assert "s1" in text and "t1" in text

    def test_non_agentic_single_candidate_asks_singular_question(self):
        ti = make_task_input(1)
        text = oo.input_instructions(ti, oo.OracleState(), managers=[], known=set(), agentic=False)
        assert "Does this equivalence hold?" in text

    def test_non_agentic_multiple_candidates_answer_format_is_still_ambiguous(self):
        """
        Documents an open design gap (flagged separately, not a crash): with
        more than one candidate, the non-agentic prompt still asks for a
        single True/False, with no schema tying individual verdicts to
        individual candidate pairs. This test pins today's text so a future
        fix (e.g. a per-candidate answer list) shows up here as an
        intentional change, not a silent regression.
        """
        ti = make_task_input(2)
        text = oo.input_instructions(ti, oo.OracleState(), managers=[], known=set(), agentic=False)
        assert "Do these equivalences hold? Answer with 'True' or 'False'" in text

    def test_no_candidates_raises(self):
        ti = oo.OracleTaskInput(candidate_equivalences=[], source_kg="a", target_kg="b")
        with pytest.raises(oo.FunctionCallException):
            oo.input_instructions(ti, oo.OracleState(), managers=[], known=set())


class TestInputAndState:
    def test_builds_state_from_dict_input(self):
        data = {
            "candidate_equivalences": [
                {
                    "entity_source": {"identifier": "s1", "entity": "s1"},
                    "entity_target": {"identifier": "t1", "entity": "t1"},
                }
            ],
            "source_kg": "src-kg",
            "target_kg": "tgt-kg",
        }
        instructions, state = oo.input_and_state(data, GraspConfig(), managers=[], known=set())
        assert state.task_input.source_kg == "src-kg"
        assert "s1" in instructions

    def test_invalid_input_raises_value_error(self):
        with pytest.raises(ValueError):
            oo.input_and_state({"nonsense": True}, GraspConfig(), managers=[], known=set())


# ---------------------------------------------------------------------------
# OmOracleTask
# ---------------------------------------------------------------------------


class TestOmOracleTaskSystemInformation:
    def test_reads_prompt_file(self, tmp_path):
        prompt_path = tmp_path / "prompt.txt"
        prompt_path.write_text("You are a careful ontology matcher.\n")
        config = GraspConfig(task_kwargs={"om_oracle": {"prompt_file": str(prompt_path)}})
        task = oo.OmOracleTask(managers=[], config=config, known=set())
        assert "careful ontology matcher" in task.system_information()

    def test_missing_prompt_file_config_returns_empty_string(self):
        task = oo.OmOracleTask(managers=[], config=GraspConfig(), known=set())
        assert task.system_information() == ""


class TestOmOracleTaskDone:
    def test_stop_is_done(self):
        task = oo.OmOracleTask(managers=[], config=GraspConfig(), known=set())
        assert task.done("stop") is True

    def test_confirm_is_not_done(self):
        task = oo.OmOracleTask(managers=[], config=GraspConfig(), known=set())
        assert task.done("confirm_equivalence") is False


class TestOmOracleTaskIntegration:
    def test_full_confirm_reject_cycle_with_shared_source(self):
        """
        End-to-end run through setup() -> call_function() -> output(),
        for exactly the scenario your matcher->GRASP split produces: one
        source entity split into two separate CandidateEquivalence entries
        (two different target candidates).
        """
        task = oo.OmOracleTask(managers=[], config=GraspConfig(), known=set())
        data = {
            "candidate_equivalences": [
                {
                    "entity_source": {"identifier": "s1", "entity": "s1"},
                    "entity_target": {"identifier": "t1", "entity": "t1"},
                },
                {
                    "entity_source": {"identifier": "s1", "entity": "s1"},
                    "entity_target": {"identifier": "t2", "entity": "t2"},
                },
            ],
            "source_kg": "src-kg",
            "target_kg": "tgt-kg",
        }
        instructions = task.setup(data)
        assert "t1" in instructions and "t2" in instructions

        msg1 = task.call_function(
            "confirm_equivalence", {"source_entity": "s1", "target_entity": "t1"}, task.known, None
        )
        assert "already involved" not in msg1

        msg2 = task.call_function(
            "reject_equivalence", {"source_entity": "s1", "target_entity": "t2"}, task.known, None
        )
        assert "already involved" in msg2  # s1 now appears in a second verdict

        assert task.done("stop") is True

        out = task.output(messages=[])
        verdicts = {
            (v["entity_source"]["identifier"], v["entity_target"]["identifier"]): v["equivalent"]
            for v in out["verdicts"]
        }
        assert verdicts == {("s1", "t1"): True, ("s1", "t2"): False}
