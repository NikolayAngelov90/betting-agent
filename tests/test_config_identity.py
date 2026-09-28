"""Stage 10.2 — the frozen experimental subject is one configuration, not two.

Stage 10.1 found the experiment had been audited against a configuration
production has never run. CI does `cp config/config.example.yaml
config/config.yaml`, so the example IS the deployed config; the local
`config/config.yaml` is gitignored, was never committed, and existed only on one
developer's machine. They disagreed on `betting.excluded_markets`, which is a
TRACKED_KEY and a selection rule, so the two files described genuinely different
models — and every "frozen model" reference in Stages 5-10 named the wrong one.

These tests make that divergence impossible to reintroduce silently.
"""

import pathlib

from tests.experiment_pins import CODE_REVISION_PIN

import pytest
import yaml

from src.models.model_version import (CODE_REVISION, TRACKED_KEYS,
                                      fingerprint, fingerprint_inputs,
                                      model_version)
from src.utils.config import Config

EXAMPLE = "config/config.example.yaml"
LOCAL = "config/config.yaml"

#: The frozen experimental subject, recomputed in Stage 10.2 from the
#: configuration production actually executes. The previously quoted
#: `d1b522` came from the local file and was never deployed.
from tests.experiment_pins import FROZEN_MODEL_VERSION  # noqa: F401


def _example() -> Config:
    return Config(EXAMPLE)


# ══════════════════════ the example IS the production configuration

def test_ci_builds_its_config_from_the_example():
    """If a workflow ever stops doing this, the rest of this file is moot."""
    workflows = list(pathlib.Path(".github/workflows").glob("*.yml"))
    assert workflows, "no workflows found"
    copiers = [w for w in workflows
               if "cp config/config.example.yaml config/config.yaml"
               in w.read_text(encoding="utf-8")]
    assert copiers, (
        "no workflow copies config.example.yaml over config.yaml — the "
        "production configuration source has changed and the frozen model "
        "identity must be re-derived")


def test_the_config_copy_PRECEDES_the_test_run():
    """THE ORDERING, pinned. Found 2026-09-28 while checking whether the two
    gated tests run in CI at all.

    `test_ci_builds_its_config_from_the_example` asserts only that the copy
    appears SOMEWHERE in SOME workflow. That is not enough: if the copy step
    ever moved after the pytest step, or pytest were added to a workflow with no
    copy, the two gated tests would start skipping in CI silently — a guard
    going quiet with no failure anywhere. Same shape as the workflow-state check
    placed after `if not runs: return 0`: correct code, unreachable position.

    So for every workflow that runs pytest, the copy must exist and come first.
    """
    for wf in sorted(pathlib.Path(".github/workflows").glob("*.yml")):
        text = wf.read_text(encoding="utf-8")
        if "pytest" not in text:
            continue
        i_pytest = text.index("pytest")
        assert "cp config/config.example.yaml config/config.yaml" in text, (
            f"{wf.name} runs pytest but never creates config/config.yaml — the "
            f"local-vs-example tests will SKIP there, silently")
        i_copy = text.index("cp config/config.example.yaml config/config.yaml")
        assert i_copy < i_pytest, (
            f"{wf.name} copies the config AFTER running pytest — the "
            f"local-vs-example tests skip, and the skip looks like a pass")


def test_local_config_is_not_tracked_and_carries_no_authority():
    """`config/config.yaml` is gitignored. It is a convenience file, never a
    specification — which is exactly why letting it define the frozen model was
    the defect."""
    ignore = pathlib.Path(".gitignore").read_text(encoding="utf-8")
    assert "config/config.yaml" in ignore


# ═══════════════════ the two configs agree on every tracked key

def test_example_and_local_agree_on_every_tracked_key():
    """The invariant. Divergence on any TRACKED_KEY means local development is
    exercising a different model from the one production runs and the one the
    experiment measures.

    Skipped rather than failed when there is no local config — a fresh clone
    legitimately has none, and there is nothing to diverge.

    CORRECTED 2026-09-28: this comment used to say "or a CI checkout", which is
    FALSE. `daily-picks.yml` creates `config/config.yaml` by `cp` from the
    example at step "Create config from example", 22 lines before it runs
    pytest in the SAME JOB — so the skip never fires in CI and this test always
    runs there. It cannot fail there either, because the file it compares
    against was manufactured from the file it compares to seconds earlier:
    `example == cp of example` is a tautology. That is VAC-1 in a test — a pass
    carrying no information, indistinguishable from a pass that verified
    something.

    THE GUARD IS STILL CORRECTLY PLACED, which is why it is not being changed:
    the hazard is a developer's divergent local config, and the test runs
    exactly when a local config exists — that is, exactly when the hazard
    exists. In CI there is no hazard because CI eliminates it by construction.
    What must never happen is CITING the CI pass, or a byte-copy local pass, as
    evidence of agreement. See `test_the_config_copy_PRECEDES_the_test_run`,
    which pins the ordering this rests on.
    """
    if not pathlib.Path(LOCAL).exists():
        pytest.skip("no local config/config.yaml (fresh clone)")

    a = fingerprint_inputs(Config(LOCAL))
    b = fingerprint_inputs(_example())
    differing = {k: (a.get(k), b.get(k)) for k in TRACKED_KEYS
                 if a.get(k) != b.get(k)}
    assert not differing, (
        "config.yaml and config.example.yaml describe different models:\n" +
        "\n".join(f"  {k}: local={lo!r} example={ex!r}"
                  for k, (lo, ex) in differing.items()))


def test_example_and_local_produce_one_fingerprint():
    """The property that matters, stated directly: same fingerprint, therefore
    the same experimental subject."""
    if not pathlib.Path(LOCAL).exists():
        pytest.skip("no local config/config.yaml (fresh clone or CI)")
    assert fingerprint(Config(LOCAL)) == fingerprint(_example())


# ═════════════════ the frozen identity, pinned with its rationale

def test_the_deployed_config_produces_the_frozen_model_version():
    """Pinned so that any change to a tracked setting has to be deliberate.

    This is NOT a bare string assertion: the value is computed from the file
    CI deploys, so the test fails when the *configuration* moves, not merely
    when someone edits a constant. If you are here because it failed, a tracked
    model parameter changed — decide whether that is a new experiment before
    updating the constant.
    """
    assert model_version(_example()) == FROZEN_MODEL_VERSION


def test_code_revision_is_unchanged_by_the_reconciliation():
    """Stage 10.2 changed no code. CODE_REVISION distinguishes code paths that
    configuration cannot express; reconciling two config files is not one."""
    assert CODE_REVISION == CODE_REVISION_PIN


# ═══════════ over_3.5 is excluded on purpose, and stays that way

def test_over_35_is_excluded_in_the_deployed_config():
    """Commit bc7eacc (2026-06-16) excluded over_3.5 globally, with the reason
    recorded in the commit body and beside the key. Production data agrees:
    Over 3.5 picks stop on 2026-06-15.

    Pinned because re-enabling it would silently widen the frozen model's
    selection space mid-experiment. Removing this line is a model change and
    needs a new identity, not a config tweak.
    """
    cfg = yaml.safe_load(
        pathlib.Path(EXAMPLE).read_text(encoding="utf-8"))
    excluded = set(cfg["betting"]["excluded_markets"])
    assert excluded == {"btts_no", "under_1.5", "over_3.5"}, (
        f"the frozen model's excluded markets changed: {sorted(excluded)}")


def test_excluded_markets_is_part_of_the_fingerprint():
    """If it were not tracked, two genuinely different selection policies would
    share one model_version — which is the ambiguity the fingerprint exists to
    remove."""
    assert "betting.excluded_markets" in TRACKED_KEYS

    import copy

    base = _example()
    probe = copy.deepcopy(base)
    probe.config["betting"]["excluded_markets"] = ["btts_no", "under_1.5"]
    assert fingerprint(probe) != fingerprint(base), (
        "changing excluded_markets did not move the fingerprint")


def test_excluded_markets_actually_constrains_selection():
    """It is not a label. It gates find_value_bets, find_best_bet and
    build_selection_pick — so it also constrains what Claude's CHANGE may
    switch to."""
    import inspect

    from src.betting import value_calculator as vc

    src = inspect.getsource(vc)
    assert src.count("in self.excluded_markets") >= 3, (
        "excluded_markets is no longer enforced at every selection site")
