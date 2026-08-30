"""Stage 21 item 2 — pin the hazard in `team_names_similar` that is true by luck.

THE SHAPE, from Stage 21's equality-versus-overlap rule:

    Canonicalising BOTH sides of a comparison is SAFE for an EQUALITY test — it
    can only increase agreement — and UNSAFE for an OVERLAP test, because it can
    delete the very token the overlap depended on.

`team_names._norm` applies `NAME_ALIASES` (a many-to-one map) and is called on
both sides by:

    same_team_strict    set(ta) == set(tb)        EQUALITY  -> safe by shape
    team_names_similar  matches/len(shorter)>=0.7 OVERLAP   -> UNSAFE by shape

`team_names_similar` is therefore the same defect that reached production in the
identity gate on 2026-08-29 (`Standard Liege` / `St. Liege`). It has not bitten
yet — and it has not bitten BY THE ACCIDENT OF THE TABLE'S CONTENTS, not by
construction.

WHY PIN RATHER THAN FIX. Fixing it symmetrically is a design change and belongs
with the guard work already deferred. But the alias table is GROWING: the identity
gate has found five corruptions and is still finding them one at a time, and every
false positive it produces is repaired BY ADDING AN ALIAS. Each repair is a chance
to activate this.

WHAT IS PINNED. Not "no hazard pair exists" — seven exist today and all seven are
benign. What is pinned is the INVENTORY. A new alias that creates an eighth fails
here, forcing whoever adds it to classify it rather than discover it months later
as a silent false refusal. Same shape as the exemption-count pin and the
`filter_generation` digest: the guarantee is unenforceable, so pin the claim and
make changing it deliberate.
"""

from src.utils.team_names import NAME_ALIASES, team_names_similar

#: Pairs where canonicalising ONE side deletes the only token it shares with the
#: other. Reviewed 2026-08-30; every one is benign, for the reason given.
#:
#:   both sides reach the same canonical, so equality rescues the comparison:
#:       olympiakos piraeus / olympiakos
#:       united states / united states of america   (both -> "usa")
#:       united states of america / united states
#:   genuinely different entities, where losing the overlap is CORRECT:
#:       korea republic / czech republic            (shared token "republic")
#:       republic of korea / czech republic
#:       republic of korea / united states of america  (shared token "of")
#:       united states of america / republic of korea
KNOWN_BENIGN_HAZARDS = {
    ("korea republic", "czech republic"),
    ("olympiakos piraeus", "olympiakos"),
    ("republic of korea", "czech republic"),
    ("republic of korea", "united states of america"),
    ("united states", "united states of america"),
    ("united states of america", "republic of korea"),
    ("united states of america", "united states"),
}


def _hazard_pairs():
    """(alias_key, counterpart) where rewriting the key destroys the only overlap.

    Derived from the table, never maintained beside it — so the scan cannot
    drift from the aliases it scans.
    """
    names = set(NAME_ALIASES) | set(NAME_ALIASES.values())
    out = set()
    for key, canon in NAME_ALIASES.items():
        deleted = set(key.split()) - set(canon.split())
        if not deleted:
            continue
        canon_tokens = set(canon.split())
        for other in names:
            if other in (key, canon):
                continue
            other_tokens = set(other.split())
            if (other_tokens & deleted) and not (other_tokens & canon_tokens):
                out.add((key, other))
    return out


def test_no_new_symmetric_canonicalisation_hazard():
    """Fails when an alias is added that creates a NEW hazard pair.

    If this fails you have added an alias whose canonical form deletes a token
    that some counterpart shares. Decide which it is and act accordingly:

      * the two names denote the SAME entity  -> `team_names_similar` will now
        return False for a legitimate pair. That is the Standard Liege defect.
        Do not silence it here; the site needs the union fix.
      * the two names denote DIFFERENT entities -> losing the overlap is
        correct. Add the pair to KNOWN_BENIGN_HAZARDS with the reason.
    """
    new = _hazard_pairs() - KNOWN_BENIGN_HAZARDS
    assert not new, (
        "new symmetric-canonicalisation hazard introduced by an alias:\n  "
        + "\n  ".join(f"{k!r} -> {NAME_ALIASES[k]!r} loses its only overlap "
                      f"with {o!r}" for k, o in sorted(new))
        + "\n\nCanonicalising both sides of an OVERLAP test can delete the "
          "token the overlap depended on. Classify the pair — see the docstring."
    )


def test_the_inventory_has_not_silently_shrunk():
    """A vanished hazard means the table changed; the inventory must follow.

    Without this, KNOWN_BENIGN_HAZARDS could accumulate stale entries that mask
    a genuinely new pair by coincidence of naming.
    """
    stale = KNOWN_BENIGN_HAZARDS - _hazard_pairs()
    assert not stale, (
        f"KNOWN_BENIGN_HAZARDS lists pairs that no longer exist: {sorted(stale)}. "
        "Remove them, so the inventory keeps describing the table.")


def test_the_currently_benign_pairs_still_behave_correctly():
    """The seven are benign for two DIFFERENT reasons; both are checked."""
    # same canonical on both sides -> still similar
    for a, b in (("olympiakos piraeus", "olympiakos"),
                 ("united states", "united states of america")):
        assert team_names_similar(a, b), f"{a!r} vs {b!r} stopped matching"
    # genuinely different entities -> correctly NOT similar
    for a, b in (("korea republic", "czech republic"),
                 ("republic of korea", "czech republic")):
        assert not team_names_similar(a, b), (
            f"{a!r} vs {b!r} now matches — these are different countries")


def test_same_team_strict_is_unaffected_by_shape():
    """The equality half of the rule, pinned so the distinction is not lost.

    `same_team_strict` compares token SETS for equality, so canonicalising both
    sides can only help. This is why the hazard is confined to the ratio test.
    """
    from src.utils.team_names import same_team_strict
    assert same_team_strict("PSG", "Paris Saint-Germain")
    assert same_team_strict("united states", "usa")
    assert not same_team_strict("AC Milan", "Inter Milan")
