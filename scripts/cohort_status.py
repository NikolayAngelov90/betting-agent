"""Does the current fingerprint have members? — the AMEND-or-BUMP decision.

STAGE 19. `CODE_REVISION` exists to stop picks made under different
configurations being pooled. **Pooling cannot happen in a cohort with no
members**, so a revision that has never been stamped on a pick guarantees
nothing and costs a history entry.

THE RULE:

    While `saved_picks` holds ZERO rows at the current fingerprint, a further
    prediction- or selection-affecting change AMENDS the current revision.
    If any pick carries it, BUMP.

The guarantee is untouched: no two configurations ever share a cohort that
contains anything.

WHY THIS FILE EXISTS RATHER THAN A NOTE. "Verify emptiness at commit time, not
from memory" is the standard applied to every other claim in this project, and a
standard that depends on remembering to check is the one that fails. Run this;
do not recall it.

    python -m scripts.cohort_status
"""

from __future__ import annotations

import sys


def main() -> int:
    from src.data.database import get_db
    from src.data.models import SavedPick
    from src.models.model_version import CODE_REVISION, model_version
    from src.utils.config import Config

    version = model_version(Config("config/config.yaml"))
    with get_db().get_session() as session:
        n = session.query(SavedPick).filter(
            SavedPick.model_version == version).count()
        total = session.query(SavedPick).count()

    print(f"CODE_REVISION : {CODE_REVISION}")
    print(f"model_version : {version}")
    print(f"picks stamped : {n}   (of {total} saved picks)")
    print()
    if n == 0:
        print("VERDICT: AMEND")
        print("  This cohort has no members, so nothing can be pooled with")
        print("  anything. A prediction- or selection-affecting change may")
        print("  amend this revision's history entry in place.")
    else:
        print("VERDICT: BUMP")
        print(f"  {n} pick(s) already carry this fingerprint. A further")
        print("  prediction- or selection-affecting change MUST take a new")
        print("  revision, or two configurations share a populated cohort.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
