"""Does the current fingerprint have members? — the AMEND-or-BUMP decision.

STAGE 19. `CODE_REVISION` exists to stop picks made under different
configurations being pooled. **Pooling cannot happen in a cohort with no
members**, so a revision that has never been stamped on a pick guarantees
nothing and costs a history entry.

THE RULE, AS RETIRED BY s5.14 (2026-09-17) AND RECONCILED HERE 2026-10-01:

    ALWAYS BUMP.

The rule this file shipped with was:

    While `saved_picks` holds ZERO rows at the current fingerprint, a further
    prediction- or selection-affecting change AMENDS the current revision.

and `s5.14`'s history entry retired it, in these words: "It evaluates a MUTABLE
COUNT — s5.13 held 0 picks at the amend and 16 within the hour, so ee60cd labels
two configurations. Always bump."

THIS TOOL KEPT PRINTING `VERDICT: AMEND` FOR TWO WEEKS AFTER THAT. An executable
giving retired advice is a stale reason string that RUNS — worse than a stale
comment, because it is consulted precisely when the decision is being made, and
it answers with authority. Found 2026-10-01 while taking the s5.15 bump.

So the member count is still measured and still reported — it is useful context —
but it no longer selects a verdict. Emptiness at the moment of the check is not
emptiness at the moment of the next pick.

WHY THIS FILE EXISTS RATHER THAN A NOTE. "Verify emptiness at commit time, not
from memory" is the standard applied to every other claim in this project, and a
standard that depends on remembering to check is the one that fails. Run this;
do not recall it.

    python -m scripts.cohort_status
"""

from __future__ import annotations

import sys


def main() -> int:
    # Loaded INSIDE main, never at import. A module-level load_dotenv() undoes
    # conftest's DATABASE_URL strip and has previously let a test suite reach
    # production — see tests/test_db_isolation.py and the ledger entry.
    import os
    import re
    for path in (".env", ".env.local"):
        try:
            for line in open(path, encoding="utf-8"):
                m = re.match(r"\s*([A-Z0-9_]+)\s*=\s*(.*)", line)
                if m and not os.environ.get(m.group(1)):
                    os.environ[m.group(1)] = m.group(2).strip().strip('"').strip("'")
        except FileNotFoundError:
            continue

    from src.data.database import get_db
    from src.data.models import SavedPick
    from src.models.model_version import CODE_REVISION, model_version
    from src.utils.config import Config

    # THE DEPLOYED CONFIG, not the local one. `config/config.yaml` is gitignored
    # and carries no authority — Stage 10.1 found the experiment audited against
    # a configuration production has never run, and CI builds its copy from the
    # example anyway. Reading the example is what makes this fingerprint the one
    # production stamps. (Byte-identical today; the point is that it stays so by
    # construction rather than by luck.)
    version = model_version(Config("config/config.example.yaml"))
    with get_db().get_session() as session:
        n = session.query(SavedPick).filter(
            SavedPick.model_version == version).count()
        total = session.query(SavedPick).count()

    print(f"CODE_REVISION : {CODE_REVISION}")
    print(f"model_version : {version}")
    print(f"picks stamped : {n}   (of {total} saved picks)")
    print()
    # ONE VERDICT. The count is context, never the decision — see the module
    # docstring for why s5.14 retired the AMEND branch.
    print("VERDICT: BUMP")
    if n == 0:
        print("  This cohort has no members YET, and that is not a licence to")
        print("  amend: the count is mutable between this check and the next")
        print("  pick. s5.13 held 0 here and 16 within the hour, and ee60cd")
        print("  ended up labelling two configurations.")
    else:
        print(f"  {n} pick(s) already carry this fingerprint.")
    print("  A prediction- or selection-affecting change MUST take a new")
    print("  revision, or two configurations share one cohort label.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
