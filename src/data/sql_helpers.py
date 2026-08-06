"""Query helpers that stay correct as collection sizes grow.

The problem
-----------
``Column.in_(python_list)`` renders one bind parameter per element. That is fine
for a handful of ids and a hard failure for a large batch:

* PostgreSQL's wire protocol caps a statement at **65,535 bind parameters**.
  ``preload_batch`` passes the same team-id set to *two* ``in_()`` clauses, so at
  ~13,200 fixtures (100x today) that is ~26,400 teams × 2 = ~52,800 parameters
  in one statement — already inside the danger zone, and it fails outright, not
  slowly.
* Even well under the cap, a 26k-element ``IN`` list is re-parsed and re-planned
  on every call, because no two calls share a statement shape.

``= ANY(:ids)`` with an array parameter is **one** bind parameter regardless of
size, has a stable statement shape, and PostgreSQL plans it as a hash/bitmap
lookup rather than a long OR chain.

SQLite has no array type, so there the helper falls back to ``IN`` — which is
correct because SQLite is dev-only here and its own limit
(``SQLITE_MAX_VARIABLE_NUMBER``, 32,766+ on modern builds) is not reachable by
local test data.
"""

from __future__ import annotations

from typing import Iterable

from sqlalchemy import Integer, String, bindparam

# Chosen well under PostgreSQL's 65,535 cap so a caller that still uses IN (or a
# statement carrying other parameters) cannot get close to it.
IN_CHUNK_SIZE = 5000


def _is_postgres(session) -> bool:
    bind = getattr(session, "bind", None)
    return bool(bind is not None and bind.dialect.name == "postgresql")


def id_in(session, column, ids: Iterable):
    """``column IN ids``, rendered as ``= ANY(array)`` on PostgreSQL.

    Use anywhere the collection can grow with fixture volume. For a genuinely
    small, fixed set (a market-type whitelist, say) plain ``in_()`` is clearer
    and equivalent.
    """
    ids = list(ids)
    if not _is_postgres(session):
        return column.in_(ids)

    from sqlalchemy.dialects.postgresql import ARRAY

    inner = Integer() if not ids or isinstance(ids[0], int) else String()
    return column == bindparam(None, value=ids, type_=ARRAY(inner)).any_()


def chunked(items: Iterable, size: int = IN_CHUNK_SIZE):
    """Yield successive chunks, for callers that must stay on plain ``IN``.

    Used where the query shape rules out an array parameter (a row-value ``IN``,
    for instance) but the collection is still unbounded.
    """
    items = list(items)
    for start in range(0, len(items), size):
        yield items[start:start + size]
