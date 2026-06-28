"""Regression tests for numpy → psycopg2 serialization.

Under numpy 2.x, ``repr(np.float64(2.75))`` is ``'np.float64(2.75)'``. Because
``np.float64`` subclasses ``float``, psycopg2's float adapter (which uses repr)
emitted that string verbatim into SQL, so Postgres parsed ``np.`` as a schema
name → ``InvalidSchemaName: schema "np" does not exist``. That silently aborted
every briefing CHANGE write — the tracked pick was never switched.

Importing src.data.database registers explicit adapters that fix this.
"""

import numpy as np
import pytest

# Importing the module runs _register_numpy_psycopg2_adapters() at import time.
import src.data.database  # noqa: F401

psycopg2 = pytest.importorskip("psycopg2")
from psycopg2.extensions import adapt  # noqa: E402


def _quoted(value) -> bytes:
    return adapt(value).getquoted()


def test_np_float64_renders_as_plain_number_not_repr():
    # The exact value that triggered the production failure.
    assert _quoted(np.float64(2.75)) == b"2.75"
    # And the value shape model code actually produces: round(prob, 4).
    assert _quoted(round(np.float64(0.5612), 4)) == b"0.5612"


def test_np_float64_never_leaks_schema_qualified_repr():
    # The bug signature: a 'np.' prefix (schema.function) in the SQL literal.
    assert b"np." not in _quoted(np.float64(2.75))
    assert b"np." not in _quoted(np.float32(1.5))


def test_np_int_and_bool_render_as_literals():
    assert _quoted(np.int64(3)) == b"3"
    assert _quoted(np.int32(7)) == b"7"
    assert _quoted(np.bool_(True)) == b"true"
    assert _quoted(np.bool_(False)) == b"false"


def test_np_float_nan_is_handled():
    # Must not raise and must not produce a bare 'nan' identifier.
    quoted = _quoted(np.float64("nan"))
    assert b"NaN" in quoted
