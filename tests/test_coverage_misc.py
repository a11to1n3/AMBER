"""Coverage-focused tests for small modules: contract dataclasses, the
``deprecated`` decorator, AttrDict / NPRandomCompat, and Agent read helpers.
"""

import warnings

import numpy as np
import polars as pl
import pytest

import ambr as am
from ambr._deprecation import deprecated
from ambr.base import AttrDict, NPRandomCompat, _coerce_param
from ambr.contract import (
    SEVERITY_ERROR,
    SEVERITY_WARNING,
    ContractCertificate,
    ContractViolation,
    ContractViolationError,
)


# --- contract.py ----------------------------------------------------------- #

def test_contract_violation_repr_truncates_ids():
    v = ContractViolation(
        "read_after_write", "x read after write",
        severity=SEVERITY_ERROR, columns=["x"], ids=list(range(12)),
    )
    r = repr(v)
    assert "read_after_write" in r
    assert "columns=['x']" in r
    assert "more" in r                       # >8 ids -> "(+4 more)"


def test_contract_certificate_ok_clean_and_repr():
    cert = ContractCertificate(3)
    assert cert.ok and cert.clean
    assert "ok (no violations)" in repr(cert)
    cert.add(ContractViolation("k", "warn detail", severity=SEVERITY_WARNING))
    assert cert.ok and not cert.clean        # a warning doesn't flip ok
    cert.add(ContractViolation("k2", "err detail", severity=SEVERITY_ERROR))
    assert not cert.ok
    assert len(cert.errors()) == 1 and len(cert.warnings()) == 1
    assert "errors=1 warnings=1" in repr(cert)
    err = ContractViolationError(cert)
    assert "step 3" in str(err) and "err detail" in str(err)


# --- _deprecation.py ------------------------------------------------------- #

def test_deprecated_decorator_warns_and_preserves_name():
    @deprecated("new_func")
    def old_func(x):
        return x * 2

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        assert old_func(21) == 42
    assert any(issubclass(w.category, DeprecationWarning) for w in rec)
    assert old_func.__name__ == "old_func"   # functools.wraps preserved


# --- base.py --------------------------------------------------------------- #

def test_coerce_param_none_passthrough():
    assert _coerce_param(int, None) is None
    assert _coerce_param(bool, "off") is False
    assert _coerce_param(int, "5") == 5


def test_attrdict_attribute_access_and_typed_getters():
    d = AttrDict({"a": 1})
    d.b = 2                                  # __setattr__
    assert d["b"] == 2
    del d.b                                  # __delattr__
    assert "b" not in d
    with pytest.raises(AttributeError):
        _ = d.missing                        # __getattr__ KeyError -> AttributeError
    with pytest.raises(AttributeError):
        del d.missing                        # __delattr__ KeyError -> AttributeError
    assert d.get_int("a") == 1
    assert d.get_float("a") == 1.0
    assert d.get_bool("a") is True
    assert d.get_int("z", 9) == 9            # default for absent key


def test_nprandom_compat_randint_and_delegate():
    c = NPRandomCompat(np.random.default_rng(0))
    assert 0 <= int(c.randint(0, 5)) < 5
    assert isinstance(c.random(), float)     # __getattr__ delegates to rng.random


# --- agent.py -------------------------------------------------------------- #

def test_agent_get_data_and_neighbors():
    m = am.Model({"show_progress": False})
    m.add_agents(3, agent_class=am.Agent)
    m.agents.set(x=np.array([1.0, 2.0, 3.0]))
    a = m.agents.by_id(0)
    assert a.get_data().height == 1
    assert a.get_neighbors().height == 2                       # the other two
    assert a.get_neighbors(pl.col("x") > 2.5).height == 1      # filtered
