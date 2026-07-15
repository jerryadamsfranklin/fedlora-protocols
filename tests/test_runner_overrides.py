import pytest
import torch

from scripts.run_experiment import apply_overrides, resolve_device


def test_resolve_device_auto_matches_legacy():
    expected = "mps" if torch.backends.mps.is_available() else "cpu"
    assert resolve_device(None) == expected
    assert resolve_device("auto") == expected


def test_resolve_device_cpu():
    assert resolve_device("cpu") == "cpu"


def test_resolve_device_cuda_without_gpu_exits():
    if torch.cuda.is_available():
        assert resolve_device("cuda") == "cuda"
    else:
        with pytest.raises(SystemExit):
            resolve_device("cuda")


def test_apply_overrides_simple():
    cfg = {"a": {"b": 1}}
    out = apply_overrides(cfg, ["a.b=2"])
    assert out["a"]["b"] == 2


def test_apply_overrides_creates_path():
    cfg = {"a": 1}
    out = apply_overrides(cfg, ["x.y.z=hello"])
    assert out["x"]["y"]["z"] == "hello"


def test_apply_overrides_yaml_typing():
    cfg = {}
    out = apply_overrides(cfg, ["a=2", "b=true", "c=0.01", "d=foo", "e=5e-05"])
    assert out["a"] == 2 and isinstance(out["a"], int)
    assert out["b"] is True
    assert out["c"] == 0.01 and isinstance(out["c"], float)
    assert out["d"] == "foo"
    assert out["e"] == 5e-05 and isinstance(out["e"], float)

