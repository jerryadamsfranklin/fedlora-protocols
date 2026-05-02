from scripts.run_experiment import apply_overrides


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
    out = apply_overrides(cfg, ["a=2", "b=true", "c=0.01", "d=foo"])
    assert out["a"] == 2 and isinstance(out["a"], int)
    assert out["b"] is True
    assert out["c"] == 0.01 and isinstance(out["c"], float)
    assert out["d"] == "foo"

