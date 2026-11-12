import yaml

from apex.config import config_hash, deep_merge, load_config


def test_deep_merge_nested():
    out = deep_merge({"a": {"b": 1, "c": 2}, "d": 1}, {"a": {"c": 3}, "e": 4})
    assert out == {"a": {"b": 1, "c": 3}, "d": 1, "e": 4}


def test_hash_stable_and_sensitive():
    assert config_hash({"a": 1, "b": 2}) == config_hash({"b": 2, "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def test_base_has_no_hardcoded_gaps():
    cfg = load_config()
    for key in ("generator", "embedder", "router", "judge"):
        assert cfg["models"][key]
    assert cfg["generation"]["temperature"] == 0.0
    assert cfg["_meta"]["config_hash"]


def test_overlay_and_extends(tmp_path):
    mid = tmp_path / "mid.yaml"
    mid.write_text(yaml.safe_dump({"retrieval": {"top_k": 9}}))
    top = tmp_path / "top.yaml"
    top.write_text(yaml.safe_dump({"extends": "mid.yaml", "generation": {"max_tokens": 64}}))
    cfg = load_config(str(top))
    assert cfg["retrieval"]["top_k"] == 9
    assert cfg["generation"]["max_tokens"] == 64
    assert cfg["models"]["embedder"] == "nomic-embed-text"  # inherited from base
