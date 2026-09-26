from nerlex.hashing import canonical_json, sha256_hex


def test_canonical_hash_is_key_order_independent() -> None:
    left = {"b": 2, "a": {"y": 2, "x": 1}}
    right = {"a": {"x": 1, "y": 2}, "b": 2}

    assert canonical_json(left) == canonical_json(right)
    assert sha256_hex(left) == sha256_hex(right)
