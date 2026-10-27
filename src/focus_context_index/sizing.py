# Derived from Leeminjing/Focus, Apache-2.0, commit 08fe98c28896992c6195912e95b507599b9dcd76.
# See NOTICE and docs/upstream-manifest.json for provenance and modifications.
_TOKENS_PER_MESSAGE = 12


def estimate_raw_tokens(raw: str, message_count: int) -> int:
    cjk = sum(1 for char in raw if "㐀" <= char <= "鿿")
    other = sum(1 for char in raw if not char.isspace() and not ("㐀" <= char <= "鿿"))
    return cjk + (other + 3) // 4 + _TOKENS_PER_MESSAGE * (message_count + 2)


def exceeds_limit(value: int, limit: int | None) -> bool:
    return limit is not None and value > limit
