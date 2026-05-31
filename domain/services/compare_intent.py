import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CompareTargets:
    target_a: str
    target_b: str
    aspect: Optional[str]

# The order of patterns matters. More specific ones first.
_PATTERNS = [
    # どちら・どっち・べき・向いている
    re.compile(
        r"(?P<a>.+?)\s*[とど]\s*(?P<b>.+?)\s*(?:は|では|って|を)?\s*.*"
        r"(?P<aspect>どちら|どっち|べき|向いている)"
    ),
    # AとBのそれぞれの〇〇 / AとBの〇〇の違い
    re.compile(
        r"(?P<a>.+?)\s*[とど]\s*(?P<b>.+?)\s*の?\s*それぞれの?\s*"
        r"(?P<aspect>違い|差|比較|使い分け|特徴|メリット|デメリット)"
    ),
    re.compile(
        r"(?P<a>.+?)\s*[とど]\s*(?P<b>.+?)\s*の?\s*"
        r"(?P<aspect>違い|差|比較|使い分け|特徴|メリット|デメリット)"
    ),
    # A vs B
    re.compile(
        r"(?P<a>.+?)\s*(?:vs\.?|versus|VS|Vs|v\.s\.|vs)\s*(?P<b>.+)"
    ),
    # AとBを比較して
    re.compile(
        r"(?P<a>.+?)\s*[とど]\s*(?P<b>.+?)\s*を?\s*"
        r"(?P<aspect>比較|比べ)"
    ),
]


def extract_targets(query: str) -> Optional[CompareTargets]:
    """
    比較クエリから対象 A/B と比較観点を抽出する。
    抽出できない場合は None を返す（既存パスへのフォールバック用）。
    """
    normalized = query.strip()
    if not normalized:
        return None

    # Try mapping patterns
    for pattern in _PATTERNS:
        m = pattern.search(normalized)
        if m:
            groups = m.groupdict()
            a = groups.get("a", "").strip()
            # For B, it might capture the rest of the string including extra particles
            # like "Bの違いを教えて" if not careful.
            b = groups.get("b", "").strip()
            
            # Clean up trailing particles for better retrieval quality
            for particle in ["を教えてください", "を教えて", "って何", "について", "を"]:
                if b.endswith(particle):
                    b = b[: -len(particle)].strip()
            
            if a and b and len(a) <= 50 and len(b) <= 50:
                return CompareTargets(
                    target_a=a,
                    target_b=b,
                    aspect=groups.get("aspect"),
                )

    return None
