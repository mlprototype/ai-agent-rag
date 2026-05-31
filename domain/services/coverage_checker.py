import re
from dataclasses import dataclass, field


_COMPARE_PATTERNS = (
    re.compile(r"(?P<left>.+?)と(?P<right>.+?)(?:の)?(?:違い|比較|差|使い分け)", re.IGNORECASE),
    re.compile(r"(?P<left>.+?)\s+vs\.?\s+(?P<right>.+)", re.IGNORECASE),
    re.compile(r"(?P<left>.+?)\s+versus\s+(?P<right>.+)", re.IGNORECASE),
    re.compile(r"(?P<left>.+?)と(?P<right>.+?)を比較", re.IGNORECASE),
)
_LOOKUP_SPLITTERS = (
    "とは何ですか",
    "とは",
    "について教えてください",
    "について",
    "を教えてください",
    "を教えて",
    "って何ですか",
    "って何",
)
_NOISE_SUFFIXES = (
    "を教えてください",
    "を教えて",
    "について教えてください",
    "について",
    "とは何ですか",
    "とは",
    "って何ですか",
    "って何",
    "の違い",
    "の比較",
    "を比較",
    "比較",
    "違い",
    "差",
    "使い分け",
    "ですか",
    "でしょうか",
    "は何ですか",
    "は？",
    "は?",
    "？",
    "?",
)
_AXIS_KEYWORDS = {
    "違い": ("違い", "比較", "差", "使い分け", "vs", "versus"),
    "定義": ("とは", "定義", "何ですか", "意味"),
    "仕組み": ("仕組み", "どう動く", "動作", "仕組"),
    "ユースケース": ("ユースケース", "用途", "使いどころ", "使う場面", "活用"),
}
_COMPARE_MARKERS = tuple(_AXIS_KEYWORDS["違い"]) + ("一方", "対して", "比べて")
_INSUFFICIENT_MARKERS = (
    "情報が不足",
    "情報不足",
    "根拠が不足",
    "ソースが不足",
    "判断できません",
    "比較できません",
    "確認できません",
    "見つかりません",
    "不明",
    "わかりません",
)
_ALIAS_GROUPS = (
    ("RAG", "Retrieval-Augmented Generation", "retrieval augmented generation", "検索拡張生成"),
    ("Fine-tuning", "fine tuning", "fine-tuning", "finetuning", "fine tune", "fine-tune", "ファインチューニング", "微調整", "追加学習"),
)


@dataclass(frozen=True)
class CoveragePlan:
    entities: list[str] = field(default_factory=list)
    intent: str = "lookup"
    comparison_axes: list[str] = field(default_factory=list)
    expected_aspects: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CoverageAssessment:
    covered_entities: list[str] = field(default_factory=list)
    missing_entities: list[str] = field(default_factory=list)
    covered_axes: list[str] = field(default_factory=list)
    missing_axes: list[str] = field(default_factory=list)
    covered_aspects: list[str] = field(default_factory=list)
    missing_aspects: list[str] = field(default_factory=list)
    required_missing_aspects: list[str] = field(default_factory=list)
    coverage_score: float = 0.0
    summary: str = ""


def _dedupe(items: list[str]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for item in items:
        cleaned = item.strip()
        if not cleaned:
            continue
        key = cleaned.lower()
        if key in seen:
            continue
        seen.add(key)
        ordered.append(cleaned)
    return ordered


def _normalize(text: str) -> str:
    lowered = text.strip().lower()
    lowered = re.sub(r"[\s\-_:/\\\[\]\(\)\{\}（）「」『』'\"`.,!?！？・]+", "", lowered)
    return lowered


def _aspect_variants(aspect: str) -> list[str]:
    normalized_aspect = _normalize(aspect)
    variants = [aspect]
    for group in _ALIAS_GROUPS:
        normalized_group = {_normalize(item) for item in group}
        if normalized_aspect in normalized_group:
            variants.extend(group)
            break
    return _dedupe(variants)


def _cleanup_entity(text: str) -> str:
    cleaned = text.strip(" 　\n\t:：,、/()（）[]{}\"'`")
    for suffix in _NOISE_SUFFIXES:
        if cleaned.endswith(suffix):
            cleaned = cleaned[: -len(suffix)].strip()
    return cleaned.strip(" 　\n\t:：,、/()（）[]{}\"'`")


def _extract_compare_entities(query: str) -> list[str]:
    for pattern in _COMPARE_PATTERNS:
        match = pattern.search(query)
        if not match:
            continue
        left = _cleanup_entity(match.group("left"))
        right = _cleanup_entity(match.group("right"))
        return _dedupe([left, right])
    return []


def _extract_lookup_entity(query: str) -> list[str]:
    for splitter in _LOOKUP_SPLITTERS:
        if splitter not in query:
            continue
        candidate = _cleanup_entity(query.split(splitter, 1)[0])
        if candidate:
            return [candidate]
    return []


def build_coverage_plan(query: str) -> CoveragePlan:
    entities = _extract_compare_entities(query)
    intent = "compare" if len(entities) >= 2 else "lookup"

    if not entities:
        entities = _extract_lookup_entity(query)

    comparison_axes: list[str] = []
    lowered = query.lower()
    if intent == "compare":
        comparison_axes.append("違い")
        comparison_axes.append("ユースケース")
        if any(keyword in lowered for keyword in _AXIS_KEYWORDS["定義"]):
            comparison_axes.append("定義")
        if any(keyword in lowered for keyword in _AXIS_KEYWORDS["仕組み"]):
            comparison_axes.append("仕組み")

    expected_aspects = _dedupe(entities + comparison_axes)

    return CoveragePlan(
        entities=_dedupe(entities),
        intent=intent,
        comparison_axes=_dedupe(comparison_axes),
        expected_aspects=expected_aspects,
    )


def aspect_in_text(aspect: str, text: str) -> bool:
    if not aspect or not text:
        return False

    normalized_text = _normalize(text)
    for variant in _aspect_variants(aspect):
        normalized_variant = _normalize(variant)
        if normalized_variant and normalized_variant in normalized_text:
            return True

    lowered_text = text.lower()
    return any(variant.lower() in lowered_text for variant in _aspect_variants(aspect))


def answer_uses_comparison_language(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _COMPARE_MARKERS + ("異なる", "異なり", "異なります", "違います", "それぞれ"))


def answer_admits_insufficient_support(answer: str) -> bool:
    lowered = answer.lower()
    return any(marker in lowered for marker in _INSUFFICIENT_MARKERS)


def aspects_mentioned_in_text(aspects: list[str], text: str) -> list[str]:
    return [aspect for aspect in aspects if aspect_in_text(aspect, text)]


def assess_coverage(plan: CoveragePlan, text: str) -> CoverageAssessment:
    covered_entities = [entity for entity in plan.entities if aspect_in_text(entity, text)]
    missing_entities = [entity for entity in plan.entities if entity not in covered_entities]

    covered_axes: list[str] = []
    missing_axes: list[str] = []
    lowered = text.lower()
    for axis in plan.comparison_axes:
        keywords = _AXIS_KEYWORDS.get(axis, (axis,))
        if any(keyword in lowered for keyword in keywords):
            covered_axes.append(axis)
        else:
            missing_axes.append(axis)

    covered_aspects = _dedupe(covered_entities + covered_axes)
    missing_aspects = _dedupe(missing_entities + missing_axes)
    required_missing_aspects = _dedupe(missing_entities)

    if plan.entities:
        entity_ratio = len(covered_entities) / max(1, len(plan.entities))
    else:
        entity_ratio = 1.0 if text.strip() else 0.0

    if plan.comparison_axes:
        axis_ratio = len(covered_axes) / max(1, len(plan.comparison_axes))
        coverage_score = round(min(1.0, 0.75 * entity_ratio + 0.25 * axis_ratio), 2)
    else:
        coverage_score = round(entity_ratio, 2)

    entity_summary = ", ".join(
        f"{entity}:{'ok' if entity in covered_entities else 'missing'}"
        for entity in plan.entities
    ) or "none"
    axis_summary = ", ".join(
        f"{axis}:{'ok' if axis in covered_axes else 'missing'}"
        for axis in plan.comparison_axes
    ) or "none"

    return CoverageAssessment(
        covered_entities=covered_entities,
        missing_entities=missing_entities,
        covered_axes=covered_axes,
        missing_axes=missing_axes,
        covered_aspects=covered_aspects,
        missing_aspects=missing_aspects,
        required_missing_aspects=required_missing_aspects,
        coverage_score=coverage_score,
        summary=f"intent={plan.intent}; entities={entity_summary}; axes={axis_summary}",
    )
