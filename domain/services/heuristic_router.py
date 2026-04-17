import re
from typing import Literal
from pydantic import BaseModel

from domain.services.compare_intent import extract_targets

class RouteDecision(BaseModel):
    query_type: Literal["direct", "calc", "compare", "definition", "retrieval_complex"]
    route: Literal["direct_answer", "calculator", "agentic_retrieval", "fallback_retrieval"]
    routing_layer: Literal["heuristic", "llm", "fallback"]
    source: Literal["heuristic_match", "llm_success", "llm_timeout_fallback", "llm_error_fallback"]
    confidence: float
    heuristic_matched: bool
    heuristic_rule: str
    llm_router_invoked: bool
    reason: str


class HeuristicRouter:
    _CALC_PATTERN = re.compile(r"^[\d\s\.\(\)\+\-\*/%]+(は|は？|です|ですか|はいくつ)?$")
    _CALC_SYMBOL_PATTERN = re.compile(r"[\+\-\*/%]")
    _CALC_JP_PATTERN = re.compile(r"^[\d\s\.]+(足す|たす|引く|ひく|かける|わる|割る|プラス|マイナス)[\d\s\.]+(は|は？|です|ですか|はいくつ)?$")
    
    _DIRECT_GREETINGS = [
        "こんにちは", "こんばんは", "おはよう", "ありがとう", "thank you", "thanks", "hello", "hi"
    ]
    
    _COMPARE_KEYWORDS = ["違い", "比較", "差", "使い分け", "vs", "versus", "メリット", "デメリット", "どちら", "どっち", "べき", "向いている"]
    _DEFINITION_KEYWORDS = ["とは何", "とは", "って何", "の意味", "定義"]

    @classmethod
    def route(cls, query: str, enable_compare: bool = True) -> RouteDecision | None:
        normalized = query.strip().lower()
        if not normalized:
            return cls._build_decision("direct", "direct_answer", "empty_query", 1.0)
            
        # 1. Direct (Greetings/Short)
        if len(normalized) <= 20 and any(normalized == g or normalized.startswith(g) for g in cls._DIRECT_GREETINGS):
            # Avoid cases like "こんにちは、RAGについて教えて"
            if len(normalized) <= 15 and "教えて" not in normalized and "について" not in normalized:
                return cls._build_decision("direct", "direct_answer", "direct_greeting", 0.9)

        # 2. Calc
        calc_candidate = "".join(re.findall(r"[\d\s\.\(\)\+\-\*/%]+", query)).strip()
        if cls._CALC_JP_PATTERN.match(normalized):
            return cls._build_decision("calc", "calculator", "calc_expression_jp", 1.0)
        elif calc_candidate and any(ch.isdigit() for ch in calc_candidate) and cls._CALC_SYMBOL_PATTERN.search(calc_candidate):
            # If the user's explicit intent is just to calculate this formula
            if cls._CALC_PATTERN.match(normalized):
                return cls._build_decision("calc", "calculator", "calc_expression", 1.0)

        # 3. Compare
        if enable_compare:
            has_compare = any(k in normalized for k in cls._COMPARE_KEYWORDS)
            has_and = any(k in normalized for k in ["と", "vs", "and"])
            if has_compare and has_and and "併用" not in normalized:
                # Need stronger heuristics to avoid false positives, e.g. "AとBの違い" 
                # Very simple heuristic: length < 50
                if len(normalized) < 50:
                    targets = extract_targets(query)
                    if targets is not None:
                        return cls._build_decision("compare", "agentic_retrieval", "compare_keywords", 0.9)

        # 4. Definition
        # "Xとは何ですか", "Xとは"
        has_def = any(k in normalized for k in cls._DEFINITION_KEYWORDS)
        if has_def:
            # "教えて" only if definition keyword is also present
            # Avoid "RAGのチューニングポイントを教えてください" -> no "とは"
            if len(normalized) < 40 and "ポイント" not in normalized:
                return cls._build_decision("definition", "agentic_retrieval", "definition_keywords", 0.85)

        return None

    @classmethod
    def _build_decision(cls, query_type: str, route: str, rule: str, confidence: float) -> RouteDecision:
        return RouteDecision(
            query_type=query_type, # type: ignore
            route=route, # type: ignore
            routing_layer="heuristic",
            source="heuristic_match",
            confidence=confidence,
            heuristic_matched=True,
            heuristic_rule=rule,
            llm_router_invoked=False,
            reason=f"Matched heuristic rule: {rule}"
        )
