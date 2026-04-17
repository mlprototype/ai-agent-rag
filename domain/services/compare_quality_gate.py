from typing import Tuple, Literal

class CompareQualityGate:
    """
    Compare Fast-Path 向けの軽量な品質評価・Confidence推定ロジック。
    LLMを呼び出さず、ルールベースで高速に評価する。
    """
    
    @staticmethod
    def evaluate(
        answer: str,
        target_a: str,
        target_b: str,
        doc_count_a: int,
        doc_count_b: int,
        coverage_ok: bool,
        sources_count: int,
        extract_success: bool
    ) -> Tuple[Literal["pass", "warning", "fail"], float, str | None, list[str]]:
        """
        戻り値: (verdict, confidence, warning, missing_aspects)
        verdict: "pass", "warning", "fail"
        """
        missing_aspects = []
        
        # 1. ターゲット言及チェック (大小文字・全角半角の揺れ吸収は簡易的にlowerのみ)
        a_mentioned = target_a.lower() in answer.lower()
        b_mentioned = target_b.lower() in answer.lower()
        
        if not a_mentioned:
            missing_aspects.append(f"{target_a} への言及がありません")
        if not b_mentioned:
            missing_aspects.append(f"{target_b} への言及がありません")
            
        # 2. 構造化チェック (プロンプトで指示している項目)
        has_common = "共通点" in answer
        has_diff = "相違点" in answer or "違い" in answer
        
        if not has_common:
            missing_aspects.append("共通点の記述がありません")
        if not has_diff:
            missing_aspects.append("相違点の記述がありません")
            
        answer_structure_ok = has_common and has_diff
        
        # 3. Confidence 推定
        # ベース: 抽出成功かつカバレッジOKなら高めにスタート
        conf = 0.85 if extract_success and coverage_ok else 0.5
        
        # ペナルティ: ドキュメント不足
        if doc_count_a == 0 or doc_count_b == 0:
            conf -= 0.3
            
        # ペナルティ: ソース数不足
        if sources_count < 2:
            conf -= 0.1
            
        # ペナルティ: 構造不足
        if not answer_structure_ok:
            conf -= 0.15
            
        # ペナルティ: ターゲット言及漏れ
        if not (a_mentioned and b_mentioned):
            conf -= 0.2
            
        conf = max(0.0, min(conf, 1.0))
        
        # 4. Verdict & Warning 判定
        if not a_mentioned or not b_mentioned or conf < 0.4:
            verdict = "fail"
            warning = "This answer may be severely incomplete or incorrect."
        elif not answer_structure_ok or conf < 0.65:
            verdict = "warning"
            warning = "This answer may be incomplete."
        else:
            verdict = "pass"
            warning = None
            
        return verdict, round(conf, 2), warning, missing_aspects
