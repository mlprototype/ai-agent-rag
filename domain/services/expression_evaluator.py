# ファイルの責務: 四則演算と括弧を含む数式の決定論的な評価
# 主な入出力: 数式文字列を受け取り、評価結果（float）を返す
# 特に重要な副作用や設計上の注意点: ast.parse を使用して許可された構文のみを評価し、任意コード実行を防止する
# calc と structured_query の境界: 本サービスは純粋な数式評価のみを担当し、データセットに基づく集計は扱わない

import ast
import operator
import re

_BINARY_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

def calculate_expression(expression: str) -> float:
    """
    四則演算と括弧を安全に評価する。
    
    何を受け取り、何を返すか:
    - expression (str): 数式文字列（例: "2 + 3 * 4", "1 + 1"）
    - 戻り値 (float): 評価結果
    
    設計上の注意点:
    - 日本語の演算子を標準的な記号に置換してから評価する。
    - 許可されていない構文（関数呼び出し、属性アクセス等）が含まれる場合は ValueError を投げる。
    """
    # 日本語演算子の簡易置換
    processed = expression.replace("足す", "+").replace("たす", "+").replace("プラス", "+")
    processed = processed.replace("引く", "-").replace("ひく", "-").replace("マイナス", "-")
    processed = processed.replace("かける", "*").replace("掛ける", "*").replace("タイムズ", "*")
    processed = processed.replace("割る", "/").replace("わる", "/").replace("スラッシュ", "/")
    processed = processed.replace("＝", "=").replace("？", "?")
    
    # 計算に関係ない記号を削除
    processed = re.sub(r"[=\?？]", "", processed).strip()

    def _eval(node):
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPERATORS:
            return _BINARY_OPERATORS[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPERATORS:
            return _UNARY_OPERATORS[type(node.op)](_eval(node.operand))
        # 任意コード実行を避けるため、上記以外は一切許可しない
        raise ValueError(f"Unsupported expression component: {type(node).__name__}")

    try:
        parsed = ast.parse(processed, mode="eval")
        value = _eval(parsed)
        return float(value)
    except (SyntaxError, ZeroDivisionError, ValueError) as e:
        raise ValueError(f"Failed to evaluate expression: {str(e)}")
