import os
from typing import List
from unstructured.partition.html import partition_html
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text
from unstructured.cleaners.core import clean, clean_extra_whitespace, replace_unicode_quotes

class DocumentIngestionPipeline:
    """
    ドキュメントのロード、パース、クリーンアップを行う一連のパイプラインコンポーネント。
    """

    @staticmethod
    def load(file_path: str) -> str:
        """
        1. ローダー: ファイルの存在確認とパスの返却を行います（PoC用）。
        （実運用ではここでS3からのダウンロードやバイナリのメモリ展開を行います）
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"ファイルが見つかりません: {file_path}")
        return file_path

    @staticmethod
    def parse(file_path: str) -> List[str]:
        """
        2. パーサー: unstructuredを利用して各フォーマットからテキスト要素を抽出します。
        """
        ext = os.path.splitext(file_path)[1].lower()
        if ext in ['.md', '.markdown']:
            elements = partition_md(filename=file_path)
        elif ext in ['.html', '.htm']:
            elements = partition_html(filename=file_path)
        elif ext in ['.txt']:
            elements = partition_text(filename=file_path)
        else:
            raise ValueError(f"サポートされていないファイル形式です: {ext}")
        
        # unstructuredのElementからテキスト文字列のみを抽出
        return [str(el) for el in elements if str(el).strip()]

    @staticmethod
    def clean(parsed_elements: List[str]) -> str:
        """
        3. クリーナー: 抽出されたテキスト要素を結合し、不要な空白や文字を正規化します。
        """
        raw_text = "\n\n".join(parsed_elements)
        # 不要な制御文字や余分な空白を削除し、ユニコードクォートを正規化
        cleaned_text = clean(raw_text, extra_whitespace=True, dashes=True, bullets=True, trailing_punctuation=False)
        cleaned_text = clean_extra_whitespace(cleaned_text)
        cleaned_text = replace_unicode_quotes(cleaned_text)
        
        return cleaned_text
