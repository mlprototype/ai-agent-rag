import os

# 1. 除外するフォルダ（ここにあるものは中身を見ない）
IGNORE_DIRS = {
    ".git",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".idea",
    ".vscode",
    "node_modules",
    "build",
    "dist",
    ".DS_Store",
    ".pytest_cache",
    ".gemini"
}

# 2. 読み込むファイルの拡張子
TARGET_EXTS = {".py", ".md", ".txt", ".yml", ".yaml", ".json", ".sh"}

# 3. 出力ファイル名
OUTPUT_FILE = "_all_project_code.txt"

def merge_files():
    # .envファイル等の機密情報やバイナリが含まれないようにセキュリティ対策
    EXCLUDE_FILES = {
        OUTPUT_FILE, 
        "merge_code.py", 
        ".env", 
        "uv.lock",
        "docker-compose.yml" # もし含めたければ外す
    }

    with open(OUTPUT_FILE, "w", encoding="utf-8") as outfile:
        # プロジェクトのルートディレクトリから探索開始
        for root, dirs, files in os.walk("."):
            # 除外ディレクトリを探索対象から外す
            dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

            for file in files:
                # ファイル名での除外チェック
                if file in EXCLUDE_FILES:
                    continue
                
                # 拡張子チェック
                ext = os.path.splitext(file)[1]
                if ext in TARGET_EXTS:
                    file_path = os.path.join(root, file)

                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            content = infile.read()

                            # AIが読みやすいヘッダーを付与
                            outfile.write(f"\n\n{'=' * 50}\n")
                            outfile.write(f"FILE_PATH: {file_path}\n")
                            outfile.write(f"{'=' * 50}\n")
                            outfile.write(content + "\n")

                        print(f"Added: {file_path}")
                    except Exception as e:
                        print(f"Skipped (Error): {file_path} - {e}")

    print(f"\nDone! すべてのコードを結合しました: {OUTPUT_FILE}")

if __name__ == "__main__":
    merge_files()
