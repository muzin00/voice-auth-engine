# 対話形式で Git ブランチを作成・切り替えする

## 手順

1. AskUserQuestion でブランチの目的をヒアリングする:
   - ブランチの種類（feature / fix / refactor / chore / docs など）
   - 変更内容の簡潔な説明

2. ヒアリング内容からブランチ名を生成する:
   - 形式: `<種類>/<kebab-case の説明>`
   - 例: `feature/add-voice-recognition`, `fix/login-timeout-error`
   - 英語・小文字・kebab-case で統一

3. 生成したブランチ名をユーザーに提示し、確認を取る

4. 承認後、`git checkout -b <ブランチ名>` を実行する
