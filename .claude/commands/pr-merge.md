# 現在のブランチの Pull Request をマージする

## 手順

1. `gh pr view --json state,mergeable,title` で現在のブランチのPR状態を確認する
2. PRがマージ可能でない場合はその旨を伝えて終了する
3. `gh pr merge --squash --delete-branch` でマージする
4. `git checkout main && git pull` で main に切り替えて最新化する