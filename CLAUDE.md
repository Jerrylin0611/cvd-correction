# 給 Claude 的專案指示

## Git 工作流程

當使用者要求修改程式碼、修 bug、或做任何檔案異動之後，**主動**執行：

```
git add <實際改到的檔案>
git commit -m "<說明改了什麼跟為什麼>"
git push
```

不需要每次都先問「要不要 commit/push」——這件事已經事先授權了。
只有以下情況才要先跟使用者確認：
- 需要 `git push --force`、`git reset --hard`、改寫歷史等破壞性操作
- 這次改動包含 `data/`、`checkpoints*/` 等 `.gitignore` 排除的大型/可重現檔案，
  使用者可能想單獨處理，不要硬塞進 commit
- 不確定該不該把某個新產生的檔案(例如新的實驗腳本、log)一起加進版控時

## 專案背景

色弱輔助即時校正系統，細節見 [README.md](README.md)。多人協作專案(見
README「使用方式」)，`data/` 跟 `checkpoints*/` 不進版控，clone 下來的人
需要自己準備訓練資料/重新訓練或另外拿到權重檔。
