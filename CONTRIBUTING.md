# 📘 如何參與本專案貢獻（Contributing Guide）

感謝你有興趣參與本專案的開發！本文件將指引你從安裝 Git 到提交 Pull Request（PR）的完整流程。

---

## 📦 1. 安裝 Git

請先安裝 Git，如果你已經安裝過，請跳至下一步。

### Windows：

1. 下載 Git 安裝程式：[https://git-scm.com/download/win](https://git-scm.com/download/win)
2. 安裝時保持預設選項即可。

### macOS or Linux：

```bash
# macOS
brew install git

# Ubuntu / Debian
sudo apt update && sudo apt install git

# Arch / Manjaro
sudo pacman -S git
```

## 2. 設定 Git 基本資訊 config
```bash
git config --global user.name "你的名稱"
git config --global user.email "你的Email"
```

## 3. Fork 專案
打開 GitHub 上的本專案頁面。

點選右上角的 Fork，建立一份專案副本到你的帳號。

## 4. Clone 你的專案副本
git clone https://github.com/JiaLong0209/loan-risk-predictor.git
cd 本專案名稱

## 5. 設定遠端原始倉庫（upstream）

這是為了日後可以同步主專案的更新。

``` bash

git remote add upstream https://github.com/JiaLong0209/loan-risk-predictor.git

#確認遠端設定：

git remote -v

``` 

## 6. 建立分支開始開發
不要在 main 或 master 分支直接開發，請建立新的分支：

``` bash
git checkout -b feat/新增功能名稱
``` 

## 7. 提交變更
編輯程式碼後，先將變更加入版本控制：

``` bash
git add .

# 提交：
git commit -m "feat: 新增 XX 功能"

#  Push 到你自己的 GitHub 倉庫
git push origin feat/新增功能名稱

```
## 8. 在 GitHub 上發出 Pull Request（PR）

到你的 GitHub 倉庫。

會看到有個「Compare & pull request」的按鈕，點它。

填寫說明後送出 PR。

等待專案擁有者審核合併。

## 9. 與主專案保持同步（更新 upstream）
請定期從主專案（upstream）拉取更新，確保你的分支是最新的：

```bash
git checkout main
git pull upstream main
git push origin main
```

## 10. 刪除已合併的本地與遠端分支（可選）
``` bash
# 本地分支
git branch -d feat/新增功能名稱

# 遠端分支
git push origin --delete feat/新增功能名稱
```

## 補充建議
每次開發新功能或修正錯誤都請開新分支。

Commit message 請盡量清楚、符合語意化命名，例如：

feat: 新增貸款風險模型比較功能

fix: 修正模型準確率計算錯誤

遇到衝突時請自行解決後再提交。


