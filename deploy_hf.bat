@echo off
REM ============================================================
REM  一鍵推送 Analog IC Studio 到 Hugging Face Space
REM  Space: LANCELOT7049/analog-ic-studio  (Docker SDK)
REM
REM  推送時會問帳密:
REM    Username : LANCELOT7049
REM    Password : 你的 HF Access Token (write 權限)
REM               -> https://huggingface.co/settings/tokens 產生
REM ============================================================
cd /d "C:\ANALOG AGENT"

echo [1/2] 設定 Hugging Face 遠端...
git remote remove hf 2>nul
git remote add hf https://huggingface.co/spaces/LANCELOT7049/analog-ic-studio

echo [2/2] 推送 main 分支到 Space (帳號 LANCELOT7049, 密碼貼 Access Token)...
git push hf main

echo.
echo ============================================================
echo  推送完成。回到 Space 頁面會自動開始 build Docker (約 3-5 分鐘)。
echo  build 完成後永久網址:
echo      https://huggingface.co/spaces/LANCELOT7049/analog-ic-studio
echo  之後更新只要再跑這個檔, 或 git push hf main。
echo ============================================================
pause
