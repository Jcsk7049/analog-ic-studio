@echo off
REM ============================================================
REM  Analog IC Studio - 一鍵公開上線 (Cloudflare Tunnel)
REM  會開兩個視窗: 後端 + Cloudflare 通道
REM  通道視窗會顯示 https://xxxx.trycloudflare.com 公開網址
REM ============================================================
cd /d "C:\ANALOG AGENT"

echo [1/2] 啟動 Flask 後端 (web_app.py)...
start "Analog IC Studio - Backend" cmd /k "python web_app.py"

echo 等待後端啟動...
timeout /t 6 >nul

echo [2/2] 開啟 Cloudflare 通道...
start "Cloudflare Tunnel" "C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://127.0.0.1:5000

echo.
echo ============================================================
echo  已開啟兩個視窗。
echo  在「Cloudflare Tunnel」視窗中找這一行 (即公開網址):
echo      https://xxxx-xxxx-xxxx.trycloudflare.com
echo  把它貼給任何人 / 用手機開都能連。
echo.
echo  關閉那兩個視窗即下線。每次重開網址會不同 (快速通道特性)。
echo ============================================================
pause
