@echo off
chcp 65001 >nul
cd /d "C:\ANALOG AGENT"
git remote remove hf 2>nul
git remote add hf https://huggingface.co/spaces/LANCELOT7049/analog-ic-studio
echo Username 輸入 LANCELOT7049 , Password 貼上 HF write token
git push hf main
pause
