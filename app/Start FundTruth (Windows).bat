@echo off
rem Double-click to open FundTruth in your browser (http://localhost:8501).
rem The first time it installs Python 3.12 and the libraries (3-5 minutes); after that it starts in a few seconds.
cd /d "%~dp0"
set "PATH=%USERPROFILE%\.local\bin;%PATH%"
where uv >nul 2>nul
if errorlevel 1 (
  echo Installing uv, the Python manager - one time only...
  powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
)
if not exist ".venv-app\Scripts\streamlit.exe" (
  echo Setting up the environment - first time only, a few minutes...
  uv venv --python 3.12 .venv-app || goto :err
  uv pip install --python .venv-app\Scripts\python.exe -r requirements.txt || goto :err
)
echo.
echo FundTruth is opening in your browser: http://localhost:8501
echo To close the app: close this window.
start "" cmd /c "timeout /t 6 >nul & start http://localhost:8501"
.venv-app\Scripts\streamlit.exe run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
goto :eof
:err
echo Installation failed: check your internet connection and try again.
pause
