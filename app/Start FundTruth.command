#!/bin/bash
# Double-click to open FundTruth in your browser (http://localhost:8501).
# The first time it installs Python 3.12 and the libraries (3-5 minutes); after that it starts in a few seconds.
cd "$(dirname "$0")" || exit 1
export PATH="$HOME/.local/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
  echo "Installing uv (Python manager, one time only)..."
  curl -LsSf https://astral.sh/uv/install.sh | sh || { echo "Installation failed: check your internet connection."; read -r; exit 1; }
  export PATH="$HOME/.local/bin:$PATH"
fi
if [ ! -x .venv-app/bin/streamlit ]; then
  echo "Setting up the environment (first time only, a few minutes)..."
  uv venv --python 3.12 .venv-app && uv pip install --python .venv-app/bin/python -r requirements.txt \
    || { echo "Installing the libraries failed."; read -r; exit 1; }
fi
echo ""
echo "FundTruth is opening in your browser: http://localhost:8501"
echo "To close the app: close this window or press Ctrl + C."
(sleep 6; open http://localhost:8501) &
.venv-app/bin/streamlit run app.py --server.port 8501 --server.headless true --browser.gatherUsageStats false
