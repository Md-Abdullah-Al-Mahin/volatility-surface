"""
Entry point: runs the live volatility surface dashboard.

Usage:
    python main.py          # same as: streamlit run dashboard.py
    streamlit run dashboard.py
"""

import sys
from pathlib import Path

if __name__ == "__main__":
    # Run the dashboard via Streamlit
    import streamlit.web.cli as stcli
    dashboard_path = Path(__file__).resolve().parent / "dashboard.py"
    sys.argv = ["streamlit", "run", str(dashboard_path), "--server.headless", "true"]
    stcli.main()
