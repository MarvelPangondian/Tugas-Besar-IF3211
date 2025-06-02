#!/usr/bin/env python3
"""
Streamlit App for Marine Ecosystem Simulation

This is the main Streamlit application file that should be run directly
with: streamlit run streamlit_app.py
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.interface import MarineEcosystemInterface

def main():
    """Main function to run the Streamlit interface."""
    interface = MarineEcosystemInterface()
    interface.run()

if __name__ == "__main__":
    main()