import sys
import os

# Robustly find the project root
# If __file__ is src/app.py, then dirname is src, and dirname(dirname) is root
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)

# DEBUG: Print paths to helps us see what's wrong in Cloud logs if it fails again
print(f"DEBUG: Current Directory: {current_dir}")
print(f"DEBUG: Project Root: {project_root}")
print(f"DEBUG: sys.path: {sys.path}")

import streamlit as st
from src.components.sidebar import render_sidebar
from src.components.chat import render_chat

def main():
    st.set_page_config(
        page_title="Local Nexus",
        page_icon="🧠",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "active_tables" not in st.session_state:
        st.session_state.active_tables = []

    # Layout
    render_sidebar()
    render_chat()

if __name__ == "__main__":
    main()
