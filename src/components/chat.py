import streamlit as st
import pandas as pd

import os
from src.core.database import DatabaseManager

def render_chat():
    st.header("Local Nexus Intelligence")
    
    # Check for Cloud/Memory mode
    # Streamlit Cloud mounts repos at /mount/src/...
    is_cloud_path = "/mount/src" in os.path.abspath(__file__)
    
    if DatabaseManager().is_in_memory or is_cloud_path:
        st.warning("⚠️ **Cloud Demo Mode**: This instance is running on ephemeral cloud storage. Data will be lost on reboot. For production use, run locally.")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "data" in message:
                st.dataframe(message["data"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Real Gemini Response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from src.core.llm import get_gemini_response
                response_text = get_gemini_response(st.session_state.messages)
                
            st.markdown(response_text)
            st.session_state.messages.append({"role": "assistant", "content": response_text})
