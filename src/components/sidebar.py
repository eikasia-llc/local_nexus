import streamlit as st
from src.core.ingestion import IngestionService

def render_sidebar():
    with st.sidebar:
        st.title("Data Management")
        
        # File Uploader
        uploaded_file = st.file_uploader("Upload CSV or Excel", type=['csv', 'xlsx'])
        if uploaded_file is not None:
            if st.button("Ingest Data"):
                service = IngestionService()
                success, message = service.process_file(uploaded_file)
                if success:
                    st.success(message)
                else:
                    st.error(message)

        st.divider()
        
        # Available Tables
        st.subheader("Active Tables")
        
        db = IngestionService().db # Access the singleton DB via the service or directly
        tables = db.get_active_tables()
        
        if not tables:
            st.info("No tables loaded.")
        else:
            for filename, rows, _ in tables:
                with st.expander(f"📄 {filename}"):
                    st.caption(f"Rows: {rows}")
                    st.button("Preview", key=f"btn_{filename}", help="Preview not implemented yet")
