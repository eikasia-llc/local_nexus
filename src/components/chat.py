import streamlit as st
import os
from src.core.database import DatabaseManager


@st.cache_resource
def get_db_connection():
    """Cached database connection."""
    return DatabaseManager().get_connection()

@st.cache_resource
def get_vector_store():
    """Cached vector store."""
    from src.core.vector_store import VectorStore
    return VectorStore(db_path="data/vectordb")

@st.cache_resource
def get_graph_store():
    """Cached graph store."""
    from src.core.graph_store import InstitutionalGraph
    return InstitutionalGraph(storage_path="data/graph")

@st.cache_resource
def get_llm_func():
    """Cached LLM function."""
    import google.generativeai as genai
    from src.core.llm import init_gemini
    if init_gemini():
        def gemini_call(prompt: str) -> str:
            model = genai.GenerativeModel('gemini-flash-latest')
            response = model.generate_content(prompt)
            return response.text
        return gemini_call
    return None

@st.cache_resource(show_spinner="Initializing Unified Engine...")
def get_engine_instance():
    """Cached Unified Engine instance."""
    try:
        from src.core.unified_engine import UnifiedEngine
        
        # Get components (cached automatically by their own decorators)
        db_conn = get_db_connection()
        try:
            vector_store = get_vector_store()
        except Exception:
            vector_store = None
            
        try:
            graph_store = get_graph_store()
        except Exception:
            graph_store = None
            
        llm_func = get_llm_func()

        engine = UnifiedEngine(
            vector_store=vector_store,
            db_connection=db_conn,
            graph_store=graph_store,
            llm_func=llm_func
        )
        print("DEBUG: UnifiedEngine initialized successfully (Cached).")
        return engine
    except Exception as e:
        print(f"ERROR: UnifiedEngine initialization failed: {e}")
        return None

def get_unified_engine():
    """Wrapper to get the cached engine."""
    return get_engine_instance()


def render_chat():
    st.header("Local Nexus Intelligence")

    # Check for Cloud/Memory mode
    is_cloud_path = "/mount/src" in os.path.abspath(__file__)

    if DatabaseManager().is_in_memory or is_cloud_path:
        st.warning("Cloud Demo Mode: Data will be lost on reboot. For production use, run locally.")

    # Engine mode toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        use_unified = st.toggle("Unified Engine", value=True, help="Use RAG + SQL routing")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Show query metadata if available
            if "query_type" in message:
                render_query_badge(message["query_type"])

            if "sql_query" in message and message["sql_query"]:
                with st.expander("SQL Query"):
                    st.code(message["sql_query"], language="sql")

            if "sources" in message and message["sources"]:
                with st.expander(f"Sources ({len(message['sources'])})"):
                    for i, src in enumerate(message["sources"][:3]):
                        st.caption(f"[{src.source}] {src.content[:200]}...")

    # Chat input
    if prompt := st.chat_input("Ask a question about your data..."):
        # Add user message to history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            if use_unified:
                response = generate_unified_response(prompt)
            else:
                response = generate_simple_response(prompt)

            st.markdown(response["content"])

            # Show metadata
            if response.get("query_type"):
                render_query_badge(response["query_type"])

            if response.get("sql_query"):
                with st.expander("SQL Query"):
                    st.code(response["sql_query"], language="sql")

            # Save to history
            st.session_state.messages.append({
                "role": "assistant",
                "content": response["content"],
                "query_type": response.get("query_type"),
                "sql_query": response.get("sql_query"),
                "sources": response.get("sources", [])
            })


def render_query_badge(query_type: str):
    """Render a badge showing the query type."""
    badges = {
        "structured": ("SQL", "blue"),
        "unstructured": ("RAG", "green"),
        "hybrid": ("Hybrid", "orange"),
        "error": ("Error", "red")
    }

    label, color = badges.get(query_type, ("Unknown", "gray"))
    st.caption(f"Query type: **{label}**")


def generate_unified_response(prompt: str) -> dict:
    """Generate response using the Unified Engine."""
    engine = get_unified_engine()

    if not engine:
        return {
            "content": "Unified Engine not available. Please check your configuration.",
            "query_type": "error"
        }

    with st.spinner("Analyzing query..."):
        try:
            print(f"DEBUG: calling engine.query with prompt: {prompt}")
            result = engine.query(prompt)
            print(f"DEBUG: engine.query returned: {result}")

            if result.error:
                return {
                    "content": f"Error: {result.error}",
                    "query_type": "error"
                }

            return {
                "content": result.answer,
                "query_type": result.query_type,
                "sql_query": result.sql_query,
                "sources": result.sources
            }

        except Exception as e:
            return {
                "content": f"An error occurred: {str(e)}",
                "query_type": "error"
            }


def generate_simple_response(prompt: str) -> dict:
    """Generate response using simple Gemini chat (fallback)."""
    with st.spinner("Thinking..."):
        try:
            from src.core.llm import get_gemini_response

            # Build message history for Gemini
            messages = st.session_state.messages + [{"role": "user", "content": prompt}]
            response_text = get_gemini_response(messages)

            return {"content": response_text}

        except Exception as e:
            return {
                "content": f"Error communicating with LLM: {str(e)}",
                "query_type": "error"
            }
