import os
import google.generativeai as genai
from dotenv import load_dotenv

def init_gemini():
    """Initializes the Gemini API with the key from environment variables."""
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY not found in environment variables.")
        return False
    
    genai.configure(api_key=api_key)
    return True

def get_gemini_response(messages) -> str:
    """
    Sends a chat history to Gemini and returns the response text.
    
    Args:
        messages (list): List of dicts with "role" ('user' or 'model') and "parts" (content).
                         Streamlit uses "role" ('user' or 'assistant') and "content".
                         We need to convert formats.
    """
    try:
        model = genai.GenerativeModel('gemini-flash-latest')
        
        # Convert Streamlit history to Gemini history
        # Streamlit: [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
        # Gemini: [{"role": "user", "parts": ["hi"]}, {"role": "model", "parts": ["hello"]}]
        gemini_history = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            content = msg["content"]
            gemini_history.append({"role": role, "parts": [content]})

        # The last message is the current prompt, so we separate it for chat.send_message if we were maintaining a chat object.
        # However, it's easier to just start a chat with history excluding the last message, then send the last message.
        
        if not gemini_history:
            return "Error: No messages to send."

        last_message = gemini_history[-1]
        history = gemini_history[:-1]

        chat = model.start_chat(history=history)
        response = chat.send_message(last_message["parts"][0])
        
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {str(e)}"
