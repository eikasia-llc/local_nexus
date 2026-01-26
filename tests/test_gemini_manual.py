import google.generativeai as genai
from src.core.llm import init_gemini, get_gemini_response

def test_gemini_connection():
    print("Initializing Gemini...")
    if not init_gemini():
        print("Failed to initialize. Check API Key.")
        return

    print("Listing available models...")
    try:
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(m.name)
    except Exception as e:
        print(f"Error listing models: {e}")

    print("Sending test message...")
    messages = [{"role": "user", "content": "Hello, this is a test from Local Nexus. Reply with 'Connection Successful'."}]
    response = get_gemini_response(messages)
    print(f"Response: {response}")

if __name__ == "__main__":
    test_gemini_connection()
