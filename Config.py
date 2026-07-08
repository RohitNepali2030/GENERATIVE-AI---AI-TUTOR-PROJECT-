import os
from dotenv import load_dotenv

load_dotenv()

# ============================================================
# SWITCH BETWEEN ONLINE (Gemini) AND OFFLINE (Ollama) HERE
# Set to "gemini" for online or "ollama" for offline
# ============================================================
MODE = "ollama"  # change to "ollama" when you want offline mode


# --- Gemini settings ---
GEMINI_MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

# --- Ollama settings ---
OLLAMA_MODELS = [
    "gemma:2b",      # lightest — best for 4GB VRAM
    "mistral",       # better quality but slower
]


def create_llm(model_name):
    if MODE == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=0.7
        )


def invoke_with_fallback(messages):
    models = GEMINI_MODELS if MODE == "gemini" else OLLAMA_MODELS

    for model_name in models:
        try:
            llm = create_llm(model_name)
            response = llm.invoke(messages)
            if hasattr(response, 'content'):
                content = response.content
                if isinstance(content, list):
                    return "\n".join(
                        block.get("text", "") if isinstance(block, dict) else str(block)
                        for block in content
                        if (isinstance(block, dict) and block.get("type") == "text") or isinstance(block, str)
                    )
                return content
            return str(response)
        except Exception as e:
            error = str(e)
            if MODE == "gemini":
                if "429" in error or "RESOURCE_EXHAUSTED" in error:
                    print(f"  [{model_name} quota exceeded, trying next model...]")
                    continue
                elif "503" in error or "UNAVAILABLE" in error:
                    print(f"  [{model_name} temporarily busy, trying next model...]")
                    continue
                else:
                    return f"An error occurred: {error}"
            else:
                if "connection" in error.lower() or "refused" in error.lower():
                    return "Ollama is not running. Open a terminal and run: ollama serve"
                else:
                    return f"An error occurred: {error}"

    if MODE == "gemini":
        return "All models quota exceeded. Please wait until 1:15 PM Nepal time tomorrow."
    else:
        return "All Ollama models failed. Make sure Ollama is running with: ollama serve"