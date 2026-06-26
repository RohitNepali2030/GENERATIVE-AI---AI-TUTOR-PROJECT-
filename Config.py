import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

# Model fallback list — tries each one if quota is exceeded
MODELS = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-flash",
]

def create_llm(model_name):
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.7
    )

def invoke_with_fallback(messages):
    for model_name in MODELS:
        try:
            llm = create_llm(model_name)
            response = llm.invoke(messages)
            # Handle both string and list responses
            if hasattr(response, 'content'):
                content = response.content
                # If content is a list (raw blocks), extract text only
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
            if "429" in error or "RESOURCE_EXHAUSTED" in error:
                print(f"  [{model_name} quota exceeded, trying next model...]")
                continue
            elif "503" in error or "UNAVAILABLE" in error:
                print(f"  [{model_name} temporarily busy, trying next model...]")
                continue
            else:
                return f"An error occurred: {error}"
    return "All models quota exceeded. Please wait until 1:15 PM Nepal time tomorrow."