import os
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
from Agent.Tools import calculator, study_planner, summarizer
from Config import GEMINI_MODELS, OLLAMA_MODELS, MODE

load_dotenv()

tools = [calculator, study_planner, summarizer]
tools_by_name = {t.name: t for t in tools}

SYSTEM_PROMPT = """You are a smart academic tutor assistant with access to these tools:
- calculator: for ANY math calculation
- study_planner: for study schedules or plans
- summarizer: for summarizing long text

For general academic questions, answer directly without tools.
Always be helpful and encouraging."""


def create_llm(model_name: str):
    """Create LLM based on current MODE."""
    if MODE == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.3
        )
    else:
        from langchain_ollama import ChatOllama
        return ChatOllama(
            model=model_name,
            temperature=0.3
        )


def run_agent(user_input: str, chat_history: list = None) -> str:
    """Run the agent on a user query using manual tool-calling loop."""
    if chat_history is None:
        chat_history = []

    models = GEMINI_MODELS if MODE == "gemini" else OLLAMA_MODELS

    for model_name in models:
        try:
            llm = create_llm(model_name)
            llm_with_tools = llm.bind_tools(tools)

            # Build message list
            messages = [SystemMessage(content=SYSTEM_PROMPT)]
            messages += chat_history
            messages.append(HumanMessage(content=user_input))

            # Step 1: Let LLM decide if it needs a tool
            response = llm_with_tools.invoke(messages)

            # Step 2: If LLM wants to use a tool, execute it
            if response.tool_calls:
                messages.append(response)

                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"  [Using tool: {tool_name}]")

                    if tool_name in tools_by_name:
                        tool_result = tools_by_name[tool_name].invoke(tool_args)
                    else:
                        tool_result = f"Tool '{tool_name}' not found."

                    messages.append(ToolMessage(
                        content=str(tool_result),
                        tool_call_id=tool_call["id"]
                    ))

                # Step 3: Final answer using tool result
                final_response = llm_with_tools.invoke(messages)
                content = final_response.content
            else:
                content = response.content

            # Clean up content if it's a list
            if isinstance(content, list):
                return "\n".join(
                    block.get("text", "") for block in content
                    if isinstance(block, dict) and block.get("type") == "text"
                )
            return content

        except Exception as e:
            error = str(e)
            if MODE == "gemini":
                if "429" in error or "RESOURCE_EXHAUSTED" in error:
                    print(f"  [{model_name} quota exceeded, trying next...]")
                    continue
                elif "503" in error or "UNAVAILABLE" in error:
                    print(f"  [{model_name} busy, trying next...]")
                    continue
                else:
                    return f"Agent error: {error}"
            else:
                if "connection" in error.lower() or "refused" in error.lower():
                    return "Ollama is not running. Make sure Ollama is installed and running."
                else:
                    return f"Agent error: {error}"

    if MODE == "gemini":
        return "All models quota exceeded. Please wait until 1:15 PM Nepal time tomorrow."
    else:
        return "All Ollama models failed. Make sure Ollama is running."