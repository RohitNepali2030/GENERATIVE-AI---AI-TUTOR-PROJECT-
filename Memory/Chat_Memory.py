from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from Config import invoke_with_fallback


class ChatMemory:
    """Maintains conversation history for one session."""

    def __init__(self, student_name: str, topics_studied: list):
        # Build a system message personalised to this student
        topics = ", ".join(topics_studied) if topics_studied else "no topics yet"
        self.history = [
            SystemMessage(content=f"""You are a personal academic tutor.
You are talking to {student_name}.
They have previously studied: {topics}.
Remember what they say during this conversation and refer back to it when relevant.
Be encouraging and personalised in your responses.""")
        ]

    def chat(self, user_input: str) -> str:
        """Send a message and get a response, maintaining full history."""
        self.history.append(HumanMessage(content=user_input))
        response = invoke_with_fallback(self.history)
        self.history.append(AIMessage(content=response))
        return response

    def clear(self):
        """Clear conversation history (keeps system message)."""
        self.history = self.history[:1]