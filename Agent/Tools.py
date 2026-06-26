from langchain_core.tools import tool
from Config import invoke_with_fallback
from langchain_core.messages import HumanMessage


@tool
def calculator(expression: str) -> str:
    """
    Evaluates a mathematical expression and returns the result.
    Use this for any arithmetic, algebra, or numerical calculation.
    Examples: '2 + 2', '15 * 8', '(100 / 4) * 3', '2 ** 10'
    """
    try:
        # Safe evaluation — only allows math operations
        allowed = {
            "__builtins__": {},
            "abs": abs, "round": round,
            "min": min, "max": max,
            "pow": pow
        }
        result = eval(expression.strip(), allowed)
        return f"Result: {expression} = {result}"
    except Exception as e:
        return f"Could not calculate '{expression}': {str(e)}"


@tool
def study_planner(topic: str, days: int, hours_per_day: float) -> str:
    """
    Creates a structured day-by-day study plan for a given topic.
    Use this when a student asks for a study schedule or study plan.
    Arguments:
        topic: the subject to study
        days: number of days available
        hours_per_day: study hours available per day
    """
    prompt = f"""Create a detailed day-by-day study plan for '{topic}'.
Available: {days} days, {hours_per_day} hours per day.

Format exactly like this:
Day 1 ({hours_per_day}hrs): [specific tasks]
Day 2 ({hours_per_day}hrs): [specific tasks]
...continue for all {days} days...

End with:
TOTAL HOURS: [total]
KEY MILESTONES: [what student should know by end]"""
    return invoke_with_fallback([HumanMessage(content=prompt)])


@tool
def summarizer(text: str) -> str:
    """
    Summarizes a long piece of text into clear, concise bullet points.
    Use this when a student provides a passage and wants it condensed.
    """
    prompt = f"""Summarize the following text into clear bullet points.
Be concise — capture only the most important ideas.

Text:
\"\"\"{text}\"\"\"

Format:
SUMMARY:
- [key point 1]
- [key point 2]
- [key point 3]
(add more if needed)

ONE LINE TAKEAWAY: [single most important idea]"""
    return invoke_with_fallback([HumanMessage(content=prompt)])