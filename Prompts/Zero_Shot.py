from langchain_core.messages import HumanMessage
from Config import invoke_with_fallback

def explain_topic(topic):
    prompt = f"""Explain the topic "{topic}" to a student in simple, clear language.
Use short paragraphs and avoid jargon. If a technical term is necessary, define it in plain English.
Assume the student is learning this for the first time."""
    return invoke_with_fallback([HumanMessage(content=prompt)])


def summarize_material(text):
    prompt = f"""Summarize the following study material into clear, concise bullet points.
Only include information that is actually present in the text below — do not add outside facts.

Study material:
\"\"\"{text}\"\"\""""
    return invoke_with_fallback([HumanMessage(content=prompt)])


def simplify_concept(concept):
    prompt = f"""Simplify the concept of "{concept}" so a complete beginner can understand it.
Use a simple everyday analogy. Keep the explanation under 150 words."""
    return invoke_with_fallback([HumanMessage(content=prompt)])