from langchain_core.messages import HumanMessage
from Config import invoke_with_fallback

def generate_quiz(topic, num_questions=3):
    prompt = f"""You are a quiz generator. Generate quiz questions in the EXACT format shown in the example below.

Example:
Topic: Photosynthesis
Q1. What is the primary pigment used in photosynthesis?
A) Chlorophyll
B) Carotene
C) Xanthophyll
D) Anthocyanin
Answer: A

Q2. Which gas is absorbed by plants during photosynthesis?
A) Oxygen
B) Nitrogen
C) Carbon Dioxide
D) Hydrogen
Answer: C

Now generate {num_questions} questions in the exact same format for this topic: "{topic}"."""
    return invoke_with_fallback([HumanMessage(content=prompt)])


def categorize_concepts(topics_text):
    prompt = f"""Categorize the given topics into Beginner, Intermediate, and Advanced difficulty levels. Follow this EXACT format.

Example:
Topics: HTML basics, Recursion, Variables, Dynamic programming, Loops, Graph algorithms

Beginner:
- HTML basics
- Variables
- Loops

Intermediate:
- Recursion

Advanced:
- Dynamic programming
- Graph algorithms

Now categorize these topics in the exact same format:
Topics: {topics_text}"""
    return invoke_with_fallback([HumanMessage(content=prompt)])