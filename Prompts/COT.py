from langchain_core.messages import HumanMessage
from Config import invoke_with_fallback

def solve_problem_cot(problem):
    prompt = f"""Solve the following problem. Do NOT jump straight to the answer.
First, break the problem down into smaller logical steps. Work through each step one at a time,
showing your reasoning clearly. Only give the final answer after all steps are complete.

Format your response exactly like this:
Step 1: [first part of reasoning]
Step 2: [next part of reasoning]
Step 3: [continue as needed]
...
Final Answer: [clear, direct final answer]

Problem: "{problem}" """
    return invoke_with_fallback([HumanMessage(content=prompt)])