from langchain_core.prompts import ChatPromptTemplate
from Config import invoke_with_fallback

explanation_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful academic tutor who explains concepts clearly."),
    ("human", """Explain the topic '{topic}' to a {level} student.
Use simple language appropriate for their level.
Include:
- A clear definition
- How it works
- A real-world example
- One key takeaway""")
])

quiz_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic quiz generator. Always follow the exact format given."),
    ("human", """Generate {num_questions} multiple choice quiz questions on '{topic}'.
Difficulty level: {difficulty}

Use this exact format for each question:
Q[number]. [question text]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [correct letter]
Explanation: [one sentence why]""")
])

notes_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic note-maker who creates structured revision notes."),
    ("human", """Create concise revision notes on '{topic}' for a {level} student.

Structure the notes exactly like this:
TOPIC: {topic}
KEY CONCEPTS:
- [concept 1]
- [concept 2]
- [concept 3]

IMPORTANT FORMULAS/RULES (if any):
- [formula or rule]

QUICK SUMMARY:
[2-3 sentences summarizing the whole topic]

EXAM TIPS:
- [tip 1]
- [tip 2]""")
])

study_plan_template = ChatPromptTemplate.from_messages([
    ("system", "You are an academic study coach who creates practical study plans."),
    ("human", """Create a study plan for a student who wants to learn '{topic}'.
Available time: {days} days
Daily study hours available: {hours} hours per day

Structure the plan day by day like this:
Day 1: [what to study]
Day 2: [what to study]
...and so on

End with:
RESOURCES NEEDED: [list what the student should gather]
SUCCESS TIP: [one motivational practical tip]""")
])


def run_template(template, variables: dict):
    prompt = template.invoke(variables)
    return invoke_with_fallback(prompt.to_messages())