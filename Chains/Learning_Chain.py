from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from Config import create_llm

# We use the best available model for chains
# If this hits quota, change the model name here
llm = create_llm("gemini-3.1-flash-lite")
parser = StrOutputParser()

# --- Chain 1: Topic → Explanation ---
explanation_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a clear and concise academic tutor."),
    ("human", """Explain the topic '{topic}' to a student in simple language.
Cover:
- What it is
- How it works
- A real-world example
Keep it under 200 words.""")
])

# --- Chain 2: Explanation → Revision Notes ---
notes_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert at creating structured revision notes."),
    ("human", """Based on this explanation, create concise revision notes:

{explanation}

Format exactly like this:
KEY POINTS:
- [point 1]
- [point 2]
- [point 3]

REMEMBER:
- [most important thing to remember]""")
])

# --- Chain 3: Notes → Quiz ---
quiz_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a quiz generator."),
    ("human", """Based on these revision notes, generate 3 multiple choice questions:

{notes}

Use this exact format:
Q1. [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]

Q2. [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]

Q3. [question]
A) [option]
B) [option]
C) [option]
D) [option]
Answer: [letter]""")
])

# --- Build individual chains using LCEL (LangChain Expression Language) ---
# The | operator means "pipe output into next step"
explanation_chain = explanation_prompt | llm | parser
notes_chain = notes_prompt | llm | parser
quiz_chain = quiz_prompt | llm | parser


def run_learning_chain(topic: str) -> dict:
    """
    Runs the full Topic → Explanation → Notes → Quiz pipeline.
    Each step's output feeds into the next step automatically.
    Returns all three outputs as a dictionary.
    """
    print("  Step 1/3: Generating explanation...")
    explanation = explanation_chain.invoke({"topic": topic})

    print("  Step 2/3: Creating revision notes...")
    notes = notes_chain.invoke({"explanation": explanation})

    print("  Step 3/3: Generating quiz...")
    quiz = quiz_chain.invoke({"notes": notes})

    return {
        "topic": topic,
        "explanation": explanation,
        "notes": notes,
        "quiz": quiz
    }