import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# --- Zero-shot prompt functions ---
# Each one is just ONE clear instruction, no examples included.

def explain_topic(topic):
    prompt = f"""Explain the topic "{topic}" to a student in simple, clear language.
Use short paragraphs and avoid jargon. If a technical term is necessary, define it in plain English.
Assume the student is learning this for the first time."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def summarize_material(text):
    prompt = f"""Summarize the following study material into clear, concise bullet points.
Only include information that is actually present in the text below — do not add outside facts.

Study material:
\"\"\"{text}\"\"\""""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def simplify_concept(concept):
    prompt = f"""Simplify the concept of "{concept}" so a complete beginner can understand it.
Use a simple everyday analogy. Keep the explanation under 150 words."""
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# --- Few-shot prompt functions (Step 3) ---
# Each one gives the model an instruction PLUS a worked example, so output format is consistent.

def generate_quiz(topic, num_questions=3):
    prompt = f"""You are a quiz generator. Generate quiz questions in the EXACT format shown in the example below — same structure, same labels, same style.

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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


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
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content
# Step 4: Chain of Thought Prompting CoT
def solve_problem_cot(problem):
    prompt = f"""Solve the following problem. Do NOT jump straight to the answer.
First, break the problem down into smaller logical steps. Work through each step one at a time, showing your reasoning clearly. Only give the final answer after all steps are complete.

Format your response exactly like this:
Step 1: [first part of reasoning]
Step 2: [next part of reasoning]
Step 3: [continue as needed]
...
Final Answer: [clear, direct final answer]

Problem: "{problem}" """
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content

# Step 5: Role Based Prompting
# Role definitions — each one sets a distinct persona and behaviour
ROLES = {
    "1": {
        "name": "Teacher",
        "system": """You are a patient, friendly school teacher. Your job is to explain 
concepts clearly to students who are learning for the first time. 
Use simple language, relatable examples, and encourage the student. 
Never make them feel bad for not knowing something. 
Always end with a quick summary of the key point."""
    },
    "2": {
        "name": "Examiner",
        "system": """You are a strict academic examiner. Your job is to evaluate the 
student's understanding critically and objectively. 
Point out gaps in knowledge, ask follow-up probing questions, 
and give direct feedback on whether the answer is correct or incorrect. 
Do not encourage guessing — demand precise, accurate answers."""
    },
    "3": {
        "name": "Study Coach",
        "system": """You are an energetic, motivational study coach. Your job is to 
help students stay focused, build good study habits, and maintain confidence. 
Be encouraging and practical. Give actionable advice. 
When explaining topics, connect them to the student's exam goals and 
remind them why the topic matters for their success."""
    },
    "4": {
        "name": "Subject Expert",
        "system": """You are a highly experienced subject matter expert and researcher. 
Your job is to give deep, technically accurate, detailed explanations. 
Do not oversimplify — use correct terminology and go beyond surface-level answers. 
If relevant, mention edge cases, exceptions, or advanced aspects of the topic."""
    }
}


def role_based_response(role_key, user_query):
    role = ROLES[role_key]
    messages = [
        SystemMessage(content=role["system"]),
        HumanMessage(content=user_query)
    ]
    try:
        response = llm.invoke(messages)
        return role["name"], response.content
    except Exception as e:
        if "503" in str(e) or "UNAVAILABLE" in str(e):
            return role["name"], "The AI server is temporarily busy. Please try again in a moment."
        elif "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
            return role["name"], "Daily quota exceeded. Please wait until midnight Pacific Time or switch models."
        else:
            return role["name"], f"An error occurred: {str(e)}"
        

# --- Prompt Templates (Step 6) ---
# Reusable prompt blueprints with dynamic placeholders in {curly braces}

# Template 1: Explanation
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

# Template 2: Quiz generation
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

# Template 3: Revision notes
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

# Template 4: Study plan
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
    """Generic runner — takes any template and a dict of variables, returns response."""
    prompt = template.invoke(variables)
    response = llm.invoke(prompt)
    return response.content

# --- Console menu ---

def main_menu():
    print("=== AI Tutor — Step 6: Prompt Templates ===\n")
    while True:
        print("What would you like to do?")
        print("1. Explain a topic")
        print("2. Summarize study material")
        print("3. Simplify a complex concept")
        print("4. Generate a quiz")
        print("5. Categorize topics by difficulty")
        print("6. Solve a problem step-by-step")
        print("7. Ask a role-based tutor")
        print("8. Explain topic (Template)")
        print("9. Generate quiz (Template)")
        print("10. Get revision notes (Template)")
        print("11. Create a study plan (Template)")
        print("12. Exit")
        choice = input("\nEnter choice (1-12): ").strip()

        if choice == "1":
            topic = input("Enter the topic: ")
            print("\nThinking...\n")
            print(explain_topic(topic))

        elif choice == "2":
            text = input("Paste the study material:\n")
            print("\nSummarizing...\n")
            print(summarize_material(text))

        elif choice == "3":
            concept = input("Enter the concept to simplify: ")
            print("\nSimplifying...\n")
            print(simplify_concept(concept))

        elif choice == "4":
            topic = input("Enter quiz topic: ")
            print("\nGenerating quiz...\n")
            print(generate_quiz(topic))

        elif choice == "5":
            topics = input("Enter topics, comma-separated: ")
            print("\nCategorizing...\n")
            print(categorize_concepts(topics))

        elif choice == "6":
            problem = input("Enter the problem to solve: ")
            print("\nWorking through it step-by-step...\n")
            print(solve_problem_cot(problem))

        elif choice == "7":
            print("\nChoose a tutor role:")
            print("1. Teacher       — simple, patient explanations")
            print("2. Examiner      — strict evaluation and probing questions")
            print("3. Study Coach   — motivational, exam-focused advice")
            print("4. Subject Expert — deep, technical, detailed answers")
            role_choice = input("\nEnter role (1-4): ").strip()
            if role_choice not in ROLES:
                print("Invalid role choice.\n")
                continue
            query = input("What is your question? ")
            print("\nThinking...\n")
            role_name, answer = role_based_response(role_choice, query)
            print(f"[{role_name}]: {answer}")

        elif choice == "8":
            topic = input("Enter topic: ")
            level = input("Student level (beginner/intermediate/advanced): ")
            print("\nGenerating explanation...\n")
            print(run_template(explanation_template, {
                "topic": topic,
                "level": level
            }))

        elif choice == "9":
            topic = input("Enter quiz topic: ")
            num = input("Number of questions (e.g. 3): ")
            difficulty = input("Difficulty (easy/medium/hard): ")
            print("\nGenerating quiz...\n")
            print(run_template(quiz_template, {
                "topic": topic,
                "num_questions": num,
                "difficulty": difficulty
            }))

        elif choice == "10":
            topic = input("Enter topic for revision notes: ")
            level = input("Student level (beginner/intermediate/advanced): ")
            print("\nCreating revision notes...\n")
            print(run_template(notes_template, {
                "topic": topic,
                "level": level
            }))

        elif choice == "11":
            topic = input("Enter topic to study: ")
            days = input("How many days do you have? ")
            hours = input("How many hours per day can you study? ")
            print("\nCreating study plan...\n")
            print(run_template(study_plan_template, {
                "topic": topic,
                "days": days,
                "hours": hours
            }))

        elif choice == "12":
            print("Goodbye! Keep studying.")
            break

        else:
            print("Invalid choice, try again.\n")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main_menu()