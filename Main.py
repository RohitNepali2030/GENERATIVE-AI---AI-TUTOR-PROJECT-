import os
import sys
import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
#sys.stderr = open(os.devnull, "w")




from Prompts.Zero_Shot import explain_topic, summarize_material, simplify_concept
from Prompts.Few_Shots import generate_quiz, categorize_concepts
from Prompts.COT import solve_problem_cot
from Prompts.Roles import role_based_response, ROLES
from Prompts.Templates import (
    run_template,
    explanation_template,
    quiz_template,
    notes_template,
    study_plan_template
)
from Chains.Learning_Chain import run_learning_chain

from Memory.Chat_Memory import ChatMemory

from Memory.User_Memory import (
    load_memory, save_memory, update_session,
    add_topic_studied, add_quiz_score, get_summary,
    record_quiz_result
)
from Agent.Tutor_Agent import run_agent

from langchain_core.messages import HumanMessage, AIMessage 

from RAG.RAG_Chain import ingest_document, ask_document, load_existing_store


def print_banner():
    print("=" * 55)
    print("   AI-POWERED ACADEMIC TUTOR")
    print("   Personalized Learning & Exam Preparation")
    print("=" * 55)
    print("   Steps implemented:")
    print("   [1] LLM Integration    [2] Zero-Shot Prompting")
    print("   [3] Few-Shot Prompting [4] Chain-of-Thought")
    print("   [5] Role Prompting     [6] Prompt Templates")
    print("   [7] LangChain Chains   [8] Memory System")
    print("   [9] Agents & Tools     [10] Final Integration")
    print("   [11] RAG Model         [Document Q&A]")  # ← only add this line
    print("=" * 55)
    print()


def setup_profile(data: dict) -> dict:
    """First time setup — ask for student name."""
    if not data["name"]:
        print("Welcome! Let's set up your profile first.")
        name = input("What is your name? ").strip()
        data["name"] = name
        save_memory(data)
        print(f"\nNice to meet you, {name}! Your profile has been created.\n")
    else:
        print(f"Welcome back, {data['name']}!")
        if data["last_seen"]:
            print(f"Last session: {data['last_seen']}")
        if data["topics_studied"]:
            print(f"Topics studied: {', '.join(data['topics_studied'])}")
        if data["weak_areas"]:
            print(f"Weak areas to review: {', '.join(data['weak_areas'])}")
        print()
    return data


def main_menu():
    print_banner()
    data = load_memory()
    data = setup_profile(data)
    update_session(data)
    chat = ChatMemory(data["name"], data["topics_studied"])
    # Load existing RAG store if available
    if load_existing_store():
        print("Previous document store loaded — RAG ready.\n")

    # Start in-session chat memory
    #chat = ChatMemory(data["name"], data["topics_studied"])

    print("=== AI Academic Tutor ===\n")
    while True:
        print(f"Hello {data['name']}! What would you like to do?")
        print("1.  Explain a topic")
        print("2.  Summarize study material")
        print("3.  Simplify a complex concept")
        print("4.  Generate a quiz")
        print("5.  Categorize topics by difficulty")
        print("6.  Solve a problem step-by-step")
        print("7.  Ask a role-based tutor")
        print("8.  Explain topic (Template)")
        print("9.  Generate quiz (Template)")
        print("10. Get revision notes (Template)")
        print("11. Create a study plan (Template)")
        print("12. Full learning chain (Topic → Explanation → Notes → Quiz)")
        print("13. Chat with tutor (remembers conversation)")
        print("14. View my learning profile")
        print("15. Ask the AI Agent (uses tools automatically)")
        print("16. Upload document for Q&A (RAG)")
        print("17. Ask question about uploaded document")
        print("18. Exit")
        choice = input("\nEnter choice (1-18): ").strip()

        if choice == "1":
            topic = input("Enter the topic: ")
            print("\nThinking...\n")
            result = explain_topic(topic)
            print(result)
            add_topic_studied(data, topic)

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
            add_topic_studied(data, topic)
            record_quiz_result(data, topic)  # ← this line was missing

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
            role_name, ansr = role_based_response(role_choice, query)
            print(f"[{role_name}]: {ansr}")

        elif choice == "8":
            topic = input("Enter topic: ")
            level = input("Student level (beginner/intermediate/advanced): ")
            print("\nGenerating explanation...\n")
            result = run_template(explanation_template, {
                "topic": topic, "level": level
            })
            print(result)
            add_topic_studied(data, topic)

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
            add_topic_studied(data, topic)
            record_quiz_result(data, topic)

        elif choice == "10":
            topic = input("Enter topic for revision notes: ")
            level = input("Student level (beginner/intermediate/advanced): ")
            print("\nCreating revision notes...\n")
            result = run_template(notes_template, {
                "topic": topic, "level": level
            })
            print(result)
            add_topic_studied(data, topic)
        elif choice == "11":
            topic = input("Enter topic to study: ")
            days = input("How many days do you have? ")
            hours = input("How many hours per day can you study? ")

            print("\nCreating study plan...\n")

            response = run_template(
                study_plan_template,
                {
                    "topic": topic,
                    "days": days,
                    "hours": hours
                }
            )

            print("\n" + "=" * 60)
            print("📚 STUDY PLAN")
            print("=" * 60)

            if isinstance(response, list):
                print(response[0]["text"])
            else:
                print(response)

            print("=" * 60)

        elif choice == "12":
            topic = input("Enter a topic to learn: ")
            print("\nRunning full learning chain...\n")
            results = run_learning_chain(topic)
            print("=" * 50)
            print(f"TOPIC: {results['topic'].upper()}")
            print("=" * 50)
            print("\n--- EXPLANATION ---")
            print(results['explanation'])
            print("\n--- REVISION NOTES ---")
            print(results['notes'])
            print("\n--- QUIZ ---")
            print(results['quiz'])
            add_topic_studied(data, topic)

        elif choice == "13":
            print("\n--- Chat Mode (type 'back' to return to menu) ---")
            print(f"The tutor knows you are {data['name']} "
                  f"and remembers this conversation as you go.\n")
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "back":
                    print("Returning to main menu...\n")
                    break
                if not user_input:
                    continue
                print("\nTutor: ", end="")
                response = chat.chat(user_input)
                print(response)
                print()

        elif choice == "14":
            print("\n--- Your Learning Profile ---")
            print(get_summary(data))
            
        elif choice == "15":
            print("\n--- AI Agent Mode ---")
            print("The agent will automatically use tools when needed.")
            print("Try: 'What is 15% of 340?'")
            print("Try: 'Make me a 5 day plan for learning Python, 2 hours a day'")
            print("Try: 'Summarize this: [paste text]'")
            print("Type 'back' to return to menu.\n")

            agent_history = []
            while True:
                user_input = input("You: ").strip()
                if user_input.lower() == "back":
                    print("Returning to main menu...\n")
                    break
                if not user_input:
                    continue
                print("\nAgent thinking...\n")
                response = run_agent(user_input, agent_history)
                print(f"Agent: {response}\n")

                # Keep conversation history for context
                agent_history.append(HumanMessage(content=user_input))
                agent_history.append(AIMessage(content=response))

       
        elif choice == "16":
            print("\n--- Document Upload (RAG) ---")
            print("Supported formats: .pdf, .txt")
            file_path = input("Enter full file path: ").strip()
            # Remove quotes if user copied path with quotes
            file_path = file_path.strip('"').strip("'")
            try:
                ingest_document(file_path)
            except FileNotFoundError as e:
                print(f"Error: {e}")
            except Exception as e:
                print(f"Error loading document: {e}")

        elif choice == "17":
            question = input("Ask a question about your document: ")
            print("\nSearching document...\n")
            print(ask_document(question))

        elif choice == "18":
            print(f"Goodbye {data['name']}! Keep studying. 📚")
            break

        else:
                print("Invalid choice, try again.\n")

        print("\n" + "-" * 50 + "\n")


if __name__ == "__main__":
    main_menu()