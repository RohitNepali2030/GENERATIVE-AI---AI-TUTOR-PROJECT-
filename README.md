# AI-Powered Academic Tutor
**Personalized Learning and Exam Preparation System**
Pokhara University — Computer Engineering — Generative AI Project

---

## Project Overview
A console-based AI academic tutor built with Python, LangChain, and the 
Google Gemini API. The system simulates a personalized tutor capable of 
explaining concepts, generating quizzes, solving problems step-by-step, 
and maintaining memory across sessions.

---

## Tech Stack
- Python 3.10+
- LangChain + LangChain Google GenAI
- Google Gemini API (Free Tier)
- python-dotenv

---

## Project Structure
AI_Tutor/

├── main.py              # Console menu — entry point

├── config.py            # LLM setup and model fallback

├── .env                 # API key (not committed to git)

├── prompts/

│   ├── zero_shot.py     # Step 2: Zero-Shot Prompting

│   ├── few_shot.py      # Step 3: Few-Shot Prompting

│   ├── cot.py           # Step 4: Chain-of-Thought

│   ├── roles.py         # Step 5: Role Prompting

│   └── templates.py     # Step 6: Prompt Templates

├── chains/

│   └── learning_chain.py  # Step 7: LangChain Chains

├── memory/

│   ├── user_memory.py   # Step 8: Persistent Memory

│   ├── chat_memory.py   # Step 8: In-Session Memory

│   └── user_data.json   # Auto-generated user profile

└── agent/

├── tools.py         # Step 9: Calculator, Planner, Summarizer

└── tutor_agent.py   # Step 9: Tool-calling Agent

---

## Setup Instructions

### 1. Clone or download the project

### 2. Create and activate virtual environment
```bash
python -m venv venv
venv\Scripts\Activate.ps1   # Windows
```

### 3. Install dependencies
```bash
pip install langchain langchain-google-genai langchain-core python-dotenv
```

### 4. Add your API key
Create a `.env` file in the project root:
GOOGLE_API_KEY=your_gemini_api_key_here
Get a free key at: https://aistudio.google.com/apikey

### 5. Run the app
```bash
python main.py
```

---

## Features Implemented

| Step | Feature | Description |
|------|---------|-------------|
| 1 | LLM Integration | Connects to Gemini API, maintains session history |
| 2 | Zero-Shot Prompting | Explains topics, summarizes material, simplifies concepts |
| 3 | Few-Shot Prompting | Generates formatted quizzes and categorizes topics |
| 4 | Chain-of-Thought | Solves problems step-by-step with visible reasoning |
| 5 | Role Prompting | 4 tutor roles: Teacher, Examiner, Study Coach, Expert |
| 6 | Prompt Templates | Reusable templates for explanations, notes, study plans |
| 7 | LangChain Chains | Topic → Explanation → Notes → Quiz pipeline |
| 8 | Memory System | Persistent profile + in-session conversation memory |
| 9 | Agents & Tools | Calculator, Study Planner, Summarizer tools with auto-selection |
| 10 | Final Integration | Modular architecture, fallback system, clean console UI |

---

## Notes
- Free tier quota resets daily at 1:15 PM Nepal Time (NPT)
- Model fallback order: gemini-3.1-flash-lite → gemini-3-flash → 
  gemini-2.5-flash-lite → gemini-2.5-flash
- User profile is saved in `memory/user_data.json`