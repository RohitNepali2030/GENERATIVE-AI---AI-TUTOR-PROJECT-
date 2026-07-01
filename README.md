# AI-Powered Academic Tutor
**Personalized Learning and Exam Preparation System**
Pokhara University — Computer Engineering — Generative AI Project

---

## Project Overview
A console-based AI academic tutor built with Python, LangChain, and the
Google Gemini API. The system simulates a personalized tutor capable of
explaining concepts, generating quizzes, solving problems step-by-step,
maintaining memory across sessions, and answering questions from uploaded
documents using RAG (Retrieval Augmented Generation).

---

## Tech Stack
- Python 3.10+
- LangChain + LangChain Google GenAI
- Google Gemini API (Free Tier)
- FAISS (Vector Store)
- Sentence Transformers (Local Embeddings)
- PyPDF (PDF Reading)
- python-dotenv

---

## Project Structure

```
AI TUTOR PROJECT/
├── Main.py                  # Console menu — entry point
├── Config.py                # LLM setup and model fallback
├── .env                     # API key (not committed to git)
├── README.md                # Project documentation
│
├── Prompts/
│   ├── Zero_Shot.py         # Step 2: Zero-Shot Prompting
│   ├── Few_Shots.py         # Step 3: Few-Shot Prompting
│   ├── COT.py               # Step 4: Chain-of-Thought
│   ├── Roles.py             # Step 5: Role Prompting
│   └── Templates.py         # Step 6: Prompt Templates
│
├── Chains/
│   └── Learning_Chain.py    # Step 7: LangChain Chains
│
├── Memory/
│   ├── User_Memory.py       # Step 8: Persistent Memory
│   ├── Chat_Memory.py       # Step 8: In-Session Memory
│   └── user_data.json       # Auto-generated user profile
│
├── Agent/
│   ├── Tools.py             # Step 9: Calculator, Planner, Summarizer
│   └── Tutor_Agent.py       # Step 9: Tool-calling Agent
│
└── RAG/
    ├── Document_Loader.py   # Step 11: PDF/TXT document loading
    ├── Vector_Store.py      # Step 11: FAISS vector storage
    └── RAG_Chain.py         # Step 11: Retrieval + generation chain
```

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
pip install langchain-community faiss-cpu pypdf sentence-transformers
```

### 4. Add your API key
Create a `.env` file in the project root:
GOOGLE_API_KEY=your_gemini_api_key_here

Get a free key at: https://aistudio.google.com/apikey

### 5. Run the app
```bash
python Main.py
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
| 9 | Agents & Tools | Calculator, Study Planner, Summarizer with auto-selection |
| 10 | Final Integration | Modular architecture, fallback system, clean console UI |
| 11 | RAG Model | Upload PDF/TXT documents and ask questions about them |

---

## How RAG Works

Student uploads a PDF or TXT file (e.g. lecture notes)
↓
RAG splits it into chunks and stores them as vectors (FAISS)
↓
Student asks a question
↓
RAG finds the most relevant chunks using semantic search
↓
Gemini answers using those chunks as context
↓
Answer is grounded in YOUR documents, not just general knowledge

---

## Menu Options
```
1. Explain a topic
2. Summarize study material
3. Simplify a complex concept
4. Generate a quiz
5. Categorize topics by difficulty
6. Solve a problem step-by-step
7. Ask a role-based tutor
8. Explain topic (Template)
9. Generate quiz (Template)
10. Get revision notes (Template)
11. Create a study plan (Template)
12. Full learning chain (Topic → Explanation → Notes → Quiz)
13. Chat with tutor (remembers conversation) 
14. View my learning profile 
15. Ask the AI Agent (uses tools automatically) 
16. Upload document for Q&A (RAG) 
17. Ask question about uploaded document 
18. Exit  
```
---

## Notes
- Free tier quota resets daily at 1:15 PM Nepal Time (NPT)
- Model fallback order: gemini-3.1-flash-lite → gemini-3-flash →
  gemini-2.5-flash-lite → gemini-2.5-flash
- User profile is saved in `Memory/user_data.json`
- Embedding model runs fully locally — no extra API key needed
- Supported document formats: PDF (.pdf) and Text (.txt)  

