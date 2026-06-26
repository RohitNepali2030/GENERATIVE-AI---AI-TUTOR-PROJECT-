import json
import os
from datetime import datetime

MEMORY_FILE = "memory/user_data.json"


def load_memory() -> dict:
    """Load user data from JSON file. Creates default if doesn't exist."""
    if os.path.exists(MEMORY_FILE):
        with open(MEMORY_FILE, "r") as f:
            return json.load(f)
    return {
        "name": None,
        "topics_studied": [],
        "quiz_scores": [],
        "weak_areas": [],
        "total_sessions": 0,
        "last_seen": None
    }


def save_memory(data: dict):
    """Save user data to JSON file."""
    with open(MEMORY_FILE, "w") as f:
        json.dump(data, f, indent=2)


def update_session(data: dict):
    """Update session count and last seen timestamp."""
    data["total_sessions"] += 1
    data["last_seen"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_memory(data)


def add_topic_studied(data: dict, topic: str):
    """Add a topic to the studied list if not already there."""
    if topic.lower() not in [t.lower() for t in data["topics_studied"]]:
        data["topics_studied"].append(topic)
        save_memory(data)


def add_quiz_score(data: dict, topic: str, score: int, total: int):
    """Save a quiz result."""
    data["quiz_scores"].append({
        "topic": topic,
        "score": score,
        "total": total,
        "date": datetime.now().strftime("%Y-%m-%d")
    })
    if score / total < 0.5:
        # Add to weak areas if not already there
        if topic not in data["weak_areas"]:
            data["weak_areas"].append(topic)
    else:
        # Remove from weak areas if score improved above 50%
        if topic in data["weak_areas"]:
            data["weak_areas"].remove(topic)
            print(f"✅ '{topic}' removed from weak areas — great improvement!")
    save_memory(data)


def get_summary(data: dict) -> str:
    """Return a readable summary of what the user has done."""
    if not data["name"]:
        return "No profile found."

    topics = ", ".join(data["topics_studied"]) if data["topics_studied"] else "None yet"
    weak = ", ".join(data["weak_areas"]) if data["weak_areas"] else "None"
    sessions = data["total_sessions"]
    last = data["last_seen"] or "Never"

    quiz_summary = ""
    if data["quiz_scores"]:
        latest = data["quiz_scores"][-1]
        quiz_summary = f"\nLast quiz: {latest['topic']} — {latest['score']}/{latest['total']}"

    return (
        f"Student: {data['name']}\n"
        f"Sessions completed: {sessions}\n"
        f"Last seen: {last}\n"
        f"Topics studied: {topics}\n"
        f"Weak areas: {weak}"
        f"{quiz_summary}"
    )

def record_quiz_result(data: dict, topic: str):
    """Ask the student how they scored and save it."""
    print(f"\nHow did you score on the {topic} quiz?")
    try:
        score = int(input("Your score (number correct): ").strip())
        total = int(input("Total questions: ").strip())
        if score < 0 or total <= 0 or score > total:
            print("Invalid score, not recorded.")
            return
        add_quiz_score(data, topic, score, total)
        percentage = round((score / total) * 100)
        print(f"\nScore recorded: {score}/{total} ({percentage}%)")
        if percentage < 50:
            print(f"⚠️  {topic} added to your weak areas — consider reviewing it.")
        elif percentage >= 80:
            print(f"✅ Great job on {topic}!")
        else:
            print(f"📝 Decent score — a bit more practice on {topic} wouldn't hurt.")
    except ValueError:
        print("Invalid input, score not recorded.")