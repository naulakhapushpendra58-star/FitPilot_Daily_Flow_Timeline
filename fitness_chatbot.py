import json, os
RECOMMENDED_QUESTIONS=["Which activity is best for building consistency?","How can I improve flexibility with a short routine?","What can I do if I only have 10 minutes?","Which activities need equipment?","How does the planner choose my activity?","Is this app giving medical advice?"]
def answer_fitness_question(question, knowledge_base, profile=None):
    activities=knowledge_base.get("activities",[])
    if os.getenv("OPENAI_API_KEY"):
        try:
            from openai import OpenAI
            r=OpenAI().responses.create(model=os.getenv("OPENAI_MODEL","gpt-4.1-mini"),instructions="You are a helpful fitness planning chatbot. Ground activity facts in the supplied library. Do not diagnose, prescribe treatment or make medical claims.",input="Activity library: "+json.dumps(activities)+"\\nProfile: "+json.dumps(profile or {},default=str)+"\\nQuestion: "+question)
            return r.output_text
        except Exception: pass
    q=question.lower()
    if "consistency" in q: return "For consistency, choose an activity you can repeat comfortably. Walking, gentle mobility, beginner bodyweight movement and recreational dancing are tagged for consistency in this activity library."
    if "flexib" in q or "stretch" in q: return "Gentle mobility and stretching is the flexibility-focused option. It is low intensity, needs no equipment, and supports 5–30 minute sessions. Avoid forcing a stretch."
    if "10 minute" in q or "10 min" in q or "short" in q: return "A 10-minute block can work for a short walk or gentle mobility session. Choose a comfortable pace."
    if "equipment" in q: return "Equipment in the activity library:\\n"+"\\n".join("• "+a["name"]+" — "+a["equipment"] for a in activities)
    if any(w in q for w in ["choose","planner","schedule"]): return "The planner ranks activities against your goal and preference, searches free time windows, then checks your time budget and busy blocks. The chatbot explains the plan; it does not change the schedule."
    if any(w in q for w in ["medical","health","pain","injur"]): return "This is not medical advice, diagnosis, or treatment. Stop if you experience pain, dizziness, or unusual discomfort, and seek professional guidance when needed."
    return "I can help with activity choices, time planning, the app's rules, and its activity library. Try a recommended question above."
