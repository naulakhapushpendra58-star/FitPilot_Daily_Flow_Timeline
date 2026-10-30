import streamlit as st
from datetime import date, datetime, timedelta
from planner import load_knowledge_base, PersonalAssistantAgent, TimeBlock, make_datetime
from llm_grounding import grounded_explanation
from fitness_chatbot import RECOMMENDED_QUESTIONS, answer_fitness_question

st.set_page_config(page_title="FitPilot AI", page_icon="🏃", layout="wide")
st.markdown("""<style>
.stApp{background:radial-gradient(ellipse at 15% 0%,#281044 0%,#0a0b14 45%,#080910 100%);color:#f7f7ff}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#171329,#0c0d17)}
.block-container{max-width:1280px;padding-top:1rem}
.hero{border:1px solid #3a2859;border-radius:22px;padding:22px;background:linear-gradient(120deg,#46206b66,#121426);margin-bottom:18px}
.hero h1{font-size:clamp(1.6rem,4vw,2.6rem)}
[data-testid="stMetric"]{background:#141525;border:1px solid #2c2c48;padding:12px;border-radius:15px}
.stButton button{border-radius:13px;border:1px solid #51317d}
@media(max-width:700px){.block-container{padding:.7rem}.hero{padding:15px}.hero h1{font-size:1.6rem}}
</style>""",unsafe_allow_html=True)
kb = load_knowledge_base()
agent = PersonalAssistantAgent(kb)

st.markdown('<div class="hero"><small>✦ YOUR PERSONAL AI FITNESS COMPANION</small><h1>Hey there 👋 Let’s make today count.</h1><p>Plan movement, chat with your AI Coach, and build a routine that fits your day.</p></div>',unsafe_allow_html=True)
st.caption("A non-clinical activity planning prototype using rules, schedule search, an assistant agent, and grounded explanations.")

with st.sidebar:
    st.header("Your preferences")
    goal = st.selectbox("Primary goal", ["Build consistency", "Improve flexibility", "Stay active", "Improve endurance"])
    preference = st.selectbox("Preferred activity", ["Any suitable activity", "Walking", "Gentle mobility and stretching", "Leisurely cycling", "Beginner bodyweight movement", "Recreational dancing"])
    experience = st.selectbox("Experience", ["Beginner", "Intermediate", "Experienced"])
    available_minutes = st.slider("Activity time budget (minutes)", 5, 120, 30, 5)
    target_duration = st.slider("Preferred activity block (minutes)", 5, 60, 20, 5)
    day = st.date_input("Plan date", date.today())
    day_start = st.time_input("Planning window starts", datetime.strptime("07:00", "%H:%M").time())
    day_end = st.time_input("Planning window ends", datetime.strptime("21:00", "%H:%M").time())
    st.subheader("Busy schedule")
    busy_start_1 = st.time_input("Busy block 1 starts", datetime.strptime("09:00", "%H:%M").time())
    busy_end_1 = st.time_input("Busy block 1 ends", datetime.strptime("15:00", "%H:%M").time())
    add_busy = st.checkbox("Add a second busy block")
    if add_busy:
        busy_start_2 = st.time_input("Busy block 2 starts", datetime.strptime("17:00", "%H:%M").time())
        busy_end_2 = st.time_input("Busy block 2 ends", datetime.strptime("18:00", "%H:%M").time())
    run = st.button("Generate activity plan", type="primary", use_container_width=True)

tabs = st.tabs(["🏠 Planner", "🤖 AI Coach", "AI Agent", "AI Paradigms", "Knowledge Base", "About & Safety"])

with tabs[0]:
    st.subheader("Feasible schedule search")
    if run:
        start_dt = make_datetime(day, day_start.strftime("%H:%M"))
        end_dt = make_datetime(day, day_end.strftime("%H:%M"))
        busy = []
        if busy_end_1 > busy_start_1:
            busy.append(TimeBlock(make_datetime(day, busy_start_1.strftime("%H:%M")), make_datetime(day, busy_end_1.strftime("%H:%M")), "Busy block 1"))
        if add_busy and busy_end_2 > busy_start_2:
            busy.append(TimeBlock(make_datetime(day, busy_start_2.strftime("%H:%M")), make_datetime(day, busy_end_2.strftime("%H:%M")), "Busy block 2"))
        profile = {"day": day, "goal": goal, "preference": preference, "experience": experience,
                   "available_minutes": available_minutes, "target_duration": target_duration,
                   "day_start": day_start.strftime("%H:%M"), "day_end": day_end.strftime("%H:%M")}
        observed = agent.observe(profile)
        plan = agent.plan(profile, busy)
        validation = agent.validate(plan, busy, available_minutes)
        st.session_state["last_plan"] = plan
        st.session_state["last_profile"] = profile
        st.session_state["last_validation"] = validation
        st.session_state["last_busy"] = busy
        st.session_state["last_observation"] = observed
    if "last_plan" in st.session_state:
        plan = st.session_state["last_plan"]
        validation = st.session_state["last_validation"]
        if validation["feasible"]:
            st.success("Plan passes the prototype's feasibility checks.")
        else:
            st.warning("The plan did not pass every feasibility check. Review the time inputs.")
        a, b, c = st.columns(3)
        a.metric("Planned minutes", validation["total_minutes"])
        b.metric("Time budget", available_minutes)
        c.metric("Feasibility", "Pass" if validation["feasible"] else "Review")
        if plan:
            st.markdown(f"""
            <div style="margin:12px 0 18px">
              <div style="font-size:1.7rem;font-weight:750;color:#8ab4ff">Daily Schedule</div>
              <div style="color:#9ca3b8">Your schedule, one activity at a time · {day.strftime('%A, %d %b %Y')}</div>
            </div>
            """, unsafe_allow_html=True)
            # Timeline cards mirror the requested Daily Flow-style reference and remain responsive.
            for idx, item in enumerate(sorted(plan, key=lambda x: x.start)):
                starts = item.start.strftime("%I:%M %p").lstrip("0")
                ends = item.end.strftime("%I:%M %p").lstrip("0")
                icon = "💪" if "strength" in item.activity_name.lower() or "bodyweight" in item.activity_name.lower() else "🚶"
                done = st.checkbox(
                    f"{icon}  {item.activity_name}   ·   {starts} – {ends}   ·   {item.duration_minutes} min",
                    key=f"timeline_done_{day.isoformat()}_{idx}"
                )
                state_label = "Completed" if done else ("Up next" if idx == 0 else "Scheduled")
                state_color = "#22c55e" if done else ("#8b5cf6" if idx == 0 else "#64748b")
                st.markdown(f"""
                <div style="margin:0 0 14px 14px;padding:16px 18px;border-radius:17px;
                    border:1px solid {state_color}55;border-left:4px solid {state_color};
                    background:linear-gradient(110deg,#111b2b,#121322);">
                  <div style="display:flex;justify-content:space-between;gap:10px;align-items:center">
                    <span style="font-weight:750;font-size:1.05rem;color:#f4f5ff">{icon} {item.activity_name}</span>
                    <span style="font-size:.75rem;color:{state_color};background:{state_color}18;
                      padding:4px 9px;border-radius:20px">{state_label}</span>
                  </div>
                  <div style="color:#aeb5c8;margin-top:8px">◷ {starts} – {ends}　•　{item.duration_minutes} mins　•　Fitness</div>
                  <div style="color:#c4b5fd;font-size:.88rem;margin-top:7px">{item.reason}</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No feasible activity slot found. Try a wider planning window or adjust busy blocks.")
        st.markdown("#### Grounded explanation")
        st.write(grounded_explanation(st.session_state["last_profile"], plan, kb["activities"]))
    else:
        st.info("Set your preferences in the sidebar and click **Generate activity plan**.")

with tabs[1]:
    st.subheader("🤖 FitPilot AI Coach")
    st.caption("Ask about activities, your plan, schedule, or the activity knowledge base.")
    if "fitness_chat" not in st.session_state: st.session_state.fitness_chat = []
    st.markdown("**Recommended questions**")
    cols=st.columns(2)
    for i,q in enumerate(RECOMMENDED_QUESTIONS):
        if cols[i%2].button(q,key=f"rec_{i}",use_container_width=True):
            st.session_state.pending_fitness_question=q
    for msg in st.session_state.fitness_chat:
        with st.chat_message(msg["role"]): st.markdown(msg["content"])
    question=st.chat_input("Type your fitness question…")
    if not question and st.session_state.get("pending_fitness_question"):
        question=st.session_state.pop("pending_fitness_question")
    if question:
        st.session_state.fitness_chat.append({"role":"user","content":question})
        with st.chat_message("user"): st.markdown(question)
        profile=st.session_state.get("last_profile",{"goal":goal,"preference":preference,"experience":experience,"available_minutes":available_minutes})
        with st.chat_message("assistant"):
            with st.spinner("FitPilot is thinking…"): answer=answer_fitness_question(question,kb,profile)
            st.markdown(answer)
        st.session_state.fitness_chat.append({"role":"assistant","content":answer})
    if st.button("Clear chat",key="clear_fitness_chat"):
        st.session_state.fitness_chat=[]
        st.rerun()

with tabs[2]:
    st.subheader("Comparison of AI paradigms")
    st.dataframe([
        {"Paradigm":"Rule-based AI","Use":"Explicit constraints and domain rules","Strength":"Predictable and explainable","Limitation":"Rules require manual updates"},
        {"Paradigm":"Search / optimisation","Use":"Find feasible time/activity combinations","Strength":"Respects hard constraints","Limitation":"Search can grow with problem size"},
        {"Paradigm":"Machine learning","Use":"Learn preferences from feedback","Strength":"Can personalise over time","Limitation":"Needs interaction or training data"},
        {"Paradigm":"LLM / RAG","Use":"Natural language and explanations grounded in retrieved facts","Strength":"Flexible interaction","Limitation":"Must be constrained and validated"},
        {"Paradigm":"Hybrid AI","Use":"Combines rules, search, preference ranking, and LLM","Strength":"Balances reliability and usability","Limitation":"More components to integrate"}
    ], use_container_width=True, hide_index=True)
    st.write("This prototype implements rule-based ranking and interval search. ML preference learning is described as a future extension; no model is trained in this build.")

with tabs[3]:
    st.subheader("Activity knowledge base")
    st.dataframe(kb["activities"], use_container_width=True, hide_index=True)
    st.subheader("Planning rules")
    st.dataframe(kb["rules"], use_container_width=True, hide_index=True)

with tabs[4]:
    st.subheader("Scope and safety")
    st.markdown("""
    - This is a **non-clinical** activity planning prototype.
    - It does not diagnose, treat, or prevent medical conditions.
    - Activity descriptions are general and should be adapted to individual comfort.
    - Stop an activity if it causes pain, dizziness, or unusual discomfort; seek appropriate professional guidance when needed.
    - The LLM is optional. Without an API key, the app uses a deterministic grounded explanation.
    - The scheduler, not the LLM, decides and validates time slots.
    """)
    st.subheader("Project modules")
    st.markdown("""
    - `app.py`: Streamlit user interface
    - `planner.py`: knowledge loading, interval search, rules, agent and validation
    - `llm_grounding.py`: optional LLM explanation with grounded fallback
    - `knowledge_base.json`: activity facts and planning rules
    - `tests/test_planner.py`: unit tests
    """)
