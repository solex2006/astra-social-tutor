# Streamlit UI for ASTRA group tutor with multiple students and tasks

import os
import json
from datetime import datetime

import streamlit as st
from openai import OpenAI

from astra_backend import LLMClient, CASMState, Message, handle_turn


# ---------- CONFIG: STUDENTS AND TASKS ----------

# You can add or remove students here
STUDENTS = {
    "Student A": "student_A",
    "Student B": "student_B",
    "Student C": "student_C",
    "Student D": "student_D",
}

# You can add more tasks here by following the same structure
TASKS = {
    "Task 1: Sum numbers from 1 to n": {
        "context": (
            "You are helping students design and debug a Python function "
            "sum_to_n(n) that returns the sum of all integers from 1 up to "
            "and including n. Focus on reasoning, not just giving them code."
        ),
        "description": (
            "You are working on this Python problem:\n\n"
            "Write a function sum_to_n(n) that returns the sum of all integers "
            "from 1 to n inclusive.\n\n"
            "Example:\n"
            "- sum_to_n(3) -> 6\n"
            "- sum_to_n(10) -> 55\n"
        ),
    },
    "Task 2: Debug a factorial function": {
        "context": (
            "You are helping students debug a Python function intended to "
            "compute factorial(n). Focus on helping them reason about loops "
            "and base cases."
        ),
        "description": (
            "You are working on this debugging problem:\n\n"
            "def factorial(n):\n"
            "    result = 0\n"
            "    for i in range(1, n):\n"
            "        result = result * i\n"
            "    return result\n\n"
            "Explain what this code currently does, why it is wrong, and how to fix it."
        ),
    },
    "Task 3: Filter even numbers from a list": {
        "context": (
            "You are helping students write and debug a Python function that "
            "filters even numbers from a list. Focus on list iteration and "
            "conditional logic."
        ),
        "description": (
            "You are working on this Python problem:\n\n"
            "Write a function filter_evens(numbers) that returns a new list "
            "containing only the even integers from the input list.\n\n"
            "Example:\n"
            "- filter_evens([1, 2, 3, 4]) -> [2, 4]\n"
            "- filter_evens([5, 7, 9]) -> []\n\n"
            "Discuss how you would design this function, and how you would test it "
            "with different inputs."
        ),
    },
}


# ---------- LLM CLIENT ----------

class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set.")
        self.client = OpenAI(api_key=api_key)
        self.model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        return resp.choices[0].message.content


# ---------- INITIALISE SESSION STATE ----------

def init_state():
    if "initialised" not in st.session_state:
        st.session_state.initialised = True

        default_task = list(TASKS.keys())[0]

        st.session_state.client = OpenAILLMClient()
        st.session_state.casm = CASMState()
        st.session_state.history = []
        # All student ids from STUDENTS mapping
        st.session_state.participants = list(STUDENTS.values())

        st.session_state.selected_task = default_task
        st.session_state.task_context = TASKS[default_task]["context"]
        st.session_state.task_description = TASKS[default_task]["description"]

        os.makedirs("logs", exist_ok=True)
        st.session_state.log_filename = os.path.join(
            "logs",
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        # Chat history: list of dicts {role, name, text}
        st.session_state.chat = []


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="ASTRA Group Tutor", page_icon="🧠")
    init_state()

    # Sidebar with study info
    with st.sidebar:
        st.header("About ASTRA")
        st.write(
            "You are interacting with **ASTRA**, a socially intelligent AI tutor "
            "designed to support small groups working on programming problems."
        )
        st.write(
            "- **Tutor** messages focus on hints, questions, and explanations.\n"
            "- **Facilitator** messages focus on how you collaborate as a group "
            "(who speaks, who explains, whether you summarise, etc.)."
        )
        st.caption(
            "Please avoid sharing personal or sensitive information. "
            "Your interactions may be logged for research and evaluation."
        )

    # Title and instructions
    st.title("ASTRA Group Tutor 🧠")

    st.markdown(
        "### How to use this tool\n\n"
        "1. Work in a small group as **Student A, Student B, Student C, Student D** (or fewer).\n"
        "2. Select the programming task you are working on.\n"
        "3. Read the task description carefully.\n"
        "4. Choose who is speaking before sending each message.\n"
        "5. Type your message explaining your thinking, code, or questions.\n"
        "6. The **Tutor** helps with the programming content; the **Facilitator** "
        "helps you collaborate effectively.\n\n"
        "Try to explain your reasoning to each other, not just ask for the final answer."
    )

    # ----- Task selection and description -----
    task_names = list(TASKS.keys())
    current_index = task_names.index(st.session_state.selected_task)
    selected_task = st.selectbox(
        "Choose your task:", options=task_names, index=current_index
    )

    if selected_task != st.session_state.selected_task:
        st.session_state.selected_task = selected_task
        st.session_state.task_context = TASKS[selected_task]["context"]
        st.session_state.task_description = TASKS[selected_task]["description"]

    st.subheader("Current programming task")
    st.text(st.session_state.task_description)

    st.markdown("---")

    # Show chat history
    for msg in st.session_state.chat:
        role = msg["role"]          # "student" or "agent"
        name = msg.get("name", "")
        text = msg["text"]

        if role == "student":
            st.markdown(f"**{name}:** {text}")
        else:
            st.markdown(
                f"<span style='color:#1f4e79; font-weight:bold;'>{name}:</span> {text}",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # Who is speaking now? (any of the configured students)
    student_labels = list(STUDENTS.keys())
    speaker = st.radio(
        "Who is speaking?",
        options=student_labels,
        index=0,
        horizontal=True,
    )

    # Input box
    user_input = st.chat_input("Type your message and press Enter")

    if user_input:
        # Map display name to internal student id
        student_id = STUDENTS[speaker]

        msg = Message(sender_id=student_id, sender_role="student", content=user_input)

        casm, history, agent_resp = handle_turn(
            client=st.session_state.client,
            new_message=msg,
            history=st.session_state.history,
            casm=st.session_state.casm,
            participant_ids=st.session_state.participants,
            task_context=st.session_state.task_context,
            cga_frequency=4,
        )

        # Update state
        st.session_state.casm = casm
        st.session_state.history = history

        # Add student message
        st.session_state.chat.append(
            {"role": "student", "name": speaker, "text": user_input}
        )

        # Add agent message if any
        if agent_resp:
            label = "Tutor" if agent_resp.agent_role == "pta" else "Facilitator"
            st.session_state.chat.append(
                {
                    "role": "agent",
                    "name": f"{label} [{agent_resp.action_tag}]",
                    "text": agent_resp.content,
                }
            )

        # Log this turn
        record = {
            "timestamp": msg.timestamp,
            "student_id": msg.sender_id,
            "student_msg": msg.content,
            "agent_role": agent_resp.agent_role if agent_resp else None,
            "agent_action": agent_resp.action_tag if agent_resp else None,
            "agent_msg": agent_resp.content if agent_resp else None,
            "task_name": st.session_state.selected_task,
        }
        with open(st.session_state.log_filename, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Refresh UI with new messages
        st.rerun()


if __name__ == "__main__":
    main()
