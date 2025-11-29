# app.py

import os
import json
from datetime import datetime

import streamlit as st
from openai import OpenAI

from astra_backend import (
    LLMClient, CASMState, Message, handle_turn
)


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

        st.session_state.client = OpenAILLMClient()
        st.session_state.casm = CASMState()
        st.session_state.history = []
        st.session_state.participants = ["student_A", "student_B"]
        st.session_state.task_context = (
            "You are helping students debug and understand short Python programs. "
            "Focus on reasoning, not just giving them code."
        )

        # Logging setup
        os.makedirs("logs", exist_ok=True)
        st.session_state.log_filename = os.path.join(
            "logs",
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
        )

        # Chat history for display
        st.session_state.chat = []  # list of dicts: {role, name, text}


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="ASTRA Group Tutor", page_icon="🧠")
    st.title("ASTRA Group Tutor 🧠")

    st.write(
        "This is your socially intelligent, multi-agent tutor prototype.\n"
        "Select who is speaking (Student A or Student B), type a message, and ASTRA will respond."
    )

    init_state()

    # Display chat history
    for msg in st.session_state.chat:
        role = msg["role"]
        name = msg.get("name", "")
        text = msg["text"]

        if role == "student":
            st.markdown(f"**{name}:** {text}")
        else:
            # Tutor or Facilitator
            st.markdown(f"**{name}:** {text}")

    st.markdown("---")

    # Who is speaking?
    speaker = st.radio(
        "Who is speaking?",
        options=["Student A", "Student B"],
        horizontal=True,
    )

    # Chat input
    user_input = st.chat_input("Type your message here")

    if user_input:
        # Map radio selection to internal student id
        if speaker == "Student A":
            student_id = "student_A"
        else:
            student_id = "student_B"

        # Create message object
        msg = Message(
            sender_id=student_id,
            sender_role="student",
            content=user_input,
        )

        # Call backend
        casm, history, agent_resp = handle_turn(
            client=st.session_state.client,
            new_message=msg,
            history=st.session_state.history,
            casm=st.session_state.casm,
            participant_ids=st.session_state.participants,
            task_context=st.session_state.task_context,
            cga_frequency=4,  # CGA tries to intervene every 4 student turns
        )

        # Update state
        st.session_state.casm = casm
        st.session_state.history = history

        # Add student message to display chat
        st.session_state.chat.append({
            "role": "student",
            "name": speaker,
            "text": user_input,
        })

        # Add agent message (if any)
        if agent_resp:
            if agent_resp.agent_role == "pta":
                label = "Tutor"
            else:
                label = "Facilitator"

            st.session_state.chat.append({
                "role": "agent",
                "name": f"{label} [{agent_resp.action_tag}]",
                "text": agent_resp.content,
            })

        # Log this turn
        record = {
            "timestamp": msg.timestamp,
            "student_id": msg.sender_id,
            "student_msg": msg.content,
            "agent_role": agent_resp.agent_role if agent_resp else None,
            "agent_action": agent_resp.action_tag if agent_resp else None,
            "agent_msg": agent_resp.content if agent_resp else None,
        }
        with open(st.session_state.log_filename, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Rerun to display updated chat
        st.rerun()


if __name__ == "__main__":
    main()
