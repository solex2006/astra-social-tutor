import os
import json
import io
import csv
from datetime import datetime

import streamlit as st
from openai import OpenAI

from solution_logging import log_solution, load_solutions, log_grade, load_grades
from astra_backend import LLMClient, CASMState, Message, handle_turn


# ---------- CONFIG: STUDENTS, TASKS, CONDITIONS ----------

STUDENTS = {
    "Student A": "student_A",
    "Student B": "student_B",
    "Student C": "student_C",
    "Student D": "student_D",
}

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

# Three configurations matching the Methods section
CONDITIONS = {
    "Alone + AI tutor (Individual–Single-Agent)": "alone_tutor",
    "Pair + AI tutor (Pair–Single-Agent)": "pair_tutor",
    "Pair + multi-agent tutor (Pair–Multi-Agent)": "pair_multiagent",
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


# ---------- INITIALISE SESSION STATE (STUDENT MODE) ----------

def init_state():
    if "initialised" not in st.session_state:
        st.session_state.initialised = True

        default_task = list(TASKS.keys())[0]
        default_condition_label = list(CONDITIONS.keys())[0]
        default_condition_code = CONDITIONS[default_condition_label]

        st.session_state.client = OpenAILLMClient()
        st.session_state.casm = CASMState()
        st.session_state.history = []
        # all student ids
        st.session_state.participants = list(STUDENTS.values())

        st.session_state.selected_task = default_task
        st.session_state.task_context = TASKS[default_task]["context"]
        st.session_state.task_description = TASKS[default_task]["description"]

        st.session_state.condition_label = default_condition_label
        st.session_state.condition_code = default_condition_code

        # researcher will type this per group
        st.session_state.group_id = ""

        os.makedirs("logs", exist_ok=True)
        st.session_state.log_filename = os.path.join(
            "logs",
            f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl",
        )

        # chat history for display: list of dicts {"role", "name", "text"}
        st.session_state.chat = []

        # storage for final solution per task
        st.session_state.solution_code = ""

        # current unsent chat message
        st.session_state.chat_message = ""


# ---------- STUDENT UI ----------

def run_student_ui():
    st.title("ASTRA Group Tutor 🧠")

    st.markdown(
        "### Session setup\n\n"
        "Before you start, please enter your group identifier and choose the configuration "
        "assigned by the instructor (alone vs pair, and whether you see a multi-agent tutor)."
    )

    # Group ID and condition selection (for the researcher / instructor)
    col1, col2 = st.columns(2)
    with col1:
        group_id_input = st.text_input(
            "Group ID (e.g. G01, LabA3)",
            value=st.session_state.group_id,
        )
    with col2:
        condition_label = st.selectbox(
            "Collaboration configuration",
            options=list(CONDITIONS.keys()),
            index=list(CONDITIONS.keys()).index(st.session_state.condition_label),
        )

    # Update session state
    st.session_state.group_id = group_id_input.strip()
    st.session_state.condition_label = condition_label
    st.session_state.condition_code = CONDITIONS[condition_label]

    st.markdown(
        "### How to use this tool\n\n"
        "1. Work **alone or in a pair** according to the configuration shown above.\n"
        "2. Select the programming task you are working on.\n"
        "3. Read the task description carefully.\n"
        "4. Choose who is speaking before sending each message.\n"
        "5. Type your message explaining your thinking, code, or questions.\n"
        "6. The Tutor helps with the programming content; in the multi-agent condition, "
        "the Facilitator helps you collaborate effectively.\n\n"
        "Try to explain your reasoning to each other (or to yourself, if working alone), "
        "not just ask for the final answer."
    )

    # ----- Task selection and description -----
    st.markdown("---")
    task_names = list(TASKS.keys())
    current_index = task_names.index(st.session_state.selected_task)
    selected_task = st.selectbox(
        "Choose your task:",
        options=task_names,
        index=current_index,
    )

    if selected_task != st.session_state.selected_task:
        st.session_state.selected_task = selected_task
        st.session_state.task_context = TASKS[selected_task]["context"]
        st.session_state.task_description = TASKS[selected_task]["description"]
        # Reset solution text when switching tasks
        st.session_state.solution_code = ""

    st.subheader("Current programming task")
    st.text(st.session_state.task_description)

    st.markdown("---")

    # Who is speaking now?
    student_labels = list(STUDENTS.keys())
    speaker = st.radio(
        "Who is speaking?",
        options=student_labels,
        index=0,
        horizontal=True,
    )

    # Chat input (positioned BEFORE transcript and final solution box)
    st.subheader("Chat with the tutor (and facilitator, if enabled)")
    st.session_state.chat_message = st.text_input(
        "Type your message and click 'Send message'",
        value=st.session_state.chat_message,
        key="chat_message_input",
    )
    send_clicked = st.button("Send message")

    # -------------------------------------------------
    # Handle chat message first (so transcript below reflects updates)
    # -------------------------------------------------
    if send_clicked:
        user_input = st.session_state.chat_message.strip()
        if user_input:
            student_id = STUDENTS[speaker]
            msg = Message(sender_id=student_id, sender_role="student", content=user_input)

            # Decide CGA frequency based on condition:
            # Only the pair + multi-agent tutor condition activates the Facilitator.
            if st.session_state.condition_code == "pair_multiagent":
                cga_frequency = 4
            else:
                # Use a very large number so the Facilitator never fires in realistic sessions
                cga_frequency = 10**9

            # Show spinner while the tutor responds
            with st.spinner("Tutor is thinking..."):
                try:
                    casm, history, agent_resp = handle_turn(
                        client=st.session_state.client,
                        new_message=msg,
                        history=st.session_state.history,
                        casm=st.session_state.casm,
                        participant_ids=st.session_state.participants,
                        task_context=st.session_state.task_context,
                        cga_frequency=cga_frequency,
                    )
                except Exception as e:
                    st.error(
                        "An error occurred while generating the tutor response. "
                        f"Details: {e}"
                    )
                    return

            st.session_state.casm = casm
            st.session_state.history = history

            # Add student message to display
            st.session_state.chat.append(
                {"role": "student", "name": speaker, "text": user_input}
            )

            # Add agent message to display
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
                "group_id": st.session_state.group_id,
                "configuration": st.session_state.condition_code,
                "student_id": msg.sender_id,
                "student_label": speaker,
                "student_msg": msg.content,
                "agent_role": agent_resp.agent_role if agent_resp else None,
                "agent_action": agent_resp.action_tag if agent_resp else None,
                "agent_msg": agent_resp.content if agent_resp else None,
                "task_name": st.session_state.selected_task,
            }
            with open(st.session_state.log_filename, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

            # Clear chat input for next message
            st.session_state.chat_message = ""

            st.rerun()

    # -------------------------------------------------
    # Conversation transcript (now BELOW the send button)
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Conversation so far")

    if not st.session_state.chat:
        st.caption("No messages yet. Start by sending a question or idea above.")
    else:
        for msg in st.session_state.chat:
            role = msg["role"]
            name = msg.get("name", "")
            text = msg["text"]

            if role == "student":
                st.markdown(f"**{name}:** {text}")
            else:
                st.markdown(
                    f"<span style='color:#1f4e79; font-weight:bold;'>{name}:</span> {text}",
                    unsafe_allow_html=True,
                )

    # -------------------------------------------------
    # Final solution capture for the CURRENT task
    # -------------------------------------------------
    st.markdown("---")
    st.subheader("Final solution for this task")

    st.session_state.solution_code = st.text_area(
        "Paste or type your FINAL code for the CURRENT task here.\n"
        "When you are happy with it, click 'Submit final solution'.",
        value=st.session_state.solution_code,
        height=200,
        key="solution_code_area",
    )

    if st.button("💾 Submit final solution for this task"):
        group_id = st.session_state.group_id.strip()
        configuration = st.session_state.condition_code.strip()
        task_id = st.session_state.selected_task.strip()
        code = st.session_state.solution_code

        if not group_id:
            st.warning("Please set a Group ID at the top before submitting a solution.")
        elif not configuration:
            st.warning("Please set the configuration before submitting.")
        elif not task_id:
            st.warning("Please select a task before submitting a solution.")
        elif not code.strip():
            st.warning("Please enter some code before submitting.")
        else:
            log_solution(
                group_id=group_id,
                configuration=configuration,
                task_id=task_id,
                solution_code=code,
            )
            st.success(
                f"Final solution saved for group {group_id}, task '{task_id}'. "
                "You can still edit and submit again; only the latest submission "
                "will be used during grading."
            )


# ---------- MARKER UI ----------

def run_marker_ui():
    st.title("ASTRA – Marker view for grading solutions")

    st.markdown(
        "This view lets you review and grade final solutions submitted "
        "through the ASTRA tutor, and export a summary of grades."
    )

    solutions = load_solutions()
    if not solutions:
        st.info("No solutions have been submitted yet. "
                "Run the Student mode with learners first.")
        return

    # Build a list of unique (group_id, task_id) combinations
    unique_pairs = sorted({(s['group_id'], s['task_id']) for s in solutions})

    label_map = {
        (g, t): f"{g} – {t}"
        for (g, t) in unique_pairs
    }
    labels = [label_map[p] for p in unique_pairs]

    st.subheader("1. Choose a group and task")
    selected_label = st.selectbox("Group and task", options=labels)

    # Recover (group_id, task_id) from label
    sel_group, sel_task = selected_label.split(" – ", 1)

    # Filter submissions for this group+task
    submissions = [
        s for s in solutions
        if s["group_id"] == sel_group and s["task_id"] == sel_task
    ]
    # Take the most recent as the final one
    solution = submissions[-1]

    st.markdown("### 2. Inspect final code")
    st.markdown(f"**Group:** {sel_group}  \n**Task:** `{sel_task}`")
    st.code(solution["solution_code"], language="python")

    st.markdown("### 3. Assign rubric score")
    st.markdown(
        "- **0** – Incorrect or no viable solution\n"
        "- **1** – Partially correct solution (core idea present, minor errors)\n"
        "- **2** – Fully correct solution"
    )

    score = st.radio(
        "Score",
        options=[0, 1, 2],
        index=1,
        horizontal=True,
    )

    comments = st.text_area(
        "Optional comments (for your analysis, not shown to students):",
        height=120,
    )

    if st.button("✅ Save grade"):
        log_grade(
            group_id=sel_group,
            task_id=sel_task,
            score=score,
            comments=comments.strip(),
        )
        st.success("Grade saved to logs/grades.jsonl.")

    # ---------- Export summary ----------
    st.markdown("---")
    st.subheader("4. Export grades summary")

    grades = load_grades()
    if not grades:
        st.info("No grades have been recorded yet.")
        return

    # Build CSV from grades
    output = io.StringIO()
    fieldnames = ["timestamp", "group_id", "task_id", "score", "comments"]
    writer = csv.DictWriter(output, fieldnames=fieldnames)
    writer.writeheader()
    for g in grades:
        row = {fn: g.get(fn, "") for fn in fieldnames}
        writer.writerow(row)

    csv_bytes = output.getvalue().encode("utf-8")

    st.download_button(
        "⬇️ Download grades as CSV",
        data=csv_bytes,
        file_name="astra_grades_summary.csv",
        mime="text/csv",
    )


# ---------- MAIN APP ----------

def main():
    st.set_page_config(page_title="ASTRA Group Tutor", page_icon="🧠")

    # Sidebar shared across modes
    with st.sidebar:
        mode = st.radio("Mode", ["Student", "Marker"], index=0)
        st.header("About ASTRA")
        st.write(
            "You are interacting with ASTRA, a socially intelligent AI tutor "
            "designed to support individuals and small groups working on "
            "introductory programming problems."
        )
        st.write(
            "- In all conditions, the Tutor focuses on hints, questions, and explanations.\n"
            "- In the multi-agent condition, the Facilitator focuses on how the pair "
            "collaborates (who speaks, who explains, whether you summarise, etc.)."
        )
        st.caption(
            "Please avoid sharing personal or sensitive information. "
            "Your interactions and submitted solutions may be logged for "
            "research and evaluation."
        )

    if mode == "Student":
        init_state()
        run_student_ui()
    else:
        run_marker_ui()


if __name__ == "__main__":
    main()
