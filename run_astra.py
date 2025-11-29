# run_astra.py

import os
import json
from datetime import datetime

from astra_backend import (
    LLMClient, CASMState, Message, handle_turn
)

from openai import OpenAI


class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        # Read API key from environment variable
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


def main():
    # Check API key exists
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set your OPENAI_API_KEY environment variable first.")
        return

    client = OpenAILLMClient()

    # ---------- LOGGING SETUP ----------
    os.makedirs("logs", exist_ok=True)
    log_filename = os.path.join(
        "logs",
        f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    )
    # -----------------------------------

    casm = CASMState()
    history = []
    participants = ["student_A", "student_B"]
    task_context = (
        "You are helping students debug and understand short Python programs. "
        "Focus on reasoning, not just giving them code."
    )

    print("ASTRA group tutor demo. Type 'quit' to exit.\n")
    print("Type messages as either:\n"
          "  A: your message here\n"
          "  B: your message here\n")

    while True:
        user_input = input("> ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        # ---------- PARSE WHICH STUDENT IS SPEAKING ----------
        if user_input.startswith("A:") or user_input.startswith("a:"):
            student_id = "student_A"
            content = user_input[2:].strip()
        elif user_input.startswith("B:") or user_input.startswith("b:"):
            student_id = "student_B"
            content = user_input[2:].strip()
        else:
            # Default to student_A if no prefix is given
            student_id = "student_A"
            content = user_input
        # -----------------------------------------------------

        msg = Message(
            sender_id=student_id,
            sender_role="student",
            content=content,
        )

        casm, history, agent_resp = handle_turn(
            client=client,
            new_message=msg,
            history=history,
            casm=casm,
            participant_ids=participants,
            task_context=task_context,
            cga_frequency=4,  # CGA tries to intervene every 4 student turns
        )

        if agent_resp:
            if agent_resp.agent_role == "pta":
                label = "Tutor"
            else:
                label = "Facilitator"  # CGA
            print(f"{label} [{agent_resp.action_tag}]: {agent_resp.content}\n")
        else:
            print("(No response from agent.)\n")

        # ---------- LOG THIS TURN ----------
        record = {
            "timestamp": msg.timestamp,
            "student_id": msg.sender_id,
            "student_msg": msg.content,
            "agent_role": agent_resp.agent_role if agent_resp else None,
            "agent_action": agent_resp.action_tag if agent_resp else None,
            "agent_msg": agent_resp.content if agent_resp else None,
        }
        with open(log_filename, "a") as f:
            f.write(json.dumps(record) + "\n")
        # -----------------------------------


if __name__ == "__main__":
    main()
