# run_astra.py

import os
from astra_backend import (
    LLMClient, CASMState, Message, handle_turn
)

from openai import OpenAI

class OpenAILLMClient(LLMClient):
    def __init__(self, model: str = "gpt-4o-mini"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
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
    if "OPENAI_API_KEY" not in os.environ:
        print("Please set your OPENAI_API_KEY environment variable first.")
        return

    client = OpenAILLMClient()

    casm = CASMState()
    history = []
    participants = ["student_A", "student_B"]
    task_context = (
        "You are helping students debug and understand short Python programs. "
        "Focus on reasoning, not just giving them code."
    )

    print("ASTRA tutor demo. Type 'quit' to exit.\n")

    while True:
        user_input = input("Student_A: ")
        if user_input.strip().lower() in {"quit", "exit"}:
            print("Goodbye.")
            break

        msg = Message(
            sender_id="student_A",
            sender_role="student",
            content=user_input,
        )

        casm, history, agent_resp = handle_turn(
            client=client,
            new_message=msg,
            history=history,
            casm=casm,
            participant_ids=participants,
            task_context=task_context,
        )

        if agent_resp:
            label = "Tutor" if agent_resp.agent_role == "pta" else "Facilitator"
            print(f"{label} [{agent_resp.action_tag}]: {agent_resp.content}\n")
        else:
            print("(No response from agent.)\n")


if __name__ == "__main__":
    main()
