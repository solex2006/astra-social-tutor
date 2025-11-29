# astra_backend.py

from dataclasses import dataclass, field
from typing import List, Dict, Literal, Optional, Tuple
import time
import json


Role = Literal["student", "pta", "cga"]
Affect = Literal["neutral", "confused", "frustrated", "engaged"]
KnowledgeLevel = Literal["low", "medium", "high"]
Talkativeness = Literal["low", "medium", "high"]


@dataclass
class CASMProfile:
    knowledge_level: KnowledgeLevel = "medium"
    misconceptions: List[str] = field(default_factory=list)
    affect: Affect = "neutral"
    talkativeness: Talkativeness = "medium"


@dataclass
class CASMState:
    semantic: Dict[str, Dict[str, float]] = field(default_factory=dict)
    episodic: Dict[str, List[str]] = field(default_factory=dict)
    social: Dict[str, CASMProfile] = field(default_factory=dict)


@dataclass
class Message:
    sender_id: str
    sender_role: Role
    content: str
    timestamp: float = field(default_factory=lambda: time.time())


@dataclass
class AgentResponse:
    content: str
    agent_role: Literal["pta", "cga"]
    action_tag: str  # e.g. "HINT", "QUESTION"


class LLMClient:
    """
    Abstract LLM client – we will subclass this for OpenAI.
    """
    def generate(self, system_prompt: str, user_prompt: str) -> str:
        raise NotImplementedError("Implement this in a subclass.")


def update_casm_from_messages(
    client: LLMClient,
    casm: CASMState,
    student_id: str,
    recent_messages: List[Message],
) -> CASMState:
    utterances = [m.content for m in recent_messages if m.sender_id == student_id]
    last_text = "\n".join(utterances[-5:])

    system_prompt = (
        "You are analysing a student's recent messages in a programming task.\n"
        "Infer a simple learner state with fields:\n"
        "- knowledge_level: one of ['low', 'medium', 'high']\n"
        "- affect: one of ['neutral', 'confused', 'frustrated', 'engaged']\n"
        "- talkativeness: one of ['low', 'medium', 'high']\n"
        "- misconceptions: list of short phrases describing misconceptions you detect.\n"
        "Respond ONLY as valid JSON with these keys."
    )

    user_prompt = (
        "Recent messages from this student:\n"
        f"{last_text or '[no recent messages]'}"
    )

    try:
        raw = client.generate(system_prompt, user_prompt)
        data = json.loads(raw)
    except Exception:
        return casm

    profile = casm.social.get(student_id, CASMProfile())

    knowledge_level = data.get("knowledge_level", profile.knowledge_level)
    affect = data.get("affect", profile.affect)
    talk = data.get("talkativeness", profile.talkativeness)
    misconceptions = data.get("misconceptions", profile.misconceptions)

    casm.social[student_id] = CASMProfile(
        knowledge_level=knowledge_level,
        affect=affect,
        talkativeness=talk,
        misconceptions=misconceptions,
    )

    return casm


def summarise_casm_for_student(casm: CASMState, student_id: str) -> str:
    profile = casm.social.get(student_id, CASMProfile())
    misconceptions = "; ".join(profile.misconceptions) if profile.misconceptions else "none noted yet"
    return (
        f"Knowledge level: {profile.knowledge_level}. "
        f"Affect: {profile.affect}. "
        f"Talkativeness: {profile.talkativeness}. "
        f"Key misconceptions: {misconceptions}."
    )


def summarise_group_state(casm: CASMState, participant_ids: List[str]) -> str:
    lines = []
    for sid in participant_ids:
        profile = casm.social.get(sid, CASMProfile())
        misconceptions = "; ".join(profile.misconceptions) if profile.misconceptions else "none"
        lines.append(
            f"{sid}: knowledge={profile.knowledge_level}, affect={profile.affect}, "
            f"talkativeness={profile.talkativeness}, misconceptions={misconceptions}"
        )
    return "\n".join(lines)


def pta_step(
    client: LLMClient,
    student_msg: Message,
    casm: CASMState,
    task_context: str,
) -> AgentResponse:
    casm_summary = summarise_casm_for_student(casm, student_msg.sender_id)

    system_prompt = (
        "You are a patient programming tutor helping students learn.\n"
        "Use Socratic questioning and hints before giving full solutions.\n"
        "Adapt your response to the learner state I give you.\n"
        "Classify your intervention as one of: QUESTION, HINT, EXPLANATION, ENCOURAGEMENT.\n"
        "At the end of your reply, add a line starting with 'ACTION_TAG:' followed by the tag."
    )

    user_prompt = (
        f"Task context:\n{task_context}\n\n"
        f"Learner state:\n{casm_summary}\n\n"
        f"Student's latest message:\n{student_msg.content}\n\n"
        "Now respond to the student, following the guidelines."
    )

    raw = client.generate(system_prompt, user_prompt)

    lines = raw.strip().splitlines()
    action_tag = "UNKNOWN"
    content_lines: List[str] = []

    for line in lines:
        if line.strip().upper().startswith("ACTION_TAG:"):
            action_tag = line.split(":", 1)[1].strip().upper()
        else:
            content_lines.append(line)

    content = "\n".join(content_lines).strip()

    return AgentResponse(
        content=content,
        agent_role="pta",
        action_tag=action_tag,
    )


def cga_step(
    client: LLMClient,
    history: List[Message],
    casm: CASMState,
    participant_ids: List[str],
) -> AgentResponse:
    last_turns = history[-15:]
    convo = "\n".join(f"{m.sender_id} ({m.sender_role}): {m.content}" for m in last_turns)
    group_state = summarise_group_state(casm, participant_ids)

    system_prompt = (
        "You are a facilitator helping a small group of programming students work together.\n"
        "Your job is to improve collaboration, ensure everyone participates, and help them "
        "resolve confusion without taking over the task.\n"
        "You may invite quieter students, prompt for summaries, or suggest turn-taking.\n"
        "Classify your intervention as one of: INVITE_QUIET_MEMBER, SUMMARISE, "
        "MEDIATE_CONFLICT, ENCOURAGE_COLLAB, NONE.\n"
        "At the end of your reply, add a line starting with 'ACTION_TAG:' and the tag."
    )

    user_prompt = (
        "Here is the recent group conversation:\n"
        f"{convo}\n\n"
        "Here is the current group state:\n"
        f"{group_state}\n\n"
        "If an intervention would help, write a short message to the group.\n"
        "If no intervention is needed, reply 'No intervention needed.' with ACTION_TAG:NONE."
    )

    raw = client.generate(system_prompt, user_prompt)

    lines = raw.strip().splitlines()
    action_tag = "NONE"
    content_lines: List[str] = []

    for line in lines:
        if line.strip().upper().startswith("ACTION_TAG:"):
            action_tag = line.split(":", 1)[1].strip().upper()
        else:
            content_lines.append(line)

    content = "\n".join(content_lines).strip()

    return AgentResponse(
        content=content,
        agent_role="cga",
        action_tag=action_tag,
    )


def handle_turn(
    client: LLMClient,
    new_message: Message,
    history: List[Message],
    casm: CASMState,
    participant_ids: List[str],
    task_context: str,
    cga_frequency: int = 5,
) -> Tuple[CASMState, List[Message], Optional[AgentResponse]]:
    history = history + [new_message]

    casm = update_casm_from_messages(client, casm, new_message.sender_id, history)

    total_student_turns = len([m for m in history if m.sender_role == "student"])

    response: Optional[AgentResponse] = None

    if "@" in new_message.content and "tutor" in new_message.content.lower():
        response = pta_step(client, new_message, casm, task_context)

    elif total_student_turns % cga_frequency == 0 and total_student_turns > 0:
        response = cga_step(client, history, casm, participant_ids)
        if response.action_tag == "NONE":
            response = None

    else:
        response = pta_step(client, new_message, casm, task_context)

    return casm, history, response
