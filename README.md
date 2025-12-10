ASTRA: Socially Intelligent Multi-Agent AI Tutor for Introductory Programming

ASTRA is a research prototype that explores socially intelligent, multi-agent AI tutoring for introductory Python programming.  
It is designed to support experimental conditions where students:
- work alone with a single AI tutor (PTA), or  
- work in pairs with a single AI tutor, or  
- work in pairs with a multi-agent tutor (Pedagogical Tutor Agent + Collaborative Group Agent).

The system logs rich interaction data (dialogue, code attempts, conditions, timing) for research into learning, collaboration and equity.

Project structure

Key files:

- `app.py`  
  Main Streamlit web app.  
  - Student mode: chat-based interface to ASTRA (task selection, condition selection, final code submission).  
  - Marker mode: view sessions, inspect final solutions, assign rubric-based grades, export summary CSV.

- `astra_backend.py`  
  Backend logic for the CASM state and multi-agent orchestration:
  - Maintains conversational state and basic social-cognitive features.
  - Routes messages to the Pedagogical Tutor Agent (PTA) and, in multi-agent conditions, to the Collaborative Group Agent (CGA).
  - Handles logging of each turn to JSONL.

- `solution_logging.py`  
  Utilities for:
  - Saving final code solutions and marking information.
  - Exporting grading summaries for analysis.

- `logs/`  
  Default directory for JSONL logs (one file per date/session).  
  Each line typically contains: timestamp, group ID, condition, task, speaker, message text, agent response, and (where applicable) CGA action tags.

- `venv/`  
  Local Python virtual environment (note tracked by Git if `.gitignore` is set correctly).

> Important: No OpenAI keys or other secrets are stored in the repository. The system reads `OPENAI_API_KEY` from the environment.


Requirements

- Python 3.9+
- A valid OpenAI API key with access to a GPT-4-class model.
- Recommended: Unix-like environment (macOS/Linux) or WSL on Windows.

Python dependencies (typical):

- `streamlit`
- `openai` (new Python client)
- `tqdm`
- `pydantic` and friends

If you have a `requirements.txt`, install with:

```bash
pip install -r requirements.txt
