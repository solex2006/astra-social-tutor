import json
from pathlib import Path
from datetime import datetime

# Directory to hold all logs
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)

# Files for solutions and grades
SOLUTIONS_LOG = LOG_DIR / "solutions.jsonl"
GRADES_LOG = LOG_DIR / "grades.jsonl"


def log_solution(group_id: str,
                 configuration: str,
                 task_id: str,
                 solution_code: str):
    """
    Append a final solution record to solutions.jsonl.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "group_id": group_id,
        "configuration": configuration,
        "task_id": task_id,
        "solution_code": solution_code,
    }
    with SOLUTIONS_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def log_grade(group_id: str,
              task_id: str,
              score: int,
              comments: str = ""):
    """
    Append a grade record to grades.jsonl.
    """
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "group_id": group_id,
        "task_id": task_id,
        "score": int(score),
        "comments": comments,
    }
    with GRADES_LOG.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")


def load_solutions():
    """
    Load all submitted solutions from solutions.jsonl.
    Returns a list of dict records.
    """
    if not SOLUTIONS_LOG.exists():
        return []

    records = []
    with SOLUTIONS_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def load_grades():
    """
    Load all grade records from grades.jsonl.
    Returns a list of dict records.
    """
    if not GRADES_LOG.exists():
        return []

    records = []
    with GRADES_LOG.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records
