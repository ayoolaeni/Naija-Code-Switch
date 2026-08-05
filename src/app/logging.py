"""
Conversation logging (README §4): every exchange (input, output, generation
params, timestamp, backend) is logged to logs/conversations/ to support the
evaluation pipeline (§7) and keep results traceable.
"""
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
LOG_DIR = ROOT / "logs" / "conversations"


class ConversationLogger:
    def __init__(self, session_id: str | None = None, log_dir: Path = LOG_DIR):
        self.session_id = session_id or str(uuid.uuid4())
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.log_dir / f"{self.session_id}.jsonl"

    def log_exchange(self, user_input: str, model_output: str, backend_name: str, gen_params: dict | None = None):
        record = {
            "session_id": self.session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "backend": backend_name,
            "user_input": user_input,
            "model_output": model_output,
            "gen_params": gen_params or {},
        }
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record
