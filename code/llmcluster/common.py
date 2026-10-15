from datetime import datetime, timezone
import json
import os
from pathlib import Path
HERE = Path(__file__).resolve().parent

def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temporary, path)


def require(condition, message):
    if not condition:
        raise ValueError(message)

