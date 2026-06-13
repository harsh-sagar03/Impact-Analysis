from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.model_comparison import train_all_models


def main() -> None:
    payload = train_all_models()
    print("Model training complete.")
    for task, model_name in payload["recommended_models"].items():
        print(f"- {task}: {model_name}")


if __name__ == "__main__":
    main()
