import re
from typing import Any, Dict, List


def get_stage_names(response_json: Dict[str, Any]) -> List[str]:
    stages = response_json["choices"][0]["message"]["custom_content"]["stages"]
    return [
        re.sub(
            r"\s*\[.*\]$", "", stage["name"]
        ).strip()  # cut [0.03s] at the end of the stage name
        for stage in stages
    ]
