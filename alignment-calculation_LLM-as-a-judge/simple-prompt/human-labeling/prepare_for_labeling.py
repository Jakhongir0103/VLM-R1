
#!/usr/bin/env python3
"""
add_human_fields.py

Load the judge's output JSON and add two empty fields for human evaluation:
  - human_aligns:  (to be filled with true/false)
  - human_comment: (optional free-text)

Outputs: alignment_with_human.json
"""

import json
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(
        description="Add human_eval fields to judge output JSON"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the original judge JSON (e.g. alignment_checks.json)"
    )
    parser.add_argument(
        "-o", "--output",
        type=Path,
        default=Path("alignment_with_human.json"),
        help="Path to write the augmented JSON"
    )
    args = parser.parse_args()

    # 1) Load the judge output
    data = json.loads(args.input.read_text())

    # 2) Append empty human evaluation fields
    for entry in data:
        entry["human_aligns"] = ""   # to be filled: "True" or "False"
        entry["human_comment"] = ""  # optional free-text

    # 3) Write out
    args.output.write_text(json.dumps(data, indent=2))
    print(f"Wrote {len(data)} entries with human fields → {args.output}")

if __name__ == "__main__":
    main()
