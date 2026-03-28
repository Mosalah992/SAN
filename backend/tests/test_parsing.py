"""
test_parsing.py — Unit tests for insight JSON parsing. No Ollama required.
Run: python -m backend.test_parsing
"""

from __future__ import annotations

import sys
from pathlib import Path

_BACKEND = Path(__file__).resolve().parent
if str(_BACKEND) not in sys.path:
    sys.path.insert(0, str(_BACKEND))

from curiosity_json import parse_json_from_llm
from curiosity_insight import extract_insight_safe


def main() -> None:
    failed = 0

    # Test 1: Valid JSON
    test1 = '{"claim":"Test claim","counter":"Test counter","confidence_delta":0.1,"should_write_to_kb":true}'
    result1 = parse_json_from_llm(test1, log_on_fail=False)
    if result1 is None:
        print("[FAIL] Test 1: Valid JSON did not parse")
        failed += 1
    else:
        print("[OK] Test 1: Valid JSON")

    # Test 2: JSON with markdown fences
    test2 = '```json\n{"claim":"Test","counter":"Test","confidence_delta":0.1,"should_write_to_kb":false}\n```'
    result2 = parse_json_from_llm(test2, log_on_fail=False)
    if result2 is None:
        print("[FAIL] Test 2: Markdown-wrapped JSON did not parse")
        failed += 1
    else:
        print("[OK] Test 2: Markdown-wrapped JSON")

    # Test 3: extract_insight_safe clamps out-of-range delta
    test3 = '{"claim":"Test","counter":"Test","confidence_delta":0.5,"should_write_to_kb":true}'
    result3 = extract_insight_safe(test3)
    if result3 is None:
        print("[FAIL] Test 3: extract_insight_safe rejected valid JSON")
        failed += 1
    elif result3.get("confidence_delta") != 0.3:
        print("[FAIL] Test 3: Delta not clamped (got %s)" % result3.get("confidence_delta"))
        failed += 1
    else:
        print("[OK] Test 3: Delta clamped to 0.3")

    # Test 4: Empty response
    result4 = parse_json_from_llm("", log_on_fail=False)
    if result4 is not None:
        print("[FAIL] Test 4: Empty string should return None")
        failed += 1
    else:
        print("[OK] Test 4: Empty returns None")

    # Test 5: synthesis optional (missing from input → we add None)
    test5 = '{"claim":"X","counter":"Y","confidence_delta":0.0,"should_write_to_kb":false}'
    result5 = extract_insight_safe(test5)
    if result5 is None:
        print("[FAIL] Test 5: Missing synthesis should still pass")
        failed += 1
    else:
        assert result5.get("synthesis") is None
        print("[OK] Test 5: synthesis optional")

    if failed:
        print("\n[FAIL] %d test(s) failed" % failed)
        sys.exit(1)
    print("\n[OK] All parsing tests passed!")


if __name__ == "__main__":
    main()
