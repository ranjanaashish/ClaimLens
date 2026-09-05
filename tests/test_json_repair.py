"""Unit tests for robust JSON parsing and repair."""
import json
import re

def clean_and_parse(raw: str) -> dict:
    text = raw.strip()
    # Strip markdown fences
    m = re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", text)
    if m:
        text = m.group(1).strip()
    else:
        s = text.find("{")
        e = text.rfind("}")
        if s != -1 and e != -1 and e > s:
            text = text[s : e + 1]
        elif s != -1:
            text = text[s:]

    # Try direct parse
    try:
        return json.loads(text)
    except Exception:
        pass

    # Remove trailing commas
    cleaned = re.sub(r",\s*([}\]])", r"\1", text)
    try:
        return json.loads(cleaned)
    except Exception:
        pass

    # Repair unclosed quotes and brackets
    repaired = cleaned
    quotes = len(re.findall(r'(?<!\\)"', repaired))
    if quotes % 2 != 0:
        repaired += '"'
    open_b = repaired.count("[") - repaired.count("]")
    if open_b > 0:
        repaired += "]" * open_b
    open_c = repaired.count("{") - repaired.count("}")
    if open_c > 0:
        repaired += "}" * open_c
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)

    try:
        return json.loads(repaired)
    except Exception:
        pass

    # Regex fallback
    data = {}
    m_codes = re.search(r'"damage_codes"\s*:\s*\[(.*?)\]', raw, re.DOTALL)
    if m_codes:
        data["damage_codes"] = re.findall(r'"([A-Za-z0-9_]+)"', m_codes.group(1))
    m_sev = re.search(r'"severity"\s*:\s*"([^"]+)"', raw)
    if m_sev:
        data["severity"] = m_sev.group(1).strip()
    m_cost = re.search(r'"cost_range"\s*:\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]', raw)
    if m_cost:
        data["cost_range"] = [int(m_cost.group(1)), int(m_cost.group(2))]
    m_conf = re.search(r'"confidence"\s*:\s*([0-9.]+)', raw)
    if m_conf:
        data["confidence"] = float(m_conf.group(1))
    m_reason = re.search(r'"reasoning"\s*:\s*"([^"]*)', raw)
    if m_reason:
        data["reasoning"] = m_reason.group(1).strip()
    return data

broken_examples = [
    # 1. Truncated at reasoning
    '{"damage_codes": ["FRONT_BUMPER_CRUSH"], "severity": "Moderate", "cost_range": [1500, 3000], "confidence": 0.85, "fraud_flag": false, "reasoning": "The front bumper',
    # 2. Markdown fence with conversational preamble and postamble
    'Here is the assessment:\n```json\n{"damage_codes": ["HOOD_CREASE"], "severity": "Minor", "cost_range": [500, 1200], "confidence": 0.9, "reasoning": "Minor dent."}\n```\nHope this helps!',
    # 3. Unterminated string inside array
    '{"damage_codes": ["DOOR_FL_DENT',
    # 4. Trailing commas
    '{"damage_codes": ["PAINT_TRANSFER",], "severity": "Minor", "cost_range": [200, 800],}'
]

for idx, ex in enumerate(broken_examples):
    res = clean_and_parse(ex)
    print(f"Example {idx+1} parsed:", res)
    assert isinstance(res, dict) and len(res) > 0

print("\nAll JSON repair tests passed successfully!")
