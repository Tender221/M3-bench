"""Verbatim frozen safe-exact router functions; see source_provenance.json."""
import re
import json
import hashlib

def normalize_safe(text):
    # Preserve decimal points, signs, word order, internal punctuation and negation.
    text = re.sub(r'\s+', ' ', str(text).strip().casefold()).strip('"\'“”‘’')
    text = text.rstrip('.!?').strip()
    return text


def safe_exact(prediction, references):
    value = normalize_safe(prediction)
    return bool(value) and any(value == normalize_safe(r) for r in references if normalize_safe(r))


def initial_decision(record, references):
    if record.get('status') != 'ok':
        return 'inference_error', None
    if safe_exact(record.get('pred', ''), references) and len(record.get('pred','').split())<=4:
        return 'safe_exact', True
    return 'judge_pending', None


def judge_key(question, references, prediction):
    value = [question, sorted(set(references)), prediction]
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, separators=(',', ':')).encode()).hexdigest()


