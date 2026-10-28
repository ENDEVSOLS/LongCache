"""Conservative admission, strict response codecs and explainable local verification."""

from __future__ import annotations
import inspect
import json
import re
from decimal import Decimal
from .decision import PolicyDecision
from .identity import freeze


def strict_json(value):
    def check(item):
        if item is None or type(item) in (str, bool, int, float):
            return
        if type(item) is list:
            for x in item:
                check(x)
            return
        if type(item) is dict and all(type(k) is str for k in item):
            for x in item.values():
                check(x)
            return
        raise ValueError("Unsupported JSON response type")

    check(value)
    encoded = json.dumps(value, allow_nan=False, ensure_ascii=False)
    encoded.encode("utf-8")
    return json.loads(encoded)


def encode_response(response, codec):
    if response is None or isinstance(response, BaseException):
        raise ValueError("No complete response")
    if codec == "json-v1":
        return strict_json(response)
    if codec not in ("openai-dict-v1", "openai-sdk-v1"):
        raise ValueError("Unsupported response codec")
    if codec == "openai-sdk-v1":
        from openai.types.chat import ChatCompletion

        if not isinstance(response, ChatCompletion):
            raise ValueError("Unexpected SDK response type")
        response = response.model_dump(mode="json")
    data = strict_json(response)
    if type(data) is not dict or type(data.get("choices")) is not list or len(data["choices"]) != 1:
        raise ValueError("Invalid completion choices")
    choice = data["choices"][0]
    if type(choice) is not dict or choice.get("finish_reason") != "stop":
        raise ValueError("Response is not a completed text answer")
    msg = choice.get("message")
    if (
        not isinstance(msg, dict)
        or msg.get("role") != "assistant"
        or type(msg.get("content")) is not str
    ):
        raise ValueError("Unsupported completion message")
    if msg.get("refusal") or msg.get("tool_calls") or msg.get("function_call") or msg.get("audio"):
        raise ValueError("Refusal or unsupported modality")
    return data


def decode_response(payload, codec):
    if codec == "openai-sdk-v1":
        from openai.types.chat import ChatCompletion

        response = ChatCompletion.model_validate(payload)
        encode_response(response, codec)
        return response
    return encode_response(payload, codec)


def call_hook(hook, *args):
    try:
        result = hook(*args)
        if inspect.isawaitable(result):
            if inspect.iscoroutine(result):
                result.close()
            return PolicyDecision("reject", "hook_failed")
        if not isinstance(result, PolicyDecision):
            return PolicyDecision("reject", "hook_failed")
        return result
    except Exception:
        return PolicyDecision("reject", "hook_failed")


def admit(response, context, hook=None):
    try:
        payload = encode_response(response, context.codec)
    except (ValueError, TypeError, ImportError, AttributeError):
        return None, PolicyDecision("reject", "response_inadmissible")
    if hook is not None:
        decision = call_hook(hook, context, freeze(payload))
        if decision.action != "accept":
            return None, decision
    return payload, PolicyDecision("accept", "accepted")


_NEGATION = re.compile(
    r"\b(?:not|no|never|without|cannot|can't|won't|don't|doesn't|isn't|aren't|shouldn't|wouldn't|couldn't|didn't|hasn't|haven't|wasn't|weren't)\b",
    re.I,
)
_NUMBER = re.compile(
    r"(?<!\w)([$€£]?)([+-]?\d+(?:\.\d+)?)(?:\s*-?\s*(%|percent\b|dollars?\b|euros?\b|pounds?\b|usd\b|eur\b|gbp\b|days?\b|hours?\b|minutes?\b|months?\b|years?\b|kg\b|km\b|mg\b))?",
    re.I,
)
_STOP_ENTITIES = set(
    "What When Where Who Why How Can Could Should Would Does Do Is Are Will The A An Please Tell Explain Give Show Which I My Our Your It This That For In On Of And Or To If May".split()
)
_ENTITY = re.compile(r"\b[A-Z][A-Za-z]+\b")
_RELATIONS = (
    {"before", "earlier", "prior"},
    {"after", "later"},
    {"above", "greater", "higher"},
    {"below", "less", "lower"},
    {"increase", "raise", "grow"},
    {"decrease", "reduce", "shrink"},
)


def numbers(text):
    result = []
    units = {
        "$": "usd",
        "dollar": "usd",
        "dollars": "usd",
        "€": "eur",
        "euro": "eur",
        "euros": "eur",
        "£": "gbp",
        "pound": "gbp",
        "pounds": "gbp",
        "percent": "%",
    }
    for m in _NUMBER.finditer(text):
        unit = (m[3] or m[1] or "").lower()
        unit = units.get(unit, unit.rstrip("s"))
        result.append((str(Decimal(m[2]).normalize()), unit))
    result.extend(
        (x.lower(), "identifier")
        for x in re.findall(r"\b(?=\w*\d)(?=\w*[A-Za-z])[A-Za-z0-9_-]+\b", text)
    )
    result.extend((x, "date") for x in re.findall(r"\b\d{4}-\d{2}-\d{2}\b", text))
    return sorted(result)


def verify(query, candidate, response, similarity, context, hook=None):
    if context.language != "en" or not query.isascii() or not candidate.isascii():
        return PolicyDecision("abstain", "verification_abstained")
    a, b = query.replace("’", "'"), candidate.replace("’", "'")
    if len(_NEGATION.findall(a)) != len(_NEGATION.findall(b)):
        return PolicyDecision("reject", "negation_mismatch")
    if numbers(a) != numbers(b):
        return PolicyDecision("reject", "number_mismatch")
    entities_a = {x.casefold() for x in _ENTITY.findall(a) if x not in _STOP_ENTITIES}
    entities_b = {x.casefold() for x in _ENTITY.findall(b) if x not in _STOP_ENTITIES}
    if entities_a != entities_b:
        return PolicyDecision("reject", "entity_mismatch")
    for entity in context.protected_entities:
        pattern = r"(?<!\w)" + re.escape(entity) + r"(?!\w)"
        if bool(re.search(pattern, a, re.I)) != bool(re.search(pattern, b, re.I)):
            return PolicyDecision("reject", "entity_mismatch")
    words_a, words_b = set(re.findall(r"[a-z]+", a.lower())), set(re.findall(r"[a-z]+", b.lower()))
    if [bool(words_a & group) for group in _RELATIONS] != [
        bool(words_b & group) for group in _RELATIONS
    ]:
        return PolicyDecision("reject", "relation_mismatch")
    if hook is not None:
        return call_hook(hook, query, candidate, freeze(response), similarity, context)
    return PolicyDecision("accept", "accepted")
