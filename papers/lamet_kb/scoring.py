from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Sequence, Set, Tuple


def normalize_text(text: str) -> str:
    lowered = text.lower()
    lowered = lowered.replace("-", " ")
    lowered = re.sub(r"\s+", " ", lowered)
    return lowered


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen: Set[str] = set()
    result: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return result


def match_term(haystack: str, patterns: Sequence[str]) -> Tuple[bool, List[str]]:
    hits: List[str] = []
    for pattern in patterns:
        normalized_pattern = normalize_text(pattern)
        if normalized_pattern in haystack:
            hits.append(pattern)
    return bool(hits), unique_preserve_order(hits)


def build_haystack(record: Dict[str, Any]) -> str:
    authors = " ".join(record.get("authors", []))
    categories = " ".join(record.get("categories", []))
    pieces = [
        record.get("title", ""),
        record.get("summary", ""),
        authors,
        categories,
        record.get("comment", ""),
        record.get("journal_ref", ""),
    ]
    return normalize_text(" ".join(piece for piece in pieces if piece))


def determine_confidence(score: int, label: str) -> str:
    if label == "core" and score >= 36:
        return "high"
    if label == "secondary" and score >= 18:
        return "high"
    if label == "irrelevant" and score <= 0:
        return "high"
    if label == "irrelevant":
        return "low"
    return "medium"


def score_record(
    record: Dict[str, Any],
    config: Dict[str, Any],
    manual_seed_ids: Set[str],
) -> Dict[str, Any]:
    haystack = build_haystack(record)
    score = 0
    reasons: List[str] = []
    anchor_hits = 0
    matched_groups: List[str] = []

    if record["arxiv_id"] in manual_seed_ids:
        score += 100
        anchor_hits += 1
        reasons.append("manual seed")

    for item in config["anchor_terms"]:
        matched, hits = match_term(haystack, item["patterns"])
        if matched:
            score += int(item["weight"])
            anchor_hits += 1
            matched_groups.append(item["name"])
            reasons.append(f"anchor:{item['name']} [{', '.join(hits[:3])}]")

    for item in config["secondary_terms"]:
        matched, hits = match_term(haystack, item["patterns"])
        if matched:
            score += int(item["weight"])
            matched_groups.append(item["name"])
            reasons.append(f"secondary:{item['name']} [{', '.join(hits[:3])}]")

    primary_category = record.get("primary_category", "")
    category_bonus = int(config.get("category_bonus", {}).get(primary_category, 0))
    if category_bonus:
        score += category_bonus
        reasons.append(f"category:{primary_category}")

    for item in config.get("negative_terms", []):
        matched, hits = match_term(haystack, item["patterns"])
        if matched:
            score += int(item["weight"])
            reasons.append(f"negative:{item['name']} [{', '.join(hits[:3])}]")

    thresholds = config["thresholds"]
    if anchor_hits > 0 and score >= thresholds["core_score"]:
        label = "core"
    elif score >= thresholds["secondary_score"] and (anchor_hits > 0 or score >= thresholds["core_score"]):
        label = "secondary"
    elif anchor_hits > 0 and score >= thresholds["review_score"]:
        label = "secondary"
        reasons.append("borderline anchor hit promoted to secondary")
    else:
        label = "irrelevant"

    record["score"] = score
    record["label"] = label
    record["confidence"] = determine_confidence(score, label)
    record["reasons"] = unique_preserve_order(reasons)
    record["matched_groups"] = unique_preserve_order(matched_groups)
    return record
