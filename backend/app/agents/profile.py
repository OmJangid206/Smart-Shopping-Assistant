"""
Preference profile updater. Owned by P1.
Learns from each message so recommendations get more personal over time.

EXPOSES:  update_profile(profile, message, intent) -> PreferenceProfile

Deterministic and simple on purpose (no ML). The profile biases P3's ranking.
Key behaviour: it LISTENS and ADAPTS - if the user changes their mind, update.
"""
import re

from app.contracts.models import Intent, PreferenceProfile

_BRANDS = ["samsung", "google", "apple", "pixel", "iphone", "galaxy"]
_BRAND_CANON = {"pixel": "Google", "google": "Google", "samsung": "Samsung",
                "galaxy": "Samsung", "apple": "Apple", "iphone": "Apple"}

# Name extraction patterns (same as intent.py's chitchat detector).
_NAME_PATTERNS = [
    r"\bmy name is\s+(\w+)",
    r"\bi(?:'?m| am)\s+(\w+)",
    r"\bcall me\s+(\w+)",
    r"\bthis is\s+(\w+)",
    r"\bi go by\s+(\w+)",
    r"\bname('?s| is)\s+(\w+)",
]
_NAME_DISQUALIFY = {
    "looking", "searching", "interested", "wanting", "trying", "thinking",
    "hoping", "planning", "needing", "wondering", "here", "back", "new",
    "not", "going", "happy", "fine", "good", "great", "okay", "ok",
    "doing", "sure", "ready", "done",
}


def update_profile(profile: PreferenceProfile, message: str, intent: Intent) -> PreferenceProfile:
    msg = message.lower()

    # Extract user name from introductions
    if intent.is_chitchat and intent.chitchat_type == "introduction":
        msg_clean = re.sub(r"[^\w\s'']", "", msg)
        for pattern in _NAME_PATTERNS:
            m = re.search(pattern, msg_clean, re.IGNORECASE)
            if m:
                name = m.group(m.lastindex)
                if name.lower() not in _NAME_DISQUALIFY:
                    profile.user_name = name.capitalize()
                    break

    if intent.budget_monthly_max is not None:
        profile.budget_monthly_max = intent.budget_monthly_max

    for f in intent.priority_features:
        if f not in profile.features_mentioned:
            profile.features_mentioned.append(f)

    # Brand affinity: from the LLM-extracted brand (most reliable) + raw keywords.
    if intent.brand and intent.brand not in profile.brands_viewed:
        profile.brands_viewed.append(intent.brand)
    for b in _BRANDS:
        if b in msg:
            canon = _BRAND_CANON[b]
            if canon not in profile.brands_viewed:
                profile.brands_viewed.append(canon)

    for t in intent.product_types:
        cat = t + "s"
        if cat not in profile.categories_browsed:
            profile.categories_browsed.append(cat)

    # Rejection signal: "not the iphone", "too expensive", "don't want apple"
    if any(w in msg for w in ["not the", "don't want", "dont want", "too expensive", "skip"]):
        for b in _BRANDS:
            if b in msg:
                canon = _BRAND_CANON[b]
                if canon not in profile.rejected:
                    profile.rejected.append(canon)

    return profile

