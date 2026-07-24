"""
Intent extraction. Owned by P1.

EXPOSES:  extract_intent(message, history, profile) -> Intent

MOCK:  regex/keyword parse - fast, zero cost, zero infra.
REAL:  OpenAI extracts structured JSON matching the Intent contract. Any failure
       (missing/bad key, rate limit, network, malformed JSON) falls back to the
       mock parser automatically - flipping MOCK_MODE=false can never 500 the
       pipeline, it just prefers the real extraction when it's available.

`history` is the CURRENT conversation thread's turns (passed in by the
orchestrator), already loaded fresh from the persisted session - so multi-turn
context survives restarts without bleeding across unrelated threads.
"""
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from app.config import MOCK_MODE
from app.contracts.models import Intent, PreferenceProfile

logger = logging.getLogger(__name__)

# Simple feature keyword map for the mock.
_FEATURE_WORDS = {
    "camera": ["camera", "photo", "photos", "picture"],
    "eu_roaming": ["travel", "roaming", "europe", "abroad"],
    "gaming": ["gaming", "game", "games"],
    "5g": ["5g", "fast internet"],
    "unlimited": ["unlimited", "lots of data", "stream", "streaming"],
}

# Map the words a customer uses to the canonical brand names in the catalog.
_BRAND_WORDS = {
    "Apple": ["apple", "iphone", "ios"],
    "Samsung": ["samsung", "galaxy"],
    "Google": ["google", "pixel"],
    "Telekom": ["telekom", "magenta"],
}
_VALID_BRANDS = set(_BRAND_WORDS.keys())

# ── Chitchat detection patterns ────────────────────────────────────────────────
_GREETING_WORDS = [
    "hello", "hi", "hey", "hola", "howdy", "yo", "sup",
    "good morning", "good afternoon", "good evening", "good night",
    "namaste", "namaskar", "hii", "hiii", "hiiii",
    "greetings", "what's up", "whats up", "wassup",
]
_THANKS_WORDS = [
    "thanks", "thank you", "thankyou", "thx", "ty", "cheers",
    "appreciated", "much appreciated", "great thanks",
    "thanks a lot", "thank you so much",
]
_SMALLTALK_PATTERNS = [
    r"\bhow are you\b", r"\bhow('?s| is) it going\b", r"\bwhat('?s| is) up\b",
    r"\bhow('?s| is) your day\b", r"\bhow do you do\b",
    r"\bnice to (meet|talk)\b", r"\bgood to (see|hear)\b",
    r"\bhow('?s| is) everything\b", r"\bwhat('?s| is) new\b",
    r"\bhow('?s| is) life\b",
]
_NAME_PATTERNS = [
    r"\bmy name is\s+(\w+)",
    r"\bi(?:'?m| am)\s+(\w+)",
    r"\bcall me\s+(\w+)",
    r"\bthis is\s+(\w+)",
    r"\bi go by\s+(\w+)",
    r"\bname('?s| is)\s+(\w+)",
]
# Words that disqualify "I'm ..." from being a name introduction
# (e.g. "I'm looking for a phone" should NOT be treated as name = "looking").
_NAME_DISQUALIFY = {
    "looking", "searching", "interested", "wanting", "trying", "thinking",
    "hoping", "planning", "needing", "wondering", "here", "back", "new",
    "not", "going", "happy", "fine", "good", "great", "okay", "ok",
    "doing", "sure", "ready", "done",
}


def _detect_chitchat(message: str) -> tuple[bool, Optional[str], Optional[str]]:
    """Detect if a message is chitchat rather than a product query.

    Returns (is_chitchat, chitchat_type, extracted_name).
    chitchat_type is one of: "greeting", "introduction", "smalltalk", "thanks", or None.
    """
    msg = message.strip().lower()
    # Remove punctuation for matching
    msg_clean = re.sub(r"[^\w\s'']", "", msg)

    # 1. Check for name introductions first (most specific)
    for pattern in _NAME_PATTERNS:
        m = re.search(pattern, msg_clean, re.IGNORECASE)
        if m:
            # Get the last group (some patterns have 2 groups)
            name = m.group(m.lastindex)
            if name.lower() not in _NAME_DISQUALIFY:
                return True, "introduction", name.capitalize()

    # 2. Check for thanks
    if any(w in msg_clean for w in _THANKS_WORDS):
        # Make sure it's primarily a thanks message, not "thanks, show me phones"
        # If the message is short or dominated by thanks words, treat as chitchat
        words = msg_clean.split()
        if len(words) <= 6:
            return True, "thanks", None

    # 3. Check for greetings
    for greeting in _GREETING_WORDS:
        if greeting in msg_clean:
            # Make sure the message is primarily a greeting (not "hey show me iphones")
            words = msg_clean.split()
            if len(words) <= 8 and not any(
                any(w in msg for w in kw_list)
                for kw_list in _FEATURE_WORDS.values()
            ) and not any(
                any(w in msg for w in kw_list)
                for kw_list in _BRAND_WORDS.values()
            ):
                return True, "greeting", None

    # 4. Check for small talk patterns
    for pattern in _SMALLTALK_PATTERNS:
        if re.search(pattern, msg_clean, re.IGNORECASE):
            return True, "smalltalk", None

    return False, None, None


def _canonical_brand(raw: Optional[str]) -> Optional[str]:
    """Normalise a free-text brand ('iPhone', 'galaxy') to a catalog brand."""
    if not raw:
        return None
    low = raw.strip().lower()
    for canon, words in _BRAND_WORDS.items():
        if low == canon.lower() or any(w in low for w in words):
            return canon
    return None


# Structured output schema - OpenAI fills this; profile is added by us afterwards.
class _IntentOutput(BaseModel):
    use_case: str = ""
    budget_monthly_max: Optional[float] = None
    brand: Optional[str] = None
    priority_features: list[str] = Field(default_factory=list)
    product_types: list[str] = Field(default_factory=list)
    clarification_needed: bool = False
    clarification_question: Optional[str] = None
    is_chitchat: bool = False
    chitchat_type: Optional[str] = None


_SYSTEM_PROMPT = """You extract shopping intent from a customer's message for a \
Telekom phone/plan/accessory shop. Fill the response schema with:

  use_case: short paraphrase of what they want
  budget_monthly_max: EUR/month number if mentioned (e.g. "under 40" -> 40), else null
  brand: the canonical brand IF the customer names a specific brand or model -
         one of [Apple, Samsung, Google, Telekom] (map "iPhone"->Apple,
         "Galaxy"->Samsung, "Pixel"->Google), else null
  priority_features: subset of [camera, eu_roaming, gaming, 5g, unlimited]
  product_types: subset of [phone, plan, accessory]
  clarification_needed: true ONLY if message is too vague to act on at all
  clarification_question: a single short question if clarification_needed, else null
  is_chitchat: true if the message is a greeting ("hello", "hi"), an introduction
               ("my name is X"), small talk ("how are you?"), or thanks ("thank you").
               These are NOT product queries.
  chitchat_type: one of ["greeting", "introduction", "smalltalk", "thanks"] if
                 is_chitchat is true, else null.

Rules:
- clarification_needed=true only if no budget, no brand, no feature, no product type, fewer than ~6 words.
- Set brand ONLY when the customer explicitly names one; never guess a brand from features.
- Never invent features/types not implied by the message.
- If is_chitchat is true, leave product-related fields empty/null (they're not shopping yet).
"""


def extract_intent(message: str, history: list[dict], profile: PreferenceProfile) -> Intent:
    if MOCK_MODE:
        return _extract_mock(message, history, profile)
    try:
        return _extract_real(message, history, profile)
    except Exception as e:  # noqa - real extraction must never crash the pipeline
        logger.warning("OpenAI intent extraction failed (%s); falling back to mock parsing.", e)
        return _extract_mock(message, history, profile)


def _extract_mock(message: str, history: list[dict], profile: PreferenceProfile) -> Intent:
    msg = message.lower()

    # ── Chitchat check (before any product extraction) ─────────────────────
    is_chitchat, chitchat_type, extracted_name = _detect_chitchat(message)
    if is_chitchat:
        return Intent(
            use_case=message,
            is_chitchat=True,
            chitchat_type=chitchat_type,
            profile=profile,
        )

    # budget: find "40 euro", "under 30", "€25"
    budget = profile.budget_monthly_max
    m = re.search(r"(?:under|below|max|€|eur|euro[s]?)\D{0,6}(\d{1,4})", msg)
    if not m:
        m = re.search(r"(\d{1,4})\s*(?:€|eur|euro)", msg)
    if m:
        budget = float(m.group(1))

    features = [feat for feat, words in _FEATURE_WORDS.items() if any(w in msg for w in words)]

    brand = None
    for canon, words in _BRAND_WORDS.items():
        if any(w in msg for w in words):
            brand = canon
            break

    product_types = []
    if any(w in msg for w in ["phone", "device", "smartphone"]):
        product_types.append("phone")
    if any(w in msg for w in ["plan", "tariff", "data", "sim"]):
        product_types.append("plan")
    if any(w in msg for w in ["case", "earbuds", "charger", "accessory", "accessories"]):
        product_types.append("accessory")

    # Clarification: too vague ("good and cheap" with nothing concrete)
    clarify = (
        not features and not product_types and not brand
        and budget is None and len(msg.split()) < 6
    )

    return Intent(
        use_case=message,
        budget_monthly_max=budget,
        brand=brand,
        priority_features=features,
        product_types=product_types,
        clarification_needed=clarify,
        clarification_question=(
            "Happy to help! Is this mainly for calls and messaging, or do you also want "
            "a strong camera, gaming, or lots of data? And what monthly budget works for you?"
            if clarify else None
        ),
        profile=profile,
    )


def _extract_real(message: str, history: list[dict], profile: PreferenceProfile) -> Intent:
    """Real extraction via openai's native Pydantic structured output.

    Uses client.beta.chat.completions.parse(response_format=_IntentOutput)
    instead of fragile json.loads(). `history` is the current thread's turns,
    already loaded from the persisted session by the orchestrator.
    """
    from openai import OpenAI

    from app.config import OPENAI_API_KEY, OPENAI_MODEL

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")

    past = [
        {"role": t.get("role", "user"), "content": t.get("content", "")}
        for t in (history[-6:] if history else [])
    ]
    messages = (
        [{"role": "system", "content": _SYSTEM_PROMPT}]
        + past
        + [{"role": "user", "content": message}]
    )

    client = OpenAI(api_key=OPENAI_API_KEY)
    resp = client.beta.chat.completions.parse(
        model=OPENAI_MODEL,
        messages=messages,
        response_format=_IntentOutput,
        temperature=0,
        timeout=15,
    )
    result: _IntentOutput = resp.choices[0].message.parsed

    valid_features = set(_FEATURE_WORDS.keys())
    valid_types = {"phone", "plan", "accessory"}
    brand = _canonical_brand(result.brand)

    # Use OpenAI's chitchat detection, but also cross-check with our regex
    # detector for robustness (belt-and-suspenders).
    is_cc = result.is_chitchat
    cc_type = result.chitchat_type
    if not is_cc:
        is_cc, cc_type, _ = _detect_chitchat(message)

    return Intent(
        use_case=result.use_case or message,
        budget_monthly_max=result.budget_monthly_max or profile.budget_monthly_max,
        brand=brand,
        priority_features=[f for f in result.priority_features if f in valid_features],
        product_types=[t for t in result.product_types if t in valid_types],
        clarification_needed=result.clarification_needed,
        clarification_question=result.clarification_question,
        is_chitchat=is_cc,
        chitchat_type=cc_type,
        profile=profile,
    )
