"""
The pipeline orchestrator. Owned by P1.

This is the BACKBONE that ties all 4 slices together:
    intent (P1) -> retrieve (P2) -> filter_eligible (P2) -> recommend (P3) -> response

Right now it's a plain function calling each step. That's intentional for mock-first:
the whole app works today. P1's REAL job is to convert this into a LangGraph StateGraph
(nodes = steps, edges = flow, with a branch for clarification and retry). The node
functions stay the same - only the wiring changes.

GUARDRAIL enforced here: recommendations may only reference eligible product ids.
"""
import random
import uuid

from app.agents.intent import extract_intent
from app.agents.profile import update_profile
from app.contracts.models import ChatResponse, Receipts, Recommendation
from app.engine.eligibility import filter_eligible
from app.recommend.recommender import recommend
from app.retrieval.retriever import retrieve
from app.retrieval.catalog import load_catalog


# ── Chitchat response templates (sales-steering) ─────────────────────────────

_GREETING_RESPONSES = [
    (
        "Hey there! 👋 Welcome to DTDL — I'm your personal shopping assistant! "
        "We've got some amazing deals on phones, plans, and accessories right now. "
        "Are you looking for a new phone, a mobile plan, or maybe some cool accessories today?"
    ),
    (
        "Hello! 😊 Great to have you here at DTDL! Whether you need the latest smartphone, "
        "an unlimited data plan, or some handy accessories — I've got you covered. "
        "What are you looking for today?"
    ),
    (
        "Hi there! 🎉 Welcome to DTDL's shopping assistant! "
        "We've got everything from flagship phones to budget-friendly plans. "
        "Tell me what you're after and I'll find the perfect match for you!"
    ),
]

_GREETING_RESPONSES_NAMED = [
    (
        "Hey there, {name}! 👋 Welcome to DTDL — I'm your personal shopping assistant! "
        "We've got some incredible deals right now. "
        "Are you looking for a new phone, a mobile plan, or maybe some cool accessories?"
    ),
    (
        "Hello, {name}! 😊 Great to have you at DTDL! "
        "Whether you need the latest smartphone or an affordable plan — I've got you covered. "
        "What can I help you find today?"
    ),
]

_INTRODUCTION_RESPONSES = [
    (
        "Nice to meet you, {name}! 😊 I'm here to help you find the perfect deal on DTDL. "
        "We've got some amazing phones and plans right now — the Google Pixel 9 just dropped "
        "with incredible AI camera features! What catches your eye?"
    ),
    (
        "Great to meet you, {name}! 🎉 Welcome to DTDL! "
        "I can help you find the best phone, plan, or accessories to fit your needs. "
        "What's most important to you — a great camera, fast data, or staying within budget?"
    ),
    (
        "Hey {name}, pleasure to meet you! 😄 "
        "I'm your DTDL shopping guide. We've got phones from Samsung, Google, and Apple, "
        "plus flexible plans starting at just €20/month. "
        "What would you like to explore first?"
    ),
]

_SMALLTALK_RESPONSES = [
    (
        "I'm doing great, thanks for asking! 😄 "
        "By the way, we just got some fantastic new arrivals at DTDL — "
        "including the latest smartphones with amazing cameras. "
        "Want me to find something perfect for you?"
    ),
    (
        "All good here, thanks! 🙌 "
        "I've been helping customers find amazing deals today. "
        "Speaking of which — are you in the market for a new phone or data plan? "
        "I'd love to help you find the best deal!"
    ),
    (
        "Doing wonderful, thank you! 😊 "
        "Always excited to help shoppers find the best tech. "
        "We have some great bundle deals right now that can save you money — "
        "interested in hearing more?"
    ),
]

_SMALLTALK_RESPONSES_NAMED = [
    (
        "I'm doing great, {name}! Thanks for asking 😄 "
        "By the way, we've got some fantastic deals right now at DTDL. "
        "Want me to find something perfect for you?"
    ),
    (
        "All good here, {name}! 🙌 "
        "Speaking of which — are you looking for a new phone, plan, or accessories? "
        "I'd love to help you find the best deal!"
    ),
]

_THANKS_RESPONSES = [
    (
        "You're welcome! 🙌 Is there anything else I can help you find? "
        "We've got some great bundle deals that could save you money — "
        "like combining a phone with an unlimited data plan!"
    ),
    (
        "Happy to help! 😊 Before you go — have you checked out our accessories? "
        "A protective case or wireless earbuds could be a great add-on! "
        "Or maybe you'd like to explore a new data plan?"
    ),
    (
        "Anytime! 🎉 If you need anything else, I'm right here. "
        "We've got new phones, flexible plans, and handy accessories — "
        "just say the word and I'll find you the best options!"
    ),
]

_THANKS_RESPONSES_NAMED = [
    (
        "You're welcome, {name}! 🙌 Is there anything else I can help you find? "
        "We've got some amazing bundle deals that could save you money!"
    ),
    (
        "Happy to help, {name}! 😊 Let me know if you'd like to explore "
        "phones, plans, or accessories — I'm here for you!"
    ),
]


def _get_trending_products(n: int = 2) -> list[Recommendation]:
    """Pick a few 'trending' products from the catalog to seed interest."""
    catalog = load_catalog()
    # Prefer in-stock phones and popular plans
    interesting = [
        p for p in catalog
        if p.in_stock and p.type.value in ("phone", "plan", "bundle")
    ]
    if not interesting:
        interesting = [p for p in catalog if p.in_stock]
    picks = random.sample(interesting, min(n, len(interesting)))
    recs = []
    for i, p in enumerate(picks):
        price_str = (
            f"€{p.price_monthly}/mo" if p.price_monthly
            else f"€{p.price_onetime}"
        )
        recs.append(Recommendation(
            product_id=p.id,
            rank=i + 1,
            score=0.0,
            why=f"🔥 Trending now — {p.name} ({price_str}). {p.description}",
            bundle=[],
        ))
    return recs


def _pick_response(templates: list[str], named_templates: list[str] | None, name: str | None) -> str:
    """Pick a random response, using the named variant if a name is known."""
    if name and named_templates:
        return random.choice(named_templates).format(name=name)
    return random.choice(templates)


def _handle_chitchat(intent, session, conversation_id: str, message: str) -> ChatResponse:
    """Handle chitchat messages with warm, sales-steering responses."""
    conv_history = session.get_conversation(conversation_id)
    name = session.profile.user_name
    chitchat_type = intent.chitchat_type or "greeting"

    if chitchat_type == "greeting":
        reply = _pick_response(_GREETING_RESPONSES, _GREETING_RESPONSES_NAMED, name)
        recs = _get_trending_products(2)
        nba = ["🛍️ Check out our latest phones!", "📱 Explore data plans starting at €20/mo"]
    elif chitchat_type == "introduction":
        reply = _pick_response(_INTRODUCTION_RESPONSES, None, None)
        # Always use the name for introductions
        name = session.profile.user_name  # freshly updated by update_profile
        if name:
            reply = reply.replace("{name}", name)
        else:
            reply = reply.replace("{name}, ", "").replace("{name}", "friend")
        recs = _get_trending_products(2)
        nba = ["📱 Tell me what features matter most to you!", "💰 Set a budget and I'll find the best fit"]
    elif chitchat_type == "smalltalk":
        reply = _pick_response(_SMALLTALK_RESPONSES, _SMALLTALK_RESPONSES_NAMED, name)
        recs = _get_trending_products(1)
        nba = ["🔥 Ask me about our latest phones!", "📦 Check out our bundle deals"]
    elif chitchat_type == "thanks":
        reply = _pick_response(_THANKS_RESPONSES, _THANKS_RESPONSES_NAMED, name)
        recs = []  # Don't push products on "thanks" — just nudge
        nba = ["🛒 Browse more products", "📱 Need help with phones, plans, or accessories?"]
    else:
        reply = _pick_response(_GREETING_RESPONSES, _GREETING_RESPONSES_NAMED, name)
        recs = _get_trending_products(2)
        nba = []

    # Record turn in conversation history
    conv_history.append({"role": "user", "content": message})
    conv_history.append({
        "role": "assistant",
        "content": reply,
        "recommendations": [r.model_dump() for r in recs],
    })

    return ChatResponse(
        reply_text=reply,
        recommendations=recs,
        nba=nba,
        cart=session.cart,
        receipts=Receipts(),
        conversation_id=conversation_id,
    )


def run_pipeline(message: str, session, conversation_id: str) -> ChatResponse:
    # Each conversation is a separate thread with its own history for AI context.
    # Profile is global across conversations (accumulated learnings persist).
    conv_history = session.get_conversation(conversation_id)

    # 1. Understand + learn (P1)
    # Pass THIS thread's history (loaded fresh from the persisted session) so
    # multi-turn context survives restarts without bleeding across unrelated
    # conversation threads.
    intent = extract_intent(message, conv_history, session.profile)
    session.profile = update_profile(session.profile, message, intent)
    intent.profile = session.profile

    # 1a. CHITCHAT BRANCH — engage the user warmly and steer toward a sale.
    # This fires BEFORE the product pipeline so casual messages never reach
    # the retriever/recommender (which would produce confusing results).
    if intent.is_chitchat:
        return _handle_chitchat(intent, session, conversation_id, message)

    # 1b. Ask instead of guess (trust behaviour)
    if intent.clarification_needed:
        conv_history.append({"role": "user", "content": message})
        conv_history.append({"role": "assistant", "content": intent.clarification_question, "recommendations": []})
        return ChatResponse(
            reply_text=intent.clarification_question,
            recommendations=[],
            nba=[],
            cart=session.cart,
            receipts=Receipts(),
            conversation_id=conversation_id,
        )

    # 2. Find candidates (P2)
    candidates = retrieve(intent)

    # 3. Filter to what's actually offerable - SOURCE OF TRUTH (P2)
    evaluated = filter_eligible(candidates, intent)
    eligible = [e for e in evaluated if e.eligible]

    # 4. Rank + explain + nudge (P3)
    recs, nba = recommend(eligible, session.profile, session.cart)

    # 4a. GUARDRAIL: the LLM can never introduce a product the engine rejected.
    eligible_ids = {e.product.id for e in eligible}
    recs = [r for r in recs if r.product_id in eligible_ids]

    # 5. Compose reply + receipts
    reply = _compose_reply(recs, eligible, intent)
    receipts = Receipts(
        retrieved_ids=[c.id for c in candidates],
        rules_fired=sorted({rule for e in evaluated for rule in (e.reasons + e.failed_rules)}),
        shown_ids=[r.product_id for r in recs],
    )

    # Record turn; persist recommendations so the frontend can restore product cards from history.
    conv_history.append({"role": "user", "content": message})
    conv_history.append({
        "role": "assistant",
        "content": reply,
        "recommendations": [r.model_dump() for r in recs],
    })

    return ChatResponse(
        reply_text=reply,
        recommendations=recs,
        nba=nba,
        cart=session.cart,
        receipts=receipts,
        conversation_id=conversation_id,
    )


def _compose_reply(recs, eligible, intent) -> str:
    if not eligible:
        # Brand-aware, honest refusal - the deterministic engine found nothing
        # offerable, and we say so rather than silently swapping in something else.
        subject = f"{intent.brand} option" if intent.brand else "option"
        budget = f" under €{int(intent.budget_monthly_max)}/mo" if intent.budget_monthly_max else ""
        return (
            f"I couldn't find an in-stock {subject}{budget} that fits. "
            "Want me to relax the budget, or show the closest alternatives?"
        )
    if not recs:
        return "I found some options but need a bit more detail to recommend the best one."
    lead = "Here's my top pick" if len(recs) == 1 else "Here are my top picks"
    return f"{lead} — {recs[0].why}"

