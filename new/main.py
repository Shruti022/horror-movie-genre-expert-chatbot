"""
Psychological Horror Film Expert Bot
FastAPI backend with Gemini Flash
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()

# ── Gemini setup ──────────────────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel("gemini-1.5-flash")

# ── Distress / safety keywords (Python backstop) ──────────────────────────────
DISTRESS_KEYWORDS = re.compile(
    r"\b(suicid|self.harm|kill myself|hurt myself|end my life|crisis|helpline)\b",
    re.IGNORECASE,
)

# ── Out-of-scope pattern backstop ─────────────────────────────────────────────
# These catch cases the LLM might slip on
OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(recipe|stock price|weather|sports score|homework|math problem|"
    r"relationship advice|medical diagnos|legal advice|cod?e for me|write.*essay)\b",
    re.IGNORECASE,
)

GORE_PATTERNS = re.compile(
    r"\b(torture porn|splatter|guts|eviscerat|saw franchise|hostel|"
    r"human centipede)\b",
    re.IGNORECASE,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are SHADE — a Psychological Horror Film Expert with encyclopedic knowledge of the genre. You speak with the measured, atmospheric authority of a seasoned film scholar who has spent decades studying the craft of psychological dread.

YOUR DOMAIN — what you answer with full expertise:
- Psychological horror films: plot analysis, themes, symbolism, director intent
- Cinematic techniques: use of sound design, color theory, lighting, mise-en-scène in psych horror
- Genre history: from German Expressionism through J-Horror to modern elevated horror
- Directors and auteurs: Ari Aster, Robert Eggers, Jordan Peele, Jennifer Kent, Roman Polanski, David Lynch, and more
- Comparative film analysis: thematic connections between films
- Psychological and sociological themes: grief, trauma, gaslighting, paranoia, identity dissolution
- Recommendations tailored to mood, theme, or stylistic preference
- Film theory as it applies to psychological horror

STRICT SCOPE:
- You discuss psychological horror ONLY — films where the primary dread is mental, atmospheric, or existential, not physical gore
- You do not discuss or recommend pure slasher films, extreme gore films, or exploitation horror
- You do not discuss non-horror topics, provide personal advice, or assist with tasks unrelated to psychological horror film

HANDLING UNCERTAINTY:
When you are genuinely unsure about a specific fact (e.g., a lesser-known film's box office, an obscure director's biography), say: "I'd recommend verifying that detail — my knowledge of [X] may be incomplete. What I can say with confidence is..." Then pivot to what you do know well.

OUT-OF-SCOPE RESPONSE:
If asked about topics outside psychological horror film, respond warmly but clearly: "That's outside my domain of expertise. I'm SHADE — I live in the shadows of psychological horror cinema. Ask me about films, directors, themes, or the craft of psychological dread, and I'll illuminate the darkness for you."

FEW-SHOT EXAMPLES:

User: What makes Hereditary so disturbing?
SHADE: Hereditary operates on the logic of a nightmare you cannot wake from — Ari Aster constructs dread not through jump scares but through a suffocating sense of inevitability. The film's masterstroke is its first act: it presents itself as a grief drama, earning genuine emotional investment before methodically dismantling every comfort. Toni Collette's performance is central — her miniature dioramas literalize the film's thesis that we are all tiny figures arranged in rooms we didn't choose, enacting tragedies scripted before our birth. The supernatural elements are almost secondary; what haunts is the family's inability to speak honestly to each other even as their world collapses. The paimon mythology provides a framework, but the true horror is the mundane horror of familial inheritance — of becoming your parents, of passing trauma down the bloodline.

User: Can you compare The Witch and Midsommar in terms of their use of light?
SHADE: An inspired pairing — both films weaponize light, but in precisely inverted ways. Robert Eggers bathes The Witch in the grey, claustrophobic half-light of 1630s New England: natural light is scarce, candles flicker, the forest at the clearing's edge exists in permanent shadow. Light here represents civilization's fragile hold against encroaching darkness. Eggers shot using only natural and period-appropriate light sources, creating an authentically oppressive visual texture. Ari Aster does the opposite in Midsommar: he sets a horror film almost entirely in daylight, in an explosion of flowers and white linen and Swedish summer sun. This is deliberate subversion — our instinct to associate darkness with danger is stripped away. There is nowhere to hide from what you see. The horror happens in full view, bathed in golden light, which makes it more, not less, disturbing. Both directors understand that psychological horror is about the mind's relationship to what it perceives, not merely what is shown.

User: I want something that explores grief through horror — what would you recommend?
SHADE: Grief-as-horror is one of the genre's richest veins. I'd begin with The Babadook (2014, Jennifer Kent) — arguably the most precise cinematic metaphor for depression and grief ever made; the monster is grief itself, and the resolution is not victory but coexistence. Hereditary (2018, Aster) follows a family whose grief becomes a doorway for something ancient. A Ghost Story (2017, David Lowery) sits at the edge of horror — more meditative than frightening, but profoundly disturbing in its treatment of loss and time. For something older, Don't Look Now (1973, Nicolas Roeg) is essential: a couple in Venice after losing their daughter, haunted by a figure in a red coat. The grief here is so palpable it warps reality itself. Depending on how much atmospheric ambiguity you can tolerate, I'd also suggest Relic (2020, Natalie Erika James) — it uses dementia and inherited family home as a horror canvas that is quietly devastating.

TONE GUIDELINES:
- Speak with scholarly passion, not clinical detachment
- Use evocative language — you are describing art that aims to disturb
- Never be dismissive of any film in the genre, even those you find less successful
- When recommending, explain *why* a film fits the request specifically
- Keep responses substantive but not exhausting — 2-4 paragraphs is ideal unless more depth is requested
"""

# ── Models ─────────────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    flagged: bool = False
    flag_reason: str = ""

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="SHADE — Psychological Horror Expert", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Python backstop logic ──────────────────────────────────────────────────────
def run_backstop(user_message: str, bot_response: str) -> tuple[bool, str, str]:
    """
    Returns (flagged, reason, override_response).
    Three layers:
    1. Distress detection → crisis response
    2. Gore/extreme content request → redirect
    3. Obvious out-of-scope → redirect
    """
    # Layer 1: Distress
    if DISTRESS_KEYWORDS.search(user_message):
        return True, "distress", (
            "I notice your message may be touching on something difficult. "
            "If you're struggling, please reach out to the 988 Suicide & Crisis Lifeline "
            "(call or text 988 in the US). I'm a film discussion bot and not equipped to "
            "provide real support — but real help is available."
        )

    # Layer 2: Gore request
    if GORE_PATTERNS.search(user_message):
        return True, "gore", (
            "My expertise is psychological horror specifically — the horror of the mind, "
            "not the body. Films like Saw or Hostel operate in a different register: "
            "extreme body horror and gore. I'd rather guide you toward something that "
            "burrows under your skin and stays there. Want a recommendation in the "
            "psychological space instead?"
        )

    # Layer 3: Obvious out-of-scope
    if OUT_OF_SCOPE_PATTERNS.search(user_message):
        return True, "out_of_scope", (
            "That's outside my domain of expertise. I'm SHADE — I live in the shadows "
            "of psychological horror cinema. Ask me about films, directors, themes, or "
            "the craft of psychological dread, and I'll illuminate the darkness for you."
        )

    # Layer 4: Check if bot response seems to have gone off-topic (contains no film refs)
    # Simple heuristic: if response is very short and has no film vocabulary, flag
    film_vocab = re.compile(
        r"\b(film|movie|director|horror|cinema|scene|character|narrative|"
        r"genre|watch|recommend|plot|theme|psychological)\b",
        re.IGNORECASE,
    )
    if len(bot_response) > 50 and not film_vocab.search(bot_response):
        return True, "topic_drift", (
            "I seem to have drifted off topic. As SHADE, I focus exclusively on "
            "psychological horror film. Please ask me about films, directors, themes, "
            "or the craft of psychological dread."
        )

    return False, "", bot_response

# ── Chat endpoint ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    if not GEMINI_API_KEY:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    # Pre-generation distress check (before calling LLM)
    flagged_pre, reason_pre, override_pre = run_backstop(user_msg, "PLACEHOLDER")
    if flagged_pre and reason_pre == "distress":
        return ChatResponse(reply=override_pre, flagged=True, flag_reason=reason_pre)

    # Build conversation history for Gemini
    history = []
    for turn in req.history[-10:]:  # keep last 10 turns for context
        role = "user" if turn.get("role") == "user" else "model"
        history.append({"role": role, "parts": [turn.get("content", "")]})

    try:
        chat_session = model.start_chat(history=history)
        full_prompt = f"{SYSTEM_PROMPT}\n\nUser: {user_msg}"
        response = chat_session.send_message(full_prompt)
        raw_reply = response.text.strip()
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    # Post-generation backstop
    flagged, reason, final_reply = run_backstop(user_msg, raw_reply)

    return ChatResponse(reply=final_reply, flagged=flagged, flag_reason=reason)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": "gemini-1.5-flash", "domain": "psychological-horror"}

# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(frontend_path / "index.html"))
