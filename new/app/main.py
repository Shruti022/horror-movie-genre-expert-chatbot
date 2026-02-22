"""
Psychological Horror Film Expert Bot — SHADE
FastAPI backend with Gemini Flash (google-genai SDK)
"""

import os
import re
import logging
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Gemini setup ───────────────────────────────────────────────────────────────
# GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "id")
# if not GEMINI_API_KEY:
#     logger.warning("GEMINI_API_KEY is not set!")
# # client = genai.Client(api_key=GEMINI_API_KEY)


GCP_PROJECT = os.getenv("GCP_PROJECT", "id")
client = genai.Client(vertexai=True, project=GCP_PROJECT, location="us-central1")
GEMINI_MODEL = "gemini-2.0-flash"


# ── Safety backstop patterns ───────────────────────────────────────────────────
DISTRESS_KEYWORDS = re.compile(
    r"\b(suicid|self.harm|kill myself|hurt myself|end my life|crisis|helpline)\b",
    re.IGNORECASE,
)
OUT_OF_SCOPE_PATTERNS = re.compile(
    r"\b(recipe|stock price|weather|sports score|homework|math problem|"
    r"relationship advice|medical diagnos|legal advice|cod?e for me|write.*essay)\b",
    re.IGNORECASE,
)
GORE_PATTERNS = re.compile(
    r"\b(torture porn|splatter|eviscerat|saw franchise|hostel|human centipede)\b",
    re.IGNORECASE,
)
FILM_VOCAB = re.compile(
    r"\b(film|movie|director|horror|cinema|scene|character|narrative|"
    r"genre|watch|recommend|plot|theme|psychological)\b",
    re.IGNORECASE,
)

# ── System prompt ──────────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are SHADE — a Psychological Horror Film Expert with encyclopedic knowledge of the genre. You speak with the measured, atmospheric authority of a seasoned film scholar who has spent decades studying the craft of psychological dread.

YOUR DOMAIN — what you answer with full expertise:
- Psychological horror films: plot analysis, themes, symbolism, director intent
- Cinematic techniques: use of sound design, color theory, lighting, mise-en-scene in psych horror
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
When you are genuinely unsure about a specific fact, say: "I'd recommend verifying that detail — my knowledge of [X] may be incomplete. What I can say with confidence is..." Then pivot to what you do know well.

OUT-OF-SCOPE RESPONSE:
If asked about topics outside psychological horror film, respond warmly but clearly: "That's outside my domain of expertise. I'm SHADE — I live in the shadows of psychological horror cinema. Ask me about films, directors, themes, or the craft of psychological dread, and I'll illuminate the darkness for you."

FEW-SHOT EXAMPLES:

User: What makes Hereditary so disturbing?
SHADE: Hereditary operates on the logic of a nightmare you cannot wake from — Ari Aster constructs dread not through jump scares but through a suffocating sense of inevitability. The film's masterstroke is its first act: it presents itself as a grief drama, earning genuine emotional investment before methodically dismantling every comfort. Toni Collette's performance is central — her miniature dioramas literalize the film's thesis that we are all tiny figures arranged in rooms we didn't choose, enacting tragedies scripted before our birth. The supernatural elements are almost secondary; what haunts is the family's inability to speak honestly to each other even as their world collapses.

User: Can you compare The Witch and Midsommar in terms of their use of light?
SHADE: An inspired pairing — both films weaponize light, but in precisely inverted ways. Robert Eggers bathes The Witch in the grey, claustrophobic half-light of 1630s New England: natural light is scarce, candles flicker, the forest at the clearing's edge exists in permanent shadow. Ari Aster does the opposite in Midsommar: he sets a horror film almost entirely in daylight, in an explosion of flowers and white linen and Swedish summer sun. This is deliberate subversion — our instinct to associate darkness with danger is stripped away. The horror happens in full view, bathed in golden light, which makes it more, not less, disturbing.

User: I want something that explores grief through horror — what would you recommend?
SHADE: Grief-as-horror is one of the genre's richest veins. I'd begin with The Babadook (2014, Jennifer Kent) — arguably the most precise cinematic metaphor for depression and grief ever made; the monster is grief itself, and the resolution is not victory but coexistence. Hereditary (2018, Aster) follows a family whose grief becomes a doorway for something ancient. For something older, Don't Look Now (1973, Nicolas Roeg) is essential: a couple in Venice after losing their daughter, haunted by a figure in a red coat. The grief here is so palpable it warps reality itself.

TONE GUIDELINES:
- Speak with scholarly passion, not clinical detachment
- Use evocative language — you are describing art that aims to disturb
- Keep responses substantive — 2-4 paragraphs is ideal
"""

# ── Pydantic models ────────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    history: list[dict] = []

class ChatResponse(BaseModel):
    reply: str
    flagged: bool = False
    flag_reason: str = ""

# ── FastAPI app ────────────────────────────────────────────────────────────────
app = FastAPI(title="SHADE", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Python backstop ────────────────────────────────────────────────────────────
def run_backstop(user_message: str, bot_response: str) -> tuple[bool, str, str]:
    if DISTRESS_KEYWORDS.search(user_message):
        return True, "distress", (
            "I notice your message may be touching on something difficult. "
            "If you're struggling, please reach out to the 988 Suicide & Crisis Lifeline "
            "(call or text 988 in the US). I'm a film discussion bot and not equipped to "
            "provide real support — but real help is available."
        )
    if GORE_PATTERNS.search(user_message):
        return True, "gore", (
            "My expertise is psychological horror specifically — the horror of the mind, not the body. "
            "Want a recommendation in the psychological space instead?"
        )
    if OUT_OF_SCOPE_PATTERNS.search(user_message):
        return True, "out_of_scope", (
            "That's outside my domain of expertise. I'm SHADE — I live in the shadows "
            "of psychological horror cinema. Ask me about films, directors, themes, or "
            "the craft of psychological dread, and I'll illuminate the darkness for you."
        )
    if bot_response != "PLACEHOLDER" and len(bot_response) > 50:
        if not FILM_VOCAB.search(bot_response):
            return True, "topic_drift", (
                "I seem to have drifted off topic. As SHADE, I focus exclusively on "
                "psychological horror film."
            )
    return False, "", bot_response

# ── Chat endpoint ──────────────────────────────────────────────────────────────
@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    # if not GEMINI_API_KEY:
    #     raise HTTPException(status_code=500, detail="GEMINI_API_KEY not configured")

    user_msg = req.message.strip()
    if not user_msg:
        raise HTTPException(status_code=400, detail="Message cannot be empty")

    flagged_pre, reason_pre, override_pre = run_backstop(user_msg, "PLACEHOLDER")
    if flagged_pre and reason_pre == "distress":
        return ChatResponse(reply=override_pre, flagged=True, flag_reason=reason_pre)

    contents = []
    for turn in req.history[-10:]:
        role = "user" if turn.get("role") == "user" else "model"
        contents.append(
            types.Content(role=role, parts=[types.Part(text=turn.get("content", ""))])
        )
    contents.append(
        types.Content(role="user", parts=[types.Part(text=user_msg)])
    )

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                temperature=0.7,
                max_output_tokens=1024,
            ),
        )
        raw_reply = response.text.strip()
        logger.info(f"Gemini OK — {len(raw_reply)} chars")
    except Exception as e:
        logger.error(f"Gemini error: {e}")
        raise HTTPException(status_code=502, detail=f"LLM error: {str(e)}")

    flagged, reason, final_reply = run_backstop(user_msg, raw_reply)
    return ChatResponse(reply=final_reply, flagged=flagged, flag_reason=reason)

# ── Health check ───────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "model": GEMINI_MODEL, "domain": "psychological-horror"}

# ── Serve frontend ─────────────────────────────────────────────────────────────
frontend_path = Path(__file__).parent.parent / "frontend"
if frontend_path.exists():
    app.mount("/static", StaticFiles(directory=str(frontend_path)), name="static")

    @app.get("/")
    async def root():
        return FileResponse(str(frontend_path / "index.html"))
