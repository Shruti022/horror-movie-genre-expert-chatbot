# SHADE — Psychological Horror Film Expert Bot

> *"The oldest and strongest emotion of mankind is fear, and the oldest and strongest kind of fear is fear of the unknown."* — H.P. Lovecraft

**SHADE** is a domain-expert chatbot specializing exclusively in **psychological horror cinema** — the films that burrow into the subconscious and linger there. Built for Columbia's Agentic AI for Analytics course.

**Live URL:** `https://shade-psych-horror-XXXX.run.app` ← replace after deploy  
**GitHub:** this repo

---

## Domain

**Psychological Horror Film** — specifically the subgenre where dread is mental, atmospheric, and existential rather than physical gore. Films like:

- *Hereditary* (2018, Ari Aster)
- *The Witch* (2015, Robert Eggers)  
- *Midsommar* (2019, Ari Aster)
- *Get Out* (2017, Jordan Peele)
- *The Babadook* (2014, Jennifer Kent)
- *Rosemary's Baby* (1968, Roman Polanski)
- *Black Swan* (2010, Darren Aronofsky)
- *Ringu* (1998, Hideo Nakata)

SHADE can discuss: film analysis, director techniques, thematic comparisons, genre history, cinematography, sound design, and tailored recommendations.

---

## Architecture

```
┌─────────────────┐     POST /chat      ┌──────────────────────┐
│   Frontend UI   │ ──────────────────► │   FastAPI Backend    │
│  (dark cinema   │                     │                      │
│   aesthetic)    │ ◄────────────────── │  ┌────────────────┐  │
└─────────────────┘    JSON response    │  │ Python Backstop│  │
                                        │  │ (pre+post gen) │  │
                                        │  └───────┬────────┘  │
                                        │          │           │
                                        │  ┌───────▼────────┐  │
                                        │  │ Gemini Flash   │  │
                                        │  │ (with system   │  │
                                        │  │  prompt + FSP) │  │
                                        │  └────────────────┘  │
                                        └──────────────────────┘
```

---

## Prompting Strategy

### 1. Role/Persona
SHADE has a distinct voice: measured, scholarly, atmospheric. Defined as a film expert who has spent decades studying psychological dread. Boundaries are encoded as **positive constraints** — what SHADE *can* answer, not what it can't.

### 2. Few-Shot Prompting (3 examples)
Three rich examples in the system prompt covering:
- Single-film deep analysis (*Hereditary*)
- Comparative technique analysis (*The Witch* vs *Midsommar* — lighting)
- Thematic recommendation (grief-horror films)

### 3. Positive Constraints
The system prompt defines SHADE's domain affirmatively:
- "Films where the primary dread is mental, atmospheric, or existential"
- Lists specific directors, topics, and question types SHADE handles
- No negative framing ("don't talk about X")

### 4. Escape Hatch
When uncertain about specific facts (box office, obscure bios), SHADE says:
> *"I'd recommend verifying that detail — my knowledge of [X] may be incomplete. What I can say with confidence is..."*

---

## Out-of-Scope Handling

Three layers of protection:

**Layer 1 — Pre-generation Python backstop (distress detection):**
```python
DISTRESS_KEYWORDS = re.compile(
    r"\b(suicid|self.harm|kill myself|hurt myself|...)\b", re.IGNORECASE
)
```
→ Bypasses LLM entirely, returns crisis resources immediately.

**Layer 2 — Pre-generation pattern matching:**
- Gore requests → redirected to psychological horror
- Obvious off-topic (recipes, code, sports scores) → redirect

**Layer 3 — Post-generation validation:**
- Checks if response contains any film-domain vocabulary
- If response lacks film terms entirely, overrides with redirect

### Out-of-Scope Categories (positive framing)
1. **Physical gore / extreme body horror** — SHADE focuses on *psychological* dread
2. **Non-entertainment topics** — SHADE's expertise is *cinema*, not life advice  
3. **Safety-sensitive content** — SHADE routes to crisis resources immediately

---

## Running Locally

### Prerequisites
- Python 3.11+
- [uv](https://github.com/astral-sh/uv) package manager
- A GCP project with Vertex AI API enabled
- `gcloud` CLI authenticated (`gcloud auth application-default login`)

### Setup

```bash
# Clone the repo
git clone https://github.com/YOUR_USERNAME/shade-psych-horror-bot
cd shade-psych-horror-bot

# Create .env
cp .env.example .env
# Edit .env and set GCP_PROJECT to your GCP project ID

# Install dependencies
uv sync

# Run the server
uv run uvicorn app.main:app --reload --port 8080
```

Open http://localhost:8080 in your browser.

---

## Running the Eval Harness

```bash
# Run full eval against local server
uv run eval/run_eval.py --url http://localhost:8000

# Run against deployed GCP instance
uv run eval/run_eval.py --url https://shade-psych-horror-XXXX.run.app

# Run with verbose output (shows bot responses)
uv run eval/run_eval.py --url http://localhost:8000 --verbose

# Run specific category only
uv run eval/run_eval.py --url http://localhost:8000 --category in_domain
uv run eval/run_eval.py --url http://localhost:8000 --category out_of_scope
uv run eval/run_eval.py --url http://localhost:8000 --category adversarial
```

### Eval Dataset Structure (`eval/golden_dataset.json`)

| Category | Count | Description |
|---|---|---|
| `in_domain` | 10 | Film questions with expected answers + rubrics |
| `out_of_scope` | 5 | Off-topic requests (expect refusal) |
| `adversarial` | 5 | Prompt injections, jailbreaks, distress signals |
| **Total** | **20** | |

### Metrics Used

| Metric | Type | Used For |
|---|---|---|
| Keyword/regex check | **Deterministic** | Refusal verification (must_not_contain, must_contain_any) |
| MaaJ golden-reference | **LLM judge** | In-domain answer quality vs. expected answer |
| MaaJ rubric | **LLM judge** | Structured criterion-by-criterion grading |

Pass threshold: ≥70% criteria met for rubric; ≥6/10 score for golden-reference.

---

## Deploying to GCP Cloud Run

### Using Cloud Build (recommended)

```bash
# Submit build — uses cloudbuild.yaml to build, push, and deploy
gcloud builds submit --config cloudbuild.yaml
```

### Manual deploy

```bash
export PROJECT_ID=your-gcp-project-id
export REGION=us-central1

# Build and push
gcloud builds submit --tag gcr.io/$PROJECT_ID/shade-bot

# Deploy to Cloud Run
gcloud run deploy shade-bot \
  --image gcr.io/$PROJECT_ID/shade-bot \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated \
  --set-env-vars GCP_PROJECT=$PROJECT_ID \
  --port 8080 \
  --memory 512Mi
```

---

## Repo Structure

```
shade-psych-horror-bot/
├── README.md
├── pyproject.toml          # uv-based dependencies
├── Dockerfile              # GCP Cloud Run deployment
├── .env.example
├── app/
│   └── main.py             # FastAPI app + prompts + backstop logic
├── frontend/
│   └── index.html          # Cinematic dark UI (single file)
└── eval/
    ├── golden_dataset.json # 20 test cases
    └── run_eval.py         # Eval harness (deterministic + MaaJ)
```

---

## Design Decisions

**Why Gemini Flash?** Fast, cost-effective for a chatbot, great instruction-following for persona consistency. Free tier via Google AI Studio works for development.

**Why single-file frontend?** Zero build complexity, instant deployment, no Node.js needed. The cinematic dark aesthetic is achieved with pure CSS.

**Why three layers of backstop?** Defense in depth: distress signals need pre-LLM handling (never route crisis to a film bot), obvious off-topic can be caught deterministically, post-generation catches subtle drift.

**Why positive constraints?** Negative constraints ("don't talk about X") create anchoring — the LLM thinks about X. Positive constraints define the space SHADE inhabits without surfacing what it excludes.
