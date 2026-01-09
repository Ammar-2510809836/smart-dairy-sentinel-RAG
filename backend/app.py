import os
import json
from pathlib import Path
from typing import List, Dict, Optional

# --- Load environment variables ---
from dotenv import load_dotenv
load_dotenv() 

import numpy as np
import requests
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ---------- CONFIG ----------
BASE_DIR = Path(__file__).resolve().parent
INDEX_DIR = BASE_DIR / "rag_index"

CHUNKS_PATH = INDEX_DIR / "chunks.json"
EMB_PATH = INDEX_DIR / "embeddings.npy"

# --- SWITCH TO GROQ (Free & Fast) ---
# Docs: https://console.groq.com/docs/models
GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"
# Reliable free models on Groq: 
# "llama-3.3-70b-versatile" or "deepseek-r1-distill-llama-70b"
MODEL_ID = "llama-3.3-70b-versatile"

# Get the key from .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    print("[WARN] GROQ_API_KEY not found in .env. The app will fail to chat.")

# ---------- DATA MODELS ----------
class SensorState(BaseModel):
    scenario: str
    scenarioKey: str
    bodyTempC: float
    heartRateBpm: float
    activityStepsPerHour: float
    milkConductivity: float
    milkPh: float

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    sensorState: Optional[SensorState] = None
    history: List[Message] = []

class ChatResponse(BaseModel):
    reply: str
    sources: List[str] = []

# ---------- LOAD RAG INDEX ----------
print("[INFO] Loading RAG index...")
try:
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        CHUNKS = json.load(f)
    EMBEDDINGS = np.load(EMB_PATH).astype("float32")
    print(f"[INFO] Loaded {len(CHUNKS)} chunks.")
except FileNotFoundError:
    print(f"[ERROR] Index not found at {INDEX_DIR}. Run build_index.py first.")
    CHUNKS = []
    EMBEDDINGS = np.array([])

# ---------- EMBEDDING MODEL ----------
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
print("[INFO] Loading embedding model...")
EMBED_MODEL = SentenceTransformer(EMBED_MODEL_NAME)

def retrieve_chunks(query: str, k: int = 5) -> List[Dict]:
    if not CHUNKS: return []
    q_vec = EMBED_MODEL.encode([query], convert_to_numpy=True)[0].astype("float32")
    norms = np.linalg.norm(EMBEDDINGS, axis=1) * (np.linalg.norm(q_vec) + 1e-8)
    sims = np.dot(EMBEDDINGS, q_vec) / (norms + 1e-8)
    top_idx = np.argsort(-sims)[:k]
    return [CHUNKS[i] for i in top_idx]

# ---------- PROMPTS & CLASSIFICATION ----------
class QueryCategory:
    EMERGENCY = "EMERGENCY"       # 🚨 High Urgency
    HEALTH = "HEALTH"             # 🟧 Moderate Urgency
    REPRODUCTION = "REPRODUCTION" # 🟨 Long-term
    OPERATIONAL = "OPERATIONAL"   # 🟩 Educational
    SYSTEM = "SYSTEM"             # 🟦 Explanatory
    GREETING = "GREETING"         # 👋 Casual
    OUT_OF_SCOPE = "OUT_OF_SCOPE" # ⛔ Non-Dairy
    UNKNOWN = "UNKNOWN"

def classify_question(query: str) -> str:
    q = query.lower()
    
    # ⛔ Signal Z: Out of Scope (Strict Guardrails)
    forbidden_topics = [
        # Non-Cow Animals
        "dog", "cat", "horse", "chicken", "pig", "sheep", "human", "baby", "child",
        # Irrelevant Topics
        "cook", "recipe", "baking", "food", "lunch", "dinner", "breakfast", # Food
        "weather", "forecast", "rain", "sun", "climate", "usa", # Weather/Geo
        "wear", "clothes", "fashion", "dress", "workout", "gym", "weight loss", # Lifestyle
        "crypto", "bitcoin", "blockchain", "finance", "stock", "money", # Finance
        "python", "java", "coding", "debug", "programming", "script", # Tech
        "movie", "song", "music", "game", "politics", "president", "election", "vote", "minister", "government", # Pop Culture & Politics
        "trump", "biden", "obama", "putin" # Specific Names
    ]
    if any(topic in q for topic in forbidden_topics):
        return QueryCategory.OUT_OF_SCOPE
    
    # 👋 Signal G: Greetings (Check first to avoid 'System' or 'Operational' traps)
    import re
    greeting_pattern = r"^\s*(hi|hello|hey|hi there|hello there|greetings|good morning|good afternoon|good evening|thanks|thank you|ok|okay|how are you|how are you doing)\s*[!.,?]*\s*$"
    if re.match(greeting_pattern, q):
        return QueryCategory.GREETING
    
    # 🚨 Signal A: Emergency Keywords
    emergency_keywords = [
        "not eating", "stopped eating", "won't eat", "refusing feed",
        "down cow", "won't stand", "can't stand", "is down", "lying down",
        "blood", "bleeding", "high fever", "40c", "41c", "42c", "cold", "hypothermia",
        "severe", "dying", "emergency", "bloat", "prolapse"
    ]
    if any(k in q for k in emergency_keywords): return QueryCategory.EMERGENCY
    if "2 days" in q and "eat" in q: return QueryCategory.EMERGENCY # Specific time combo
    
    # 🟩 Signal E: Operational / Educational (Priority: Check DEFINITIONS before SYMPTOMS)
    # If user asks "What is temp?" or "Normal range?", it's educational, not an alarm.
    # Also catch "hungry", "feed" to give practical advice instead of medical alerts.
    # Removed 'how are you' to let it fall to GREETING.
    op_keywords = ["how to", "what is", "explain", "normal", "range", "define", "meaning", "should be", "hungry", "feed", "food"]
    if any(k in q for k in op_keywords):
        return QueryCategory.OPERATIONAL

    # 🟧 Signal B: Health Concern
    health_keywords = [
        "mastitis", "lame", "limping", "swollen", "diarrhea",
        "coughing", "snot", "discharge", "conductivity", "temperature",
        "sick", "ill", "pain"
    ]
    if any(k in q for k in health_keywords): return QueryCategory.HEALTH

    # 🟨 Signal C: Reproduction
    repro_keywords = [
        "pregnant", "infertility", "heat", "estrus", "calving", 
        "breeding", "cycle", "insemination", "abortion"
    ]
    if any(k in q for k in repro_keywords): return QueryCategory.REPRODUCTION
    
    # 🟦 Signal D: System / AI
    system_keywords = [
        "sensor", "confidence", "accuracy", "model", "ai", 
        "how do you know", "reliability", "battery", "connection"
    ]
    if any(k in q for k in system_keywords): return QueryCategory.SYSTEM

    return QueryCategory.UNKNOWN


    return QueryCategory.UNKNOWN

def is_out_of_scope(query: str) -> bool:
    """
    Hard Gate: Returns True if the query does NOT contain any dairy-related keywords.
    This prevents the LLM from even seeing irrelevant questions.
    """
    allowed_keywords = [
        "cow", "cows", "dairy", "milk", "mastitis", "calving", "calf", "bull", "heifer",
        "estrus", "heat", "udder", "farm", "barn", "lactation", "hoof", "lameness",
        "conductivity", "ph", "temperature", "rumination", "sensor", "rumen", "feed",
        "ration", "silage", "grass", "field", "herd", "vet", "veterinarian", "disease",
        "symptom", "treatment", "medicine", "antibiotic", "prolapse", "bloat", "fever",
        "ketosis", "acidosis", "metritis", "pneumonia", "scours", "colostrum", "calcium",
        "leptospirosis", "abscess", "infection", "virus", "bacteria", "parasite", "vaccine",
        "brucellosis", "tuberculosis", "foot", "mouth", "anthrax", "blackleg",
        "johne", "listeriosis", "salmonella", "bvd", "ibr", "rhinotracheitis",
        "laminitis", "dermatitis", "downer", "placenta", "endometritis", "infertility",
        "dystocia", "clots", "flakes", "campylobacter",
        "hello", "hi", "hey", "greetings", "good morning", "good afternoon", "good evening",
        "thanks", "thank you", "how are you", "how are you doing", 
        "temperature","conductivity","ph","activity","steps","heart rate","bpm","steps per hour","temp",
        "understand", "mean", "talking", "said", "saying", "repeat", "clarify", "explain", "why", "what",
        "summary", "history", "previous", "last", "context",
    ]
    
    q_lower = query.lower()
    
    # If explicitly mentioning forbidden topics, it's OUT, regardless of allowed words
    forbidden_topics = [
        "dog", "cat", "horse", "chicken", "pig", "sheep", "human", "baby", "child",
        "cook", "recipe", "baking", "food", "lunch", "dinner", "breakfast",
        "weather", "forecast", "rain", "sun", "climate", "usa",
        "wear", "clothes", "fashion", "dress", "workout", "gym", "weight loss",
        "crypto", "bitcoin", "blockchain", "finance", "stock", "money",
        "python", "java", "coding", "debug", "programming", "script",
        "movie", "song", "music", "game", "politics", "president", "election", "vote", "minister", "government",
        "trump", "biden", "obama", "putin"
    ]
    if any(topic in q_lower for topic in forbidden_topics):
        return True

    # Otherwise, require at least one DAIRY keyword
    # If NO dairy keyword is found, it is Out of Scope.
    has_keyword = any(keyword in q_lower for keyword in allowed_keywords)
    return not has_keyword


def get_system_prompt(category: str) -> str:
    base_identity = "You are Smart Dairy Sentinel, a Virtual Dairy Farm Assistant."
    
    if category == QueryCategory.OUT_OF_SCOPE:
        return (
            f"{base_identity}\n"
            "You are a specialized system for DAIRY COWS only.\n"
            "The user asked about a different animal or topic.\n"
            "RULES:\n"
            "1. Politely refuse to answer.\n"
            "2. State clearly that you only support Dairy Cattle.\n"
            "3. Do NOT provide any medical advice for other animals.\n"
            "4. Structure:\n"
            "   ⛔ **Out of Scope**\n"
            "   [I am designed only for dairy cows...]\n"
            "   💡 **Suggestion**\n"
            "   [Please contact a general vet...]\n"
            "STOP. Do not chatter."
        )

    elif category == QueryCategory.EMERGENCY:
        return (
            f"{base_identity} 🚨 MODE: EMERGENCY RESPONSE.\n"
            "The user is reporting a serious, possibly life-threatening situation.\n"
            "RULES:\n"
            "1. Be fast, direct, and authoritative.\n"
            "2. TRUST the Sensor Status provided in context (e.g. if 'NORMAL ✅', do not say it is high).\n"
            "3. If user says 'cold' but temp is Normal, suspect MILK FEVER (Hypocalcemia) or SHOCK, NOT Heat Stress.\n"
            "4. Structure:\n"
            "   🚨 **CRITICAL WARNING**\n"
            "   [State the likely issue based on signs. e.g. 'Symptoms suggest Milk Fever...']\n\n"
            "   🩺 **Immediate Actions**\n"
            "   [Bulleted list of first aid]\n\n"
            "   ❗ **Recommendation**\n"
            "   [Call Vet / Emergency procedure]\n"
            "STOP. Do not chatter."
        )
    
    elif category == QueryCategory.HEALTH:
        return (
            f"{base_identity} 🟧 MODE: HEALTH MONITORING.\n"
            "The user is asking about a potential illness or symptom.\n"
            "RULES:\n"
            "1. Use sensor data to validate (if available).\n"
            "2. Suggest observations, not just diagnosis.\n"
            "3. Structure:\n"
            "   🚨 **Why this matters**\n"
            "   [Brief impact]\n\n"
            "   🩺 **Common causes**\n"
            "   [Bulleted list]\n\n"
            "   🔍 **Check immediately**\n"
            "   [Physical signs to look for]\n\n"
            "   🤖 **Sensor Check**\n"
            "   [Compare user query to current sensor stats]\n\n"
            "   ❗ **Action**\n"
            "   [Monitor/Treat/Vet]\n"
            "STOP. Do not chatter."
        )

    elif category == QueryCategory.REPRODUCTION:
        return (
            f"{base_identity} 🟨 MODE: REPRODUCTION MANAGEMENT.\n"
            "The user is asking about breeding, cycles, or fertility.\n"
            "RULES:\n"
            "1. Be calm, time-aware, and management-focused.\n"
            "2. Focus on records, trends, and protocols.\n"
            "3. Structure:\n"
            "   📊 **Analysis**\n"
            "   [Context of the issue]\n"
            "   📅 **Timeline/Cycle**\n"
            "   [Explain relevant cycle details]\n"
            "   💡 **Suggestion**\n"
            "   [Management advice]\n"
        )
        
    elif category == QueryCategory.SYSTEM:
        return (
            f"{base_identity} 🟦 MODE: SYSTEM EXPLANATION.\n"
            "The user is asking about the AI, sensors, or reliability.\n"
            "RULES:\n"
            "1. Be transparent and explainable.\n"
            "2. Admit limitations. Explain how the sensors work.\n"
            "3. No biology guessing here, just tech explanation.\n"
        )

    elif category == QueryCategory.GREETING:
        return (
            f"{base_identity} 👋 MODE: GREETING.\n"
            "The user is saying hello or asking how you are.\n"
            "RULES:\n"
            "1. Be friendly, polite, and professional.\n"
            "2. Keep it short (1-2 sentences).\n"
            "3. DO NOT use headers like 'Introduction' or 'Structure'. Just text.\n"
            "4. Ask how you can help with their dairy farm.\n"
        )

    else: # OPERATIONAL / UNKNOWN / GENERAL
        return (
            f"{base_identity} 🟩 MODE: GENERAL ASSISTANCE / EDUCATIONAL.\n"
            "The user is asking for a definition, a normal range, or PRACTICAL ADVICE (e.g. feeding).\n"
            "RULES:\n"
            "1. Be objective, concise, and use the user's requested format.\n"
            "2. DO NOT be alarmist. This is a reference lookup.\n"
            "3. IF asking about FEED/HUNGRY/SURVIVAL:\n"
            "   - List ✅ SAFE ALTERNATIVES (Grass, Rice/Wheat Straw, Banana peels, Veg scraps, Sugarcane leaves).\n"
            "   - List ❌ DO NOT FEED (Moldy food, Plastic, spicy/oily food).\n"
            "   - Emphasize WATER is critical.\n"
            "4. Structure (General):\n"
            "   ℹ️ **Info / Normal Range**\n"
            "   [Definition or 'Normal adult cow temperature: 38.0–39.3°C']\n\n"
            "   🤖 **Sensor Reading**\n"
            "   [Compare current sensor value to normal. e.g. '38.5°C -> Normal']\n\n"
            "   ✅ **Practical Advice / Alternatives**\n"
            "   [Bulleted list of actionable steps or safe foods]\n\n"
            "   ❗ **Call a Vet if...**\n"
            "   [Specific thresholds]\n"
            "CRITICAL: If the question is not about dairy farming, REFUSE TO ANSWER."
        )

def validate_sensor_data(sensor: SensorState) -> list[str]:
    """Checks for physiologically impossible sensor values."""
    warnings = []
    
    # Temperature Sanity Check (Dead cow < 30C, Boiling cow > 45C)
    if sensor.bodyTempC < 30.0 or sensor.bodyTempC > 45.0:
        warnings.append(f"CRITICAL ERROR: Body Temp {sensor.bodyTempC}C is PHYSIOLOGICALLY IMPOSSIBLE (Sensor Failure likely).")
    
    # Heart Rate Sanity Check (Dead < 30, Exploding > 200)
    if sensor.heartRateBpm < 30 or sensor.heartRateBpm > 200:
        warnings.append(f"CRITICAL ERROR: Heart Rate {sensor.heartRateBpm} BPM is OUT OF RANGE.")

    # Conductivity Sanity Check
    if sensor.milkConductivity < 2 or sensor.milkConductivity > 15:
        warnings.append(f"WARNING: Conductivity {sensor.milkConductivity} mS is highly abnormal.")

    return warnings

def analyze_sensor_health(s: SensorState) -> str:
    """Analyzes sensor data against hard medical thresholds and returns a status string."""
    # Temperature (Adult Cow)
    if s.bodyTempC >= 39.5:
        temp_status = "FEVER 🥵"
    elif s.bodyTempC <= 37.5:
        temp_status = "HYPOTHERMIA 🥶"
    elif 38.0 <= s.bodyTempC <= 39.3:
        temp_status = "NORMAL ✅"
    else:
        temp_status = "SLIGHTLY ABNORMAL ⚠️" # 37.6-37.9 or 39.4

    # Activity
    if s.activityStepsPerHour < 100: # Arbitrary low threshold for 'downer'?
        act_status = "LOW ACTIVITY ⚠️"
    else:
        act_status = "Active"

    return f"[Temp: {s.bodyTempC}C ({temp_status}), HR: {s.heartRateBpm}, Steps: {s.activityStepsPerHour} ({act_status})]"

def build_context(sensor: Optional[SensorState], query: str) -> tuple[str, list[str]]:
    # 1. Sensor Data
    s_text = "No sensor data."
    if sensor:
        # Check for integrity
        warnings = validate_sensor_data(sensor)
        if warnings:
            # If data is impossible, FORCE the AI to see the error
            s_text = f"⚠ SENSOR FAILURE DETECTED ⚠\n" + "\n".join(warnings) + "\n(Do NOT hallucinate normal values. Report this error.)"
        else:
            # Generate smart summary with medical ranges
            health_summary = analyze_sensor_health(sensor)
            s_text = f"Sensors: {health_summary}"
    
    # 2. RAG Data
    # Skip RAG for simple greetings/short phrases to avoid noise
    import re
    # Regex to match exact greetings with optional punctuation/whitespace
    greeting_pattern = r"^\s*(hi|hello|hey|hi there|hello there|greetings|good morning|good afternoon|good evening|thanks|thank you|ok|okay|how are you|how are you doing)\s*[!.,?]*\s*$"
    
    is_greeting = re.match(greeting_pattern, query, re.IGNORECASE)
    
    if is_greeting:
        docs = []
    else:
        docs = retrieve_chunks(query, k=3)
    rag_text = "\n".join([f"- {d['text']}" for d in docs]) if docs else "No specific manual info."
    
    # Extract sources (files) from chunks
    sources = list(set([d.get("source", "Unknown") for d in docs]))
    
    return f"{s_text}\n\nReference Manual Info:\n{rag_text}", sources

def strip_think(text: str) -> str:
    import re
    return re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.IGNORECASE).strip()

# ---------- API ----------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if not GROQ_API_KEY:
        return ChatResponse(reply="Server Error: GROQ_API_KEY is missing.")

    # 0. HARD GATE: Strict Scope Check
    if is_out_of_scope(req.message):
        print(f"[INFO] HARD GATE REFUSAL: '{req.message}'")
        return ChatResponse(reply=(
            "⛔ **Out of Scope**\n\n"
            "I’m sorry, I can’t help with that. I’m a virtual dairy farm assistant and "
            "only answer questions related to dairy cows, milk production, and farm health.\n\n"
            "💡 **Suggestion**\n"
            "Please ask about cow health, sensors, or herd management."
        ))

    # 1. Classify
    category = classify_question(req.message)
    print(f"[INFO] Query: '{req.message}' -> Category: {category}")

    # 2. Context & Prompt
    context_str, sources = build_context(req.sensorState, req.message)
    system_prompt = get_system_prompt(category)
    
    # Construct conversation history
    # System Prompts -> History -> Current Context -> User Message
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "system", "content": f"Current Context:\n{context_str}"}
    ]
    
    # Append history
    for msg in req.history:
        messages.append({"role": msg.role, "content": msg.content})
        
    # Append current user message
    messages.append({"role": "user", "content": req.message})

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }
    
    body = {
        "model": MODEL_ID,
        "messages": messages,
        "temperature": 0.3 if category == QueryCategory.EMERGENCY else 0.6,
        "max_tokens": 600
    }

    try:
        resp = requests.post(GROQ_API_URL, headers=headers, json=body)
        resp.raise_for_status()
        data = resp.json()
        raw_reply = data["choices"][0]["message"]["content"]
        return ChatResponse(reply=strip_think(raw_reply), sources=sources)
        
    except Exception as e:
        print(f"[ERROR] Groq API Failed: {e}")
        # Fallback error message for UI
        if 'resp' in locals(): print(resp.text)
        return ChatResponse(reply="I'm having trouble reaching the Groq cloud. Please check your API key.")