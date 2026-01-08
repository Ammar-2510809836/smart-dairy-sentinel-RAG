Below is a **clean, professional README.md** you can **directly copy into your GitHub repo**.
It matches your **actual implementation**, **RAG logic**, **IoT focus**, and is written at a **student + engineer level** (perfect for professors, recruiters, and reviewers).

---

# 🐄 Smart Dairy Sentinel

**A Real-Time Virtual Veterinary Assistant using IoT & RAG**

Smart Dairy Sentinel is an AI-powered decision-support system for dairy farming that combines **IoT sensor data** with **Retrieval-Augmented Generation (RAG)** to translate raw farm metrics into **clear, safe, and actionable insights** for cow health and milk quality monitoring.

This system is designed to **support farmers**, not replace veterinarians, by explaining sensor readings, detecting risks early, and escalating emergencies responsibly.

---

## 🚩 Problem Statement

Modern dairy farms use IoT sensors to monitor cow health and milk quality, but farmers are often presented with **raw numerical data** (temperature, activity, milk conductivity, pH) without clear explanations.
As a result, farmers struggle to understand **what the data means**, **whether a problem exists**, and **what action to take**, leading to delayed care, poor animal welfare, milk contamination, and economic losses.

---

## ✅ Proposed Solution

Smart Dairy Sentinel bridges this gap by:

* Collecting real-time cow and milk sensor data
* Interpreting sensor values using an **AI assistant**
* Grounding responses in **verified veterinary knowledge via RAG**
* Classifying queries by urgency and enforcing safety guardrails

---

## 🧠 Key Features

* **Retrieval-Augmented Generation (RAG)**
  Uses indexed veterinary manuals (PDFs) to prevent hallucinations and ensure evidence-based responses.

* **Query Classification & Urgency Detection**
  Automatically categorizes queries into:

  * 🚨 Emergency
  * 🟧 Health
  * 🟨 Reproduction
  * 🟦 System / Educational
  * ⛔ Out of Scope (blocked before AI generation)

* **Sensor-Aware Context Injection**
  Combines user questions with live IoT data (temperature, activity, milk conductivity, pH).

* **Dynamic Persona Switching**
  Emergency responses are short and directive; educational responses are calm and explanatory.

* **Safety Guardrails**
  No diagnoses, no drug dosages, strict dairy-only scope, and mandatory veterinary escalation.

---

## 🛠️ How It Works (RAG Pipeline)

```
User Query
   ↓
Out-of-Scope Check (Hard Gate)
   ↓
Query Classification (Urgency & Category)
   ↓
Sensor Context Injection
   ↓
Knowledge Retrieval (RAG)
   ↓
LLM Response Generation
   ↓
Safety & Formatting Layer
```

---

## 🧪 Example (Emergency Case)

**User:**

> My cow is down and cold after calving. What should I do?

**System Behavior:**

1. Classified as 🚨 Emergency
2. Sensor context injected (low activity, abnormal temperature)
3. Relevant veterinary protocol retrieved (Milk Fever)
4. Short, authoritative response generated
5. Immediate veterinary escalation enforced

---

## 🏗️ Technology Stack

* **Frontend:** HTML5, JavaScript (Real-time dashboard & chat UI)
* **Backend:** Python, FastAPI
* **AI Model:** LLaMA 3.3 70B (via Groq API)
* **RAG Embeddings:** SentenceTransformers
* **Vector Store:** Local NumPy-based vector index
* **Data Sources:** Curated veterinary PDFs & IoT documentation

---

## 📂 Project Structure

```
smart-dairy-sentinel/
│
├── app.py                # FastAPI backend (classification, RAG, safety)
├── build_index.py        # PDF ingestion, chunking & embedding
├── frontend.html         # Dashboard & chat interface
├── data/
│   ├── health_basics/    # Veterinary health PDFs
│   └── iot_sensors/      # IoT & sensor documentation
├── embeddings/           # Stored vector embeddings (NumPy)
└── README.md
```

---

## ⚠️ Limitations

* Uses simulated sensor data (prototype stage)
* Limited document set (manual curation required)
* No real farm deployment yet
* No confidence scoring or uncertainty estimation

---

## 🔮 Future Improvements

* Real farm pilot testing
* Source citation per response
* Confidence scoring for recommendations
* Multilingual farmer support
* Integration with real IoT hardware

---

## ⚖️ Ethics & Responsibility

* Designed as a **decision-support system**, not a medical authority
* Always escalates emergencies to veterinarians
* Prioritizes farmer observations over sensor data
* Enforces strict domain and safety boundaries
