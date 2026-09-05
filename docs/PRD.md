<USER_REQUEST>
# PRD: Vehicle Damage Assessment & Adjuster Support Dashboard

**Project:** Foundation-model-based automated vehicle damage assessment for insurance
**Component:** Interactive dashboard (Gradio / Streamlit)
**Owner:** Ashish

---

## 1. Purpose

The dashboard is the human-facing surface for the damage-assessment system. It has two audiences and two jobs:

1. **Show the automated damage assessment output** (Vehicle Damage Code, severity rating, cost estimate, evidence) from the VLM+LoRA pipeline, in a form a claims handler can quickly verify or override.
2. **Let adjusters/underwriters/CS staff query the RAG-backed knowledge agent** — policy rules, taxonomy definitions, similar past claims — in natural language, grounded and cited.

It is not the model training environment. It's the review/demo layer that proves the pipeline is usable by a human in the loop, which matters for your pilot-deployment phase and for demoing the project's value to your guide/stakeholders.

## 2. Goals & Non-Goals

**Goals**
- Upload or select a vehicle image (or a batch) and view the model's structured damage assessment
- Let a human accept, edit, or reject the assessment (human-in-the-loop review)
- Surface confidence/uncertainty and flag low-confidence or fraud-suspicious cases for manual review
- Provide a chat panel backed by RAG for policy/taxonomy/precedent questions, with citations
- Show basic evaluation metrics (accuracy, false-positive rate, per-class performance) for transparency, since this is a research prototype being demoed

**Non-goals (out of scope for 2-month build)**
- Production authentication/SSO, multi-tenant access control
- Real claims-system integration (policy admin, payments)
- Mobile app
- Full audit-trail/compliance logging (note as future work only)

## 3. Users / Personas

| Persona | Needs from dashboard |
|---|---|
| Claims Adjuster | Fast visual review of damage assessment vs. photos; override capability; cost estimate breakdown |
| Underwriter | Aggregate view across claims; risk/severity distribution; policy Q&A |
| Customer Service | Simple, non-technical view of claim status and plain-language explanation |
| You / your guide (demo audience) | Evaluation metrics view; ability to show model reasoning and system architecture |

## 4. Functional Requirements

### 4.1 Damage Assessment View
- Image upload (single image, or multi-image per claim for multi-angle vehicles)
- Display: annotated image (bounding box/highlight if available), predicted Vehicle Damage Code(s), severity rating, estimated repair cost range, model confidence
- Editable fields for adjuster override, with a "reason for override" free-text field (useful later as feedback data)
- Flag banner for low-confidence or fraud-suspicious predictions

### 4.2 Knowledge Agent (RAG chat panel)
- Chat interface, persona-aware (optional persona selector: adjuster / underwriter / CS) to adjust tone/detail
- Answers grounded in curated KB (taxonomy, rubrics, policy docs, past-claim precedents)
- Citations/source snippets shown alongside each answer
- Suggested prompt chips for common queries ("Why this severity?", "Similar past claims", "Policy on pre-existing damage")

### 4.3 Evaluation / Metrics View
- Summary metrics: accuracy, mIoU (if segmentation used), false-positive rate, per-damage-type breakdown
- Confusion matrix or per-class chart
- Comparison against INS-MMBench auto-insurance subset results (your held-out eval)

### 4.4 Claims Batch View (stretch, if time permits)
- Table of processed claims with status, severity, flag state
- Filter/sort by damage type, confidence, flag status

## 5. Non-Functional Requirements

- **Latency:** single-image assessment should return in a few seconds for a usable demo; RAG chat responses similarly interactive
- **Deployability:** must run locally / on a single GPU instance for demo purposes; no heavy infra assumptions
- **Data handling:** treat uploaded images as sensitive; no external logging of images beyond the session unless explicitly stored for the human-feedback loop
- **Simplicity over polish:** given the timeline, prioritize a working, honest demo over pixel-perfect UI

## 6. System Architecture (dashboard's place in it)

```
                     ┌─────────────────────────────┐
   Image upload  →   │   VLM + LoRA backbone         │  →  Structured JSON
                     │   (damage code / severity /   │     (damage code, severity,
                     │   cost estimate)               │      cost, confidence)
                     └─────────────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   Dashboard (Streamlit/Gradio)│
                     │   - Assessment view            │
                     │   - Human override              │
                     │   - Metrics view                │
                     └─────────────────────────────┘
                                    │
                                    ▼
                     ┌─────────────────────────────┐
                     │   RAG Knowledge Agent          │
                     │   (vector store + curated KB)  │
                     │   - Chat panel                  │
                     └─────────────────────────────┘
```

The dashboard is a thin orchestration layer: it calls the VLM inference function and the RAG agent's query function, and renders both. It does not itself contain model logic.

## 7. Tech Stack Decision: Gradio vs. Streamlit

| | Gradio | Streamlit |
|---|---|---|
| Strength | Fastest way to wrap an ML model's inputs/outputs; built-in `ChatInterface` component is ideal for the RAG chat panel | Better for multi-page, multi-panel dashboards with tables, metrics, filters |
| Weakness | Less flexible for complex multi-view layouts (adjuster view vs. metrics view vs. batch table) | Slightly more boilerplate for a pure chat UI |
| Fit here | Great if you want a fast single-page demo focused on "upload image → see result → ask a question" | Better if you want the fuller dashboard: separate tabs/pages for assessment, metrics, and batch review |

**Recommendation:** Use **Streamlit** as the primary dashboard shell (multi-page: Assessment / Knowledge Agent / Metrics), and embed the RAG chat as a Streamlit chat component (`st.chat_message`/`st.chat_input`) rather than switching frameworks. This keeps one codebase and avoids the overhead of running two separate apps. If your priority shifts to a very fast standalone chat demo of just the RAG agent, Gradio's `ChatInterface` is the quicker path — but for the combined adjuster-facing tool described above, Streamlit's layout flexibility wins given your stated requirements.

## 8. Page-by-Page Spec (Streamlit)

**Page 1 — Damage Assessment**
- Sidebar: image uploader, "Run Assessment" button
- Main panel: image preview, results card (damage code, severity, cost range, confidence), override form
- Flag banner (conditional) for low-confidence/fraud-suspicious cases

**Page 2 — Knowledge Agent**
- Persona selector (optional)
- Chat interface with citation display under each response
- Suggested-prompt buttons

**Page 3 — Evaluation Metrics**
- Summary stat cards (accuracy, FP rate, mIoU)
- Per-class bar chart
- Confusion matrix
- Note on INS-MMBench held-out comparison

**Page 4 — Batch View (stretch)**
- Table of recent assessments with filter/sort


## 9. Risks & Mitigations

| Risk | Mitigation |
|---|---|
| Inference latency too slow for live demo | Pre-run a set of "known good" example images as fallback demo path |
| RAG KB too sparse to answer realistic questions | Scope KB narrowly (taxonomy + rubrics + a handful of curated policy/precedent docs) rather than trying to cover everything |
| Time pressure pushes dashboard polish over correctness | Prioritize Pages 1–2 (core value) over Page 4 (batch view, stretch only) |
| Confusing override UX distracts from core demo | Keep override minimal (edit fields + reason text) rather than building a full workflow |

## 11. Include the feedback loop as well for future fine tuning.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-03T21:55:19+05:30.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from None to Claude Sonnet 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>