"""ticket-triage — FastAPI + Gradio (Milestone 5 fills this in; see CLAUDE.md §6).

Run: uvicorn app:app --port 7860
"""

from __future__ import annotations

from fastapi import FastAPI

from ticket_triage import __version__

app = FastAPI(title="ticket-triage", version=__version__)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": __version__}
