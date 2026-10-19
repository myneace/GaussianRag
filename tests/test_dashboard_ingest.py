"""test_dashboard_ingest.py — Integration tests for /api/ingest in dashboard_server.py.

Validates BUG-01:
  - POST /api/ingest instantiates KnowledgeStore without NameError crash
  - Files uploaded are parsed, chunked, and saved to disk
  - Returns HTTP 200 with valid session statistics
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import pytest
import httpx

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dashboard_server import app, BASE_STORE_DIR


@pytest.mark.asyncio
async def test_dashboard_ingest_success():
    """Verify POST /api/ingest processes a valid text file and writes to session store."""
    transport = httpx.ASGITransport(app=app)
    file_content = (
        b"Law 1: Never outshine the master.\n"
        b"Always make those above you feel comfortably superior. In your desire to please "
        b"or impress them, do not go too far in displaying your talents or you might accomplish "
        b"the opposite - inspire fear and insecurity."
    )
    files = {"file": ("power_law_1.txt", file_content, "text/plain")}

    created_session_id = None
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/ingest", files=files)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data.get("status") == "success"
        assert "session_id" in data
        assert data.get("nodes_added", 0) > 0
        created_session_id = data["session_id"]

        session_path = BASE_STORE_DIR / created_session_id
        assert session_path.exists(), f"Session directory {session_path} was not created"
        # Find document subdirectory
        subdirs = [p for p in session_path.iterdir() if p.is_dir()]
        assert len(subdirs) >= 1
        doc_dir = subdirs[0]
        assert (doc_dir / "knowledge.json").exists(), "knowledge.json missing in session doc_dir"
        assert (doc_dir / "lexical.db").exists(), "lexical.db missing in session doc_dir"

        payload = json.loads((doc_dir / "knowledge.json").read_text(encoding="utf-8"))
        assert len(payload) == data["nodes_added"]

    finally:
        if created_session_id:
            session_path = BASE_STORE_DIR / created_session_id
            if session_path.exists():
                shutil.rmtree(session_path, ignore_errors=True)


@pytest.mark.asyncio
async def test_dashboard_ingest_latin1_fallback():
    """Verify POST /api/ingest handles non-utf8 files gracefully via fallback."""
    transport = httpx.ASGITransport(app=app)
    # Byte 0xE9 is 'é' in latin-1, invalid UTF-8 without multibyte sequence
    latin1_content = "Café standard knowledge and historical treatise.".encode("latin-1")
    files = {"file": ("cafe_history.txt", latin1_content, "text/plain")}

    created_session_id = None
    try:
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            resp = await client.post("/api/ingest", files=files)

        assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert data.get("status") == "success"
        created_session_id = data.get("session_id")
    finally:
        if created_session_id:
            session_path = BASE_STORE_DIR / created_session_id
            if session_path.exists():
                shutil.rmtree(session_path, ignore_errors=True)
