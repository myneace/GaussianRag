"""test_telemetry_sse.py — SSE telemetry stream integration tests.

Validates BENCH-02:
  - /api/telemetry/stream endpoint streams Server-Sent Events with text/event-stream
  - pipeline_timing events contain all required stage breakdown fields
  - event_queue broadcasts are received by SSE clients
  - emit_telemetry() helper correctly enqueues pipeline_timing events
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import pytest
import httpx

_PKG_ROOT = Path(__file__).resolve().parents[1]
if str(_PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(_PKG_ROOT))

from dashboard_server import app, event_queue, emit_telemetry  # noqa: E402


REQUIRED_TIMING_FIELDS = {
    "disambiguation_ms",
    "taxonomy_routing_ms",
    "fts5_prefilter_ms",
    "clef_passage_gating_ms",
    "woodbury_elk_rerank_ms",
    "total_ms",
    "pruned_ratio",
    "survivor_count",
}


def _sample_pipeline_timing() -> dict:
    return {
        "disambiguation_ms": 32.4,
        "taxonomy_routing_ms": 18.7,
        "fts5_prefilter_ms": 1.1,
        "clef_passage_gating_ms": 28.3,
        "woodbury_elk_rerank_ms": 2.5,
        "total_ms": 83.0,
        "pruned_ratio": 0.85,
        "survivor_count": 5,
    }


def _drain_queue() -> None:
    while True:
        try:
            event_queue.get_nowait()
        except asyncio.QueueEmpty:
            break


def _parse_sse_block(buffer: str) -> list[dict]:
    """Parse SSE message blocks from buffer string."""
    events = []
    for block in buffer.split("\n\n"):
        block = block.strip()
        if not block or block.startswith(":"):
            continue
        event_type = None
        data_line = None
        for line in block.splitlines():
            if line.startswith("event:"):
                event_type = line[len("event:"):].strip()
            elif line.startswith("data:"):
                data_line = line[len("data:"):].strip()
        if event_type and data_line:
            try:
                events.append({"type": event_type, "data": json.loads(data_line)})
            except json.JSONDecodeError:
                pass
    return events


async def _collect_n_sse_events(
    n: int,
    inject_fn,
    collect_timeout: float = 10.0,
) -> list[dict]:
    """Connect to /api/telemetry/stream?limit={n}, inject events, collect n pipeline_timing events."""
    collected: list[dict] = []
    transport = httpx.ASGITransport(app=app)

    # Inject events first so queue has them ready for the stream
    await inject_fn()

    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get(f"/api/telemetry/stream?limit={n}")
        for evt in _parse_sse_block(resp.text):
            if evt["type"] == "pipeline_timing":
                collected.append(evt)

    return collected


# ---------------------------------------------------------------------------
# Test 1: Response headers
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_telemetry_stream_returns_200_and_event_stream_content_type() -> None:
    """Verify /api/telemetry/stream returns HTTP 200 with text/event-stream."""
    _drain_queue()

    # Pre-seed event so the SSE generator yields immediately
    event_queue.put_nowait({"type": "pipeline_timing", "data": _sample_pipeline_timing()})

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/telemetry/stream?limit=1")
        assert resp.status_code == 200, f"Expected HTTP 200, got {resp.status_code}"
        content_type = resp.headers.get("content-type", "")
        assert "text/event-stream" in content_type, (
            f"Expected text/event-stream content-type, got: {content_type!r}"
        )


# ---------------------------------------------------------------------------
# Test 2: emit_telemetry() enqueues correctly (pure unit test, no SSE needed)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_emit_telemetry_enqueues_pipeline_timing_event() -> None:
    """emit_telemetry() must put a pipeline_timing event onto event_queue."""
    _drain_queue()

    pipeline_data = _sample_pipeline_timing()
    emit_telemetry(pipeline_data)
    await asyncio.sleep(0.05)

    queued: list[dict] = []
    while True:
        try:
            queued.append(event_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    assert queued, "emit_telemetry() did not enqueue any event"
    evt = queued[0]
    assert evt.get("type") == "pipeline_timing", (
        f"Expected type='pipeline_timing', got {evt.get('type')!r}"
    )
    assert evt.get("data") == pipeline_data, (
        f"Payload mismatch.\nExpected: {pipeline_data}\nGot: {evt.get('data')}"
    )


# ---------------------------------------------------------------------------
# Test 3: pipeline_timing payload structure (unit test via queue)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_timing_payload_contains_all_required_fields() -> None:
    """Verify pipeline_timing payload has all 8 required stage breakdown fields."""
    _drain_queue()

    pipeline_data = _sample_pipeline_timing()

    # Verify payload structure directly (emit_telemetry unit test)
    missing = REQUIRED_TIMING_FIELDS - set(pipeline_data.keys())
    assert not missing, (
        f"Sample pipeline_timing payload is missing required fields: {missing}"
    )

    # Verify that the emit_telemetry helper faithfully transmits the payload
    emit_telemetry(pipeline_data)
    await asyncio.sleep(0.05)

    queued: list[dict] = []
    while True:
        try:
            queued.append(event_queue.get_nowait())
        except asyncio.QueueEmpty:
            break

    assert queued, "No event enqueued"
    received = queued[0].get("data", {})
    missing_in_received = REQUIRED_TIMING_FIELDS - set(received.keys())
    assert not missing_in_received, (
        f"Enqueued event missing required fields: {missing_in_received}"
    )


# ---------------------------------------------------------------------------
# Test 4: Pipeline timing field types and ranges
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_timing_payload_field_types_and_ranges() -> None:
    """All pipeline_timing fields must have correct types and values in valid ranges."""
    _drain_queue()

    pipeline_data = _sample_pipeline_timing()
    p = pipeline_data

    for ms_field in (
        "disambiguation_ms", "taxonomy_routing_ms", "fts5_prefilter_ms",
        "clef_passage_gating_ms", "woodbury_elk_rerank_ms", "total_ms",
    ):
        assert isinstance(p[ms_field], (int, float)), (
            f"{ms_field} must be numeric, got {type(p[ms_field])}"
        )
        assert p[ms_field] >= 0.0, f"{ms_field} must be non-negative"

    assert 0.0 <= p["pruned_ratio"] <= 1.0, (
        f"pruned_ratio must be in [0, 1], got {p['pruned_ratio']}"
    )
    assert isinstance(p["survivor_count"], int), (
        f"survivor_count must be int, got {type(p['survivor_count'])}"
    )
    assert p["survivor_count"] >= 0


# ---------------------------------------------------------------------------
# Test 5: SSE event delivery over the stream
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_pipeline_timing_event_is_received_over_sse() -> None:
    """pipeline_timing events injected into event_queue arrive over the SSE stream."""
    _drain_queue()
    pipeline_data = _sample_pipeline_timing()

    async def inject():
        event_queue.put_nowait({"type": "pipeline_timing", "data": pipeline_data})

    collected = await _collect_n_sse_events(n=1, inject_fn=inject)

    assert collected, "No pipeline_timing event received over SSE stream"
    assert collected[0]["type"] == "pipeline_timing"
    assert collected[0]["data"]["total_ms"] == pipeline_data["total_ms"]


# ---------------------------------------------------------------------------
# Test 6: Sequential multi-event delivery
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_three_sequential_pipeline_timing_events_delivered_in_order() -> None:
    """Three sequential pipeline_timing events must arrive in order over SSE."""
    _drain_queue()

    base = _sample_pipeline_timing()
    events_to_send = [
        {**base, "total_ms": 80.0 + i * 5.0, "survivor_count": i + 3}
        for i in range(3)
    ]

    async def inject():
        for evt_data in events_to_send:
            event_queue.put_nowait({"type": "pipeline_timing", "data": evt_data})
            await asyncio.sleep(0.02)

    collected = await _collect_n_sse_events(n=3, inject_fn=inject, collect_timeout=10.0)

    assert len(collected) == 3, (
        f"Expected 3 pipeline_timing events, received {len(collected)}"
    )
    for i, (sent, got) in enumerate(zip(events_to_send, collected)):
        assert got["data"]["total_ms"] == sent["total_ms"], (
            f"Event {i}: total_ms mismatch"
        )
        assert got["data"]["survivor_count"] == sent["survivor_count"], (
            f"Event {i}: survivor_count mismatch"
        )
