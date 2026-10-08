import os
import json
import asyncio
import time
import re
import shutil
from contextlib import asynccontextmanager
from typing import List, Optional, Any
from pathlib import Path
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from gaussian_rag.config import load_project_config, load_provider_env, load_runtime_env
from gaussian_rag.core.gaussian_embedding import GaussianEmbedder
from gaussian_rag.creator.ingestion import IngestionPipeline
from gaussian_rag.core.store import KnowledgeStore
from gaussian_rag.rag.retrieval.retriever import GaussianRetriever
from gaussian_rag.rag.disambiguation.pipeline import disambiguate
from gaussian_rag.rag.disambiguation.router import route
from gaussian_rag.rag.sense_cache import SenseCache
from gaussian_rag.rag.generation.provider import MistralChatClient
from gaussian_rag.rag.generation.generator import ProviderBackedGenerator
from gaussian_rag.training.online_trainer import OnlineRadiationCollector

# Configuration
BASE_STORE_DIR = Path(".full_run_store")
BASE_STORE_DIR.mkdir(exist_ok=True)

# Global state
config = load_project_config()
runtime_env = load_runtime_env()
active_session: Optional[str] = None
store = KnowledgeStore()

# SSE Event Queue
event_queue = asyncio.Queue()

# Continual learning — initialized at startup once the event loop is running
radiation_collector: Optional[OnlineRadiationCollector] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize singletons that need the running event loop."""
    global radiation_collector
    loop = asyncio.get_event_loop()

    def _sse_emit(event_type: str, data: dict) -> None:
        loop.call_soon_threadsafe(
            event_queue.put_nowait, {"type": event_type, "data": data}
        )

    cl_config = config.get("continual_learning", {"enabled": True, "train_every_n": 5})
    if cl_config.get("enabled", True):
        radiation_collector = OnlineRadiationCollector(
            train_every_n=cl_config.get("train_every_n", 5), emit_fn=_sse_emit
        )
    else:
        radiation_collector = None

    yield  # application runs here


app = FastAPI(title="GSKG Manifold Dashboard", lifespan=lifespan)

# Enable CORS for the Vite frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.lower()).strip("_")


def _build_embedder(on_event: Optional[Any] = None) -> GaussianEmbedder:
    encoder_config = config["encoder"]
    return GaussianEmbedder(
        dimension=encoder_config["dimension"],
        sigma_mode=encoder_config["sigma_mode"],
        rank=encoder_config["low_rank"],
        embedding_backend=encoder_config.get("embedding_backend", "local"),
        on_mismatch=encoder_config.get("on_mismatch", "error"),
        on_event=on_event,
    )


def _get_pipeline(on_event: Optional[Any] = None):
    encoder_config = config["encoder"]
    embedder = GaussianEmbedder(
        dimension=encoder_config["dimension"],
        sigma_mode=encoder_config["sigma_mode"],
        rank=encoder_config["low_rank"],
        embedding_backend=encoder_config.get("embedding_backend", "local"),
        on_mismatch=encoder_config.get("on_mismatch", "error"),
        on_event=on_event,
    )

    llm_caller = None
    try:
        provider_env = load_provider_env()
        client = MistralChatClient(provider_env)
        llm_caller = client.generate
    except Exception:
        pass

    pipeline = IngestionPipeline(
        embedder,
        llm_caller=llm_caller,
        extract_entities=True,
        ai_chunking=(llm_caller is not None),
        on_event=on_event,
    )
    return pipeline


@app.get("/api/nodes")
async def get_nodes():
    nodes = []
    for k in store.values():
        nodes.append(
            {
                "id": k.id,
                "text": k.text,
                "mu": k.mu.tolist(),
                "sigma": float(k.sigma_diag.sum()),
                "metadata": k.metadata,
            }
        )
    return nodes


@app.get("/api/radiation/stats")
async def get_radiation_stats():
    """Return the cumulative radiation absorption and training stats."""
    if radiation_collector is None:
        return {"error": "Radiation collector not initialized"}
    return radiation_collector.get_stats()


@app.get("/api/sessions")
async def list_sessions():
    if not BASE_STORE_DIR.exists():
        return []

    manifolds = {}
    dirs = [d for d in BASE_STORE_DIR.iterdir() if d.is_dir()]
    dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    # First pass: Base manifolds
    for sess_dir in dirs:
        metadata_file = sess_dir / "metadata.json"
        if not metadata_file.exists():
            manifold_id = sess_dir.name
            manifolds[manifold_id] = {
                "id": manifold_id,
                "name": manifold_id,
                "chats": [
                    {
                        "id": manifold_id,
                        "name": "Original Chat",
                        "timestamp": sess_dir.stat().st_mtime,
                        "active": manifold_id == active_session,
                    }
                ],
            }

    # Second pass: Pointer chats
    for sess_dir in dirs:
        metadata_file = sess_dir / "metadata.json"
        if metadata_file.exists():
            try:
                metadata = json.loads(metadata_file.read_text())
                manifold_path = Path(metadata["manifold_path"])
                manifold_id = manifold_path.parent.name

                if manifold_id in manifolds:
                    manifolds[manifold_id]["chats"].append(
                        {
                            "id": sess_dir.name,
                            "name": "Branch Chat",
                            "timestamp": sess_dir.stat().st_mtime,
                            "active": sess_dir.name == active_session,
                        }
                    )
            except Exception as e:
                print(f"Error parsing metadata for {sess_dir}: {e}")

    # Sort chats within each manifold by timestamp descending
    for m in manifolds.values():
        m["chats"].sort(key=lambda x: x["timestamp"], reverse=True)

    return list(manifolds.values())


class SwitchSessionRequest(BaseModel):
    session_id: str


@app.post("/api/sessions/switch")
async def switch_session(req: SwitchSessionRequest):
    global active_session, store
    sess_path = BASE_STORE_DIR / req.session_id
    if not sess_path.exists():
        raise HTTPException(status_code=404, detail="Session not found")

    metadata_file = sess_path / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        data_path = Path(metadata["manifold_path"])
    else:
        subdirs = [d for d in sess_path.iterdir() if d.is_dir()]
        if not subdirs:
            raise HTTPException(status_code=404, detail="Session data not found")
        data_path = subdirs[0]

    store = KnowledgeStore.load(str(data_path))
    active_session = req.session_id
    return {"status": "success", "session_id": active_session}


@app.post("/api/sessions/new_chat")
async def create_new_chat():
    global active_session, store
    if not active_session:
        raise HTTPException(status_code=400, detail="No active session to branch from")

    # Extract base doc name
    parts = active_session.split("_")
    doc_slug = "_".join(parts[:-1]) if len(parts) > 1 else active_session

    timestamp = int(time.time())
    new_session_id = f"{doc_slug}_{timestamp}"
    new_session_path = BASE_STORE_DIR / new_session_id

    old_session_path = BASE_STORE_DIR / active_session
    # Create pointer instead of copying
    new_session_path.mkdir(parents=True, exist_ok=True)
    metadata_file = new_session_path / "metadata.json"

    # Check if the old session is already a pointer
    old_metadata = old_session_path / "metadata.json"
    if old_metadata.exists():
        # Point to the original manifold
        shutil.copy(old_metadata, metadata_file)
    else:
        # Create pointer to the old session's data
        old_subdirs = [d for d in old_session_path.iterdir() if d.is_dir()]
        if not old_subdirs:
            raise HTTPException(
                status_code=404, detail="Current session data not found"
            )
        old_data_path = old_subdirs[0]
        metadata_file.write_text(json.dumps({"manifold_path": str(old_data_path)}))

    active_session = new_session_id
    return {"status": "success", "session_id": active_session}


@app.get("/api/sessions/{session_id}/history")
async def get_chat_history(session_id: str):
    history_file = BASE_STORE_DIR / session_id / "chat_history.json"
    if not history_file.exists():
        return []
    try:
        return json.loads(history_file.read_text())
    except:
        return []


def save_chat_history(
    session_id: str, user_msg: str, ai_msg: str, retrieved: List[dict]
):
    if not session_id:
        return
    history_file = BASE_STORE_DIR / session_id / "chat_history.json"
    history = []
    if history_file.exists():
        try:
            history = json.loads(history_file.read_text())
        except:
            history = []

    history.append(
        {
            "timestamp": time.time(),
            "user": user_msg,
            "ai": ai_msg,
            "retrieved": retrieved,
        }
    )

    history_file.write_text(json.dumps(history, indent=2))


@app.post("/api/ingest")
async def ingest_document(file: UploadFile = File(...)):
    global active_session, store

    content = await file.read()
    text = content.decode("utf-8")
    original_filename = file.filename or "uploaded_doc"
    doc_slug = _slugify(Path(original_filename).stem)

    # Create new session directory
    timestamp = int(time.time())
    session_id = f"{doc_slug}_{timestamp}"
    session_path = BASE_STORE_DIR / session_id
    data_path = session_path / doc_slug
    data_path.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_event_loop()

    def sync_on_event(event_type, data):
        loop.call_soon_threadsafe(
            event_queue.put_nowait, {"type": event_type, "data": data}
        )

    pipeline = _get_pipeline(on_event=sync_on_event)

    # Run ingestion
    new_store = KnowledgeStore()

    def run_ingestion():
        return pipeline.ingest_document(text, document_id=doc_slug)

    new_nodes = await asyncio.to_thread(run_ingestion)

    for node in new_nodes:
        new_store.add([node])

    new_store.save(str(data_path))

    store = new_store
    active_session = session_id

    # Collect radiation: every ingested node is a perfect {text, domain} training sample
    if radiation_collector is not None:
        ingestion_radiation = [
            {"text": node.text, "domain": node.metadata.get("node_type", "anchor")}
            for node in new_nodes
            if node.text
        ]
        radiation_collector.collect(ingestion_radiation)

    return {
        "status": "success",
        "session_id": session_id,
        "nodes_added": len(new_nodes),
    }


class QueryRequest(BaseModel):
    text: str
    top_k: int = 5


@app.post("/api/query")
async def query_graph(req: QueryRequest):
    if not store:
        raise HTTPException(status_code=400, detail="No active knowledge store")

    loop = asyncio.get_event_loop()

    def emit(event_type, data):
        loop.call_soon_threadsafe(
            event_queue.put_nowait, {"type": event_type, "data": data}
        )

    emit("query_received", {"text": req.text})

    def run_query():
        # 1. Setup components
        embedder = _build_embedder(on_event=emit)
        retriever = GaussianRetriever(store)
        retriever.refresh()

        # Sense cache (local to session)
        cache_dir = Path(runtime_env["store_path"]) / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = SenseCache.load(cache_dir)

        provider_env = load_provider_env()
        client = MistralChatClient(provider_env)

        def llm_caller(prompt: str) -> str:
            return client.generate(prompt)

        # 2. Disambiguation Step
        emit("query_disambiguation_started", {"text": req.text})
        packet = disambiguate(
            req.text,
            embedder,
            cache,
            retriever=retriever,
            llm_caller=llm_caller,
            min_confidence=0.05,
        )
        cache.save(cache_dir)

        emit(
            "query_senses_detected",
            {
                "entropy": packet.entropy,
                "senses": [
                    {
                        "id": s.sense_id,
                        "domain": s.domain,
                        "weight": packet.field_weights.get(s.sense_id, 0.0),
                    }
                    for s in packet.senses
                ],
            },
        )

        # 3. Parallel Retrieval (MMASA)
        emit(
            "query_retrieval_started",
            {
                "top_k": req.top_k,
                "mode": "mmasa" if len(packet.senses) > 1 else "standard",
                "adaptive": True,
            },
        )

        sense_results, fallback = route(
            packet,
            embedder,
            retriever,
            top_k=req.top_k,
            metric=config.get("retrieval", {}).get("metric", "elk"),
        )

        # 4. Flatten / Fuse results for the dashboard response
        # (The router already interleaved them if multi-sense)
        all_results: list[RetrievedChunk] = []
        seen_ids = set()

        # We preserve the order provided by the router (which handles fusion)
        # sense_results is sid -> list[RetrievedChunk]
        if len(sense_results) == 1:
            all_results = list(next(iter(sense_results.values())))
        else:
            # For multi-sense, we use the interleave order from the values
            # However, the router's return type for multi-sense is sid -> list
            # We want to flatten it based on the weighted order
            from gaussian_rag.rag.retrieval.manifold_ranker import rerank_multi_sense

            per_sense = [
                (packet.field_weights.get(sid, 0.0), chunks)
                for sid, chunks in sense_results.items()
            ]
            all_results = rerank_multi_sense(per_sense, fusion="interleave")

        print(
            f"DEBUG: MMASA Retrieval returned {len(all_results)} results (senses={len(sense_results)})"
        )

        emit(
            "query_retrieval_complete",
            {
                "count": len(all_results),
                "fallback": fallback,
                "nodes": [r.knowledge.id for r in all_results],
            },
        )

        # 5. Generation
        emit("query_generation_started", {})
        generation_config = config["generation"]
        generator = ProviderBackedGenerator(generation_config, provider_env)

        answer_data = generator.generate(req.text, all_results)
        emit("query_generation_complete", {})

        # Save to history
        save_chat_history(
            active_session,
            req.text,
            answer_data["answer"],
            [
                {
                    "id": r.knowledge.id,
                    "text": r.knowledge.text,
                    "confidence": r.confidence,
                }
                for r in all_results
            ],
        )

        # Collect radiation for continual learning
        if radiation_collector is not None:
            radiation_collector.record_event(
                {
                    "query_text": req.text,
                    "sense_queries": [s.retrieval_query for s in packet.senses],
                    "retrieved_docs": [
                        {
                            "id": r.knowledge.id,
                            "text": r.knowledge.text,
                            "domain": r.knowledge.metadata.get("node_type", "anchor"),
                            "confidence": r.confidence,
                        }
                        for r in all_results
                    ],
                    "feedback_signal": "implicit_generation",
                    "fallback_triggered": fallback,
                }
            )
            radiation: list[dict] = [{"text": req.text, "domain": "query"}]
            for s in packet.senses:
                radiation.append({"text": s.retrieval_query, "domain": s.domain})
            for r in all_results:
                node_type = r.knowledge.metadata.get("node_type", "anchor")
                radiation.append({"text": r.knowledge.text, "domain": node_type})
            radiation_collector.collect(radiation)

        return {
            "answer": answer_data["answer"],
            "retrieved": [
                {
                    "id": r.knowledge.id,
                    "text": r.knowledge.text,
                    "confidence": r.confidence,
                }
                for r in all_results
            ],
        }

    return await asyncio.to_thread(run_query)


@app.get("/api/events")
async def events():
    async def event_generator():
        while True:
            event = await event_queue.get()
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8001)
