# Multi-Agent Disambiguation Pipeline

The Multi-Agent Disambiguation Pipeline is the core reasoning layer of GaussianRAG. It transforms ambiguous natural language queries into precise, manifold-aware retrieval strategies by orchestrating multiple specialized AI agents.

## Architecture Overview

The pipeline operates as a directed acyclic graph (DAG) of agents, each refining the semantic resolution of the query.

### 0. Agent 0: Query Dissolver (Uncertainty Routing)
The Query Dissolver applies the "Agent Dissolver" pattern to the query itself. If the query is structurally ambiguous (e.g., "The field was ready for harvest"), it projects the query into multiple high-resolution rewrites.
- **Output**: An `ambiguity_score` and a set of `SenseNode` interpretations representing discrete query intents.
- **Manifold Relation**: Query interpretations are automatically linked to the closest existing manifold nodes (Anchors and Interpretations) discovered during ingestion.

### 1. Agent 1: Sense Generator (Term-Level Polysemy)
If the query-level ambiguity is low, the pipeline falls back to the Sense Generator to handle term-level polysemy (e.g., "What is the latest Apple earnings?").
- **Output**: A list of `SenseNode` objects with domain labels and descriptions.
- **Constraint**: It maps the full semantic breadth of specific ambiguous tokens.

### 2. Agent 2: Domain Grounder (Manifold Anchoring)
The Grounder projects each identified sense into the Gaussian Manifold. It calculates the specific geometric coordinates ($ \mu_{anchor} $) and variance ($ \sigma_{anchor} $) for the sense.
- **Benefit**: This allows the system to carry "anchors" through the pipeline, avoiding redundant re-encoding at query time.

### 3. Agent 3: Field Mapper
The Field Mapper analyzes the grounded senses, computes compatibility weights, and prepares the `DisambiguationPacket` for routing:
- **Single-Sense**: If one sense dominates (High Confidence, Low Entropy).
- **Multi-Sense**: If the query is genuinely polysemous (e.g., "Apple" without context).
- **Fallback**: If no sense meets the confidence threshold.

### 4. Agent 4: Memory & Router Packager
The final agent applies session-memory continuity, packages the retrieval-ready queries, and hands the packet to the entropy router for ELK-first retrieval.

## Key Mathematical Signals

### Semantic Entropy ($ H $)
We calculate the Shannon Entropy of the sense distribution to measure ambiguity:
$$ H = - \sum p_i \log_2(p_i) $$
- **$ H < 1.5 $**: Sharp query, proceed with single-sense retrieval.
- **$ H \geq 1.5 $**: Ambiguous query, trigger multi-sense interleave or clarification.

### Manifold Sharpening
By using the pre-computed `sigma_anchor`, the retriever can "sharpen" its focus on specific regions of the manifold that correspond to the intended domain, effectively filtering out noise from competing senses.

## Implementation Details

- **Core Module**: `gaussian_rag.rag.disambiguation.pipeline`
- **Agents**:
    - `query_dissolver.py`: Agent 0 ambiguity routing.
    - `sense_generator.py`: Agent 1 sense shaping.
    - `domain_grounder.py`: Agent 2 geometric anchoring.
    - `field_mapper.py`: Agent 3 compatibility + entropy mapping.
    - `memory_packager.py`: Agent 4 session continuity packaging.
    - `router.py`: Entropy-based strategic dispatch.
- **Caching**: `SenseCache` persists grounded senses across sessions, enabling "contextual memory" for frequent terms.

## Usage

To run a disambiguated query via the CLI:

```bash
python3 gaussian_rag/main.py polysemy-query --text "Your query here" --store ./your_store
```

To run the integration demo:

```bash
python3 scripts/polysemy_demo.py
```
