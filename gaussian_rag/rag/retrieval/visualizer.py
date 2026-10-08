import json
import numpy as np
from pathlib import Path
import textwrap
from gaussian_rag.core.store import KnowledgeStore

def pca(X, n_components=3):
    """Simple PCA using pure NumPy."""
    X_centered = X - np.mean(X, axis=0)
    # Handle cases where all values are the same
    if np.allclose(X_centered, 0):
        return np.zeros((X.shape[0], n_components))
        
    cov_matrix = np.cov(X_centered, rowvar=False)
    # If it's a scalar or 1D array, cov might return something weird or 0
    if np.isscalar(cov_matrix) or cov_matrix.ndim == 0:
        return np.zeros((X.shape[0], n_components))
        
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Sort eigenvalues and eigenvectors in descending order
    sorted_index = np.argsort(eigenvalues)[::-1]
    sorted_eigenvectors = eigenvectors[:, sorted_index]
    
    # Select the first n_components
    eigenvector_subset = sorted_eigenvectors[:, 0:n_components]
    
    # Project the data
    X_reduced = np.dot(X_centered, eigenvector_subset)
    
    return X_reduced

def generate_visualization(store_path: str | Path, output_dir: str | Path) -> str:
    """
    Generates an interactive HTML visualization of the Gaussian field.
    Returns the path to the generated HTML file.
    """
    store_path = Path(store_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "index.html"
    
    print(f"Loading Gaussian store from {store_path}...")
    store = KnowledgeStore.load(store_path)
    data = store.values()
    
    if not data:
        raise ValueError("No data found in store.")
        
    print(f"Processing {len(data)} Gaussian embeddings...")
    
    ids = []
    texts = []
    mus = []
    sigmas = []
    sources = []
    node_types = []
    anchor_ids = []
    
    for item in data:
        ids.append(item.id)
        wrapped_text = "<br>".join(textwrap.wrap(item.text, width=60))
        texts.append(wrapped_text)
        mus.append(item.mu)
        sigmas.append(float(np.sum(item.sigma_diag)))
        sources.append(item.metadata.get("source_path", "Unknown"))
        node_types.append(item.metadata.get("node_type", "anchor"))
        anchor_ids.append(item.metadata.get("anchor_id"))
        
    mus = np.array(mus)
    id_to_idx = {node_id: i for i, node_id in enumerate(ids)}
    
    # Project to 3D using PCA
    if len(mus) <= 1:
        reduced_mus = np.zeros((len(mus), 3))
    else:
        n_comp = min(3, len(mus) - 1, mus.shape[1])
        reduced_mus = pca(mus, n_components=n_comp)
        if reduced_mus.shape[1] < 3:
            padding = np.zeros((reduced_mus.shape[0], 3 - reduced_mus.shape[1]))
            reduced_mus = np.hstack([reduced_mus, padding])
            
    x, y, z = reduced_mus[:, 0], reduced_mus[:, 1], reduced_mus[:, 2]
    
    # Manifold Filaments (lines between interpretations and anchors)
    line_x, line_y, line_z = [], [], []
    for i, anchor_id in enumerate(anchor_ids):
        if anchor_id and anchor_id in id_to_idx:
            anchor_idx = id_to_idx[anchor_id]
            # Add line segment: Interpretation -> Anchor -> None (to break the line)
            line_x.extend([x[i], x[anchor_idx], None])
            line_y.extend([y[i], y[anchor_idx], None])
            line_z.extend([z[i], z[anchor_idx], None])

    # Sizing and Colors
    norm_sigmas = []
    colors = []
    for i, n_type in enumerate(node_types):
        if n_type == "anchor":
            norm_sigmas.append(25)
            colors.append("#38bdf8") # Bright Blue
        else:
            norm_sigmas.append(12)
            colors.append("#818cf8") # Indigo
            
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Manifold | Gaussian Field Explorer</title>
    <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600&display=swap" rel="stylesheet">
    <style>
        :root {{
            --bg: #020617;
            --panel: rgba(15, 23, 42, 0.8);
            --border: #1e293b;
            --text: #f8fafc;
            --accent: #38bdf8;
            --accent-secondary: #818cf8;
        }}
        body {{ 
            margin: 0; padding: 0; 
            font-family: 'Inter', sans-serif; 
            background-color: var(--bg); 
            color: var(--text); 
            overflow: hidden; 
        }}
        #plot {{ width: 100vw; height: 100vh; }}
        #ui-panel {{ 
            position: absolute; top: 24px; left: 24px; 
            background: var(--panel); 
            backdrop-filter: blur(16px); 
            padding: 24px; 
            border-radius: 16px; 
            border: 1px solid var(--border); 
            box-shadow: 0 20px 50px rgba(0,0,0,0.6); 
            z-index: 1000; 
            max-width: 380px;
        }}
        h1 {{ 
            margin: 0 0 8px 0; 
            font-size: 1.5rem; 
            font-weight: 600;
            background: linear-gradient(135deg, var(--accent), var(--accent-secondary)); 
            -webkit-background-clip: text; 
            -webkit-text-fill-color: transparent; 
        }}
        .badge {{
            display: inline-block;
            padding: 4px 10px;
            background: rgba(56, 189, 248, 0.1);
            border: 1px solid rgba(56, 189, 248, 0.2);
            border-radius: 99px;
            font-size: 0.75rem;
            color: var(--accent);
            margin-bottom: 16px;
        }}
        p {{ margin: 0 0 20px 0; font-size: 0.95rem; color: #94a3b8; line-height: 1.6; }}
        .legend-item {{ display: flex; align-items: center; margin-bottom: 12px; font-size: 0.85rem; color: #cbd5e1; }}
        .legend-color {{ width: 14px; height: 14px; border-radius: 50%; margin-right: 12px; }}
    </style>
</head>
<body>
    <div id="ui-panel">
        <h1>Gaussian Field Explorer</h1>
        <div class="badge">{len(data)} Knowledge Nodes</div>
        <p>A topological view of the information manifold. Anchors represent core chunks; interpretations represent dissolved semantic variants.</p>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background: var(--accent); border: 2px solid #fff;"></div>
                <span><strong>Anchor:</strong> Core Knowledge Hub</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background: var(--accent-secondary);"></div>
                <span><strong>Interpretation:</strong> Dissolved Semantic View</span>
            </div>
            <div class="legend-item">
                <div style="width: 14px; height: 2px; background: rgba(255,255,255,0.2); margin-right: 12px;"></div>
                <span><strong>Filament:</strong> Manifold Connection</span>
            </div>
        </div>
    </div>
    <div id="plot"></div>

    <script>
        const x = {json.dumps(x.tolist())};
        const y = {json.dumps(y.tolist())};
        const z = {json.dumps(z.tolist())};
        const texts = {json.dumps(texts)};
        const ids = {json.dumps(ids)};
        const colors = {json.dumps(colors)};
        const sizes = {json.dumps(norm_sigmas)};
        
        const lineX = {json.dumps(line_x)};
        const lineY = {json.dumps(line_y)};
        const lineZ = {json.dumps(line_z)};

        const filaments = {{
            x: lineX, y: lineY, z: lineZ,
            mode: 'lines',
            line: {{ color: 'rgba(255, 255, 255, 0.1)', width: 1 }},
            hoverinfo: 'none',
            type: 'scatter3d'
        }};

        const nodes = {{
            x: x, y: y, z: z,
            mode: 'markers',
            marker: {{
                size: sizes,
                color: colors,
                opacity: 0.9,
                line: {{ color: 'rgba(255, 255, 255, 0.3)', width: 1 }}
            }},
            text: texts,
            hoverinfo: 'text',
            type: 'scatter3d'
        }};

        const layout = {{
            scene: {{
                xaxis: {{ visible: false }}, yaxis: {{ visible: false }}, zaxis: {{ visible: false }},
                bgcolor: '#020617'
            }},
            paper_bgcolor: '#020617',
            margin: {{ l: 0, r: 0, b: 0, t: 0 }},
            showlegend: false
        }};

        Plotly.newPlot('plot', [filaments, nodes], layout);
    </script>
</body>
</html>
    """
    output_file.write_text(html_content, encoding="utf-8")
    return str(output_file)
