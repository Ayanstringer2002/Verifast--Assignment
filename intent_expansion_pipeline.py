

import os
import json
import argparse
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sentence_transformers import SentenceTransformer

# Gemini imports loader
def _load_gemini_models():
    from google.generativeai import GenerativeModel, TextEmbeddingModel
    return GenerativeModel, TextEmbeddingModel

# Attempt optional imports
try:
    import umap
    has_umap = True
except ImportError:
    has_umap = False

try:
    import hdbscan
    has_hdbscan = True
except ImportError:
    has_hdbscan = False


# ----------------------------
# Preprocessing
# ----------------------------
def basic_clean(text: str) -> str:
    return " ".join(text.lower().strip().split())


# ----------------------------
# Embeddings (Gemini or fallback)
# ----------------------------
def embed_texts(texts, use_gemini=True, gemini_key=None):
    if use_gemini and gemini_key:
        try:
            from google.generativeai import configure, TextEmbeddingModel
            configure(api_key=gemini_key)

            model = TextEmbeddingModel("models/embedding-001")
            vectors = []

            for t in texts:
                resp = model.embed_content(t)
                vectors.append(resp["embedding"])

            return np.array(vectors)

        except Exception as e:
            print("Gemini embedding failed — falling back to sentence-transformers.")
            print("ERROR:", e)

    # Fallback to sentence-transformers (local)
    model = SentenceTransformer("all-MiniLM-L6-v2")
    return model.encode(texts, show_progress_bar=False)


# ----------------------------
# Clustering
# ----------------------------
def cluster_embeddings(X, min_cluster_size=15, n_clusters=20):
    if has_umap:
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.0, metric="cosine")
        X_red = reducer.fit_transform(X)
    else:
        X_red = X

    if has_hdbscan:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric="euclidean")
        labels = clusterer.fit_predict(X_red)
    else:
        km = KMeans(n_clusters=n_clusters, random_state=0)
        labels = km.fit_predict(X_red)

    return labels


# ----------------------------
# Cluster keyword extraction
# ----------------------------
def extract_keywords(texts, top_k=6):
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    X = vectorizer.fit_transform(texts)

    scores = np.asarray(X.mean(axis=0)).ravel()
    idx = scores.argsort()[::-1][:top_k]
    feats = vectorizer.get_feature_names_out()

    return [feats[i] for i in idx]


# ----------------------------
# Gemini-based LLM proposal generator
# ----------------------------
def generate_proposal(cluster_summary, gemini_key):
    from google.generativeai import configure
    GenerativeModel, _ = _load_gemini_models()

    configure(api_key=gemini_key)
    model = GenerativeModel("gemini-pro")

    prompt = f"""
You are an expert in intent taxonomy design.
Given this cluster summary:
{cluster_summary}

Return:
- Proposed primary intent (if new)
- Proposed secondary intents
- Rationale
- Overlaps with existing mapped intents (if any)
"""

    response = model.generate_text(prompt)
    return response.text


# ----------------------------
# Main pipeline
# ----------------------------
def process(input_json, output_dir, gemini_key):
    data = json.load(open(input_json))
    df = pd.DataFrame(data)

    df["clean"] = df["text"].apply(basic_clean)

    print("Embedding…")
    X = embed_texts(df["clean"].tolist(), use_gemini=True, gemini_key=gemini_key)

    print("Clustering…")
    labels = cluster_embeddings(X)
    df["cluster"] = labels

    clusters = {}

    for c in sorted(df["cluster"].unique()):
        sub = df[df["cluster"] == c]

        clusters[c] = {
            "count": len(sub),
            "examples": sub["text"].head(10).tolist(),
            "keywords": extract_keywords(sub["clean"].tolist()),
        }

    proposals = {}

    print("Generating proposals with Gemini…")
    for cid, obj in clusters.items():
        summary = {
            "cluster_id": cid,
            "count": obj["count"],
            "keywords": obj["keywords"],
            "examples": obj["examples"][:5],
        }

        proposals[cid] = generate_proposal(summary, gemini_key)

    os.makedirs(output_dir, exist_ok=True)

    json.dump(clusters, open(f"{output_dir}/cluster_examples.json", "w"), indent=2)
    json.dump(proposals, open(f"{output_dir}/proposals.json", "w"), indent=2)

    # Write Markdown Report
    with open(f"{output_dir}/report.md", "w") as f:
        for cid in clusters:
            f.write(f"## Cluster {cid}\n")
            f.write(f"Count: {clusters[cid]['count']}\n")
            f.write(f"Keywords: {clusters[cid]['keywords']}\n")
            f.write("### Proposal\n")
            f.write(proposals[cid])
            f.write("\n---\n")


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output_dir", default="out")
    parser.add_argument("--gemini_key", required=True)

    args = parser.parse_args()

    process(args.input, args.output_dir, args.gemini_key)
