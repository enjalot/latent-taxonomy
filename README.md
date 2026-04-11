# Latent Taxonomy

Interactive exploration of Sparse Autoencoder (SAE) features trained on text embedding models.

**[Live Demo](https://enjalot.github.io/latent-taxonomy/)** | **[Methodology](https://enjalot.github.io/latent-taxonomy/articles/about)**

## Related Repositories

- [latent-data-modal](https://github.com/enjalot/latent-data-modal): data preparation pipeline (chunking, embedding, SAE feature extraction on Modal)
- [latent-sae](https://github.com/enjalot/latent-sae): sparse autoencoder training (TopK, Gated, JumpReLU, LISTA)
- [latent-scope](https://github.com/enjalot/latent-scope): dataset visualization

## Interpretability Pipeline

The `pipeline/` directory contains a scriptable pipeline for SAE feature interpretation:

```
pipeline/
  label.py           # Phase 3: LLM-based feature labeling (Ollama, OpenAI, Anthropic)
  llm.py             # Configurable async LLM client
  tokens.py          # TF-IDF top token computation (no model inference needed)
  triage.py          # SAE model loading (HuggingFace Hub, local, Modal)
  label_compare.py   # Compare label sets across LLMs (generates HTML report)
  local_reduce.py    # Local top-10 reduce when Modal billing is exhausted
  benchmark_sae.py   # CPU vs GPU encode throughput benchmarking
  config.py          # Shared pipeline configuration
  review/            # FastAPI review apps for inspecting results
```

### Quick Start

```bash
# Install dependencies (or use an existing venv with torch + latentsae)
pip install -r pipeline/requirements.txt

# Label features with a local model (Ollama)
ollama pull qwen2.5:7b
python -m pipeline.label \
  --samples pipeline/data/samples_combined.parquet \
  --model-source hub \
  --model-repo enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M \
  --model-name 128_4 \
  --provider ollama --model qwen2.5:7b \
  --output pipeline/output/labels_qwen25_7b.parquet

# Or with an API model
python -m pipeline.label \
  --provider openai --model gpt-5.4-mini \
  --output pipeline/output/labels_gpt54_mini.parquet \
  ...

# Compare label sets
python -m pipeline.label_compare \
  pipeline/output/labels_qwen25_7b.parquet \
  pipeline/output/labels_gpt54_mini.parquet \
  --output pipeline/output/comparison.html

# Review labels interactively
python -m pipeline.review.label_app \
  --labels pipeline/output/labels_qwen25_7b.parquet \
  --samples pipeline/data/samples_combined.parquet \
  --port 8503
```

### SAE Models

| Model | Embedding | Features | Active | HuggingFace |
|-------|-----------|----------|--------|-------------|
| 128_4 | MiniLM-L6-v2 | 1,536 | 402 | [enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M](https://huggingface.co/enjalot/sae-all-MiniLM-L6-v2-FineWeb-RedPajama-Pile-150M) |
| 128_8 | MiniLM-L6-v2 | 3,072 | 771 | same repo |
| 64_32 | nomic-embed-v1.5 | 24,576 | ~24K | [enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT](https://huggingface.co/enjalot/sae-nomic-text-v1.5-FineWeb-edu-100BT) |

## Web App

The `web/` directory contains a Next.js app for interactive feature exploration.

```bash
cd web
npm install
npm run dev
```
