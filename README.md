# Academic Skeleton Miner: Unbiased Literature Extraction Engine

> *"Academics don't need an AI to write for them. They need an AI to read for them."*

## The Philosophy Behind the Project

Imagine you are writing a research paper. First, you develop a **"paper skeleton"** that covers all the topics and subtopics you need to discuss section by section. Naturally, before reaching this step, you have already done the reading and hold an entire conceptual map in your head. However, that map is highly generic—you see the broad connections and the big picture, but you cannot remember every granular detail, nor do you need to at that stage.

But then comes the actual writing. You have to fill those empty spaces below your skeleton subheadings with theory, empirical evidence, and specific citations.

Now, imagine you also have a folder packed with over 100 PDFs—research papers, literature reviews, and book chapters. At this point, you have completely lost the motivation to dive back into those files and grind out the prose. Why? Because the most exciting, curious part, connecting the dots in your mind is already done. What remains feels like a tedious formality required by academic writing.

### Why Standard AI Fails
Standard AI tools fail spectacularly at this step. They create a black box, generating text out of thin air. To me, that is a complete disaster. A total disaster for academic integrity.

It doesn't matter whether the traditional research paper format survives the AI "revolution" or not. The reality is that we have the technology to make this process less boring. **I built this modest, but useful tool to do exactly that.**

You map out your paper skeleton, feed those subtopics into this script as network nodes, and the pipeline extracts the exact information you need—**fully quoted, completely unmodified, and entirely free of AI hallucinations.**

---

## How It Works (The 2-Stage Forensic Search)

Instead of passing your thoughts to an LLM to synthesize data blindly, this system operates deterministically on your local files:

1. **Stage 1 (Bi-Encoder Dense Retrieval):** Uses `all-MiniLM-L6-v2` to vectorize your outline subtopics and pull the top N candidate paragraphs across thousands of PDF pages in milliseconds.
2. **Stage 2 (Cross-Encoder Attention Re-Ranking):** Uses a deep `ms-marco-MiniLM-L-6-v2` cross-attention map to score each candidate passage against your query, surfacing the best matches with confidence scores.
3. **The Deliverable:** Outputs a clean, interactive HTML dashboard with collapsible sections, keyword highlighting, search filtering, page citations, confidence scores, and duplicate detection.

---

## Features

- **Page-level citation** — Every extracted passage shows the exact PDF page number for direct lookup.
- **Sentence-aware chunking** — Text is split at sentence boundaries, preserving coherent passages instead of cutting mid-sentence.
- **Embedding cache** — Document embeddings are cached locally. Change your subtopics without re-encoding your entire PDF library.
- **Confidence scores** — Cross-encoder scores displayed per result so you can immediately tell strong matches from weak ones.
- **Duplicate detection** — Flags when the same passage surfaces under multiple skeleton nodes.
- **Keyword highlighting** — Subtopic terms are highlighted in the extracted text for quick scanning.
- **Collapsible sections** — Click any skeleton node header to collapse/expand its results.
- **Live search** — Filter all evidence blocks by keyword in real time.
- **CLI interface** — All parameters configurable via command-line arguments.
- **External subtopics** — Load your skeleton from a JSON or YAML file.

---

## Quick Start

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Basic usage (uses default folder and built-in subtopics):
```bash
python miner.py
```

3. Full usage with all options:
```bash
python miner.py --folder "path/to/pdfs" --subtopics subtopics.json --output results.html --top-k 5 --candidates 30
```

---

## CLI Options

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--folder` | `-f` | Built-in path | Directory containing your PDF files |
| `--output` | `-o` | `literature_review_index.html` | Output HTML file |
| `--subtopics` | `-s` | Built-in list | Path to JSON/YAML file with subtopics |
| `--top-k` | `-k` | 3 | Number of top matches per skeleton node |
| `--candidates` | `-c` | 20 | Bi-encoder candidates passed to cross-encoder |
| `--no-cache` | — | False | Force re-encoding (ignore cached embeddings) |

---

## Subtopics File Format

**JSON:**
```json
{
    "subtopics": [
        "HPA axis calibration cortisol damage to hippocampus",
        "Allostatic load childhood poverty stress mediating pathway"
    ]
}
```

**YAML:**
```yaml
subtopics:
  - "HPA axis calibration cortisol damage to hippocampus"
  - "Allostatic load childhood poverty stress mediating pathway"
```

Or simply a flat list in either format.

---

## Interpreting the Output

- **Score badge (blue):** High confidence match (score > 3.0)
- **Score badge (yellow):** Moderate match (score 0–3.0) — read to verify relevance
- **Score badge (red):** Low confidence (score < 0) — likely noise, included only when top-k demands it
- **"ALSO MATCHED ABOVE" badge:** This passage already appeared under a previous node — consider whether you're citing it twice

---

## Requirements

- Python 3.8+
- ~500MB disk for model downloads on first run (cached afterwards)
- Works fully offline after initial model download
