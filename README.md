# CatEye

CatEye is a deterministic evidence extraction tool for researchers. It maps your PDF library against your paper's argument structure, and shows you exactly which passages from which papers support each section.

**Zero generation. Zero hallucination. Every word comes from your documents.**

## What makes CatEye different

| Standard AI tools | CatEye |
|---|---|
| Generate text from training data | Shows exact original text from YOUR papers |
| Black box — you get paraphrased passages  | Every result links to the exact page |
| Risk of hallucinated citations | Impossible to hallucinate — text comes directly from PDFs |
| Write for you | Searches for you |

## How it works

CatEye uses a two-stage neural retrieval system that operates deterministically on your local files:

1. **Stage 1 — Bi-encoder retrieval:** Vectorises your research questions and finds the top candidate passages across all your PDFs in milliseconds.
2. **Stage 2 — Cross-encoder reranking:** Deep attention scoring ranks candidates by true semantic relevance, surfacing the best matches with confidence scores.
3. **The output:** Exact quoted passages with source file, page number, and confidence score. Nothing is generated, paraphrased, or summarised.

## Features

- **Live Search** — Type any question, get instant results from your library
- **Skeleton Map** — Define your paper structure, map your entire library against it
- **Coverage Analysis** — See which papers contribute most and where your evidence is thin
- **Gap Detection** — Identifies skeleton nodes your library doesn't cover well
- **Confidence Scores** — Know immediately if a match is strong or weak
- **Duplicate Detection** — Flags when the same passage appears under multiple sections
- **Keyword Highlighting** — Query terms highlighted in results for quick scanning
- **Upload or Local** — Upload PDFs directly or point to a local folder

## Try it

Visit the live demo: [cat-eye.streamlit.app](https://cat-eye.streamlit.app)

Or run locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Who is this for

- PhD students writing dissertations
- Researchers writing literature reviews
- Academics who want AI to search for them, not write for them
- Anyone who values academic integrity over convenience

## Philosophy

Every AI writing tool asks: "How can we generate text faster?"

CatEye asks a different question: "How can we find the right evidence faster?"

The most intellectually demanding part of academic writing, connecting ideas, building arguments, developing theory  is already done by the time you sit down to write. What remains is the tedious retrieval: finding which paper said what, on which page, to support the point you already know you want to make.

CatEye automates the tedious part without touching the intellectual part.

---

Built by [Khatia Nadaraia](https://github.com/Aliceka9)
