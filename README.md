# WheelWise
Local document-grounded tire knowledge assistant for Jiale and Thew's Generative AI course project.

## Status
Initial implementation. Syntax checked; dependencies could not be installed in the authoring environment because package-network access was unavailable. End-to-end Ollama testing is REQUIRED before submission. No PDFs have been indexed yet.

## Setup (Mac)
Install Python 3.10+ and Ollama from their official websites. Start Ollama, then open Terminal in this folder:

```sh
ollama pull qwen3:1.7b
ollama pull nomic-embed-text
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python app.py
```
Open http://127.0.0.1:5000 . If port 5000 is occupied, change the port at the bottom of app.py (for example 5001).

Windows: use `py -m venv .venv`, then `.venv\Scripts\activate`, followed by the same pip and app commands.

## Use
Download four PDFs using SOURCES.md. Upload them one at a time. Wait for each Ready message. Ask questions, inspect retrieved excerpts and open source links. Remove files to remove their indexed passages.

## Architecture
Flask extracts PDF pages, DOCX paragraphs or UTF-8 text; overlapping 1,200-character chunks are embedded by nomic-embed-text. SQLite stores documents, passages and vectors locally. A question retrieves five passages by cosine similarity and passes them to qwen3:1.7b with recent conversation history. Answers are prompted to cite excerpt numbers; the interface independently lists the actual retrieved sources.

## Limitations
No OCR; use text-readable PDFs. Table extraction can lose layout: verify numeric fitment tables manually. Retrieval always returns the five closest passages, which may not answer the question. Generated citations and answers require checking. The model is prompted to acknowledge missing evidence, but this is not a guarantee. Source links download the originals. PDF page numbers refer to file pages; DOCX/text use section 1. Local demo, no public deployment or authentication. Stored documents are not encrypted. No automated vehicle fitment approval.

## GitHub
Create a PUBLIC repository wheelwise-ai. Upload app.py, templates/index.html, requirements.txt, README.md, SOURCES.md, TESTING.md and .gitignore, preserving folders. Do NOT upload data/, .venv/, PDFs or personal information. The source ZIP contains code only. Both members submit the same repository URL individually on MCV.
