# Acceptance checks — pending real execution
Record actual outcome and evidence for each check; do not mark passed without running it.

- Upload PDF, DOCX, TXT and MD; verify indexing and readable excerpts.
- Restart app; verify documents persist.
- Ask what load index means; compare answer and cited PDF page.
- Ask a follow-up; verify conversation context.
- Ask a question unsupported by the collection; check acknowledgment of missing evidence.
- Ask for a table value; manually compare with original PDF layout.
- Open each source link.
- Delete a document; confirm subsequent retrieval excludes it.
- Stop Ollama; confirm clear error and no false success.
- Upload an unsupported file or image-only PDF; confirm clear rejection.
- Test a fresh install following README.md.

Authoring checks: Python syntax passed. Dependency installation and real Ollama checks blocked in authoring environment; run on team laptop before claiming functional code.
