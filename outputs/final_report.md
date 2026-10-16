# Atlas Industries Enterprise Assistant: Final Report

Agentic internal-knowledge chatbot for HR, IT, and Finance questions, in English and Arabic. Repository: https://github.com/Mariam-abd-elhady/atlas-enterprise-assistant

## 1. Architecture

Three layers (diagram: `docs/architecture.png`):

1. **Ingestion and index.** 30 fictional documents (MD, PDF, DOCX; English, Arabic, bilingual) are loaded, cleaned, chunked, embedded, and stored in Chroma with `domain`, `source_filename`, and `language` metadata (139 chunks).
2. **LangGraph agent.** A typed graph: `load_memory -> route -> retrieve -> agent -> answer -> save_memory`. The state carries `run_id` and `session_id` through every node.
3. **Chat UI and observability.** A Chainlit UI shows the domain, tool calls, and the cited answer with the source passages in a side panel. Each turn is logged as one JSON line tagged with its `run_id`, without document text.

## 2. Stack choices and why

| Part | Choice | Why |
|---|---|---|
| LLM | Gemini (`langchain-google-genai`), default `gemini-3.8-flash` | Free tier, native tool calling, and it answered the Arabic gold questions correctly. The free-tier quota ran out during testing, so the model is set through the `LLM_MODEL` and `JUDGE_MODEL` environment variables. |
| Embeddings | `intfloat/multilingual-e5-small` | Multilingual (the corpus is 30% Arabic) and small enough for fast CPU ingestion. The trade-off is weaker ranking than a larger model. |
| Vector store | Chroma, persisted locally | Zero cost, metadata filtering by domain, and no server. |
| Orchestration | LangGraph | Explicit nodes, typed state, and bounded loops, which makes each step easy to inspect. |
| Memory | Own JSON store | Simple and inspectable; see section 3. |
| UI | Chainlit | Built-in steps for tool calls, side panels for sources, starter questions. |
| Validation | Pydantic | Typed tool inputs with automatic error messages. |
| Evaluation | DeepEval plus custom checks | Required metrics, plus routing accuracy and citation checks. |

## 3. LangGraph design and memory

- **Nodes.** `load_memory` reads prior turns; `route` picks the domain; `retrieve` fetches chunks; `agent` calls Gemini (with tools bound); `answer` finalizes the reply and the verified sources; `save_memory` stores the turn.
- **Typed state.** `AgentState` holds the question, ids, domain, history, retrieved chunks, tool calls, answer, sources, and a step counter.
- **Fallbacks.** If the router's confidence is below 0.6 the graph skips retrieval and asks the user to clarify. A model error returns a message instead of a stack trace. If nothing relevant is retrieved and no tool succeeded, the fixed message "I couldn't find this in the company documents." is returned.
- **Limits.** `MAX_STEPS = 10` and `MAX_TOOL_CALLS = 3`; the agent loop stops two steps early to leave room for the answer and memory nodes, and a tool that fails twice is not called again.
- **Memory mechanism.** `src/memory.py` keeps each session's user and assistant turns in `outputs/session_memory.json`, keyed by `session_id`, guarded by a lock and written atomically through a temporary file. The history is loaded into the state and into the prompt, so a follow-up such as "And what's the cap for a domestic trip?" resolves against the previous turn. A new chat gets a new session id (empty memory) and the UI clears the session when the chat ends.

## 4. RAG design

- **Ingestion.** Markdown, PDF, and DOCX (including DOCX tables) are cleaned, and each document gets a language label from its share of Arabic letters. Text is split with a recursive splitter (800 characters, 100 overlap).
- **Retrieval.** The question is embedded and searched inside the routed domain only. The retriever fetches 30 candidates (`TOP_K` x 5), then keeps the best 6 with at most 2 chunks per file so one document cannot fill every slot.
- **Answering.** The prompt tells the model to use only the supplied context and tool results, to answer in the user's language, and to say so when the answer is missing. The source list is rebuilt from the chunks and tool outputs actually used, so cited filenames are real.

## 5. Routing logic

The router scores each domain by keyword matches in English and Arabic lists (single words and phrases, with phrases weighted higher). Confidence is `0.60 + 0.05 x best score + 0.10 x margin over the second domain`. With no match it reuses the domain of the last user turn (confidence 0.65) or returns low confidence, and the graph then asks the user to clarify. This is cheap, fast, and needs no extra model call, but it only knows the words in its lists (see limitations).

## 6. Tools

Three tools, each with a Pydantic input schema and a typed function: `policy_lookup` (search policies by ID or topic), `reimbursement_calculator` (sum a list of items with decimal arithmetic), and `list_leave_types` (retrieve HR leave information). Bad input is caught and returned to the agent as a plain-text validation error so it can correct the call. Tool choice is left to the model, guided by short tool descriptions, with the cap of three calls per question. None of the 10 gold questions triggered a tool call, so the tools are not covered by the evaluation.

## 7. DeepEval results

`python -m tests.evaluate` runs the 10 gold questions through the real graph and writes `outputs/eval_report.json`. Judge and assistant were both `gemini-3.1-flash-lite` (the default model's quota was exhausted). Two full runs after the router fix (the second on the final repository commit) produced identical answers and scores because the temperature is 0.

| Metric | Result | Comment |
|---|---|---|
| Routing accuracy | **100%** (10 of 10; Arabic 3 of 3) | The first run, with English-only keyword lists, scored 40% and sent Arabic questions to `unknown`. The current lists overlap the gold questions, so this may be optimistic. |
| Citation recall | **95%** | One expected file missing: the international travel policy in EVAL-008. |
| Faithfulness | **0.93** | Every answer passed. Deductions come from an approval threshold that conflicts with the context (EVAL-009), a late-claim rule the retrieved chunks do not support (EVAL-007), and the filename being read as a policy ID (EVAL-002, EVAL-006). |
| Answer Relevancy | **0.93** | 9 of 10 passed. EVAL-007 scored 0.57 because it added documentation rules nobody asked for. |
| Hallucination (1 = none) | **1.00** | Answers do not contradict the retrieved context. This does not catch answers built on the wrong context. |
| Contextual Precision | **0.82** (7 of 10 pass) | Irrelevant chunks are often ranked first (for example VPN SLA for a password question). |
| Contextual Recall | **0.82** (8 of 10 pass) | EVAL-001 scores 0.0: the answer "15 days" is right but the retrieved chunks lack the accrual rate and probation detail. |

**A manual check against the reference answers shows the metrics overstate quality.** EVAL-008 states the international per diem as 500 EGP per day, but the reference is 75 USD. The answer is faithful to the domestic per diem chunk it received, which is why Faithfulness and Hallucination are both 1.0. EVAL-007 says hotel and taxi caps are not specified, but the policy gives 1,500 EGP per night and 800 EGP per leg. A simple key-figure check on the answers finds both failures (8 of 10 answers contain all key figures).

## 8. Known limitations

- **Retrieval is the main weakness.** The right passage is sometimes missing (hotel and taxi caps, international per diem), and `MAX_CHUNKS_PER_FILE = 2` may cut off later sections of long policies. A multi-query helper (`retrieve_multi`, question plus its translation) exists in `src/retriever.py` but is not connected to the graph, which could help English questions whose answer lives in an Arabic document.
- **The router is keyword based.** An unknown phrasing ends in a clarification request instead of an answer.
- **Evaluation scope.** Only 10 cases, the same model family as assistant and judge, no multi-turn or tool cases, and scores compare answers with retrieved context rather than ground truth.
- **Source lists in the answer text.** The answer node appends a verified "Sources" list to every answer, but its filter only removes an English "Sources" heading, so a model-written Arabic "المصادر" block can remain and Arabic answers may show sources twice. Judges also sometimes read a filename as an incorrect policy ID, which lowers Faithfulness and Relevancy for a few cases.
- **Logging.** Events are written by the UI layer (one per turn: run id, domain, tools, retrieved chunk ids and scores, sources, latency); the graph nodes do not log their own events.
- **Free-tier quota.** Gemini limits made full evaluation runs slow and forced a lighter model.

**Next steps:** connect multi-query retrieval, raise the per-file chunk limit and test larger chunks or a reranker, add an exact-match check for key figures to the evaluation, and extend the router with vocabulary taken from the corpus itself.
