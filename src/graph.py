"""LangGraph workflow for the Atlas Enterprise Assistant."""

import re
from math import isfinite
from uuid import uuid4

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import StructuredTool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph

from src.memory import load_history, save_turn
from src.prompts import ANSWER_PROMPT
from src.retriever import retrieve
from src.router import route
from src.schemas import AgentState, ToolCall
from src.settings import (
    LLM_MODEL,
    MAX_STEPS,
    MAX_TOOL_CALLS,
    NOT_FOUND_MESSAGE,
    ROUTER_CONFIDENCE_THRESHOLD,
)
from src.tools import TOOLS

VALID_DOMAINS = {"hr", "it", "finance"}

TOOL_DESCRIPTIONS = {
    "policy_lookup": "Look up company policies by topic or policy ID.",
    "reimbursement_calculator": ("Calculate reimbursement totals from supplied items."),
    "list_leave_types": "List available employee leave types.",
}


def _get_model():
    """Create the configured Gemini chat model."""
    return ChatGoogleGenerativeAI(
        model=LLM_MODEL,
        temperature=0,
    )


def _build_tools():
    """Convert typed project tools into LangChain tools."""
    built_tools = []

    for name, definition in TOOLS.items():
        if isinstance(definition, tuple) and len(definition) == 2:
            input_schema, function = definition

            def make_wrapper(tool_function, schema):
                def wrapper(**kwargs):
                    """Validate inputs and execute a project tool."""
                    validated_input = schema(**kwargs)
                    return tool_function(validated_input)

                return wrapper

            tool_wrapper = make_wrapper(function, input_schema)

            built_tools.append(
                StructuredTool.from_function(
                    func=tool_wrapper,
                    name=name,
                    description=TOOL_DESCRIPTIONS.get(
                        name,
                        f"Execute the {name} tool.",
                    ),
                    args_schema=input_schema,
                )
            )

        elif hasattr(definition, "invoke") and hasattr(definition, "name"):
            built_tools.append(definition)

        else:
            raise TypeError(
                f"Unsupported definition for tool '{name}'. "
                "Expected (input_schema, function)."
            )

    return built_tools


def _content_to_text(content) -> str:
    """Convert model output into plain text."""
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and item.get("text"):
                parts.append(item["text"])

        return "\n".join(parts).strip()

    return str(content).strip() if content is not None else ""


def _chunk_value(chunk, field, default=""):
    """Read a field from a dictionary or a Pydantic model."""
    if isinstance(chunk, dict):
        return chunk.get(field, default)

    return getattr(chunk, field, default)


def _format_chunks(chunks) -> str:
    """Format retrieved chunks as document context."""
    if not chunks:
        return "No relevant document context was retrieved."

    formatted = []

    for chunk in chunks:
        filename = _chunk_value(chunk, "source_filename", "Unknown source")
        domain = _chunk_value(chunk, "domain", "unknown")
        content = _chunk_value(chunk, "text", "")

        formatted.append(
            f"Source: {filename}\n" f"Domain: {domain}\n" f"Content:\n{content}"
        )

    return "\n\n---\n\n".join(formatted)


def _format_history(history) -> str:
    """Format previous conversation turns."""
    if not history:
        return "No previous conversation."

    lines = []

    for message in history:
        if not isinstance(message, dict):
            continue

        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, str) and content.strip():
            lines.append(f"{role.capitalize()}: {content}")

    return "\n".join(lines) or "No previous conversation."


def _is_tool_error(result: str) -> bool:
    """Check whether a tool returned an error or validation message."""
    error_prefixes = (
        "VALIDATION_ERROR:",
        "RETRIEVAL_ERROR:",
        "Tool execution failed:",
        "Unknown tool requested:",
    )
    return result.startswith(error_prefixes)


def build_graph():
    """Build and compile the enterprise assistant workflow."""
    if MAX_STEPS < 5:
        raise ValueError("MAX_STEPS must be at least 5 for the configured workflow.")

    if MAX_TOOL_CALLS < 0:
        raise ValueError("MAX_TOOL_CALLS cannot be negative.")

    model = _get_model()
    tools = _build_tools()
    tools_by_name = {tool.name: tool for tool in tools}

    model_with_tools = model.bind_tools(tools) if tools else model

    def load_memory_node(state: AgentState) -> dict:
        """Load conversation history for the current session."""
        try:
            history = load_history(state["session_id"])
        except Exception:
            history = []

        return {
            "history": history,
            "step_count": state.get("step_count", 0) + 1,
        }

    def route_node(state: AgentState) -> dict:
        """Classify the question and handle uncertain routing."""
        try:
            domain, confidence = route(
                question=state["question"],
                history=state.get("history", []),
            )
            confidence = float(confidence)

            valid_confidence = isfinite(confidence) and 0.0 <= confidence <= 1.0

            if (
                domain not in VALID_DOMAINS
                or not valid_confidence
                or confidence < ROUTER_CONFIDENCE_THRESHOLD
            ):
                domain = "unknown"

        except Exception:
            domain = "unknown"

        return {
            "domain": domain,
            "step_count": state.get("step_count", 0) + 1,
        }

    def retrieve_node(state: AgentState) -> dict:
        """Retrieve relevant chunks for the selected domain."""
        try:
            chunks = retrieve(
                query=state["question"],
                domain=state["domain"],
            )
        except Exception:
            chunks = []

        return {
            "retrieved_chunks": chunks or [],
            "step_count": state.get("step_count", 0) + 1,
        }

    def agent_node(state: AgentState) -> dict:
        """Generate a response and execute tools within configured limits."""
        question = state["question"]
        history = state.get("history", [])
        chunks = state.get("retrieved_chunks", [])

        prompt = ANSWER_PROMPT.format(
            history=_format_history(history),
            context=_format_chunks(chunks),
            tool_results=("Tool results will be provided in subsequent messages."),
            question=question,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=question),
        ]

        recorded_tool_calls = list(state.get("tool_calls", []))
        failed_attempts = {}

        tool_call_count = 0
        steps = int(state.get("step_count", 0))
        final_answer = ""

        # Reserve two steps for answer processing and memory saving.
        while steps < MAX_STEPS - 2:
            try:
                if not tools or tool_call_count >= MAX_TOOL_CALLS:
                    response = model.invoke(messages)
                else:
                    response = model_with_tools.invoke(messages)

                steps += 1

            except Exception as exc:
                # Keep technical details in server logs, not in user output.
                print("Language model request failed: " f"{type(exc).__name__}: {exc}")
                final_answer = (
                    "I could not complete the request because of a "
                    "language model error. Please try again."
                )
                break

            requested_calls = getattr(response, "tool_calls", []) or []

            if not requested_calls:
                final_answer = _content_to_text(response.content)
                break

            messages.append(response)

            for requested in requested_calls:
                tool_name = requested.get("name", "")
                tool_args = requested.get("args", {})
                tool_id = requested.get("id") or str(uuid4())

                if not isinstance(tool_args, dict):
                    tool_args = {"raw_input": str(tool_args)}

                if tool_call_count >= MAX_TOOL_CALLS:
                    result_text = (
                        "Tool-call limit reached. No additional tool "
                        "was executed. Answer using existing information."
                    )

                else:
                    tool_call_count += 1

                    if tool_name not in tools_by_name:
                        result_text = (
                            f"Unknown tool requested: {tool_name}. "
                            "Choose an available tool."
                        )

                    elif failed_attempts.get(tool_name, 0) >= 2:
                        result_text = (
                            "This tool has already failed twice. "
                            "Do not retry it again."
                        )

                    else:
                        try:
                            result = tools_by_name[tool_name].invoke(tool_args)
                            result_text = str(result)

                        except Exception as exc:
                            print(
                                f"Tool '{tool_name}' failed: "
                                f"{type(exc).__name__}: {exc}"
                            )
                            result_text = (
                                "Tool execution failed. "
                                "Correct the input and retry once if possible."
                            )

                    if _is_tool_error(result_text):
                        failed_attempts[tool_name] = (
                            failed_attempts.get(tool_name, 0) + 1
                        )

                    recorded_tool_calls.append(
                        ToolCall(
                            name=tool_name or "unknown",
                            input=tool_args,
                            output=result_text,
                        )
                    )

                messages.append(
                    ToolMessage(
                        content=result_text,
                        tool_call_id=tool_id,
                    )
                )

        if not final_answer:
            final_answer = (
                "I could not complete the request within the configured "
                "step limit. Please try a more specific question."
            )

        return {
            "answer": final_answer,
            "tool_calls": recorded_tool_calls,
            "step_count": steps,
        }

    def answer_node(state: AgentState) -> dict:
        """Finalize the answer and attach verified document sources."""
        if state.get("domain") == "unknown":
            return {
                "answer": (
                    "I could not confidently classify this question. "
                    "Please clarify whether it concerns HR, IT, or Finance."
                ),
                "sources": [],
                "step_count": state.get("step_count", 0) + 1,
            }

        chunks = state.get("retrieved_chunks", [])
        tool_calls = state.get("tool_calls", [])
        answer = (state.get("answer") or "").strip()

        # Collect source filenames from retrieved document chunks.
        source_names = [_chunk_value(chunk, "source_filename", "") for chunk in chunks]

        # Include source filenames explicitly returned by policy lookup.
        for call in tool_calls:
            if call.name != "policy_lookup" or _is_tool_error(call.output):
                continue

            for match in re.finditer(
                r"^Source:\s*(.+)$",
                call.output,
                flags=re.MULTILINE,
            ):
                source_names.append(match.group(1).strip())

        sources = list(dict.fromkeys(source for source in source_names if source))

        # A tool call alone is not proof that useful information was found.
        successful_tools = [
            call
            for call in tool_calls
            if (
                not _is_tool_error(call.output)
                and call.output.strip()
                and not call.output.startswith("This tool has already failed twice.")
                and "No relevant information found" not in call.output
                and "No results found" not in call.output
            )
        ]

        if not chunks and not successful_tools:
            answer = NOT_FOUND_MESSAGE
            sources = []

        else:
            # Remove any Sources section generated by the language model.
            answer = re.split(
                r"(?im)^\s*(?:#{1,6}\s*)?Sources\s*:?.*$",
                answer,
                maxsplit=1,
            )[0].strip()

            if not answer:
                answer = NOT_FOUND_MESSAGE

        # Always include the verified source list in the final answer.
        source_section = (
            "\n".join(f"- {source}" for source in sources)
            if sources
            else "- No document sources used."
        )

        answer = f"{answer}\n\nSources:\n{source_section}"

        return {
            "answer": answer,
            "sources": sources,
            "step_count": state.get("step_count", 0) + 1,
        }

    def save_memory_node(state: AgentState) -> dict:
        """Save the final question and answer."""
        try:
            save_turn(
                session_id=state["session_id"],
                question=state["question"],
                answer=state["answer"],
            )
            answer = state["answer"]

        except Exception as exc:
            print(
                "Conversation history could not be saved: "
                f"{type(exc).__name__}: {exc}"
            )
            answer = (
                state.get("answer", "")
                + "\n\nNote: Conversation history could not be saved."
            )

        return {
            "answer": answer,
            "step_count": state.get("step_count", 0) + 1,
        }

    def after_route(state: AgentState) -> str:
        """Select retrieval or uncertainty handling."""
        if state.get("domain") == "unknown":
            return "answer"

        return "retrieve"

    workflow = StateGraph(AgentState)

    workflow.add_node("load_memory", load_memory_node)
    workflow.add_node("route", route_node)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("agent", agent_node)
    workflow.add_node("answer", answer_node)
    workflow.add_node("save_memory", save_memory_node)

    workflow.add_edge(START, "load_memory")
    workflow.add_edge("load_memory", "route")

    workflow.add_conditional_edges(
        "route",
        after_route,
        {
            "answer": "answer",
            "retrieve": "retrieve",
        },
    )

    workflow.add_edge("retrieve", "agent")
    workflow.add_edge("agent", "answer")
    workflow.add_edge("answer", "save_memory")
    workflow.add_edge("save_memory", END)

    return workflow.compile()


def run_question(question: str, session_id: str) -> AgentState:
    """Run a question and return the final workflow state."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must be a non-empty string.")

    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("Session ID must be a non-empty string.")

    graph = build_graph()

    initial_state: AgentState = {
        "question": question.strip(),
        "session_id": session_id.strip(),
        "run_id": str(uuid4()),
        "domain": "",
        "history": [],
        "retrieved_chunks": [],
        "tool_calls": [],
        "answer": "",
        "sources": [],
        "step_count": 0,
    }

    return graph.invoke(initial_state)
