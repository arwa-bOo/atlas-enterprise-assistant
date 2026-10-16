"""Typed tools for the Atlas Enterprise Assistant."""

from decimal import Decimal, InvalidOperation
from typing import Any

from pydantic import BaseModel, Field

from src.retriever import retrieve
from src.settings import DOMAINS, TOP_K


class PolicyLookupInput(BaseModel):
    """Input for looking up a company policy by ID or topic."""

    query: str = Field(
        min_length=1,
        max_length=500,
        description="A company policy ID, topic, or question.",
    )


class ReimbursementInput(BaseModel):
    """Input for calculating reimbursement totals."""

    items: list[dict[str, Any]] = Field(
        min_length=1,
        description=(
            "Items containing a numeric amount and optionally quantity, "
            "description, and currency."
        ),
    )


class ListLeaveTypesInput(BaseModel):
    """Input for retrieving available leave types."""


def _format_chunks(chunks: list, limit: int = TOP_K) -> str:
    """Format document chunks with their source metadata."""
    if not chunks:
        return "NOT_FOUND: No relevant company documents were found."

    unique_chunks = []
    seen_ids = set()

    sorted_chunks = sorted(
        chunks,
        key=lambda item: getattr(item, "score", 0.0),
        reverse=True,
    )

    for chunk in sorted_chunks:
        chunk_id = getattr(chunk, "chunk_id", None)

        if chunk_id and chunk_id in seen_ids:
            continue

        if chunk_id:
            seen_ids.add(chunk_id)

        unique_chunks.append(chunk)

        if len(unique_chunks) >= limit:
            break

    parts = []

    for chunk in unique_chunks:
        source = getattr(chunk, "source_filename", "Unknown source")
        domain = getattr(chunk, "domain", "unknown")
        content = getattr(chunk, "text", "")

        parts.append(f"Source: {source}\n" f"Domain: {domain}\n" f"Content:\n{content}")

    return "\n\n---\n\n".join(parts)


def policy_lookup(args: PolicyLookupInput) -> str:
    """Search company documents for a policy ID or topic."""
    query = args.query.strip()

    if not query:
        return "VALIDATION_ERROR: Provide a non-empty policy ID or topic."

    found = []
    seen_ids = set()

    try:
        for domain in DOMAINS:
            chunks = retrieve(
                query=query,
                domain=domain,
                k=TOP_K,
            )

            for chunk in chunks:
                chunk_id = getattr(chunk, "chunk_id", None)

                if chunk_id and chunk_id in seen_ids:
                    continue

                if chunk_id:
                    seen_ids.add(chunk_id)

                found.append(chunk)

    except FileNotFoundError:
        return (
            "RETRIEVAL_ERROR: The document index is missing. "
            "Run `python -m src.ingest` to build it first."
        )

    except Exception as exc:
        return (
            "RETRIEVAL_ERROR: Policy search failed. "
            f"Error type: {type(exc).__name__}."
        )

    if not found:
        return (
            "NOT_FOUND: No relevant policy was found in the indexed "
            "company documents. Do not invent a policy."
        )

    found.sort(
        key=lambda item: getattr(item, "score", 0.0),
        reverse=True,
    )

    return f"Policy search results for '{query}':\n\n" f"{_format_chunks(found)}"


def reimbursement_calculator(args: ReimbursementInput) -> str:
    """Calculate a reimbursement total from validated line items.

    Each item requires an amount. Quantity defaults to 1.
    This tool calculates totals but does not determine policy eligibility.
    """
    if not args.items:
        return "VALIDATION_ERROR: Provide at least one reimbursement item."

    validated_items = []
    currencies = set()

    for index, item in enumerate(args.items, start=1):
        if not isinstance(item, dict):
            return f"VALIDATION_ERROR: Item {index} must be an object."

        if "amount" not in item:
            return (
                f"VALIDATION_ERROR: Item {index} is missing 'amount'. "
                "Provide a numeric amount and retry once."
            )

        amount_value = item["amount"]
        quantity_value = item.get("quantity", 1)

        allowed_types = (int, float, str, Decimal)

        if isinstance(amount_value, bool) or not isinstance(
            amount_value, allowed_types
        ):
            return f"VALIDATION_ERROR: Item {index} amount must be numeric."

        if isinstance(quantity_value, bool) or not isinstance(
            quantity_value, allowed_types
        ):
            return f"VALIDATION_ERROR: Item {index} quantity must be numeric."

        try:
            amount = Decimal(str(amount_value))
            quantity = Decimal(str(quantity_value))

        except (InvalidOperation, ValueError):
            return f"VALIDATION_ERROR: Item {index} amount or quantity " "is invalid."

        if not amount.is_finite() or amount < 0:
            return (
                f"VALIDATION_ERROR: Item {index} amount must be finite "
                "and non-negative."
            )

        if not quantity.is_finite() or quantity <= 0:
            return (
                f"VALIDATION_ERROR: Item {index} quantity must be finite "
                "and greater than zero."
            )

        line_total = amount * quantity

        if not line_total.is_finite():
            return (
                f"VALIDATION_ERROR: Item {index} total is outside "
                "the supported numeric range."
            )

        currency_value = item.get("currency")

        if currency_value is not None:
            if not isinstance(currency_value, str) or not currency_value.strip():
                return (
                    f"VALIDATION_ERROR: Item {index} currency must be "
                    "a non-empty string."
                )

            currencies.add(currency_value.strip().upper())

        validated_items.append(
            {
                "description": str(item.get("description") or f"Item {index}"),
                "amount": amount,
                "quantity": quantity,
                "line_total": line_total,
            }
        )

    if len(currencies) > 1:
        return (
            "VALIDATION_ERROR: Items use different currencies. "
            "Convert all items to one currency before calculating "
            "a combined total."
        )

    total = sum(
        (item["line_total"] for item in validated_items),
        Decimal(0),
    )

    if not total.is_finite():
        return "VALIDATION_ERROR: The total is outside the supported " "numeric range."

    currency_label = next(iter(currencies)) if currencies else "currency unspecified"

    lines = ["Reimbursement calculation:"]

    for item in validated_items:
        lines.append(
            f"- {item['description']}: "
            f"{item['amount']} x {item['quantity']} "
            f"= {item['line_total']}"
        )

    lines.append(f"Total: {total} {currency_label}")
    lines.append(
        "Note: This tool calculates totals only; "
        "policy eligibility was not evaluated."
    )

    return "\n".join(lines)


def list_leave_types(args: ListLeaveTypesInput) -> str:
    """Retrieve leave-type information from indexed HR documents."""
    query = (
        "employee leave types categories annual sick parental personal "
        "bereavement and other leave policies"
    )

    try:
        chunks = retrieve(
            query=query,
            domain="hr",
            k=TOP_K,
        )

    except FileNotFoundError:
        return (
            "RETRIEVAL_ERROR: The document index is missing. "
            "Run `python -m src.ingest` to build it first."
        )

    except Exception as exc:
        return (
            "RETRIEVAL_ERROR: Leave-type search failed. "
            f"Error type: {type(exc).__name__}."
        )

    if not chunks:
        return (
            "NOT_FOUND: No leave-type information was found in the "
            "indexed HR documents. Do not invent leave categories."
        )

    return (
        "Leave-type information from company HR documents:\n\n"
        f"{_format_chunks(chunks)}"
    )


TOOLS = {
    "policy_lookup": (PolicyLookupInput, policy_lookup),
    "reimbursement_calculator": (
        ReimbursementInput,
        reimbursement_calculator,
    ),
    "list_leave_types": (
        ListLeaveTypesInput,
        list_leave_types,
    ),
}
