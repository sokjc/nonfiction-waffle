"""Shared error types and LLM invocation helper.

Every agent should use ``invoke_llm`` instead of calling ``chain.invoke()``
directly.  This ensures connection failures produce actionable error messages
instead of raw tracebacks.
"""

from __future__ import annotations


class LLMConnectionError(RuntimeError):
    """The LLM endpoint is unreachable or timed out."""


_CONNECTION_KEYWORDS = ("connect", "timeout", "refused", "unreachable", "reset", "closed")


def invoke_llm(chain, inputs: dict, *, endpoint_url: str) -> str:
    """Call *chain*.invoke() with clean error handling for connection failures.

    Args:
        chain: A LangChain runnable (prompt | llm | parser).
        inputs: The dict of template variables to pass.
        endpoint_url: The base URL shown in the error message.

    Returns:
        The string result from the chain.

    Raises:
        LLMConnectionError: If the endpoint is unreachable.
    """
    try:
        return chain.invoke(inputs)
    except Exception as e:
        msg = str(e).lower()
        if any(kw in msg for kw in _CONNECTION_KEYWORDS):
            raise LLMConnectionError(
                f"LLM endpoint unreachable at {endpoint_url}. "
                "Ensure your model server is running and the URL is correct."
            ) from e
        raise
