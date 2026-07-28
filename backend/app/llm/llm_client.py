# llm/llm_client.py

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq


class LLMWithFallback:
    """Wraps two LLMs: Gemini as primary, Groq as fallback if primary fails."""

    def __init__(self):
        self.primary = ChatGoogleGenerativeAI(
            model="gemini-3.1-flash-lite",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0
        )
        self.fallback = ChatGroq(
            model="llama-3.3-70b-versatile",
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=0
        )

    def invoke(self, messages):
        try:
            return self.primary.invoke(messages)
        except Exception as e:
            print(f"[LLM] Primary (Gemini) failed: {e}. Falling back to Groq.")
            return self.fallback.invoke(messages)

    def with_structured_output(self, schema):
        return _StructuredLLMWrapper(self.primary, self.fallback, schema)

    def bind_tools(self, tools):
        """Binds tools to both primary and fallback models, returns a wrapper
        that tries primary first, falls back to secondary on failure."""
        primary_with_tools = self.primary.bind_tools(tools)
        fallback_with_tools = self.fallback.bind_tools(tools)
        return _ToolBoundLLMWrapper(primary_with_tools, fallback_with_tools)


class _StructuredLLMWrapper:
    def __init__(self, primary, fallback, schema):
        self.primary_structured = primary.with_structured_output(schema)
        self.fallback_structured = fallback.with_structured_output(schema)

    def invoke(self, prompt):
        try:
            return self.primary_structured.invoke(prompt)
        except Exception as e:
            print(f"[LLM Structured] Primary failed: {e}. Falling back to Groq.")
            return self.fallback_structured.invoke(prompt)


class _ToolBoundLLMWrapper:
    """Wraps a tool-bound LLM with fallback support."""

    def __init__(self, primary_with_tools, fallback_with_tools):
        self.primary_with_tools = primary_with_tools
        self.fallback_with_tools = fallback_with_tools
        self.last_used = "primary"  # track which model handled the last call

    def invoke(self, messages):
        try:
            result = self.primary_with_tools.invoke(messages)
            self.last_used = "primary"
            return result
        except Exception as e:
            print(f"[LLM Tools] Primary failed: {e}. Falling back to Groq.")
            self.last_used = "fallback"
            return self.fallback_with_tools.invoke(messages)


# Singleton — reused across the whole app
_llm_instance = None
_llm_with_tools = None  # reset when prompt changes

def get_llm() -> LLMWithFallback:
    global _llm_instance
    if _llm_instance is None:
        _llm_instance = LLMWithFallback()
    return _llm_instance