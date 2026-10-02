"""toolagent — a small, transparent tool-using agent built on the Claude Messages API."""

from toolagent.agent import Agent, AgentResult, Tracer
from toolagent.builtin import Workspace, calculator, current_time, default_tools
from toolagent.llm import ClaudeClient, ModelTurn, ScriptedLLM
from toolagent.tools import Tool, ToolError, ToolRegistry, tool

__all__ = [
    "Agent",
    "AgentResult",
    "ClaudeClient",
    "ModelTurn",
    "ScriptedLLM",
    "Tool",
    "ToolError",
    "ToolRegistry",
    "Tracer",
    "Workspace",
    "calculator",
    "current_time",
    "default_tools",
    "tool",
]

__version__ = "0.1.0"
