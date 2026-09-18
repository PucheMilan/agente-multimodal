"""Las aristas condicionales: quien decide por donde sigue el grafo."""

from typing import Literal

from langgraph.graph import END

from graph.state import AgentState
from settings import settings
from tools import logger


def select_workflow(
    state: AgentState,
) -> Literal["conversation_node", "image_node", "audio_node"]:
    """Traduce lo que decidio el router al nodo que toca ejecutar."""
    return {
        "image": "image_node",
        "audio": "audio_node",
    }.get(state.get("workflow", "conversation"), "conversation_node")


def should_summarize_conversation(
    state: AgentState,
) -> Literal["summarize_conversation_node", "__end__"]:
    """Resume cuando la conversacion pasa del umbral configurado."""
    n = len(state["messages"])
    if n > settings.TOTAL_MESSAGES_SUMMARY_TRIGGER:
        logger.info("%d mensajes acumulados: toca resumir", n)
        return "summarize_conversation_node"
    return END
