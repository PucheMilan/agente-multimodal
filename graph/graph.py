"""Definicion del grafo de estados."""

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from graph.edges import select_workflow, should_summarize_conversation
from graph.nodes import (
    audio_node,
    conversation_node,
    image_node,
    memory_extraction_node,
    memory_injection_node,
    router_node,
    summarize_conversation_node,
)
from graph.state import AgentState
from tools import logger


@lru_cache(maxsize=1)
def create_workflow_graph() -> StateGraph:
    """Construye el grafo del agente.

        START -> memory_extraction -> router -> memory_injection
              -> {conversation | image | audio} -> (resumir?) -> END

    Se cachea porque el grafo es el mismo siempre; lo que cambia en cada
    ejecucion es el estado, no la topologia.
    """
    g = StateGraph(AgentState)

    g.add_node("memory_extraction_node", memory_extraction_node)
    g.add_node("router_node", router_node)
    g.add_node("memory_injection_node", memory_injection_node)
    g.add_node("conversation_node", conversation_node)
    g.add_node("image_node", image_node)
    g.add_node("audio_node", audio_node)
    g.add_node("summarize_conversation_node", summarize_conversation_node)

    g.add_edge(START, "memory_extraction_node")
    g.add_edge("memory_extraction_node", "router_node")
    g.add_edge("router_node", "memory_injection_node")
    g.add_conditional_edges("memory_injection_node", select_workflow)
    for nodo in ("conversation_node", "image_node", "audio_node"):
        g.add_conditional_edges(nodo, should_summarize_conversation)
    g.add_edge("summarize_conversation_node", END)

    logger.info("Grafo creado con %d nodos", len(g.nodes))
    return g


graph_builder = create_workflow_graph()
