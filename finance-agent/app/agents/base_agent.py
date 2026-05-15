"""
agents/base_agent.py — Abstract base class for all agents in the platform.

Every agent (Finance, Hypothesis, MarketResearch, Strategy, Legal, Reporting...)
inherits from BaseAgent. This ensures:
  - Consistent interface for the Orchestrator
  - Structured input/output (no raw strings between agents)
  - Uniform logging and error handling
  - Easy registration in the agent registry

HOW TO ADD A NEW AGENT:
  1. Create app/agents/your_agent.py
  2. Inherit from BaseAgent
  3. Implement process() and can_handle()
  4. Register in AGENT_REGISTRY below

COMMUNICATION PROTOCOL:
  Agents communicate through AgentMessage objects — structured JSON, never raw strings.
  The Orchestrator (future: app/orchestrator.py) reads can_handle() to route messages.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Optional


logger = logging.getLogger(__name__)


@dataclass
class AgentMessage:
    """
    Universal message format between agents.
    Replaces ad-hoc string passing between components.
    """
    # Routing
    sender_agent_id: str
    target_agent_id: Optional[str] = None    # None = broadcast / orchestrator decides
    session_id: Optional[str] = None

    # Content
    intent: str = "chat"                     # "chat" | "analyze" | "plan" | "research" | ...
    user_message: Optional[str] = None       # Raw user text (for chat path)
    structured_payload: Optional[dict] = None  # Structured data (agent-to-agent path)
    context: dict = field(default_factory=dict)  # Cross-agent context (entity_type, sector, etc.)

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    trace_id: Optional[str] = None           # For request tracing across agents


@dataclass
class AgentResponse:
    """
    Universal response format from agents.
    Always includes structured data + optional human-readable text.
    """
    agent_id: str
    session_id: Optional[str]
    intent: str

    # Text response (for chat/user-facing output)
    message: Optional[str] = None

    # Structured data (for agent-to-agent handoff)
    structured_output: Optional[dict] = None

    # Status
    success: bool = True
    error: Optional[str] = None

    # Metadata
    agent_mode: str = "responding"           # "gathering_info" | "analyzing" | "answering"
    metrics_calculated: Optional[dict] = None
    business_state: Optional[dict] = None
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseAgent(ABC):
    """
    Abstract base class for all platform agents.

    Enforces:
    - Unique agent_id per agent type
    - Semantic version tracking
    - Structured process() interface
    - can_handle() for orchestrator routing
    - Uniform logging
    """

    agent_id: str = "base"
    agent_version: str = "0.1.0"
    description: str = "Base agent"

    def __init__(self):
        self.logger = logging.getLogger(f"agent.{self.agent_id}")

    @abstractmethod
    def process(self, message: AgentMessage) -> AgentResponse:
        """
        Core agent logic. Must be implemented by every agent.
        Receives a structured AgentMessage, returns a structured AgentResponse.
        Never raises — catches internally and returns error AgentResponse.
        """
        ...

    def can_handle(self, intent: str) -> bool:
        """
        Return True if this agent can handle the given intent.
        Orchestrator calls this on each registered agent to route messages.
        Override in subclasses.
        """
        return False

    def _make_error_response(
        self, message: AgentMessage, error: str
    ) -> AgentResponse:
        self.logger.error(f"Agent {self.agent_id} error: {error}")
        return AgentResponse(
            agent_id=self.agent_id,
            session_id=message.session_id,
            intent=message.intent,
            success=False,
            error=error,
            message=f"Une erreur est survenue dans l'agent {self.agent_id}: {error}",
        )

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} id={self.agent_id} v{self.agent_version}>"


# ── Agent Registry ────────────────────────────────────────────────────────────
# Add new agents here. The Orchestrator imports this registry.

def build_agent_registry() -> dict[str, type[BaseAgent]]:
    """
    Returns the registry of available agent classes keyed by agent_id.
    Import lazily to avoid circular imports.

    HOW TO REGISTER A NEW AGENT:
      1. Create app/agents/your_agent.py inheriting from BaseAgent
      2. Uncomment its line below and import it
      3. Implement can_handle() so the Orchestrator routes to it correctly
    """
    registry: dict[str, type[BaseAgent]] = {}

    # Finance agent currently uses the functional pattern (handle_chat).
    # When migrated to BaseAgent subclass, register it here:
    # from app.agents.finance_agent_v2 import FinanceAgentV2
    # registry["finance"] = FinanceAgentV2

    # Future agents — uncomment as they are built:
    # from app.agents.hypothesis_agent import HypothesisAgent
    # registry["hypothesis"] = HypothesisAgent

    # from app.agents.market_research_agent import MarketResearchAgent
    # registry["market_research"] = MarketResearchAgent

    # from app.agents.strategy_agent import StrategyAgent
    # registry["strategy"] = StrategyAgent

    return registry