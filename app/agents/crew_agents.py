from crewai import Agent
from langchain.tools import Tool
from typing import Dict, Optional
from app.core.agent_config import (
    DEFAULT_PROMPT_MANAGER_CONFIG,
    DEFAULT_AI_ANALYST_CONFIG,
    DEFAULT_QUALITY_CONTROLLER_CONFIG,
    DEFAULT_GENERAL_MANAGER_CONFIG,
    merge_agent_configs,
    DEFAULT_MODEL
)

class AgentFactory:
    @staticmethod
    def create_prompt_manager(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_PROMPT_MANAGER_CONFIG, agent_params)
        return Agent(
            role='Prompt Manager',
            goal='Reformuler et optimiser les questions des utilisateurs pour une meilleure clarté',
            backstory=config["context"],
            verbose=config.get("verbose", True),
            allow_delegation=False,
            temperature=config.get("temperature", 0.7),
            model=DEFAULT_MODEL,
            max_iterations=config.get("max_iterations", 3),
            **{k: v for k, v in config.items() if k not in ["context", "verbose", "temperature", "model", "max_iterations"]}
        )

    @staticmethod
    def create_ai_analyst(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_AI_ANALYST_CONFIG, agent_params)
        return Agent(
            role='AI Analyst',
            goal='Générer des réponses précises et pertinentes aux questions',
            backstory=config["context"],
            verbose=config.get("verbose", True),
            allow_delegation=False,
            temperature=config.get("temperature", 0.5),
            model=DEFAULT_MODEL,
            max_iterations=config.get("max_iterations", 3),
            **{k: v for k, v in config.items() if k not in ["context", "verbose", "temperature", "model", "max_iterations"]}
        )

    @staticmethod
    def create_quality_controller(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_QUALITY_CONTROLLER_CONFIG, agent_params)
        return Agent(
            role='Quality Controller',
            goal='Évaluer la qualité des réponses et fournir un score numérique entre 0 et 1',
            backstory="""Expert en contrôle qualité chargé d'évaluer la pertinence et l'exactitude des réponses.
            Vous devez toujours fournir un score numérique entre 0 et 1, où 1 représente une réponse parfaite.
            Format de réponse requis : "Score: [0-1]". Exemple : "Score: 0.95" """,
            verbose=config.get("verbose", True),
            allow_delegation=False,
            temperature=config.get("temperature", 0.3),
            model=DEFAULT_MODEL,
            max_iterations=config.get("max_iterations", 3),
            **{k: v for k, v in config.items() if k not in ["context", "verbose", "temperature", "model", "max_iterations"]}
        )

    @staticmethod
    def create_general_manager(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_GENERAL_MANAGER_CONFIG, agent_params)
        return Agent(
            role='General Manager',
            goal='Superviser le processus et valider les réponses finales',
            backstory=config["context"],
            verbose=config.get("verbose", True),
            allow_delegation=True,
            temperature=config.get("temperature", 0.4),
            model=DEFAULT_MODEL,
            max_iterations=config.get("max_iterations", 3),
            **{k: v for k, v in config.items() if k not in ["context", "verbose", "temperature", "model", "max_iterations"]}
        ) 