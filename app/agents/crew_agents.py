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
            goal='Analyser et reformuler les questions pour une meilleure compréhension',
            backstory="""Vous êtes un expert en analyse et reformulation de questions.
            Votre rôle est de comprendre l'intention derrière chaque question et de la reformuler
            de manière claire et précise. Vous devez toujours fournir une question reformulée,
            même si la question originale est déjà claire.""",
            tools=[],
            allow_delegation=False,
            verbose=True
        )

    @staticmethod
    def create_ai_analyst(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_AI_ANALYST_CONFIG, agent_params)
        return Agent(
            role='AI Analyst',
            goal='Générer des réponses précises et informatives',
            backstory="""Vous êtes un expert en analyse et en génération de réponses.
            Votre rôle est de fournir des réponses détaillées, précises et factuelles aux questions posées.
            Vous devez toujours structurer vos réponses de manière claire et concise, en vous concentrant
            sur les informations les plus pertinentes.""",
            tools=[],
            allow_delegation=False,
            verbose=True
        )

    @staticmethod
    def create_quality_controller(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_QUALITY_CONTROLLER_CONFIG, agent_params)
        return Agent(
            role='Quality Controller',
            goal='Évaluer la qualité des réponses et fournir un score numérique',
            backstory="""Vous êtes un expert en contrôle qualité.
            Votre rôle est d'évaluer la qualité et la pertinence des réponses.
            Vous devez fournir un score numérique entre 0 et 1, où:
            - 1.0 représente une réponse parfaite
            - 0.0 représente une réponse totalement inadéquate
            
            IMPORTANT: Votre réponse doit TOUJOURS être au format exact:
            Final Answer: Score: X.XX
            
            Exemple: Final Answer: Score: 0.95""",
            tools=[],
            allow_delegation=False,
            verbose=True
        )

    @staticmethod
    def create_general_manager(agent_params: Optional[Dict] = None) -> Agent:
        config = merge_agent_configs(DEFAULT_GENERAL_MANAGER_CONFIG, agent_params)
        return Agent(
            role='General Manager',
            goal='Valider et finaliser les réponses',
            backstory="""Vous êtes le manager général responsable de la validation finale des réponses.
            Votre rôle est de vous assurer que les réponses sont de haute qualité et appropriées.
            
            IMPORTANT: Votre réponse doit TOUJOURS être au format exact:
            Final Answer: validé|[RÉPONSE]
            ou
            Final Answer: rejeté|[RAISON]
            
            Exemple de validation:
            Final Answer: validé|La ville de Grasse est la capitale mondiale du parfum.
            
            Exemple de rejet:
            Final Answer: rejeté|La réponse est incomplète.""",
            tools=[],
            allow_delegation=False,
            verbose=True
        ) 