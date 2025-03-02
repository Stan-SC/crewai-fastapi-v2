from crewai import Agent
from langchain.tools import Tool
from typing import Dict, Optional

class AgentFactory:
    @staticmethod
    def create_prompt_manager(agent_params: Optional[Dict] = None) -> Agent:
        return Agent(
            role='Prompt Manager',
            goal='Reformuler et optimiser les questions des utilisateurs pour une meilleure clarté',
            backstory="""Expert en NLP et en reformulation de questions. 
            Votre rôle est d'analyser et d'améliorer la clarté des questions tout en préservant leur intention.""",
            verbose=True,
            allow_delegation=False,
            **agent_params if agent_params else {}
        )

    @staticmethod
    def create_ai_analyst(agent_params: Optional[Dict] = None) -> Agent:
        return Agent(
            role='AI Analyst',
            goal='Générer des réponses précises et pertinentes aux questions',
            backstory="""Analyste IA expérimenté spécialisé dans la génération de réponses 
            complètes et précises basées sur les questions optimisées.""",
            verbose=True,
            allow_delegation=False,
            **agent_params if agent_params else {}
        )

    @staticmethod
    def create_quality_controller(agent_params: Optional[Dict] = None) -> Agent:
        return Agent(
            role='Quality Controller',
            goal='Vérifier et valider la qualité des réponses générées',
            backstory="""Expert en contrôle qualité chargé d'évaluer la pertinence, 
            l'exactitude et la clarté des réponses générées.""",
            verbose=True,
            allow_delegation=False,
            **agent_params if agent_params else {}
        )

    @staticmethod
    def create_general_manager(agent_params: Optional[Dict] = None) -> Agent:
        return Agent(
            role='General Manager',
            goal='Superviser le processus et valider les réponses finales',
            backstory="""Manager expérimenté responsable de la supervision globale du processus 
            et de la validation finale des réponses avant leur transmission.""",
            verbose=True,
            allow_delegation=True,
            **agent_params if agent_params else {}
        ) 