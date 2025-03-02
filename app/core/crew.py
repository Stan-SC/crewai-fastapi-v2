from crewai import Crew, Task
from typing import Dict, Optional
from app.agents.crew_agents import AgentFactory
from loguru import logger
import re

class QuestionCrew:
    def __init__(self, agent_params: Optional[Dict] = None):
        self.agent_factory = AgentFactory()
        self.agent_params = agent_params or {}
        
        # Création des agents
        self.prompt_manager = self.agent_factory.create_prompt_manager(self.agent_params)
        self.ai_analyst = self.agent_factory.create_ai_analyst(self.agent_params)
        self.quality_controller = self.agent_factory.create_quality_controller(self.agent_params)
        self.general_manager = self.agent_factory.create_general_manager(self.agent_params)

    def _extract_score(self, quality_result: str) -> float:
        """Extrait le score numérique de la réponse du Quality Controller."""
        try:
            # Recherche un nombre entre 0 et 1 dans le texte
            match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', quality_result)
            if match:
                score = float(match.group(1))
                return min(max(score, 0.0), 1.0)  # Assure que le score est entre 0 et 1
            return 0.5  # Score par défaut si aucun score n'est trouvé
        except Exception as e:
            logger.warning(f"Erreur lors de l'extraction du score: {str(e)}")
            return 0.5

    def process_question(self, question: str) -> Dict:
        try:
            # Création des tâches
            task1 = Task(
                description=f"Analyser et reformuler la question suivante pour plus de clarté : {question}",
                agent=self.prompt_manager
            )

            task2 = Task(
                description="Générer une réponse détaillée et précise basée sur la question reformulée",
                agent=self.ai_analyst
            )

            task3 = Task(
                description="Évaluer la qualité et la pertinence de la réponse générée. Fournir un score entre 0 et 1 au format 'Score: X.XX'",
                agent=self.quality_controller
            )

            task4 = Task(
                description="Valider la réponse finale et préparer le retour à l'utilisateur",
                agent=self.general_manager
            )

            # Création et exécution du crew
            crew = Crew(
                agents=[self.prompt_manager, self.ai_analyst, self.quality_controller, self.general_manager],
                tasks=[task1, task2, task3, task4],
                verbose=True
            )

            result = crew.kickoff()

            # Traitement du résultat
            return {
                "original_question": question,
                "refined_question": result[0],  # Résultat de task1
                "answer": result[1],            # Résultat de task2
                "quality_score": self._extract_score(result[2]),  # Extraction du score numérique
                "status": result[3]             # Résultat de task4
            }

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {str(e)}")
            raise 