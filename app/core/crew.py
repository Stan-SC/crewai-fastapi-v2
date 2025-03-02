from crewai import Crew, Task
from typing import Dict, Optional
from app.agents.crew_agents import AgentFactory
from loguru import logger
import re
import json

class QuestionCrew:
    def __init__(self, agent_params: Optional[Dict] = None):
        logger.info("Initialisation de QuestionCrew")
        self.agent_factory = AgentFactory()
        self.agent_params = agent_params or {}
        
        # Création des agents
        logger.info("Création des agents...")
        self.prompt_manager = self.agent_factory.create_prompt_manager(self.agent_params)
        self.ai_analyst = self.agent_factory.create_ai_analyst(self.agent_params)
        self.quality_controller = self.agent_factory.create_quality_controller(self.agent_params)
        self.general_manager = self.agent_factory.create_general_manager(self.agent_params)
        logger.info("Tous les agents ont été créés avec succès")

    def _extract_score(self, quality_result: str) -> float:
        """Extrait le score numérique de la réponse du Quality Controller."""
        try:
            logger.info(f"Tentative d'extraction du score à partir de: {json.dumps(quality_result)}")
            
            if not quality_result or quality_result.isspace():
                logger.warning("Résultat du Quality Controller vide ou uniquement des espaces")
                return 0.5
            
            # Recherche un nombre entre 0 et 1 dans le texte
            match = re.search(r'Score:\s*([0-9]*\.?[0-9]+)', quality_result)
            if match:
                score = float(match.group(1))
                score = min(max(score, 0.0), 1.0)  # Assure que le score est entre 0 et 1
                logger.info(f"Score extrait avec succès: {score}")
                return score
            
            logger.warning(f"Aucun score trouvé dans le format attendu. Texte complet: {json.dumps(quality_result)}")
            return 0.5
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du score: {str(e)}")
            logger.error(f"Texte qui a causé l'erreur: {json.dumps(quality_result)}")
            return 0.5

    def process_question(self, question: str) -> Dict:
        try:
            logger.info(f"Début du traitement de la question: {json.dumps(question)}")
            
            # Création des tâches
            logger.info("Création des tâches...")
            task1 = Task(
                description=f"""Analysez et reformulez la question suivante pour plus de clarté : {question}
                IMPORTANT: Votre réponse doit être une question reformulée, rien d'autre.""",
                agent=self.prompt_manager
            )

            task2 = Task(
                description="""Générez une réponse détaillée et précise basée sur la question reformulée.
                IMPORTANT: Votre réponse doit être concise mais complète.""",
                agent=self.ai_analyst
            )

            task3 = Task(
                description="""Évaluez la qualité et la pertinence de la réponse générée.
                IMPORTANT: Vous devez fournir un score numérique entre 0 et 1 au format exact 'Score: X.XX'.
                Par exemple: 'Score: 0.95' pour une excellente réponse.
                Votre réponse ne doit contenir que le score, rien d'autre.""",
                agent=self.quality_controller
            )

            task4 = Task(
                description="""Validez la réponse finale et préparez le retour à l'utilisateur.
                IMPORTANT: Votre réponse doit être un statut simple : 'validé' ou 'rejeté'.""",
                agent=self.general_manager
            )

            # Création et exécution du crew
            logger.info("Création du crew...")
            crew = Crew(
                agents=[self.prompt_manager, self.ai_analyst, self.quality_controller, self.general_manager],
                tasks=[task1, task2, task3, task4],
                verbose=True
            )

            logger.info("Démarrage de l'exécution du crew")
            result = crew.kickoff()
            logger.info("Crew terminé, traitement des résultats")
            
            # Log des résultats bruts
            logger.info(f"Résultat task1 (reformulation): {json.dumps(result[0])}")
            logger.info(f"Résultat task2 (réponse): {json.dumps(result[1])}")
            logger.info(f"Résultat task3 (qualité): {json.dumps(result[2])}")
            logger.info(f"Résultat task4 (validation): {json.dumps(result[3])}")

            # Traitement du résultat
            response = {
                "original_question": question,
                "refined_question": str(result[0]).strip(),
                "answer": str(result[1]).strip(),
                "quality_score": self._extract_score(str(result[2])),
                "status": str(result[3]).strip()
            }
            
            logger.info(f"Réponse finale préparée: {json.dumps(response)}")
            return response

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {str(e)}")
            logger.exception("Détails de l'erreur:")
            raise 