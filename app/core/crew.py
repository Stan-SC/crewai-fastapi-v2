from crewai import Crew, Task
from typing import Dict, Optional, List
from app.agents.crew_agents import AgentFactory
from loguru import logger
import re
import json
import time

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

    def _safe_str(self, value: any) -> str:
        """Convertit de manière sécurisée une valeur en chaîne de caractères."""
        try:
            if isinstance(value, (list, dict)):
                return json.dumps(value, ensure_ascii=False)
            return str(value).strip()
        except Exception as e:
            logger.error(f"Erreur lors de la conversion en chaîne: {str(e)}")
            return ""

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

    def _extract_manager_response(self, manager_result: str) -> Dict[str, str]:
        """Extrait la validation et la réponse finale du General Manager."""
        try:
            logger.info(f"Analyse de la réponse du manager: {json.dumps(manager_result)}")
            
            # Format attendu: "STATUS|RÉPONSE" (ex: "validé|Voici la réponse finale...")
            parts = manager_result.split('|', 1)
            
            if len(parts) == 2:
                status = parts[0].strip().lower()
                response = parts[1].strip()
                
                # Vérifie que le status est valide
                if status not in ['validé', 'rejeté']:
                    logger.warning(f"Status invalide '{status}', utilisation de 'rejeté'")
                    status = 'rejeté'
                
                logger.info(f"Extraction réussie - Status: {status}, Réponse: {json.dumps(response)}")
                return {"status": status, "final_answer": response}
            
            logger.warning("Format de réponse du manager invalide")
            return {"status": "rejeté", "final_answer": ""}
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la réponse du manager: {str(e)}")
            return {"status": "erreur", "final_answer": ""}

    def _process_task_results(self, results: List[str]) -> Dict:
        """Traite les résultats des tâches et les formate correctement."""
        try:
            logger.info("Traitement des résultats des tâches...")
            for i, result in enumerate(results):
                logger.info(f"Résultat brut de la tâche {i+1}: {json.dumps(result)}")
                logger.info(f"Type du résultat: {type(result)}")

            refined = self._safe_str(results[0])
            answer = self._safe_str(results[1])
            quality = self._safe_str(results[2])
            
            # Extraction de la validation et de la réponse finale du manager
            manager_response = self._extract_manager_response(self._safe_str(results[3]))

            logger.info(f"Résultats traités:")
            logger.info(f"Question raffinée: {json.dumps(refined)}")
            logger.info(f"Réponse initiale: {json.dumps(answer)}")
            logger.info(f"Qualité: {json.dumps(quality)}")
            logger.info(f"Réponse du manager: {json.dumps(manager_response)}")

            return {
                "refined_question": refined,
                "initial_answer": answer,
                "quality_score": self._extract_score(quality),
                "status": manager_response["status"],
                "final_answer": manager_response["final_answer"]
            }
        except Exception as e:
            logger.error(f"Erreur lors du traitement des résultats: {str(e)}")
            logger.exception("Détails de l'erreur:")
            return {
                "refined_question": "",
                "initial_answer": "",
                "quality_score": 0.5,
                "status": "erreur",
                "final_answer": ""
            }

    def process_question(self, question: str) -> Dict:
        try:
            logger.info(f"Début du traitement de la question: {json.dumps(question)}")
            
            # Création des tâches
            logger.info("Création des tâches...")
            task1 = Task(
                description=f"""Analysez et reformulez la question suivante pour plus de clarté : {question}
                IMPORTANT: Votre réponse doit être une question reformulée, rien d'autre.
                Exemple de format attendu: "Quels sont les principaux attraits touristiques de la ville de Grasse ?"
                
                Répondez UNIQUEMENT avec la question reformulée, sans autre texte.""",
                agent=self.prompt_manager,
                output_format="La question reformulée uniquement"
            )

            # Attente de la question reformulée
            logger.info("Obtention de la question reformulée...")
            initial_crew = Crew(
                agents=[self.prompt_manager],
                tasks=[task1],
                verbose=True
            )
            refined_question = self._safe_str(initial_crew.kickoff()[0])
            logger.info(f"Question reformulée: {json.dumps(refined_question)}")

            # Création de la tâche d'analyse
            task2 = Task(
                description=f"""Générez une réponse détaillée et précise à cette question : {refined_question}
                IMPORTANT: Votre réponse doit être concise mais complète.
                Concentrez-vous sur les faits les plus importants.
                
                Répondez directement avec les informations, sans formules de politesse.""",
                agent=self.ai_analyst,
                output_format="La réponse directe sans formules de politesse"
            )

            # Obtention de la réponse
            logger.info("Obtention de la réponse...")
            analyst_crew = Crew(
                agents=[self.ai_analyst],
                tasks=[task2],
                verbose=True
            )
            answer = self._safe_str(analyst_crew.kickoff()[0])
            logger.info(f"Réponse générée: {json.dumps(answer)}")

            # Évaluation de la qualité
            task3 = Task(
                description=f"""Évaluez la qualité et la pertinence de cette réponse : {answer}
                IMPORTANT: Vous devez fournir UNIQUEMENT un score numérique au format 'Score: X.XX'.
                Exemple: 'Score: 0.95'
                Ne donnez aucune autre information ou explication.""",
                agent=self.quality_controller,
                output_format="Score: X.XX"
            )

            # Obtention du score
            logger.info("Évaluation de la qualité...")
            quality_crew = Crew(
                agents=[self.quality_controller],
                tasks=[task3],
                verbose=True
            )
            quality = self._safe_str(quality_crew.kickoff()[0])
            score = self._extract_score(quality)
            logger.info(f"Score de qualité: {score}")

            # Validation finale
            task4 = Task(
                description=f"""Validez la réponse finale et envoyez-la.
                
                Question initiale: {question}
                Question reformulée: {refined_question}
                Réponse proposée: {answer}
                Score de qualité: {score}
                
                IMPORTANT: Si le score est >= 0.7, vous devez valider la réponse.
                Si le score est < 0.7, vous devez rejeter la réponse.
                
                Votre réponse doit suivre ce format exact:
                'validé|[RÉPONSE FINALE]' ou 'rejeté|[RAISON DU REJET]'
                
                Exemple de réponse validée:
                'validé|[COPIEZ LA RÉPONSE ICI SI ELLE EST SATISFAISANTE]'
                
                Exemple de réponse rejetée:
                'rejeté|La réponse manque de précision et de sources fiables.'""",
                agent=self.general_manager,
                output_format="validé|RÉPONSE ou rejeté|RAISON"
            )

            # Obtention de la validation
            logger.info("Validation finale...")
            manager_crew = Crew(
                agents=[self.general_manager],
                tasks=[task4],
                verbose=True
            )
            manager_result = self._safe_str(manager_crew.kickoff()[0])
            manager_response = self._extract_manager_response(manager_result)
            logger.info(f"Validation du manager: {json.dumps(manager_response)}")
            
            # Préparation de la réponse finale
            response = {
                "original_question": question,
                "refined_question": refined_question,
                "initial_answer": answer,
                "quality_score": score,
                "status": manager_response["status"],
                "final_answer": manager_response["final_answer"]
            }
            
            logger.info(f"Réponse finale préparée: {json.dumps(response, ensure_ascii=False)}")
            return response

        except Exception as e:
            logger.error(f"Erreur lors du traitement de la question: {str(e)}")
            logger.exception("Détails de l'erreur:")
            raise 