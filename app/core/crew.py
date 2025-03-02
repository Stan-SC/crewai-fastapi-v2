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

    def _extract_final_answer(self, text: str) -> str:
        """Extrait la réponse finale du texte complet."""
        try:
            if not text:
                logger.warning("Texte vide reçu dans _extract_final_answer")
                return ""

            # Nettoyage du texte
            text = text.strip()
            if not text:
                logger.warning("Texte vide après nettoyage dans _extract_final_answer")
                return ""

            # Log du texte complet pour le débogage
            logger.debug(f"Texte complet reçu dans _extract_final_answer: {json.dumps(text)}")

            # Recherche différents patterns possibles
            patterns = [
                r'Final Answer:\s*(.*?)(?=\n|$)',  # Format standard
                r'Answer:\s*(.*?)(?=\n|$)',        # Format alternatif
                r'Response:\s*(.*?)(?=\n|$)',      # Autre format possible
            ]

            for pattern in patterns:
                match = re.search(pattern, text, re.DOTALL)
                if match:
                    answer = match.group(1).strip()
                    logger.info(f"Réponse extraite avec le pattern '{pattern}': {json.dumps(answer)}")
                    return answer

            # Si aucun pattern ne correspond, retourner le texte complet
            logger.info(f"Aucun pattern ne correspond, retour du texte complet: {json.dumps(text)}")
            return text

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la réponse finale: {str(e)}")
            logger.error(f"Texte qui a causé l'erreur: {json.dumps(text)}")
            return text if text else ""

    def _extract_score(self, quality_result: str) -> float:
        """Extrait le score numérique de la réponse du Quality Controller."""
        try:
            logger.info(f"Tentative d'extraction du score à partir de: {json.dumps(quality_result)}")
            
            if not quality_result or quality_result.isspace():
                logger.warning("Résultat du Quality Controller vide ou uniquement des espaces")
                return 0.5
            
            # Recherche un nombre entre 0 et 1 dans le texte
            patterns = [
                r'Score:\s*([0-9]*\.?[0-9]+)',  # Format standard
                r'([0-9]*\.?[0-9]+)/1\.0',      # Format fraction
                r'([0-9]*\.?[0-9]+)\s*sur\s*1', # Format texte
            ]

            for pattern in patterns:
                match = re.search(pattern, quality_result)
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
            
            if not manager_result or manager_result.isspace():
                logger.warning("Réponse du manager vide")
                return {"status": "rejeté", "final_answer": ""}
            
            # Extrait d'abord la réponse finale
            final_answer = self._extract_final_answer(manager_result)
            logger.debug(f"Réponse finale extraite: {json.dumps(final_answer)}")
            
            # Format attendu dans la réponse finale: "STATUS|RÉPONSE"
            parts = final_answer.split('|', 1)
            
            if len(parts) == 2:
                status = parts[0].strip().lower()
                response = parts[1].strip()
                
                # Vérifie que le status est valide
                if status not in ['validé', 'rejeté']:
                    logger.warning(f"Status invalide '{status}', utilisation de 'rejeté'")
                    status = 'rejeté'
                
                logger.info(f"Extraction réussie - Status: {status}, Réponse: {json.dumps(response)}")
                return {"status": status, "final_answer": response}
            
            # Si le format n'est pas correct, on essaie de détecter le statut dans le texte
            status = 'validé' if 'validé' in final_answer.lower() else 'rejeté'
            logger.warning(f"Format de réponse du manager non standard, statut détecté: {status}")
            return {"status": status, "final_answer": final_answer}
            
        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la réponse du manager: {str(e)}")
            return {"status": "erreur", "final_answer": ""}

    def _execute_task(self, task: Task, crew_name: str) -> str:
        """Exécute une tâche avec un équipage d'un seul agent et retourne la réponse."""
        try:
            logger.info(f"Exécution de la tâche pour {crew_name}...")
            
            # Vérification des paramètres
            if not task or not task.agent:
                logger.error(f"Tâche invalide pour {crew_name}")
                return ""
                
            # Modification de la description pour forcer le format de réponse
            task.description = f"""{task.description}

                TRÈS IMPORTANT: Votre réponse DOIT commencer par 'Final Answer:' suivi d'un espace.
                Si vous ne suivez pas ce format, votre réponse sera rejetée.
                
                Exemple de format correct:
                Final Answer: Voici ma réponse..."""
                
            # Création et exécution de l'équipage
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                verbose=True
            )
            
            # Exécution avec retry en cas d'erreur
            max_retries = 3
            retry_count = 0
            base_delay = 5  # Délai de base en secondes
            
            while retry_count < max_retries:
                try:
                    # Exécution de la tâche
                    result = crew.kickoff()
                    
                    # Vérification du résultat
                    if not result or len(result) == 0:
                        logger.error(f"Aucun résultat obtenu pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))  # Délai exponentiel
                        continue
                    
                    # Log du résultat brut
                    logger.debug(f"Résultat brut pour {crew_name}: {json.dumps(result[0])}")
                    
                    # Extraction et validation de la réponse
                    response = self._extract_final_answer(result[0])
                    if not response:
                        logger.warning(f"Réponse vide obtenue pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))
                        continue
                    
                    # Vérification de la qualité de la réponse
                    if len(response.strip()) < 5:  # Réponse trop courte
                        logger.warning(f"Réponse trop courte pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))
                        continue
                        
                    # Vérification supplémentaire pour le format Final Answer
                    if not any(pattern in result[0] for pattern in ['Final Answer:', 'Answer:', 'Response:']):
                        logger.warning(f"Format de réponse incorrect pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))
                        continue
                        
                    logger.info(f"Réponse obtenue pour {crew_name}: {json.dumps(response)}")
                    return response
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la tentative {retry_count + 1}/{max_retries} pour {crew_name}: {str(e)}")
                    retry_count += 1
                    time.sleep(base_delay * (retry_count + 1))
            
            logger.error(f"Échec de toutes les tentatives pour {crew_name}")
            return ""
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la tâche pour {crew_name}: {str(e)}")
            return ""

    def process_question(self, question: str) -> Dict:
        try:
            logger.info(f"Début du traitement de la question: {json.dumps(question)}")
            
            # Création et exécution de la tâche de reformulation
            task1 = Task(
                description=f"""Analysez et reformulez la question suivante pour plus de clarté : {question}
                IMPORTANT: Votre réponse doit être une question reformulée, rien d'autre.
                Exemple de format attendu: "Quels sont les principaux attraits touristiques de la ville de Grasse ?"
                
                Répondez UNIQUEMENT avec la question reformulée, sans autre texte.
                
                Format de réponse attendu:
                Final Answer: [VOTRE QUESTION REFORMULÉE]""",
                agent=self.prompt_manager
            )
            refined_question = self._execute_task(task1, "Prompt Manager")
            logger.info(f"Question reformulée: {json.dumps(refined_question)}")

            # Création et exécution de la tâche d'analyse
            task2 = Task(
                description=f"""Générez une réponse détaillée et précise à cette question : {refined_question}
                IMPORTANT: Votre réponse doit être concise mais complète.
                Concentrez-vous sur les faits les plus importants.
                
                Répondez directement avec les informations, sans formules de politesse.
                
                Format de réponse attendu:
                Final Answer: [VOTRE RÉPONSE]""",
                agent=self.ai_analyst
            )
            answer = self._execute_task(task2, "AI Analyst")
            logger.info(f"Réponse générée: {json.dumps(answer)}")

            # Création et exécution de la tâche d'évaluation
            task3 = Task(
                description=f"""Évaluez la qualité et la pertinence de cette réponse : {answer}
                IMPORTANT: Vous devez fournir UNIQUEMENT un score numérique au format 'Score: X.XX'.
                Exemple: 'Score: 0.95'
                Ne donnez aucune autre information ou explication.
                
                Format de réponse attendu:
                Final Answer: Score: [VOTRE SCORE]""",
                agent=self.quality_controller
            )
            quality = self._execute_task(task3, "Quality Controller")
            score = self._extract_score(quality)
            logger.info(f"Score de qualité: {score}")

            # Création et exécution de la tâche de validation
            task4 = Task(
                description=f"""Validez la réponse finale et envoyez-la.
                
                Question initiale: {question}
                Question reformulée: {refined_question}
                Réponse proposée: {answer}
                Score de qualité: {score}
                
                IMPORTANT: Si le score est >= 0.7, vous devez valider la réponse.
                Si le score est < 0.7, vous devez rejeter la réponse.
                
                Format de réponse attendu:
                Final Answer: validé|[COPIEZ LA RÉPONSE ICI SI ELLE EST SATISFAISANTE]
                ou
                Final Answer: rejeté|[RAISON DU REJET]""",
                agent=self.general_manager
            )
            manager_result = self._execute_task(task4, "General Manager")
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