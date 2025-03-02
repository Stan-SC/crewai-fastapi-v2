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
        """Extrait la réponse finale du texte."""
        if not text or not isinstance(text, str):
            logger.warning("Texte vide ou invalide reçu dans _extract_final_answer")
            return ""

        logger.debug(f"Texte complet reçu dans _extract_final_answer: {json.dumps(text)}")

        # Nettoyage initial du texte
        text = text.strip()
        
        # Recherche du pattern "Final Answer:" avec gestion des sauts de ligne
        pattern = r'Final Answer:\s*(.*?)(?=\n\s*(?:Thought:|Human:|Assistant:|$)|$)'
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        
        if match:
            answer = match.group(1).strip()
            logger.info(f"Réponse extraite avec le pattern 'Final Answer:': {json.dumps(answer)}")
            if answer:  # Vérifie que la réponse n'est pas vide
                return answer

        # Si le pattern n'a pas fonctionné, essayons de trouver la dernière occurrence de "Final Answer:"
        parts = text.split("Final Answer:")
        if len(parts) > 1:
            answer = parts[-1].strip()  # Prend la dernière partie après "Final Answer:"
            logger.info(f"Réponse extraite après le dernier 'Final Answer:': {json.dumps(answer)}")
            if answer:  # Vérifie que la réponse n'est pas vide
                return answer

        # Si aucun pattern ne correspond, retourner le texte complet
        logger.info(f"Aucun pattern ne correspond, retour du texte complet: {json.dumps(text)}")
        return text

    def _extract_score(self, quality_result: str) -> float:
        """Extrait le score numérique de la réponse du Quality Controller."""
        logger.info(f"Tentative d'extraction du score à partir de: {json.dumps(quality_result)}")

        if not quality_result or quality_result.isspace():
            logger.warning("Résultat du Quality Controller vide ou uniquement des espaces")
            return 0.5

        try:
            # Recherche d'un score au format "Score: X.XX"
            pattern = r'Score:\s*([0-9]*\.?[0-9]+)'
            match = re.search(pattern, quality_result)
            
            if match:
                score = float(match.group(1))
                # Assurer que le score est entre 0 et 1
                score = max(0.0, min(1.0, score))
                logger.info(f"Score extrait avec succès: {score}")
                return score
            else:
                logger.warning("Aucun score trouvé dans le format attendu")
                return 0.5

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction du score: {str(e)}")
            return 0.5

    def _extract_manager_response(self, manager_result: str) -> Dict[str, str]:
        """Extrait la réponse du General Manager."""
        logger.info(f"Analyse de la réponse du manager: {json.dumps(manager_result)}")

        if not manager_result or manager_result.isspace():
            logger.warning("Réponse du manager vide")
            return {"status": "rejeté", "final_answer": ""}

        try:
            # Extraction de la réponse après "Final Answer:"
            response = self._extract_final_answer(manager_result)
            
            # Séparation du statut et de la réponse
            if "|" in response:
                status, answer = response.split("|", 1)
                status = status.strip().lower()
                answer = answer.strip()
                
                if status not in ["validé", "rejeté"]:
                    logger.warning(f"Statut invalide reçu: {status}")
                    status = "rejeté"
                    
                logger.info(f"Réponse du manager extraite: status={status}, answer={json.dumps(answer)}")
                return {"status": status, "final_answer": answer}
            else:
                logger.warning("Format de réponse du manager invalide")
                return {"status": "rejeté", "final_answer": ""}

        except Exception as e:
            logger.error(f"Erreur lors de l'extraction de la réponse du manager: {str(e)}")
            return {"status": "rejeté", "final_answer": ""}

    def _execute_task(self, task: Task, crew_name: str) -> str:
        """Exécute une tâche avec un équipage d'un seul agent et retourne la réponse."""
        try:
            logger.info(f"Exécution de la tâche pour {crew_name}...")
            
            # Vérification des paramètres
            if not task or not task.agent:
                logger.error(f"Tâche invalide pour {crew_name}")
                return ""

            # Modification de la description pour forcer le format de réponse
            original_description = task.description
            task.description = f"""INSTRUCTIONS TRÈS IMPORTANTES:
                1. Votre réponse DOIT commencer par 'Final Answer: ' (avec un espace après les deux points)
                2. Votre réponse doit être complète et détaillée
                3. Ne donnez pas d'explications sur votre processus de réflexion
                4. Répondez directement à la question
                
                QUESTION OU TÂCHE:
                {original_description}
                
                FORMAT DE RÉPONSE REQUIS:
                Final Answer: [Votre réponse ici]
                
                EXEMPLE DE RÉPONSE CORRECTE:
                Final Answer: La ville de Grasse est la capitale mondiale du parfum..."""

            # Création de l'équipage avec configuration spécifique
            crew = Crew(
                agents=[task.agent],
                tasks=[task],
                verbose=True,
                process_timeout=120  # 2 minutes de timeout
            )
            
            # Variables pour le retry
            max_retries = 3
            retry_count = 0
            base_delay = 5
            min_response_length = 50  # Longueur minimale attendue pour une réponse valide
            
            while retry_count < max_retries:
                try:
                    # Exécution de la tâche avec timeout
                    start_time = time.time()
                    result = crew.kickoff()
                    execution_time = time.time() - start_time
                    
                    logger.info(f"Temps d'exécution pour {crew_name}: {execution_time:.2f} secondes")
                    
                    # Vérification du résultat
                    if not result or len(result) == 0:
                        logger.error(f"Aucun résultat obtenu pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))
                        continue
                    
                    # Log du résultat brut
                    raw_result = result[0]
                    logger.debug(f"Résultat brut pour {crew_name}: {json.dumps(raw_result)}")
                    
                    # Extraction de la réponse finale
                    if "Final Answer:" in raw_result:
                        response = raw_result.split("Final Answer:", 1)[1].strip()
                    else:
                        response = raw_result.strip()
                    
                    # Vérification de la qualité de la réponse
                    if len(response) < min_response_length:
                        logger.warning(f"Réponse trop courte pour {crew_name} (tentative {retry_count + 1}/{max_retries})")
                        retry_count += 1
                        time.sleep(base_delay * (retry_count + 1))
                        continue
                    
                    logger.info(f"Réponse valide obtenue pour {crew_name}: {json.dumps(response)}")
                    return response
                    
                except Exception as e:
                    logger.error(f"Erreur lors de la tentative {retry_count + 1}/{max_retries} pour {crew_name}: {str(e)}")
                    retry_count += 1
                    time.sleep(base_delay * (retry_count + 1))
            
            # Si toutes les tentatives ont échoué, on retourne une réponse d'erreur appropriée
            error_responses = {
                "Prompt Manager": "Je ne peux pas reformuler votre question pour le moment. Veuillez réessayer.",
                "AI Analyst": "Je ne peux pas générer une réponse appropriée pour le moment. Veuillez réessayer.",
                "Quality Controller": "Score: 0.5",
                "General Manager": "rejeté|Le système n'a pas pu générer une réponse appropriée."
            }
            
            logger.error(f"Échec de toutes les tentatives pour {crew_name}")
            return error_responses.get(crew_name, "Une erreur est survenue. Veuillez réessayer.")
            
        except Exception as e:
            logger.error(f"Erreur lors de l'exécution de la tâche pour {crew_name}: {str(e)}")
            return error_responses.get(crew_name, "Une erreur est survenue. Veuillez réessayer.")

    def process_question(self, question: str) -> Dict:
        try:
            logger.info(f"Début du traitement de la question: {json.dumps(question)}")
            
            # Création et exécution de la tâche de reformulation
            task1 = Task(
                description=f"""INSTRUCTIONS TRÈS IMPORTANTES:
                Votre tâche est de reformuler la question suivante de manière claire et précise : 
                "{question}"
                
                RÈGLES À SUIVRE:
                1. Gardez la même intention que la question originale
                2. Soyez clair et direct
                3. Ne changez pas le sens de la question
                4. Utilisez un français correct
                5. Ne donnez AUCUNE explication ou commentaire
                
                FORMAT DE RÉPONSE REQUIS:
                Final Answer: [VOTRE QUESTION REFORMULÉE]
                
                EXEMPLE:
                Si la question est "c koi Grasse?", votre réponse devrait être:
                Final Answer: Pouvez-vous décrire la ville de Grasse et ses caractéristiques principales ?""",
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