from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
import os
from dotenv import load_dotenv

from app.models.schemas import QuestionRequest, QuestionResponse
from app.core.crew import QuestionCrew

# Chargement des variables d'environnement
load_dotenv()

# Configuration de la journalisation
logger.add("app.log", rotation="500 MB")

app = FastAPI(
    title="CrewAI Question API",
    description="API pour traiter les questions avec une équipe d'agents CrewAI",
    version="1.0.0"
)

# Configuration CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/ask", response_model=QuestionResponse)
async def ask_question(request: QuestionRequest):
    """
    Traite une question utilisateur avec l'équipe CrewAI.
    
    - **question**: La question de l'utilisateur
    - **agent_params**: Paramètres optionnels pour configurer les agents
    """
    try:
        logger.info(f"Nouvelle question reçue: {request.question}")
        
        # Création de l'équipe
        crew = QuestionCrew(request.agent_params)
        
        # Traitement de la question
        result = crew.process_question(request.question)
        
        logger.info("Question traitée avec succès")
        return QuestionResponse(**result)
        
    except Exception as e:
        logger.error(f"Erreur lors du traitement de la question: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors du traitement de la question: {str(e)}"
        )

@app.get("/health")
async def health_check():
    """
    Vérifie l'état de santé de l'API
    """
    return {"status": "healthy"} 