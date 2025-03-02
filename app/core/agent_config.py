from typing import Dict, TypedDict, Optional, Union, List

class AgentConfig(TypedDict, total=False):
    # Paramètres de base de l'agent
    temperature: float  # Contrôle la créativité (0.0 à 2.0)
    model: str         # Modèle GPT à utiliser (ex: "gpt-3.5-turbo")
    max_iterations: int  # Nombre maximum d'itérations pour une tâche
    verbose: bool      # Afficher les logs détaillés
    
    # Paramètres de personnalité
    tone: str         # Ton de l'agent (ex: "professional", "friendly", "academic")
    language: str     # Langue de réponse (ex: "french", "english")
    expertise_level: str  # Niveau d'expertise (ex: "beginner", "expert")
    
    # Paramètres de réponse
    max_tokens: int   # Longueur maximale des réponses
    response_format: Dict[str, str]  # Format de réponse spécifique
    
    # Paramètres de contexte
    context: str      # Contexte supplémentaire pour l'agent
    tools: List[str]  # Outils spécifiques à utiliser

# Modèle par défaut pour tous les agents
DEFAULT_MODEL = "gpt-3.5-turbo"

# Configuration par défaut pour le Prompt Manager
DEFAULT_PROMPT_MANAGER_CONFIG = {
    "context": """Expert en reformulation et clarification de questions.
    Votre rôle est d'analyser les questions des utilisateurs et de les reformuler
    pour une meilleure compréhension et un traitement optimal.""",
    "verbose": True,
    "temperature": 0.7,
    "max_iterations": 3
}

# Configuration par défaut pour l'AI Analyst
DEFAULT_AI_ANALYST_CONFIG = {
    "context": """Expert en analyse et génération de réponses.
    Votre rôle est de fournir des réponses précises, détaillées et pertinentes
    aux questions des utilisateurs.""",
    "verbose": True,
    "temperature": 0.5,
    "max_iterations": 3
}

# Configuration par défaut pour le Quality Controller
DEFAULT_QUALITY_CONTROLLER_CONFIG = {
    "context": """Expert en contrôle qualité.
    Votre rôle est d'évaluer la qualité et la pertinence des réponses générées.
    Vous devez fournir un score numérique entre 0 et 1 au format 'Score: X.XX'.""",
    "verbose": True,
    "temperature": 0.3,
    "max_iterations": 3
}

# Configuration par défaut pour le General Manager
DEFAULT_GENERAL_MANAGER_CONFIG = {
    "context": """Superviseur général du processus.
    Votre rôle est de valider les réponses finales et d'assurer
    la cohérence globale du processus.""",
    "verbose": True,
    "temperature": 0.4,
    "max_iterations": 3
}

def merge_agent_configs(default_config: Dict, custom_config: Optional[Dict] = None) -> Dict:
    """
    Fusionne la configuration par défaut avec une configuration personnalisée.
    
    Args:
        default_config (Dict): Configuration par défaut
        custom_config (Optional[Dict]): Configuration personnalisée à fusionner
        
    Returns:
        Dict: Configuration fusionnée
    """
    if not custom_config:
        return default_config.copy()
    
    merged_config = default_config.copy()
    merged_config.update(custom_config)
    return merged_config

# Exemple d'utilisation des configurations
AGENT_CONFIG_EXAMPLES = {
    "prompt_manager": {
        "temperature": 0.8,
        "language": "french",
        "context": "Reformulez cette question pour plus de clarté"
    },
    "ai_analyst": {
        "temperature": 0.6,
        "max_tokens": 1000,
        "response_format": {
            "style": "detailed",
            "sections": ["context", "analysis", "conclusion"]
        }
    },
    "quality_controller": {
        "temperature": 0.2,
        "expertise_level": "expert",
        "context": "Évaluez la qualité de cette réponse"
    }
} 