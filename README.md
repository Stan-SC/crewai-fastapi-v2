# CrewAI Question API

Une API REST FastAPI qui utilise CrewAI pour traiter intelligemment les questions des utilisateurs avec une équipe d'agents spécialisés.

## 🚀 Fonctionnalités

- API REST avec FastAPI
- Équipe d'agents CrewAI spécialisés
- Pipeline de traitement intelligent des questions
- Documentation Swagger intégrée
- Logging complet
- Prêt pour le déploiement sur Render

## 📋 Prérequis

- Python 3.9+
- OpenAI API Key

## 🛠️ Installation

1. Cloner le repository :
```bash
git clone [votre-repo]
cd [votre-repo]
```

2. Installer les dépendances :
```bash
pip install -r requirements.txt
```

3. Configurer les variables d'environnement :
Créer un fichier `.env` à la racine du projet :
```env
OPENAI_API_KEY=votre-clé-api-openai
```

## 🏃‍♂️ Lancement local

```bash
./start.sh
```

L'API sera disponible sur `http://localhost:8000`
Documentation Swagger : `http://localhost:8000/docs`

## 📡 Utilisation de l'API

### Endpoint POST /ask

Exemple de requête :
```json
{
    "question": "Quelle est la capitale de la France ?",
    "agent_params": {
        "temperature": 0.7
    }
}
```

Exemple de réponse :
```json
{
    "original_question": "Quelle est la capitale de la France ?",
    "refined_question": "Pouvez-vous me dire quelle est la capitale officielle de la France ?",
    "answer": "Paris est la capitale de la France.",
    "quality_score": 0.95,
    "status": "validated"
}
```

## 🚀 Déploiement sur Render

1. Créer un nouveau Web Service sur Render
2. Connecter votre repository GitHub
3. Configurer le service :
   - **Environment**: Python 3.9
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `./start.sh`
   - **Variables d'environnement** :
     - `OPENAI_API_KEY`: Votre clé API OpenAI

## 📝 Logs

Les logs sont stockés dans `app.log` avec rotation automatique à 500 MB.

## 🔒 Sécurité

- Validation des entrées avec Pydantic
- Gestion sécurisée des variables d'environnement
- Middleware CORS configuré

## 📚 Documentation

La documentation complète de l'API est disponible via Swagger UI à `/docs` 