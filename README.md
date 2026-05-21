
# 📚 RAG Document Assistant - Chatbot Intelligent

Ce projet implémente une application interactive de **RAG (Retrieval-Augmented Generation)** avec une interface utilisateur moderne développée avec **Streamlit**. L'application permet de charger des documents PDF, d'indexer leur contenu et de dialoguer intelligemment avec vos documents grâce à un LLM.

## ✨ Fonctionnalités

- 📄 **Upload de documents PDF** - Chargez un ou plusieurs documents PDF
- 🔍 **Recherche sémantique** - Recherche vectorielle avancée avec ChromaDB
- 💬 **Interface conversationnelle** - Chat élégant avec historique des messages
- ⚡ **RAG temps réel** - Réponses contextuelles basées uniquement sur vos documents
- 🎨 **Design moderne** - Interface utilisateur propre et responsive
- ⌨️ **Raccourcis clavier** - Envoi de messages avec la touche Entrée

## 🛠️ Stack technique

- **Streamlit** – Framework d'interface utilisateur
- **LangChain** – Orchestration du pipeline RAG
- **ChromaDB** – Base de données vectorielle persistante
- **OpenRouter API** – Accès au modèle GPT-4o (solution alternative à OpenAI)
- **PyPDF2** – Extraction de texte depuis les PDF
- **python-dotenv** – Gestion des variables d'environnement

## 📋 Prérequis

- Python 3.13 ou supérieur
- `uv` (recommandé) ou `pip`
- Une clé API Open Router ([openrouter.ai](https://openrouter.ai/))

## 🔧 Installation

Clonez le dépôt et installez les dépendances :

```bash
git clone https://github.com/indexkboss/ChatBot_RAG.git
cd ChatBot_RAG
uv sync
```

> 💡 **Remarque** : L'utilisation de `uv` est recommandée pour une gestion plus rapide des dépendances, mais vous pouvez aussi utiliser `pip install -e .`.

## 🔑 Configuration

Créez un fichier `.env` à la racine du projet avec votre clé API Open Router :

```env
OPENROUTER_API_KEY=votre-clé-api-ici
```

> L'API Open Router est compatible avec le format OpenAI, ce qui permet d'utiliser `langchain-openai` en redirigeant simplement la base URL vers `https://openrouter.ai/api/v1`.

## 🚀 Utilisation

Lancez l'application Streamlit :

```bash
uv run streamlit run app.py
```

Ou si vous utilisez pip :

```bash
streamlit run app.py
```

### 📖 Guide d'utilisation

1. **Chargez vos documents** - Dans la barre latérale, uploader vos fichiers PDF
2. **Cliquez sur "Submit"** - Les documents sont découpés en chunks et indexés
3. **Posez vos questions** - Dans la zone de chat, tapez votre question et appuyez sur Entrée
4. **Obtenez des réponses** - L'assistant répond en se basant uniquement sur le contenu de vos documents

## 📂 Structure du projet

```
ChatBot_RAG/
├── .env                        # Variables d'environnement (clé API)
├── .gitignore                  # Fichiers ignorés par Git
├── pyproject.toml              # Dépendances et configuration du projet
├── README.md                   # Documentation
├── app.py                      # Application principale Streamlit
└── .python-version             # Version Python (pour pyenv)
```

## Personnalisation

- **Modèle LLM** - Changez le modèle dans l'initialisation de `ChatOpenAI` (actuellement `gpt-4o`)
- **Taille des chunks** - Ajustez `chunk_size` et `chunk_overlap` dans le `RecursiveCharacterTextSplitter`
- **Nombre de chunks récupérés** - Modifiez la valeur `k` dans `search_kwargs`
- **Thème et couleurs** - Personnalisez le CSS dans l'application

## 📦 Dépendances principales (extrait de `pyproject.toml`)

```toml
dependencies = [
    "chromadb>=1.5.9",          # Base vectorielle
    "ipykernel>=7.2.0",         # Support Jupyter
    "langchain-community>=0.4.1", # Composants LangChain
    "langchain-openai>=1.2.1",   # Interface OpenAI/OpenRouter
    "pypdf2>=3.0.1",            # Extraction PDF
    "streamlit>=1.57.0",        # Interface utilisateur
]
```

## 🤝 Contribution

Les suggestions et pull requests sont les bienvenues. Pour signaler un bug ou proposer une amélioration, ouvrez une **issue** sur GitHub.

---

### 👨‍💻 Auteur

Khadija Bossony
GitHub : [indexkboss](https://github.com/indexkboss)

Projet réalisé dans le cadre d'un apprentissage du RAG (Retrieval-Augmented Generation) avec Python.

---

### 🙏 Remerciements

- **Professeur Mohamed YOUSSFI** - Pour son contenu pédagogique sur le RAG

---

Développé avec ❤️ 