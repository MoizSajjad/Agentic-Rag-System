# Git Commands to Push to GitHub

Run these commands in your project directory (`project/`) to push all files to GitHub.

## Step-by-Step Commands

### 1. Navigate to Project Directory

```bash
cd C:\Users\moizs\Desktop\agnetic_project\project
```

### 2. Initialize Git Repository (if not already initialized)

```bash
git init
```

### 3. Add All Files

```bash
git add .
```

**Note**: The `.gitignore` file will automatically exclude:
- `chroma_store/` (database files)
- `logs/` (log files)
- `__pycache__/` (Python cache)
- Other temporary files

### 4. Commit Files

```bash
git commit -m "Initial commit: Multi-agent RAG system with self-correction"
```

### 5. Set Branch to Main

```bash
git branch -M main
```

### 6. Add Remote Origin (if not already added)

```bash
git remote add origin https://github.com/MoizSajjad/Agentic-Rag-System.git
```

**Note**: If you get an error saying remote already exists, use:
```bash
git remote set-url origin https://github.com/MoizSajjad/Agentic-Rag-System.git
```

### 7. Push to GitHub

```bash
git push -u origin main
```

## All Commands in One Block (Copy & Paste)

```bash
cd C:\Users\moizs\Desktop\agnetic_project\project
git init
git add .
git commit -m "Initial commit: Multi-agent RAG system with self-correction"
git branch -M main
git remote add origin https://github.com/MoizSajjad/Agentic-Rag-System.git
git push -u origin main
```

## Troubleshooting

### If remote already exists:
```bash
git remote set-url origin https://github.com/MoizSajjad/Agentic-Rag-System.git
```

### If you need to force push (be careful!):
```bash
git push -u origin main --force
```

### To check what will be committed:
```bash
git status
```

### To see what files are tracked:
```bash
git ls-files
```

## Files That Will Be Pushed

✅ **Will be included:**
- All Python source files (`backend/`, `frontend/`, `tests/`, `evaluation/`)
- Configuration files (`requirements.txt`, `config.py`)
- Documentation (`README.md`, `PROJECT_DOCUMENTATION.md`, `ARCHITECTURE_DIAGRAMS.md`)
- Data file (`data/planets_full_clean.csv`)
- License (`LICENSE`)
- Diagrams (`diagrams.mmd`)

❌ **Will be excluded** (via `.gitignore`):
- `chroma_store/` (database files - too large)
- `logs/` (log files)
- `__pycache__/` (Python cache)
- Virtual environments
- IDE files

## After Pushing

1. Visit your repository: https://github.com/MoizSajjad/Agentic-Rag-System
2. Verify all files are uploaded correctly
3. Check that README.md displays properly with badges and diagrams
4. Add repository description: "A multi-agent RAG system with advanced hallucination detection and self-correction capabilities. Features four specialized AI agents (Planner, Retriever, Verifier, Memory) working together to provide accurate, verified answers for planetary science Q&A. Includes Streamlit UI, comprehensive evaluation metrics, and automatic answer correction."

## Next Steps

1. Add topics/tags to your repository:
   - `rag`
   - `retrieval-augmented-generation`
   - `multi-agent-systems`
   - `hallucination-detection`
   - `self-correction`
   - `llm`
   - `groq`
   - `chromadb`
   - `streamlit`
   - `planetary-science`
   - `python`
   - `ai`
   - `machine-learning`

2. Enable GitHub Pages (optional) for documentation
3. Add a project description on GitHub
4. Consider adding a demo GIF or screenshot

