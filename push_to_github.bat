@echo off
REM Git commands to push project to GitHub

echo Initializing git repository...
git init

echo Adding all files...
git add .

echo Committing files...
git commit -m "Initial commit: Multi-agent RAG system with self-correction"

echo Setting branch to main...
git branch -M main

echo Adding remote origin...
git remote add origin https://github.com/MoizSajjad/Agentic-Rag-System.git

echo Pushing to GitHub...
git push -u origin main

echo Done!

