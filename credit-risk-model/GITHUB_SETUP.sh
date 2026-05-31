# GitHub Setup Guide — Credit Risk Model
# =========================================
# Run these commands IN ORDER after creating your repo on GitHub

# STEP 1: Create repo on GitHub
# ─────────────────────────────
# 1. Go to https://github.com/new
# 2. Repository name: credit-risk-model
# 3. Description: Credit Risk Probability Model for Bati Bank BNPL service
# 4. Set to PUBLIC (required for 10 Academy submission)
# 5. DO NOT initialize with README (we already have one)
# 6. Click "Create repository"

# STEP 2: Connect local repo to GitHub
# ──────────────────────────────────────
git remote add origin https://github.com/YOUR_USERNAME/credit-risk-model.git

# Verify remote
git remote -v

# STEP 3: Push main branch
# ─────────────────────────
git push -u origin main

# STEP 4: Push task branches
# ───────────────────────────
git push origin task-1
git push origin task-2

# STEP 5: Work on task-1 (already done — Business Understanding in README)
# ─────────────────────────────────────────────────────────────────────────
git checkout task-1
# Make any edits...
git add README.md
git commit -m "docs: complete credit scoring business understanding section"
git push origin task-1
# → Open Pull Request on GitHub: task-1 → main → Merge

# STEP 6: Work on task-2 (EDA notebook)
# ──────────────────────────────────────
git checkout main
git pull origin main        # Always pull latest before branching
git checkout task-2
# Edit notebooks/eda.ipynb ...
git add notebooks/eda.ipynb
git commit -m "feat: complete EDA with top 5 insights"
git push origin task-2
# → Open Pull Request on GitHub: task-2 → main → Merge

# STEP 7: Create future task branches
# ──────────────────────────────────────
git checkout main && git pull origin main
git checkout -b task-3
# ... work on feature engineering ...
git push origin task-3

git checkout main && git pull origin main
git checkout -b task-4
# ... work on proxy target variable ...
git push origin task-4

git checkout main && git pull origin main
git checkout -b task-5
# ... work on model training + MLflow ...
git push origin task-5

git checkout main && git pull origin main
git checkout -b task-6
# ... work on API + Docker + CI/CD ...
git push origin task-6

# ─────────────────────────────────────────────────────────────────────────────
# COMMON GIT COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

# Check current status
git status

# See commit history
git log --oneline --graph --all

# Stage specific file
git add src/data_processing.py

# Stage all changes
git add .

# Commit with message
git commit -m "feat: add RFM clustering for proxy target variable"

# Push current branch
git push origin $(git branch --show-current)

# Switch branch
git checkout task-3

# Pull latest from remote
git pull origin main

# See all branches (local + remote)
git branch -a
