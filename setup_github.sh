#!/bin/bash
# setup_github.sh
# Run this once from inside the mlb-prediction-model folder.
# It initializes git, creates the GitHub repo, pushes code, and sets secrets.

set -e  # Exit on any error

echo ""
echo "══════════════════════════════════════════════"
echo "  MLB Prediction Model — GitHub Setup"
echo "══════════════════════════════════════════════"
echo ""

# ── Check prerequisites ────────────────────────────────────────────────────────
if ! command -v git &> /dev/null; then
    echo "✗ git not found. Install from https://git-scm.com"
    exit 1
fi

if ! command -v gh &> /dev/null; then
    echo "✗ GitHub CLI not found. Install it first:"
    echo "   Mac:     brew install gh"
    echo "   Windows: winget install GitHub.cli"
    echo "   Then run: gh auth login"
    exit 1
fi

if ! gh auth status &> /dev/null; then
    echo "✗ Not logged into GitHub CLI. Run: gh auth login"
    exit 1
fi

echo "✓ git and gh CLI found"
echo ""

# ── Get API keys ──────────────────────────────────────────────────────────────
read -p "Enter your Odds API key (from the-odds-api.com): " ODDS_KEY
read -p "Enter your Weather API key (from weatherapi.com): " WEATHER_KEY
echo ""

# ── Git init and commit ───────────────────────────────────────────────────────
echo "1. Initializing git repo..."
git init
git add .
git commit -m "Initial commit: MLB prediction model"
echo "   ✓ Git initialized"
echo ""

# ── Create GitHub repo ────────────────────────────────────────────────────────
echo "2. Creating GitHub repo..."
gh repo create stat-trick-hockey/mlb-prediction-model \
    --public \
    --source=. \
    --remote=origin \
    --push \
    --description "MLB moneyline, run line, and O/U prediction model with Statcast features"
echo "   ✓ Repo created and code pushed"
echo ""

# ── Set secrets ───────────────────────────────────────────────────────────────
echo "3. Setting GitHub Actions secrets..."
gh secret set ODDS_API_KEY    --repo stat-trick-hockey/mlb-prediction-model --body "$ODDS_KEY"
gh secret set WEATHER_API_KEY --repo stat-trick-hockey/mlb-prediction-model --body "$WEATHER_KEY"
echo "   ✓ Secrets set"
echo ""

# ── Done ──────────────────────────────────────────────────────────────────────
echo "══════════════════════════════════════════════"
echo "  ✓ All done!"
echo ""
echo "  Repo: https://github.com/stat-trick-hockey/mlb-prediction-model"
echo ""
echo "  Next steps:"
echo "  1. pip install -r requirements.txt"
echo "  2. python setup_and_train.py  (builds data + trains models)"
echo "  3. GitHub Actions will auto-run predictions at 10am ET daily"
echo "══════════════════════════════════════════════"
echo ""
