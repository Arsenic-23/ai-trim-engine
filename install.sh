#!/usr/bin/env bash
set -e

# ANSI Colors
CYAN='\033[1;36m'
WHITE='\033[1;37m'
GREY='\033[0;90m'
GREEN='\033[1;32m'
NC='\033[0m' # No Color

echo -e "\n${CYAN}   🪐 CRAON Engine${NC}"
echo -e "${GREY}   Intelligent Video Reasoning & Trimming${NC}\n"

# 1. Install uv if not present
if ! command -v uv &> /dev/null; then
    echo -e "${WHITE} → ${NC}Installing ultra-fast dependencies manager..."
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1
    
    # Source the environment variables so uv is available in this script's subshell
    if [ -f "$HOME/.cargo/env" ]; then
        source "$HOME/.cargo/env"
    fi
    export PATH="$HOME/.local/bin:$PATH"
fi

echo -e "${WHITE} → ${NC}Building and linking CRAON globally (this may take a few seconds)..."

# Install directly from the GitHub repository quietly
uv tool install -q --force git+https://github.com/Arsenic-23/ai-trim-engine.git

echo -e "\n${GREEN} ✓ Installation completely successful!${NC}\n"
echo -e "${GREY}   You can now launch the interactive shell from anywhere by typing:${NC}"
echo -e "${CYAN}   craon${NC}\n"
