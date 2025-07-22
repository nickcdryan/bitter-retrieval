#!/bin/bash
# Comprehensive setup script for bitter-retrieval on fresh Linux servers
# This script handles all prerequisites and ensures a reproducible environment

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🔧 Bitter Retrieval - Linux Server Setup${NC}"
echo "==========================================="

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to detect Linux distribution
detect_distro() {
    if [ -f /etc/os-release ]; then
        . /etc/os-release
        echo $ID
    elif [ -f /etc/redhat-release ]; then
        echo "rhel"
    elif [ -f /etc/debian_version ]; then
        echo "debian"
    else
        echo "unknown"
    fi
}

DISTRO=$(detect_distro)
echo -e "${BLUE}📋 Detected OS: $DISTRO${NC}"

# Step 1: Update system packages
echo -e "${YELLOW}📦 Updating system packages...${NC}"
case $DISTRO in
    ubuntu|debian)
        sudo apt update && sudo apt upgrade -y
        sudo apt install -y curl wget git python3 python3-pip python3-venv
        ;;
    fedora|rhel|centos)
        sudo dnf update -y
        sudo dnf install -y curl wget git python3 python3-pip
        ;;
    arch|manjaro)
        sudo pacman -Syu --noconfirm
        sudo pacman -S --noconfirm curl wget git python python-pip
        ;;
    *)
        echo -e "${YELLOW}⚠️  Unknown distribution. Assuming basic tools are available.${NC}"
        ;;
esac
echo -e "${GREEN}✅ System packages updated${NC}"

# Step 2: Install pipx (recommended way to install Poetry)
echo -e "${YELLOW}📦 Installing pipx...${NC}"
if ! command_exists pipx; then
    case $DISTRO in
        ubuntu|debian)
            sudo apt install -y pipx
            ;;
        fedora|rhel|centos)
            if command_exists dnf; then
                sudo dnf install -y pipx
            else
                python3 -m pip install --user pipx
            fi
            ;;
        *)
            python3 -m pip install --user pipx
            ;;
    esac
    
    # Ensure pipx is in PATH
    if ! command_exists pipx; then
        export PATH="$HOME/.local/bin:$PATH"
        echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
    fi
    
    pipx ensurepath
    echo -e "${GREEN}✅ pipx installed${NC}"
else
    echo -e "${GREEN}✅ pipx already installed${NC}"
fi

# Step 3: Install Poetry via pipx
echo -e "${YELLOW}📦 Installing Poetry...${NC}"
if ! command_exists poetry; then
    pipx install poetry
    
    # Ensure poetry is in PATH
    if ! command_exists poetry; then
        export PATH="$HOME/.local/bin:$PATH"
    fi
    echo -e "${GREEN}✅ Poetry installed${NC}"
else
    echo -e "${GREEN}✅ Poetry already installed${NC}"
fi

# Step 4: Verify Poetry installation and show version
echo -e "${YELLOW}🔍 Verifying Poetry installation...${NC}"
poetry --version
echo -e "${GREEN}✅ Poetry verification successful${NC}"

# Step 5: Configure PyTorch CUDA source
echo -e "${YELLOW}⚙️  Configuring PyTorch CUDA source...${NC}"
if ! poetry source show pytorch &>/dev/null; then
    echo "Adding PyTorch CUDA source for cu128..."
    poetry source add --priority=explicit pytorch https://download.pytorch.org/whl/cu128
    echo -e "${GREEN}✅ PyTorch CUDA source added${NC}"
else
    echo -e "${GREEN}✅ PyTorch CUDA source already configured${NC}"
fi

# Step 6: Install project dependencies
echo -e "${YELLOW}📦 Installing project dependencies...${NC}"
poetry install
echo -e "${GREEN}✅ Project dependencies installed${NC}"

# Step 6b: Install flash-attn for optimized attention (requires CUDA)
echo -e "${YELLOW}⚡ Installing flash-attn for faster inference...${NC}"
if poetry run pip install flash-attn --no-build-isolation >/dev/null 2>&1; then
    echo -e "${GREEN}✅ flash-attn installed successfully${NC}"
else
    echo -e "${YELLOW}⚠️  flash-attn installation failed (this is optional)${NC}"
    echo "  This is normal on some systems - the script will continue"
fi

# Step 7: Create helpful aliases and environment setup
echo -e "${YELLOW}⚙️  Setting up environment...${NC}"

# Add poetry to PATH permanently if not already there
if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' ~/.bashrc; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
fi

# Create helpful aliases
ALIAS_FILE="$HOME/.bash_aliases"
if [ ! -f "$ALIAS_FILE" ]; then
    touch "$ALIAS_FILE"
fi

# Add project-specific aliases if not already present
if ! grep -q "alias bitter-train=" "$ALIAS_FILE"; then
    echo "# Bitter Retrieval aliases" >> "$ALIAS_FILE"
    echo "alias bitter-train='cd $(pwd) && poetry run python -m src.bitter_retrieval.train'" >> "$ALIAS_FILE"
    echo "alias bitter-shell='cd $(pwd) && poetry shell'" >> "$ALIAS_FILE"
fi

echo -e "${GREEN}✅ Environment setup complete${NC}"

# Step 8: Verify complete installation
echo -e "${YELLOW}🔍 Verifying complete installation...${NC}"
if poetry run python test_setup.py >/dev/null 2>&1; then
    echo -e "${GREEN}✅ Complete verification passed${NC}"
    echo "  All dependencies and modules working correctly"
else
    echo -e "${YELLOW}⚠️  Some verification tests failed${NC}"
    echo "  Run 'poetry run python test_setup.py' for details"
fi

# Step 9: Setup API keys
echo -e "${YELLOW}🔑 Setting up API keys...${NC}"
if [ ! -f ".env" ]; then
    echo "Creating .env file from template..."
    cp .env.example .env
    echo -e "${YELLOW}⚠️  API keys need configuration!${NC}"
    echo ""
    echo -e "${BLUE}📝 Required API Keys Setup:${NC}"
    echo "1. Hugging Face Token (required for gated models):"
    echo "   - Visit: https://huggingface.co/settings/tokens"
    echo "   - Create a new token with 'Read' access"
    echo "   - Copy the token"
    echo ""
    echo "2. Weights & Biases API Key (optional for experiment tracking):"
    echo "   - Visit: https://wandb.ai/authorize"
    echo "   - Copy your API key"
    echo ""
    echo "3. Edit your .env file:"
    echo "   nano .env  # or use your preferred editor"
    echo ""
    echo -e "${RED}⚠️  You MUST configure these keys before training!${NC}"
else
    echo -e "${GREEN}✅ .env file already exists${NC}"
fi

# Step 10: Final verification
echo ""
echo -e "${BLUE}🎉 Setup Complete!${NC}"
echo "===================="
echo -e "${GREEN}✅ Python: $(python3 --version)${NC}"
echo -e "${GREEN}✅ Poetry: $(poetry --version)${NC}"
echo -e "${GREEN}✅ Project dependencies: Installed${NC}"
echo -e "${GREEN}✅ PyTorch CUDA: Configured for cu128${NC}"
echo -e "${GREEN}✅ API key template: Created${NC}"
echo ""
echo -e "${YELLOW}💡 Helpful commands:${NC}"
echo "  poetry shell          # Activate virtual environment"
echo "  poetry run <command>  # Run command in virtual environment"
echo "  bitter-train          # Shortcut for training (after sourcing ~/.bashrc)"
echo ""
echo -e "${YELLOW}📝 Next steps:${NC}"
echo "1. Source your bashrc: source ~/.bashrc"
echo "2. Configure API keys: poetry run python setup_env.py (recommended) or nano .env"
echo "3. Test setup: poetry run python test_setup.py"
echo "4. Run training: ./run_training.sh <google_drive_file_id>"
echo ""
echo -e "${GREEN}🚀 Your server is ready for bitter-retrieval with CUDA support!${NC}" 