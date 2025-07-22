#!/bin/bash
# Automated setup and training script for bitter-retrieval

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Bitter Retrieval - Automated Setup & Training${NC}"
echo "=================================================="

# Check arguments
if [ $# -eq 0 ]; then
    echo -e "${RED}❌ Error: Google Drive file ID required${NC}"
    echo "Usage: $0 <google_drive_file_id> [training_method]"
    echo ""
    echo "Example: $0 1ABCdef123xyz456 standard_infonce"
    echo ""
    echo "Available methods:"
    echo "  - standard_infonce (default)"
    echo "  - converted_infonce" 
    echo "  - kl_soft_infonce"
    echo ""
    echo "To get file ID from Google Drive link:"
    echo "https://drive.google.com/file/d/1ABCdef123xyz.../view"
    echo "                                 ↑ This part"
    exit 1
fi

FILE_ID=$1
METHOD=${2:-standard_infonce}

echo -e "${BLUE}📋 Configuration:${NC}"
echo "  File ID: $FILE_ID"
echo "  Method: $METHOD"
echo ""

# Step 1: Install dependencies
echo -e "${YELLOW}📦 Installing dependencies...${NC}"
if command -v uv &> /dev/null; then
    echo "Using uv for faster installs..."
    uv sync
else
    pip install -r requirements.txt
fi
echo -e "${GREEN}✅ Dependencies installed${NC}"
echo ""

# Step 2: Download data
echo -e "${YELLOW}📥 Downloading data from Google Drive...${NC}"
python download_data.py "$FILE_ID"
echo ""

# Step 3: Verify data exists
DATA_PATH="data/msmarco/soft_labels_msmarco_3B_nopad_5000.json"
if [ ! -f "$DATA_PATH" ]; then
    echo -e "${RED}❌ Error: Data file not found at $DATA_PATH${NC}"
    exit 1
fi

# Get file size for info
FILE_SIZE=$(du -h "$DATA_PATH" | cut -f1)
echo -e "${GREEN}✅ Data verified: $FILE_SIZE${NC}"
echo ""

# Step 4: Start training
echo -e "${YELLOW}🔥 Starting training with method: $METHOD${NC}"
echo "Command: python -m src.bitter_retrieval.train --method $METHOD"
echo ""

# Create a run name with timestamp
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RUN_NAME="${METHOD}_${TIMESTAMP}"

# Run training with nice defaults
python -m src.bitter_retrieval.train \
    --method "$METHOD" \
    --run-name "$RUN_NAME" \
    --batch-size 2 \
    --num-epochs 2 \
    --learning-rate 2e-5

echo ""
echo -e "${GREEN}🎉 Training completed successfully!${NC}"
echo -e "${BLUE}📊 Check your results in Weights & Biases or the console output above.${NC}" 