#!/bin/bash
# EmotionSense Complete Setup and Test Script
# Runs all three services: AI Service, Backend, and Frontend

echo "=========================================="
echo "  EmotionSense - Complete Setup"
echo "=========================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0
    else
        return 1
    fi
}

# Function to wait for service
wait_for_service() {
    local port=$1
    local name=$2
    local timeout=30
    local elapsed=0
    
    echo -e "${YELLOW}Waiting for $name on port $port...${NC}"
    
    while ! check_port $port && [ $elapsed -lt $timeout ]; do
        sleep 1
        elapsed=$((elapsed + 1))
    done
    
    if check_port $port; then
        echo -e "${GREEN}✓ $name is running on port $port${NC}"
        return 0
    else
        echo -e "${RED}✗ $name failed to start${NC}"
        return 1
    fi
}

# Step 1: Install Backend Dependencies
echo -e "\n${YELLOW}[1/4] Installing Backend Dependencies...${NC}"
cd backend
if npm install; then
    echo -e "${GREEN}✓ Backend dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install backend dependencies${NC}"
    exit 1
fi
cd ..

# Step 2: Install AI Service Dependencies
echo -e "\n${YELLOW}[2/4] Installing AI Service Dependencies...${NC}"
cd ai-service
if pip install -r requirements.txt; then
    echo -e "${GREEN}✓ AI service dependencies installed${NC}"
else
    echo -e "${RED}✗ Failed to install AI service dependencies${NC}"
    exit 1
fi
cd ..

# Step 3: Start AI Service (Python)
echo -e "\n${YELLOW}[3/4] Starting AI Service (Port 5000)...${NC}"
cd ai-service
python app.py &
AI_PID=$!
cd ..
wait_for_service 5000 "AI Service"

# Step 4: Start Backend Server (Node.js)
echo -e "\n${YELLOW}[4/4] Starting Backend Server (Port 3000)...${NC}"
cd backend
npm start &
BACKEND_PID=$!
cd ..
wait_for_service 3000 "Backend Server"

# Summary
echo ""
echo "=========================================="
echo -e "${GREEN}  ✓ All Services Running${NC}"
echo "=========================================="
echo -e "Frontend:  ${GREEN}http://localhost:3000${NC}"
echo -e "Backend:   ${GREEN}http://localhost:3000/api${NC}"
echo -e "AI Service: ${GREEN}http://localhost:5000${NC}"
echo ""
echo "To stop the services, press Ctrl+C"
echo ""

# Wait for interrupt signal
trap "kill $AI_PID $BACKEND_PID 2>/dev/null; echo 'Services stopped'; exit" INT

wait

# Cleanup on exit
kill $AI_PID $BACKEND_PID 2>/dev/null
