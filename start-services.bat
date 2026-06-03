@echo off
REM EmotionSense - Windows Startup Script
REM Run this file to start all services on Windows

echo.
echo ==========================================
echo  EmotionSense - Windows Startup
echo ==========================================
echo.

REM Check if running from correct directory
if not exist "backend" (
    echo ERROR: Please run this from the EmotionSense root directory
    pause
    exit /b 1
)

echo [1/3] Starting Backend Server (Port 3000)...
echo.
start "EmotionSense Backend" cmd /k "cd backend && npm install && npm start"
timeout /t 3 /nobreak

echo [2/3] Starting AI Service (Port 5000)...
echo.
start "EmotionSense AI Service" cmd /k "cd ai-service && pip install -r requirements.txt && python app.py"
timeout /t 3 /nobreak

echo [3/3] Opening Browser...
echo.
timeout /t 2 /nobreak
start http://localhost:3000

echo.
echo ==========================================
echo  Services Starting...
echo ==========================================
echo.
echo Frontend:   http://localhost:3000
echo Backend:    http://localhost:3000/api
echo AI Service: http://localhost:5000
echo.
echo Services will open in new windows.
echo This window will close in 5 seconds...
echo.

timeout /t 5
exit /b 0
