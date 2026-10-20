@echo off
TITLE APEX Telegram Bot Server
echo ========================================================
echo        APEX RAG Mobile Telegram Bot Server
echo ========================================================
echo.

:: Check if Python is available
python --version >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [ERROR] Python is not installed or not in PATH.
    pause
    exit /b 1
)

:: Run preflight inspector
python inspector.py
if %ERRORLEVEL% neq 0 (
    echo.
    echo [WARNING] Environment health check reported an issue.
    echo Make sure 'ollama serve' is running and required models are pulled.
    echo.
)

:: Run Telegram Bot
echo.
echo Starting APEX Telegram Bot...
python telegram_bot.py

pause
