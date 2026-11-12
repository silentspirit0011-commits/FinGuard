@echo off
title FinGuard - Financial Transaction Surveillance System
echo ========================================================
echo   Launching FinGuard - Financial Anomaly Detection Suite
echo ========================================================
echo.
timeout /t 2 /nobreak >nul
start "" "http://localhost:8501"
python -m streamlit run app.py
pause
