@echo off
echo Starting Vision OS Edge Pipeline...
start /b python main.py
echo Starting Vision OS Streamlit Dashboard...
start /b python -m streamlit run dashboard/app.py
echo.
echo ==============================================
echo [SUCCESS] Vision OS is now running!
echo The dashboard should open automatically in your browser at http://localhost:8501
echo.
echo NOTE: To stop the system and turn off your camera, close the command prompt windows that popped up.
echo ==============================================
pause
