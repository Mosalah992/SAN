@echo off
cd /d "%~dp0"
set USE_LOCAL_LLM=true
python -m backend.test_insight_extraction
pause
