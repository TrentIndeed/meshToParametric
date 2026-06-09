@echo off
cd /d C:\Users\Trenton\CodeProjects\meshToParametric
set PYTHONIOENCODING=utf-8

REM Free port 8080 if a previous instance is still bound to it.
for /f "tokens=5" %%P in ('netstat -ano ^| findstr /R "127\.0\.0\.1:8080.*LISTENING"') do (
    echo Stopping existing process on port 8080 (PID %%P)...
    taskkill /F /PID %%P >nul 2>&1
)

echo Starting Paramesh AI server at http://127.0.0.1:8080 ...
python -m mesh2param.viz_server.app
