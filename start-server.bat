@echo off
cd /d C:\Users\Trenton\CodeProjects\meshToParametric
set PYTHONIOENCODING=utf-8
echo Starting Paramesh AI server at http://127.0.0.1:8080 ...
python -m mesh2param.viz_server.app
