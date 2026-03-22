@echo off
echo ╔══════════════════════════════════════════════════════╗
|      EggDetect AI Django - Starting Application      ║
╚══════════════════════════════════════════════════════╝

echo.
echo Installing dependencies into virtual environment...
if not exist venv (
    echo Creating virtual environment...
    python -m venv venv
)
venv\Scripts\python -m pip install -r requirements.txt

echo.
echo Running migrations...
venv\Scripts\python manage.py makemigrations
venv\Scripts\python manage.py migrate

echo.
echo Setting up admin user...
venv\Scripts\python setup.py

echo.
echo ══════════════════════════════════════════════════════
echo echo Starting Django server...
echo Access: http://127.0.0.1:8000
echo Admin login: Triple-click logo OR Ctrl+Shift+A
echo ══════════════════════════════════════════════════════
echo.
.\venv\Scripts\python.exe manage.py runserver
pause
