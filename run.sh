#!/bin/bash
echo "╔══════════════════════════════════════════════════════╗"
echo "║      EggDetect AI Django - Starting Application      ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""
echo "Installing dependencies..."
pip3 install -r requirements.txt
echo ""
echo "Running migrations..."
python3 manage.py makemigrations
python3 manage.py migrate
echo ""
echo "Setting up admin user..."
python3 setup.py
echo ""
echo "══════════════════════════════════════════════════════"
echo "Starting Django server..."
echo "Access: http://localhost:8000"
echo "Admin login: Triple-click logo OR Ctrl+Shift+A"
echo "══════════════════════════════════════════════════════"
echo ""
python3 manage.py runserver
