# EggDetect AI Django Project - Run Guide

## How to Run the Project

The project is now running successfully! Here's how to run it properly:

### Correct Command (PowerShell)
```bash
.\run.bat
```

### Common Issue Fixed
The main error was using `run.bat` instead of `.\run.bat`. In PowerShell, you must use the `.\` prefix to execute batch files in the current directory.

### Alternative Commands
If you prefer to run manually:

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run migrations:
```bash
python manage.py makemigrations
python manage.py migrate
```

3. Set up admin user:
```bash
python setup.py
```

4. Start the server:
```bash
python manage.py runserver
```

### Access the Application
- URL: http://localhost:8000
- Admin login: Username `admin`, Password `admin123`
- Admin panel: http://localhost:8000/admin/

### Project Status
✅ All dependencies installed (Django 5.0.0, Pillow 10.1.0, TensorFlow 2.20.0)
✅ Database migrations completed
✅ ML models loaded successfully
✅ Server running on port 8000
✅ Application responding to requests

The project is now fully functional!