import os
from django.core.management.base import BaseCommand
from users.models import User
from django.contrib.auth.hashers import make_password

class Command(BaseCommand):
    help = 'Create a default admin user for Render deployment'

    def handle(self, *args, **options):
        # Get credentials from environment or use defaults
        username = os.environ.get('ADMIN_USERNAME', 'admin')
        password = os.environ.get('ADMIN_PASSWORD', 'admin123')
        email = os.environ.get('ADMIN_EMAIL', 'admin@example.com')

        if not User.objects.filter(username=username).exists():
            self.stdout.write(f'Creating admin user: {username}...')
            User.objects.create(
                username=username,
                email=email,
                password=make_password(password),
                role='admin',
                status='Active',
                is_staff=True,
                is_superuser=True
            )
            self.stdout.write(self.style.SUCCESS(f'Successfully created admin user: {username}'))
        else:
            self.stdout.write(self.style.WARNING(f'Admin user {username} already exists.'))
