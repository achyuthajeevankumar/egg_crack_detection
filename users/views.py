from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, logout, authenticate
from django.contrib.auth.decorators import login_required, user_passes_test
from django.contrib import messages
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt, csrf_protect
from django.contrib.auth.hashers import make_password
from .models import User, LoginActivity
from django.db import models
from django.core.exceptions import ValidationError
from django.views.decorators.cache import never_cache
from django.utils import timezone
from django.conf import settings
import os
import cv2
import base64
import json
from io import BytesIO
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .utility.train import xception_model
import logging

logger = logging.getLogger(__name__)

def is_admin(user):
    """Check if user is admin"""
    return user.is_authenticated and user.role == 'admin'


def get_client_ip(request):
    """Get client IP address"""
    x_forwarded_for = request.META.get('HTTP_X_FORWARDED_FOR')
    if x_forwarded_for:
        ip = x_forwarded_for.split(',')[0]
    else:
        ip = request.META.get('REMOTE_ADDR')
    return ip


@csrf_exempt
def register_view(request):
    """User registration"""
    if request.method == 'POST':
        try:
            name = request.POST.get('name')
            username = request.POST.get('username')
            email = request.POST.get('email')
            mobile = request.POST.get('mobile')
            password = request.POST.get('password')
            confirm_password = request.POST.get('confirm_password')
            
            if not all([username, email, password, confirm_password]):
                messages.error(request, 'All fields are required!')
                return render(request, 'signup.html')
            
            if password != confirm_password:
                messages.error(request, 'Passwords do not match!')
                return render(request, 'signup.html')
            
            if User.objects.filter(username=username).exists():
                messages.error(request, 'Username already exists!')
                return render(request, 'signup.html')
            
            user = User.objects.create_user(
                username=username,
                email=email,
                password=password,
                full_name=name,
                mobile=mobile,
                role='user'
            )
            
            messages.success(request, 'Registration successful! Please login.')
            return redirect('userlogin')
        
        except Exception as e:
            messages.error(request, str(e))
            return render(request, 'signup.html')
    
    return render(request, 'signup.html')

    # --------------------------------------------------------model training views----------------------------------


def train_xception_view(request):
    if not request.session.get('logged_in'):
        return redirect('userlogin')
    result = xception_model()
    return render(request, 'train_result.html', {'result': result, 'model': 'Xception'})

def train_xception_json_view(request):
    if not request.session.get('logged_in'):
        return redirect('userlogin')
    
    json_result_path = os.path.join(settings.MEDIA_ROOT, 'models', 'xception_model_result.json')
    if os.path.exists(json_result_path):
        with open(json_result_path, 'r') as f:
            result = json.load(f)
    else:
        result = {'success': False, 'message': 'Result file not found.'}
        
    return render(request, 'train_result.html', {'result': result, 'model': 'Xception (JSON)'})

# ---------------------------------------------------DATASET VIEW--------------------------------------------------------------

def egg_dataset_view(request):
    if not request.session.get('logged_in'):
        return redirect('userlogin')
        
    print("✅ Starting Egg Dataset View...")
    data_dir = os.path.join(settings.MEDIA_ROOT, 'augmented_dataset')
    categories = ['Not Damaged', 'Damaged']
    IMG_SIZE = 299
    num_images = 5

    if not os.path.exists(data_dir):
        # Fallback if augmented_dataset doesn't exist yet
        data_dir = os.path.join(settings.MEDIA_ROOT, 'dataset')

    def load_sample_images(label):
        path = os.path.join(data_dir, label)
        images = []
        if not os.path.exists(path): return images
        count = 0
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                images.append(img)
                count += 1
            if count >= num_images:
                break
        return images

    def get_base64_samples(images, title):
        if not images: return ""
        fig, ax = plt.subplots(1, len(images), figsize=(15, 3))
        for i, img in enumerate(images):
            ax[i].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            ax[i].axis('off')
            ax[i].set_title(title)
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png')
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    def get_base64_bar_chart():
        counts = {}
        for label in categories:
            path = os.path.join(data_dir, label)
            counts[label] = len(os.listdir(path)) if os.path.exists(path) else 0
            
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(counts.keys(), counts.values(), color=['#22c55e', '#ef4444'])
        ax.set_title("Egg Crack Dataset - Class Distribution", color='white')
        ax.set_xlabel("Class", color='white')
        ax.set_ylabel("Number of Images", color='white')
        ax.tick_params(colors='white')
        
        # Transparent background for the chart
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
        
        buffer = BytesIO()
        plt.tight_layout()
        plt.savefig(buffer, format='png', transparent=True)
        plt.close(fig)
        buffer.seek(0)
        return base64.b64encode(buffer.read()).decode('utf-8')

    not_damaged_imgs = load_sample_images("Not Damaged")
    damaged_imgs = load_sample_images("Damaged")

    context = {
        'distribution_chart': get_base64_bar_chart(),
        'samples_not_damaged': get_base64_samples(not_damaged_imgs, "Not Damaged"),
        'samples_damaged': get_base64_samples(damaged_imgs, "Damaged"),
        'username': request.session.get('username')
    }

    return render(request, 'egg_dataset.html', context)


@csrf_exempt
def login_view(request):
    """User login"""
    if request.method == 'POST':
        login_input = request.POST.get('username') # This could be username, email, or mobile
        password = request.POST.get('password')
        
        # Try to find user by username, email, or mobile
        user_obj = User.objects.filter(
            models.Q(username=login_input) | 
            models.Q(email=login_input) | 
            models.Q(mobile=login_input)
        ).first()

        if user_obj:
            user = authenticate(request, username=user_obj.username, password=password)
            if user is not None:
                if user.status.lower() == 'active' or user.is_staff:
                    login(request, user)
                    messages.success(request, f'Welcome back, {user.full_name or user.username}!')
                    return redirect('dashboard')
                else:
                    messages.error(request, 'Your account is pending activation by Admin.')
                    return render(request, 'userlogin.html')
            else:
                messages.error(request, 'Invalid password')
                return render(request, 'userlogin.html')
        else:
            messages.error(request, 'Invalid credentials')
            print("🌐 Rendering login page (GET request or failed POST)")
    return render(request, 'userlogin.html')

# --------------------------------------------------------admin user management views----------------------------------

@login_required
@user_passes_test(is_admin)
def activate_user(request, user_id):
    
    user = get_object_or_404(User, id=user_id)
    user.status = 'Active'
    user.save()
    messages.success(request, f"User {user.username} has been activated successfully.")
    return redirect('dashboard')

@login_required
@user_passes_test(is_admin)
def deactivate_user(request, user_id):
    
    user = get_object_or_404(User, id=user_id)
    user.status = 'Inactive'
    user.save()
    messages.success(request, f"User {user.username} has been deactivated successfully.")
    return redirect('dashboard')

@login_required
@user_passes_test(is_admin)
def delete_user(request, user_id):
    
    user = get_object_or_404(User, id=user_id)
    username = user.username
    user.delete()
    messages.success(request, f"User {username} has been permanently deleted.")
    return redirect('dashboard')


@csrf_exempt
def admin_login_view(request):
    """Admin login"""
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')
        
        user = authenticate(request, username=username, password=password)
        if user is not None and user.role == 'admin':
            login(request, user)
            request.session['is_admin_logged_in'] = True
            return redirect('dashboard')
        else:
            messages.error(request, 'Invalid admin credentials!')
            return render(request, 'adminlogin.html')
            
    return render(request, 'adminlogin.html')


@login_required
def logout_view(request):
    """User logout"""
    logout(request)
    return redirect('login')


@login_required
def profile_view(request):
    """Get or update user profile"""
    if request.method == 'GET':
        return JsonResponse({
            'success': True,
            'username': request.user.username,
            'email': request.user.email
        })
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            
            # Check if username/email exists for other users
            if User.objects.filter(username=username).exclude(id=request.user.id).exists():
                return JsonResponse({'success': False, 'message': 'Username already exists!'})
            
            if User.objects.filter(email=email).exclude(id=request.user.id).exists():
                return JsonResponse({'success': False, 'message': 'Email already exists!'})
            
            # Update user
            request.user.username = username
            request.user.email = email
            
            if password:
                if len(password) < 6:
                    return JsonResponse({'success': False, 'message': 'Password must be at least 6 characters!'})
                request.user.password = make_password(password)
            
            request.user.save()
            
            return JsonResponse({'success': True, 'message': 'Profile updated successfully!'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})




@login_required
@user_passes_test(is_admin)
def admin_stats_view(request):
    """Get admin statistics"""
    from detection.models import Detection
    
    total_users = User.objects.count()
    total_detections = Detection.objects.count()
    cracked_eggs = Detection.objects.filter(is_cracked=True).count()
    
    return JsonResponse({
        'success': True,
        'stats': {
            'total_users': total_users,
            'total_detections': total_detections,
            'cracked_eggs': cracked_eggs
        }
    })


@login_required
@user_passes_test(is_admin)
def admin_users_view(request):
    """Manage users"""
    if request.method == 'GET':
        users = User.objects.all().values('id', 'username', 'email', 'role', 'status', 'created_at')
        users_list = list(users)
        
        # Format datetime
        for user in users_list:
            user['created_at'] = user['created_at'].strftime('%Y-%m-%d %H:%M:%S')
        
        return JsonResponse({'success': True, 'users': users_list})
    
    elif request.method == 'POST':
        try:
            data = json.loads(request.body)
            username = data.get('username')
            email = data.get('email')
            password = data.get('password')
            role = data.get('role', 'user')
            
            # Check if user exists
            if User.objects.filter(username=username).exists():
                return JsonResponse({'success': False, 'message': 'Username already exists!'})
            
            if User.objects.filter(email=email).exists():
                return JsonResponse({'success': False, 'message': 'Email already exists!'})
            
            # Create user
            User.objects.create(
                username=username,
                email=email,
                password=make_password(password),
                role=role
            )
            
            return JsonResponse({'success': True, 'message': 'User added successfully!'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    elif request.method == 'PUT':
        try:
            data = json.loads(request.body)
            user_id = data.get('id')
            user = User.objects.get(id=user_id)
            
            if 'toggle_role' in data:
                user.role = 'admin' if user.role == 'user' else 'user'
            else:
                user.username = data.get('username')
                user.email = data.get('email')
            
            user.save()
            
            return JsonResponse({'success': True, 'message': 'User updated successfully!'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    elif request.method == 'DELETE':
        try:
            data = json.loads(request.body)
            user_id = data.get('id')
            User.objects.filter(id=user_id).delete()
            
            return JsonResponse({'success': True, 'message': 'User deleted successfully!'})
        
        except Exception as e:
            return JsonResponse({'success': False, 'message': str(e)})
    
    return JsonResponse({'success': False, 'message': 'Invalid request method'})


@login_required
@user_passes_test(is_admin)
def admin_activity_view(request):
    """Get login activities"""
    activities = LoginActivity.objects.select_related('user').all()[:100]
    
    activities_list = [{
        'username': activity.user.username,
        'role': activity.user.role,
        'ip_address': activity.ip_address,
        'timestamp': activity.timestamp.strftime('%Y-%m-%d %H:%M:%S')
    } for activity in activities]
    
    return JsonResponse({'success': True, 'activities': activities_list})
