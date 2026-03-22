from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required


def index_view(request):
    """Landing page - login/register"""
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'base.html')


@login_required
def dashboard_view(request):
    """Main dashboard"""
    return render(request, 'dashboard.html', {
        'username': request.user.username,
        'role': request.user.role
    })
