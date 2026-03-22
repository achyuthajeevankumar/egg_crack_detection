from django.urls import path
from . import views

urlpatterns = [
    path('', views.index_view, name='base'),
    path('dashboard/', views.dashboard_view, name='dashboard'),
]
