from django.urls import path
from . import views

urlpatterns = [
    path('signup/', views.register_view, name='registration'),
    path('login/', views.login_view, name='userlogin'),
    path('adminlogin/', views.admin_login_view, name='adminlogin'),
    path('auth/login/', views.login_view, name='login'), # for form action mapping
    path('auth/admin-check/', views.admin_login_view, name='admin_login'), # for admin template
    path('logout/', views.logout_view, name='logout'),
    path('profile/', views.profile_view, name='profile'),
    path('admin/stats/', views.admin_stats_view, name='admin_stats'),
    path('admin/users/', views.admin_users_view, name='admin_users'),
    path('admin/activity/', views.admin_activity_view, name='admin_activity'),
    path('activate/<int:user_id>/', views.activate_user, name='activate_user'),
    path('deactivate/<int:user_id>/', views.deactivate_user, name='deactivate_user'),
    path('delete/<int:user_id>/', views.delete_user, name='delete_user'),
    path('train/xception/', views.train_xception_view, name='jsonxception_modelview'),
    path('train/xception-json/', views.train_xception_json_view, name='json_xception_model_view'),
    path('dataset/', views.egg_dataset_view, name='egg_dataset'),
]
