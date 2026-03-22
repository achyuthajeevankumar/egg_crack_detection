from django.urls import path
from . import views

urlpatterns = [
    path('upload/', views.upload_detect_view, name='upload_detect'),
    path('camera/', views.camera_detect_view, name='camera_detect'),
    path('history/', views.history_view, name='history'),
    path('report/', views.generate_report_view, name='generate_report'),
    path('performance/', views.performance_comparison_view, name='performance_comparison'),
    path('analysis/', views.graphical_analysis_view, name='graphical_analysis'),
]
