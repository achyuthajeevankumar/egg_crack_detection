from django.apps import AppConfig
import threading
import sys

class DetectionConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'detection'

    def ready(self):
        import os
        if os.environ.get('RUN_MAIN') == 'true':
            def pre_load_task():
                print("\n" + "="*50)
                print(">>> EggDetect AI: Starting Model Pre-loading...")
                print("="*50)
                from .views import get_model
                models_to_load = ['egg_validator', 'cnn', 'resnet', 'xception']
                for name in models_to_load:
                    try:
                        get_model(name)
                    except Exception as e:
                        print(f"[!] Unexpected error pre-loading {name}: {e}")
                print("="*50)
                print("[OK] Model pre-loading sequence complete!")
                print("="*50 + "\n")

            threading.Thread(target=pre_load_task, daemon=True).start()
