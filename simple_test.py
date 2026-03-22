#!/usr/bin/env python
"""
Simple test to verify the fixes are working
"""

import os
import sys
import django

# Setup Django environment
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'eggdetect.settings')
django.setup()

from django.test import Client
from django.contrib.auth.models import User
from detection.models import Detection

def test_endpoints():
    print("🚀 Testing EggDetect Fixes")
    print("=" * 40)

    # Create test user
    user = User.objects.create_user(username='testuser', password='testpass123')
    client = Client()
    client.login(username='testuser', password='testpass123')

    # Test 1: History endpoint
    print("1. Testing History Endpoint...")
    response = client.get('/detection/history/', HTTP_X_REQUESTED_WITH='XMLHttpRequest')
    if response.status_code == 200:
        data = response.json()
        if data.get('success', False):
            print("   ✅ History AJAX endpoint works")
        else:
            print("   ❌ History endpoint failed")
    else:
        print("   ❌ History endpoint failed")

    # Test 2: Performance comparison
    print("2. Testing Performance Comparison...")

    # First test with no data
    response = client.get('/detection/performance/')
    if response.status_code == 200:
        data = response.json()
        if not data.get('success', True):
            print("   ✅ Performance comparison handles no data correctly")
        else:
            print("   ❌ Performance comparison should fail with no data")
    else:
        print("   ❌ Performance comparison request failed")

    # Add test data
    Detection.objects.create(
        user=user,
        is_cracked=True,
        cnn_accuracy=95.5,
        cnn_confidence=92.3,
        resnet_accuracy=97.8,
        resnet_confidence=94.1,
        xception_accuracy=96.2,
        xception_confidence=93.5
    )

    # Test with data
    response = client.get('/detection/performance/')
    if response.status_code == 200:
        data = response.json()
        if data.get('success', False) and 'comparison' in data:
            print("   ✅ Performance comparison works with data")
        else:
            print("   ❌ Performance comparison failed with data")
    else:
        print("   ❌ Performance comparison request failed")

    # Test 3: Graphical analysis
    print("3. Testing Graphical Analysis...")

    response = client.get('/detection/analysis/')
    if response.status_code == 200:
        data = response.json()
        if data.get('success', False) and 'analysis' in data:
            analysis = data['analysis']
            if (('accuracy_history' in analysis) and
                ('loss_history' in analysis) and
                ('confusion_matrix' in analysis) and
                ('roc_curve' in analysis)):
                print("   ✅ Graphical analysis works correctly")
            else:
                print("   ❌ Graphical analysis missing data sections")
        else:
            print("   ❌ Graphical analysis failed")
    else:
        print("   ❌ Graphical analysis request failed")

    # Cleanup
    Detection.objects.all().delete()
    user.delete()

    print("=" * 40)
    print("🎉 Testing complete!")

if __name__ == '__main__':
    test_endpoints()