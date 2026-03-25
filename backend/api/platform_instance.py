# backend/api/platform_instance.py
from backend.ab_testing.managers import AdaptiveABTestingPlatform
from backend.microservices.ab_testing.service import ABPlatformProvider


def get_platform() -> AdaptiveABTestingPlatform:
    """Return shared thread-safe AdaptiveABTestingPlatform instance."""
    return ABPlatformProvider.get()
