# backend/api/platform_instance.py
from backend.ab_testing.managers import AdaptiveABTestingPlatform

_platform_instance = None

def get_platform() -> AdaptiveABTestingPlatform:
    """Return the shared singleton AdaptiveABTestingPlatform instance."""
    global _platform_instance
    if _platform_instance is None:
        _platform_instance = AdaptiveABTestingPlatform()
    return _platform_instance
