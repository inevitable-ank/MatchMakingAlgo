from typing import Dict, Any
import json
import os

from app.models.profile import Profile
from app.models.user import Viewer

# Global in-memory stores
USER_STORE: Dict[str, Any] = {}
# BANDIT_STATS stores feedback history per viewer
BANDIT_STATS: Dict[str, Dict[str, Any]] = {}

# Load mock profiles into USER_STORE on startup
def load_profiles():
    base = os.path.dirname(os.path.dirname(__file__))
    path = os.path.join(base, 'data', 'profiles.json')
    with open(path, 'r') as f:
        data = json.load(f)
    for item in data:
        profile = Profile(**item)
        USER_STORE[str(profile.id)] = profile

# Initialize store
load_profiles()