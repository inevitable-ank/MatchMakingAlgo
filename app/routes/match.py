from fastapi import APIRouter, HTTPException
from app.services.recommender import recommend_next_profile, record_feedback
from app.models.profile import Profile
from app.models.user import Feedback, Viewer
from app.db.database import USER_STORE, BANDIT_STATS
from typing import Dict, List, Any

router = APIRouter()

@router.get("/next-profile/{viewer_id}", response_model=Profile)
def get_next_profile(viewer_id: str):
    """Get the next recommended profile for a viewer"""
    # Ensure viewer exists
    viewer = USER_STORE.get(viewer_id)
    if viewer is None:
        raise HTTPException(status_code=404, detail="Viewer not found")
    # Recommend profile
    profile = recommend_next_profile(viewer_id)
    if profile is None:
        raise HTTPException(status_code=204, detail="No more profiles available")
    return profile

@router.post("/feedback")
def post_feedback(feedback: Feedback):
    """Record user feedback (like/dislike) for a profile"""
    # Validate viewer and profile
    if feedback.viewer_id not in USER_STORE:
        raise HTTPException(status_code=404, detail="Viewer not found")
    if str(feedback.profile_id) not in USER_STORE:
        raise HTTPException(status_code=404, detail="Profile not found")
    # Record feedback
    record_feedback(feedback.viewer_id, feedback.profile_id, feedback.liked)
    return {"status": "ok", "message": "Feedback recorded successfully"}

@router.get("/analytics/{viewer_id}")
def get_user_analytics(viewer_id: str) -> Dict[str, Any]:
    """Get analytics and insights for a specific user"""
    if viewer_id not in USER_STORE:
        raise HTTPException(status_code=404, detail="Viewer not found")
    
    stats = BANDIT_STATS.get(viewer_id, {})
    history = stats.get('history', [])
    
    if not history:
        return {
            "viewer_id": viewer_id,
            "total_interactions": 0,
            "like_ratio": 0.0,
            "engagement_score": 0.5,
            "preferences": {},
            "recommendation_insights": "No interaction data available"
        }
    
    # Calculate analytics
    total_interactions = len(history)
    likes = sum(1 for entry in history if entry['liked'])
    like_ratio = likes / total_interactions if total_interactions > 0 else 0.0
    
    # Get engagement metrics
    engagement = stats.get('engagement_metrics', {})
    engagement_score = engagement.get('like_ratio', 0.0)
    
    # Extract preferences
    preferences = stats.get('preference_model', {})
    
    return {
        "viewer_id": viewer_id,
        "total_interactions": total_interactions,
        "like_ratio": like_ratio,
        "engagement_score": engagement_score,
        "preferences": preferences,
        "recommendation_insights": _generate_insights(history, preferences)
    }

@router.get("/algorithm-performance")
def get_algorithm_performance() -> Dict[str, Any]:
    """Get overall algorithm performance metrics"""
    total_users = len(BANDIT_STATS)
    total_interactions = sum(len(stats.get('history', [])) for stats in BANDIT_STATS.values())
    
    # Calculate average engagement
    engagement_scores = []
    for stats in BANDIT_STATS.values():
        engagement = stats.get('engagement_metrics', {})
        if engagement.get('total_interactions', 0) > 0:
            engagement_scores.append(engagement.get('like_ratio', 0.0))
    
    avg_engagement = sum(engagement_scores) / len(engagement_scores) if engagement_scores else 0.0
    
    return {
        "total_active_users": total_users,
        "total_interactions": total_interactions,
        "average_engagement": avg_engagement,
        "algorithm_status": "operational",
        "performance_metrics": {
            "response_time_avg_ms": 45,  # Placeholder
            "cache_hit_rate": 0.78,      # Placeholder
            "exploration_rate": 0.15
        }
    }

@router.post("/create-viewer")
def create_viewer(viewer: Viewer):
    """Create a new viewer in the system"""
    if viewer.id in USER_STORE:
        raise HTTPException(status_code=400, detail="Viewer already exists")
    
    USER_STORE[viewer.id] = viewer
    return {"status": "ok", "message": f"Viewer {viewer.id} created successfully"}

@router.get("/profiles")
def get_all_profiles() -> List[Profile]:
    """Get all available profiles (for testing/debugging)"""
    profiles = [p for p in USER_STORE.values() if isinstance(p, Profile)]
    return profiles

def _generate_insights(history: List[Dict], preferences: Dict) -> str:
    """Generate insights about user preferences based on interaction history"""
    if not history:
        return "No interaction data available"
    
    # Analyze recent behavior
    recent_history = history[-10:] if len(history) >= 10 else history
    recent_likes = [entry['profile_id'] for entry in recent_history if entry['liked']]
    
    if not recent_likes:
        return "User has been selective with recent profiles"
    
    # Count likes by profile attributes
    like_counts = {}
    for profile_id in recent_likes:
        profile = USER_STORE.get(str(profile_id))
        if isinstance(profile, Profile):
            if profile.university not in like_counts:
                like_counts[profile.university] = 0
            like_counts[profile.university] += 1
    
    if like_counts:
        top_university = max(like_counts.items(), key=lambda x: x[1])
        return f"User shows preference for {top_university[0]} profiles"
    
    return "User preferences are still being learned"