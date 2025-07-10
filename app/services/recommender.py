import random
import time
import math
from typing import Optional, List, Dict, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from collections import defaultdict, Counter

from app.models.profile import Profile
from app.models.user import Viewer
from app.db.database import USER_STORE, BANDIT_STATS
from app.utils.matcher import cosine_similarity, jaccard_similarity

# Algorithm parameters
EPSILON = 0.15  # Exploration probability
ALPHA = 0.1     # Learning rate for UCB
BETA = 0.05     # Decay factor for exploration
MIN_INTERACTIONS = 5  # Minimum interactions before using learned preferences

class AdvancedProfileRecommender:
    """
    Advanced Profile Discovery Engine implementing:
    1. Multi-armed bandit with UCB (Upper Confidence Bound)
    2. Progressive filtering (familiar vs diverse)
    3. Learning from user behavior
    4. Engagement optimization
    """
    
    def __init__(self):
        self.profile_cache = {}  # Cache for computed feature vectors
        self.similarity_cache = {}  # Cache for similarity scores
        
    def recommend_next_profile(self, viewer_id: str) -> Optional[Profile]:
        """
        Main recommendation function implementing the complete algorithm.
        """
        viewer_raw = USER_STORE.get(viewer_id)
        if viewer_raw is None or not isinstance(viewer_raw, Viewer):
            raise ValueError(f"Viewer {viewer_id} not found")
        
        # Type assertion to help IDE understand the type
        viewer = viewer_raw  # type: ignore
        assert isinstance(viewer, Viewer)
            
        # Get viewer's interaction history
        stats = BANDIT_STATS.get(viewer_id, {'history': [], 'profile_scores': {}})
        history = stats['history']
        seen_ids = {entry['profile_id'] for entry in history}
        
        # Get candidate profiles
        candidates = self._get_candidates(viewer_id, seen_ids)
        if not candidates:
            return None
            
        # Determine exploration vs exploitation strategy
        exploration_strategy = self._get_exploration_strategy(viewer_id, len(history))
        
        if exploration_strategy == "random":
            return random.choice(candidates)
        elif exploration_strategy == "diverse":
            return self._select_diverse_profile(viewer, candidates, history)
        else:  # exploitation
            return self._select_optimal_profile(viewer, candidates, history, stats)
    
    def _get_candidates(self, viewer_id: str, seen_ids: set) -> List[Profile]:
        """Get candidate profiles excluding seen ones"""
        return [
            p for p in USER_STORE.values()
            if isinstance(p, Profile) and p.id not in seen_ids
        ]
    
    def _get_exploration_strategy(self, viewer_id: str, interaction_count: int) -> str:
        """
        Determine exploration strategy based on user engagement and interaction count.
        Returns: "random", "diverse", or "exploit"
        """
        # Early stage: more exploration
        if interaction_count < 10:
            if random.random() < EPSILON:
                return "random"
            elif random.random() < 0.3:
                return "diverse"
            else:
                return "exploit"
        
        # Mid stage: balanced exploration
        elif interaction_count < 30:
            if random.random() < EPSILON * 0.7:
                return "random"
            elif random.random() < 0.2:
                return "diverse"
            else:
                return "exploit"
        
        # Late stage: mostly exploitation with occasional exploration
        else:
            if random.random() < EPSILON * 0.3:
                return "random"
            elif random.random() < 0.1:
                return "diverse"
            else:
                return "exploit"
    
    def _select_diverse_profile(self, viewer: Viewer, candidates: List[Profile], 
                              history: List[Dict]) -> Profile:
        """
        Select a diverse profile to prevent filter bubbles.
        Balances between familiar and diverse backgrounds.
        """
        if not history:
            return random.choice(candidates)
        
        # Get recently seen profile IDs to avoid duplicates
        recent_seen_ids = {entry['profile_id'] for entry in history[-5:]}  # Last 5 interactions
        
        # Filter out recently seen candidates
        available_candidates = [c for c in candidates if c.id not in recent_seen_ids]
        
        # If no candidates available, use all candidates but prefer diverse ones
        if not available_candidates:
            available_candidates = candidates
        
        # Analyze user's recent preferences (liked profiles)
        recent_likes = [entry['profile_id'] for entry in history[-10:] if entry['liked']]
        recent_profiles_raw = [USER_STORE.get(str(pid)) for pid in recent_likes 
                          if str(pid) in USER_STORE and isinstance(USER_STORE[str(pid)], Profile)]
        
        # Filter out None values and ensure type safety
        recent_profiles: List[Profile] = [p for p in recent_profiles_raw if p is not None]
        
        if not recent_profiles:
            return random.choice(available_candidates)
        
        # Calculate diversity scores
        diversity_scores = []
        for candidate in available_candidates:
            diversity_score = self._calculate_diversity_score(candidate, recent_profiles)
            diversity_scores.append((candidate, diversity_score))
        
        # Select candidate with highest diversity score
        diversity_scores.sort(key=lambda x: x[1], reverse=True)
        return diversity_scores[0][0]
    
    def _calculate_diversity_score(self, candidate: Profile, recent_profiles: List[Profile]) -> float:
        """
        Calculate diversity score for a candidate profile.
        Higher score = more diverse from recently seen profiles.
        """
        if not recent_profiles:
            return 1.0  # Maximum diversity if no recent profiles
        
        similarities = []
        for recent in recent_profiles:
            # Calculate similarity between two profiles (not profile-viewer)
            sim = self._get_profile_similarity(candidate, recent)
            similarities.append(sim)
        
        avg_similarity = float(np.mean(similarities))  # Convert numpy type to Python float
        # Return diversity score (inverse of similarity)
        return 1.0 - avg_similarity
    
    def _get_profile_similarity(self, profile1: Profile, profile2: Profile) -> float:
        """
        Calculate similarity between two profiles.
        """
        # Academic similarity
        academic_sim = self._get_profile_academic_similarity(profile1, profile2)
        
        # Interest overlap
        interest_sim = self._get_profile_interest_overlap(profile1, profile2)
        
        # Personality similarity
        personality_sim = self._get_profile_personality_similarity(profile1, profile2)
        
        # Weighted combination
        similarity = (academic_sim * 0.3 + 
                     interest_sim * 0.4 + 
                     personality_sim * 0.3)
        
        return float(similarity)
    
    def _get_profile_academic_similarity(self, profile1: Profile, profile2: Profile) -> float:
        """Calculate academic similarity between two profiles"""
        score = 0.0
        if profile1.university == profile2.university:
            score += 0.4
        if profile1.degree == profile2.degree:
            score += 0.3
        if profile1.year == profile2.year:
            score += 0.2
        if profile1.city == profile2.city:
            score += 0.1
        return score
    
    def _get_profile_interest_overlap(self, profile1: Profile, profile2: Profile) -> float:
        """Calculate interest overlap between two profiles"""
        if not profile1.interests or not profile2.interests:
            return 0.0
        intersection = len(profile1.interests.intersection(profile2.interests))
        union = len(profile1.interests.union(profile2.interests))
        return intersection / union if union > 0 else 0.0
    
    def _get_profile_personality_similarity(self, profile1: Profile, profile2: Profile) -> float:
        """Calculate personality similarity between two profiles"""
        if not profile1.personality_prompts or not profile2.personality_prompts:
            return 0.0
        
        common_prompts = set(profile1.personality_prompts.keys()) & set(profile2.personality_prompts.keys())
        if not common_prompts:
            return 0.0
        
        similarities = []
        for prompt in common_prompts:
            # Simple text similarity for prompt responses
            response1 = profile1.personality_prompts[prompt].lower()
            response2 = profile2.personality_prompts[prompt].lower()
            
            # Calculate word overlap
            words1 = set(response1.split())
            words2 = set(response2.split())
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                similarities.append(overlap)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def _select_optimal_profile(self, viewer: Viewer, candidates: List[Profile], 
                              history: List[Dict], stats: Dict) -> Profile:
        """
        Select optimal profile using UCB bandit algorithm with learned preferences.
        """
        if len(history) < MIN_INTERACTIONS:
            # Not enough data, use content-based similarity
            return self._select_by_similarity(viewer, candidates)
        
        # Calculate UCB scores for each candidate
        ucb_scores = []
        for candidate in candidates:
            score = self._calculate_ucb_score(candidate.id, viewer, history, stats)
            ucb_scores.append((candidate, score))
        
        # Return candidate with highest UCB score
        ucb_scores.sort(key=lambda x: x[1], reverse=True)
        return ucb_scores[0][0]
    
    def _calculate_ucb_score(self, profile_id: int, viewer: Viewer, 
                           history: List[Dict], stats: Dict) -> float:
        """
        Calculate UCB (Upper Confidence Bound) score for a profile.
        UCB = exploitation_score + exploration_bonus
        """
        # Get profile's historical performance
        profile_history = [entry for entry in history if entry['profile_id'] == profile_id]
        
        if not profile_history:
            # New profile: high exploration bonus
            return float('inf')
        
        # Calculate exploitation score (average reward)
        likes = sum(1 for entry in profile_history if entry['liked'])
        total_views = len(profile_history)
        exploitation_score = likes / total_views if total_views > 0 else 0.0
        
        # Calculate exploration bonus
        total_interactions = len(history)
        exploration_bonus = math.sqrt(2 * math.log(total_interactions) / total_views)
        
        # Add content-based similarity bonus
        profile = USER_STORE.get(str(profile_id))
        if isinstance(profile, Profile):
            similarity_bonus = self._get_similarity_score(profile, viewer) * 0.2
        else:
            similarity_bonus = 0.0
        
        return exploitation_score + ALPHA * exploration_bonus + similarity_bonus
    
    def _select_by_similarity(self, viewer: Viewer, candidates: List[Profile]) -> Profile:
        """Select profile based on content similarity when no behavioral data exists"""
        best_candidate = None
        best_score = -1.0
        
        for candidate in candidates:
            score = self._get_similarity_score(candidate, viewer)
            if score > best_score:
                best_score = score
                best_candidate = candidate
        
        return best_candidate or random.choice(candidates)
    
    def _get_similarity_score(self, profile: Profile, viewer: Viewer) -> float:
        """
        Calculate similarity score between a profile and viewer.
        """
        # Academic similarity
        academic_sim = self._get_academic_similarity(profile, viewer)
        
        # Interest overlap
        interest_sim = self._get_interest_overlap(profile, viewer)
        
        # Personality similarity
        personality_sim = self._get_personality_similarity(profile, viewer)
        
        # Weighted combination
        similarity = (academic_sim * 0.3 + 
                     interest_sim * 0.4 + 
                     personality_sim * 0.3)
        
        return float(similarity)  # Ensure float type
    
    def _get_academic_similarity(self, profile: Profile, viewer: Viewer) -> float:
        """Calculate academic similarity between profile and viewer"""
        score = 0.0
        if profile.university == viewer.university:
            score += 0.4
        if profile.degree == viewer.degree:
            score += 0.3
        if profile.year == viewer.year:
            score += 0.2
        if profile.city == viewer.city:
            score += 0.1
        return score
    
    def _get_interest_overlap(self, profile: Profile, viewer: Viewer) -> float:
        """Calculate interest overlap between profile and viewer"""
        if not profile.interests or not viewer.interests:
            return 0.0
        intersection = len(profile.interests.intersection(viewer.interests))
        union = len(profile.interests.union(viewer.interests))
        return intersection / union if union > 0 else 0.0
    
    def _get_personality_similarity(self, profile: Profile, viewer: Viewer) -> float:
        """Calculate personality similarity between profile and viewer"""
        if not profile.personality_prompts or not viewer.personality_prompts:
            return 0.0
        
        common_prompts = set(profile.personality_prompts.keys()) & set(viewer.personality_prompts.keys())
        if not common_prompts:
            return 0.0
        
        similarities = []
        for prompt in common_prompts:
            # Simple text similarity for prompt responses
            response1 = profile.personality_prompts[prompt].lower()
            response2 = viewer.personality_prompts[prompt].lower()
            
            # Calculate word overlap
            words1 = set(response1.split())
            words2 = set(response2.split())
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                similarities.append(overlap)
        
        return sum(similarities) / len(similarities) if similarities else 0.0
    
    def record_feedback(self, viewer_id: str, profile_id: int, liked: bool) -> None:
        """
        Record user feedback and update learning models.
        """
        timestamp = time.time()
        
        # Update bandit statistics
        stats = BANDIT_STATS.setdefault(viewer_id, {
            'history': [],
            'profile_scores': {},
            'preference_model': {},
            'engagement_metrics': {
                'total_interactions': 0,
                'like_ratio': 0.0,
                'last_activity': timestamp
            }
        })
        
        # Record interaction
        stats['history'].append({
            'profile_id': profile_id,
            'liked': liked,
            'timestamp': timestamp
        })
        
        # Update engagement metrics
        engagement = stats['engagement_metrics']
        engagement['total_interactions'] += 1
        engagement['last_activity'] = timestamp
        
        # Update like ratio
        total_likes = sum(1 for entry in stats['history'] if entry['liked'])
        engagement['like_ratio'] = total_likes / engagement['total_interactions']
        
        # Update profile-specific scores
        profile_key = str(profile_id)
        if profile_key not in stats['profile_scores']:
            stats['profile_scores'][profile_key] = {'likes': 0, 'dislikes': 0}
        
        if liked:
            stats['profile_scores'][profile_key]['likes'] += 1
        else:
            stats['profile_scores'][profile_key]['dislikes'] += 1
        
        # Update preference model (simplified)
        self._update_preference_model(viewer_id, profile_id, liked)
    
    def _update_preference_model(self, viewer_id: str, profile_id: int, liked: bool) -> None:
        """
        Update user preference model based on feedback.
        This is a simplified version - in production, you'd use more sophisticated ML.
        """
        stats = BANDIT_STATS[viewer_id]
        profile = USER_STORE.get(str(profile_id))
        
        if not isinstance(profile, Profile):
            return
        
        # Extract profile attributes and update preferences
        attributes = {
            'university': profile.university,
            'degree': profile.degree,
            'city': profile.city,
            'age_group': f"{(profile.age // 5) * 5}-{(profile.age // 5) * 5 + 4}"
        }
        
        for attr, value in attributes.items():
            if attr not in stats['preference_model']:
                stats['preference_model'][attr] = {}
            if value not in stats['preference_model'][attr]:
                stats['preference_model'][attr][value] = {'likes': 0, 'dislikes': 0}
            
            if liked:
                stats['preference_model'][attr][value]['likes'] += 1
            else:
                stats['preference_model'][attr][value]['dislikes'] += 1

# Global recommender instance
recommender = AdvancedProfileRecommender()

def recommend_next_profile(viewer_id: str) -> Optional[Profile]:
    """Wrapper function for the advanced recommender"""
    return recommender.recommend_next_profile(viewer_id)

def record_feedback(viewer_id: str, profile_id: int, liked: bool) -> None:
    """Wrapper function for recording feedback"""
    recommender.record_feedback(viewer_id, profile_id, liked)