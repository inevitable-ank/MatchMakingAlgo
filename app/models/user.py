from pydantic import BaseModel
from typing import List, Tuple, Set, Dict, Optional
import numpy as np
from collections import Counter, defaultdict

class Feedback(BaseModel):
    viewer_id: str
    profile_id: int
    liked: bool
    timestamp: Optional[float] = None

class Viewer(BaseModel):
    id: str
    age: int
    gender: str
    university: str
    degree: str
    year: str
    city: str
    bio: str = ""
    interests: Set[str]
    personality_prompts: Dict[str, str] = {}
    # track history of feedback: list of (profile_id, liked)
    behavior: List[Tuple[int, bool]] = []
    
    def feature_vector(self) -> List[float]:
        """
        Convert viewer to feature vector for similarity computation.
        Similar to Profile.feature_vector() but for viewers.
        """
        # Academic features (one-hot encoded)
        universities = ["IIT Delhi", "IIT Bombay", "IIT Madras", "IIT Kharagpur", 
                       "IIT Guwahati", "IIT Roorkee", "IIT Hyderabad", "BITS Pilani",
                       "IIM Ahmedabad", "IISc Bangalore", "NIT Trichy", "Anna University",
                       "Savitribai Phule Pune University", "Delhi University"]
        degrees = ["Computer Science", "Mechanical Engineering", "Economics & Finance", 
                  "MBA", "Electrical Engineering", "Civil Engineering", "Chemical Engineering",
                  "Environmental Engineering", "Data Science", "Astronomy", 
                  "Electronics and Communication", "Information Technology", "Psychology", "English Literature"]
        years = ["1st Year", "2nd Year", "3rd Year", "4th Year"]
        
        # Geographic features
        cities = ["New Delhi", "Mumbai", "Bangalore", "Ahmedabad", "Chennai", 
                 "Kolkata", "Guwahati", "Roorkee", "Hyderabad", "Tiruchirappalli", "Pune"]
        
        # Gender features (one-hot encoded)
        genders = ["Male", "Female", "Non-binary", "Prefer not to say"]
        
        # Interest categories for better matching
        interest_categories = {
            "tech": ["Machine Learning", "Programming", "Data Viz", "Electronics", "Coding", "Robotics"],
            "creative": ["Photography", "Art", "Poetry", "Writing", "Music", "Theatre", "Calligraphy"],
            "outdoor": ["Hiking", "Trekking", "Travel", "Biking", "Sketching"],
            "social": ["Cooking", "Food", "Gaming", "Chess", "Table Tennis", "Cricket"],
            "business": ["Entrepreneurship", "Strategy", "Finance", "Startups", "Social Impact"],
            "academic": ["Chemistry", "Botany", "Literature", "Psychology", "Astronomy"],
            "lifestyle": ["Sustainability", "Volunteering", "Yoga", "Dancing", "Books", "Reading"]
        }
        
        features = []
        
        # Academic features (one-hot encoding)
        for uni in universities:
            features.append(1.0 if self.university == uni else 0.0)
        for deg in degrees:
            features.append(1.0 if self.degree == deg else 0.0)
        for yr in years:
            features.append(1.0 if self.year == yr else 0.0)
            
        # Geographic features
        for city in cities:
            features.append(1.0 if self.city == city else 0.0)
            
        # Gender features (one-hot encoding)
        for gender in genders:
            features.append(1.0 if self.gender == gender else 0.0)
            
        # Age feature (normalized)
        features.append((self.age - 18) / 10.0)  # Normalize age 18-28 to 0-1
        
        # Interest category features
        for category, category_interests in interest_categories.items():
            overlap = len(self.interests.intersection(set(category_interests)))
            features.append(min(overlap / 3.0, 1.0))  # Normalize to 0-1
            
        # Bio features
        bio_length = len(self.bio) / 200.0  # Normalize bio length
        features.append(min(bio_length, 1.0))
        
        # Personality prompts features (simplified - count of prompts answered)
        prompt_count = len(self.personality_prompts) / 5.0  # Normalize assuming max 5 prompts
        features.append(min(prompt_count, 1.0))
        
        return features
    
    def get_preference_vector(self) -> Dict[str, float]:
        """
        Extract user preferences from behavioral data.
        Returns a dictionary of preference scores for different profile attributes.
        """
        if not self.behavior:
            return {}
        
        # Count likes/dislikes for different attributes
        preferences = defaultdict(lambda: {"likes": 0, "dislikes": 0})
        
        # This would need to be implemented with actual profile data
        # For now, return empty dict
        return {}
    
    def get_engagement_score(self) -> float:
        """
        Calculate user engagement score based on activity patterns.
        Higher score = more engaged user.
        """
        if not self.behavior:
            return 0.5  # Default neutral score
        
        # Calculate engagement based on:
        # 1. Number of interactions
        # 2. Consistency of activity
        # 3. Response patterns
        
        total_interactions = len(self.behavior)
        like_ratio = sum(1 for _, liked in self.behavior if liked) / total_interactions
        
        # Simple engagement score (0-1)
        engagement = min(total_interactions / 20.0, 1.0) * 0.7 + like_ratio * 0.3
        return engagement