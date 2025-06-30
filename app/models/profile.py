from pydantic import BaseModel
from typing import List, Set, Dict, Any, Optional
import numpy as np
from collections import Counter

class Profile(BaseModel):
    id: int
    name: str
    age: int
    gender: str
    university: str
    degree: str
    year: str
    city: str
    bio: str
    interests: Set[str]
    images: List[str]
    personality_prompts: Dict[str, str] = {}
    compatibility: float
    
    def feature_vector(self) -> List[float]:
        """
        Convert profile to feature vector for similarity computation.
        Returns normalized feature vector with:
        - Academic features (university, degree, year)
        - Geographic features (city)
        - Personal features (age, gender, interests)
        - Content features (bio sentiment, length, personality prompts)
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
    
    def get_academic_similarity(self, other: 'Profile') -> float:
        """Calculate academic similarity score"""
        score = 0.0
        if self.university == other.university:
            score += 0.4
        if self.degree == other.degree:
            score += 0.3
        if self.year == other.year:
            score += 0.2
        if self.city == other.city:
            score += 0.1
        return score
    
    def get_interest_overlap(self, other: 'Profile') -> float:
        """Calculate interest overlap using Jaccard similarity"""
        if not self.interests or not other.interests:
            return 0.0
        intersection = len(self.interests.intersection(other.interests))
        union = len(self.interests.union(other.interests))
        return intersection / union if union > 0 else 0.0
    
    def get_personality_similarity(self, other: 'Profile') -> float:
        """Calculate personality similarity based on prompt responses"""
        if not self.personality_prompts or not other.personality_prompts:
            return 0.0
        
        common_prompts = set(self.personality_prompts.keys()) & set(other.personality_prompts.keys())
        if not common_prompts:
            return 0.0
        
        similarities = []
        for prompt in common_prompts:
            # Simple text similarity for prompt responses
            response1 = self.personality_prompts[prompt].lower()
            response2 = other.personality_prompts[prompt].lower()
            
            # Calculate word overlap
            words1 = set(response1.split())
            words2 = set(response2.split())
            if words1 and words2:
                overlap = len(words1 & words2) / len(words1 | words2)
                similarities.append(overlap)
        
        return sum(similarities) / len(similarities) if similarities else 0.0