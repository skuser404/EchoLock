"""
EchoLock: ML Model for Behavioral Biometric Authentication
This module handles the machine learning logic for typing pattern analysis.
"""

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import json
import os

class TypingPatternModel:
    """
    KNN-based classifier for typing pattern authentication.
    Uses dwell time (key press duration) and flight time (time between keys)
    to create a behavioral biometric profile.
    """

    def __init__(self):
        self.model = KNeighborsClassifier(n_neighbors=3)
        self.scaler = StandardScaler()
        self.is_trained = False
        self.user_profiles = {}  # Store user typing patterns

    def extract_features(self, typing_data):
        """
        Extract features from raw typing data.

        Features extracted:
        - Dwell times: How long each key is held down
        - Flight times: Time between releasing one key and pressing the next
        - Average typing speed
        - Consistency metrics

        Args:
            typing_data: List of dicts with 'key', 'keydown', 'keyup' timestamps

        Returns:
            numpy array of features
        """
        if len(typing_data) < 2:
            return None

        # Calculate dwell times (key press duration)
        dwell_times = []
        for event in typing_data:
            if 'keydown' in event and 'keyup' in event:
                dwell_times.append(event['keyup'] - event['keydown'])

        # Calculate flight times (time between key release and next key press)
        flight_times = []
        for i in range(len(typing_data) - 1):
            if 'keyup' in typing_data[i] and 'keydown' in typing_data[i + 1]:
                flight_time = typing_data[i + 1]['keydown'] - typing_data[i]['keyup']
                flight_times.append(flight_time)

        if not dwell_times or not flight_times:
            return None

        # Create feature vector
        features = [
            np.mean(dwell_times),           # Average dwell time
            np.std(dwell_times),            # Dwell time consistency
            np.mean(flight_times),          # Average flight time
            np.std(flight_times),           # Flight time consistency
            len(typing_data),               # Number of keystrokes
            np.max(dwell_times),            # Max dwell time
            np.min(dwell_times),            # Min dwell time
            np.max(flight_times),           # Max flight time
            np.min(flight_times),           # Min flight time
        ]

        return np.array(features)

    def register_user(self, username, typing_samples):
        """
        Register a new user with multiple typing samples.

        Args:
            username: Unique identifier for the user
            typing_samples: List of typing pattern data (5 samples from registration)

        Returns:
            dict: User profile with average features
        """
        features_list = []

        for sample in typing_samples:
            features = self.extract_features(sample)
            if features is not None:
                features_list.append(features)

        if not features_list:
            return None

        # Create user profile with average features and individual samples
        profile = {
            'username': username,
            'average_features': np.mean(features_list, axis=0).tolist(),
            'samples': [f.tolist() for f in features_list],
            'std_dev': np.std(features_list, axis=0).tolist(),
            'sample_count': len(features_list)
        }

        self.user_profiles[username] = profile
        return profile

    def calculate_similarity(self, input_features, username):
        """
        Calculate similarity between input typing pattern and stored profile.

        Uses weighted Euclidean distance normalized to a similarity score (0-100).

        Args:
            input_features: numpy array of input features
            username: User to compare against

        Returns:
            float: Similarity score (0-100, higher is more similar)
        """
        if username not in self.user_profiles:
            return 0.0

        profile = self.user_profiles[username]
        stored_features = np.array(profile['average_features'])

        # Calculate Euclidean distance
        distance = np.sqrt(np.sum((input_features - stored_features) ** 2))

        # Convert distance to similarity score (0-100)
        # Using exponential decay: similarity = 100 * exp(-distance/scale)
        scale = 500  # Tuning parameter - adjust based on your data
        similarity = 100 * np.exp(-distance / scale)

        return similarity

    def authenticate(self, username, typing_data, threshold_high=80, threshold_low=50):
        """
        Authenticate user based on typing pattern.

        Decision logic:
        - Similarity >= threshold_high: Access Granted
        - threshold_low <= Similarity < threshold_high: OTP Required
        - Similarity < threshold_low: Access Denied

        Args:
            username: User attempting to authenticate
            typing_data: Raw typing pattern data
            threshold_high: High confidence threshold (default 80)
            threshold_low: Low confidence threshold (default 50)

        Returns:
            dict: Authentication result with decision and score
        """
        input_features = self.extract_features(typing_data)

        if input_features is None:
            return {
                'success': False,
                'decision': 'Access Denied',
                'similarity': 0.0,
                'reason': 'Invalid typing data'
            }

        similarity = self.calculate_similarity(input_features, username)

        # Decision logic
        if similarity >= threshold_high:
            decision = 'Access Granted'
            success = True
            requires_otp = False
        elif similarity >= threshold_low:
            decision = 'OTP Required'
            success = True
            requires_otp = True
        else:
            decision = 'Access Denied'
            success = False
            requires_otp = False

        return {
            'success': success,
            'decision': decision,
            'similarity': round(similarity, 2),
            'requires_otp': requires_otp,
            'threshold_high': threshold_high,
            'threshold_low': threshold_low
        }

    def save_profiles(self, filepath='database.json'):
        """Save user profiles to JSON file."""
        data = {'users': self.user_profiles}
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load_profiles(self, filepath='database.json'):
        """Load user profiles from JSON file."""
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
                self.user_profiles = data.get('users', {})


# Simple OTP generator for demo purposes
class OTPGenerator:
    """Generate and verify One-Time Passwords for additional authentication."""

    def __init__(self):
        self.active_otps = {}  # Store active OTPs: {username: otp}

    def generate_otp(self, username, length=4):
        """Generate a random numeric OTP."""
        import random
        otp = ''.join([str(random.randint(0, 9)) for _ in range(length)])
        self.active_otps[username] = otp
        return otp

    def verify_otp(self, username, otp_input):
        """Verify if the input OTP matches the generated one."""
        if username in self.active_otps:
            return self.active_otps[username] == otp_input
        return False

    def clear_otp(self, username):
        """Clear OTP for a user after use or timeout."""
        if username in self.active_otps:
            del self.active_otps[username]


# Utility functions
def analyze_typing_pattern(typing_data):
    """
    Analyze typing pattern and return statistics for display.

    Args:
        typing_data: List of typing events

    Returns:
        dict: Statistics about the typing pattern
    """
    if not typing_data:
        return {}

    dwell_times = []
    flight_times = []

    for event in typing_data:
        if 'keydown' in event and 'keyup' in event:
            dwell_times.append(event['keyup'] - event['keydown'])

    for i in range(len(typing_data) - 1):
        if 'keyup' in typing_data[i] and 'keydown' in typing_data[i + 1]:
            flight_times.append(typing_data[i + 1]['keydown'] - typing_data[i]['keyup'])

    total_time = 0
    if typing_data and 'keydown' in typing_data[0] and 'keyup' in typing_data[-1]:
        total_time = typing_data[-1]['keyup'] - typing_data[0]['keydown']

    return {
        'keystroke_count': len(typing_data),
        'avg_dwell_time': round(np.mean(dwell_times), 2) if dwell_times else 0,
        'avg_flight_time': round(np.mean(flight_times), 2) if flight_times else 0,
        'total_time_ms': round(total_time, 2),
        'typing_speed_wpm': round((len(typing_data) / 5) / (total_time / 60000), 2) if total_time > 0 else 0
    }
