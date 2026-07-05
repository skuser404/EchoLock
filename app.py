"""
EchoLock: Flask Backend API
Handles user registration, authentication, and typing pattern analysis.
"""

from flask import Flask, render_template, request, jsonify, session
from model import TypingPatternModel, OTPGenerator, analyze_typing_pattern
import json
import os
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(16)  # Secure secret key for sessions

# Initialize ML model and OTP generator
model = TypingPatternModel()
otp_generator = OTPGenerator()
DATABASE_FILE = 'database.json'

# Load existing profiles on startup
model.load_profiles(DATABASE_FILE)


@app.route('/')
def index():
    """Home page with navigation."""
    return render_template('index.html')


@app.route('/register')
def register_page():
    """Registration page."""
    return render_template('register.html')


@app.route('/login')
def login_page():
    """Login page."""
    return render_template('login.html')


@app.route('/api/register', methods=['POST'])
def register_user():
    """
    API endpoint to register a new user with typing patterns.

    Expected JSON payload:
    {
        "username": "string",
        "pin": "string",
        "typing_samples": [  // 5 samples
            [{"key": "1", "keydown": 123, "keyup": 150}, ...],
            ...
        ]
    }
    """
    try:
        data = request.get_json()
        username = data.get('username')
        pin = data.get('pin')
        typing_samples = data.get('typing_samples', [])

        # Validation
        if not username or not pin:
            return jsonify({'success': False, 'error': 'Username and PIN required'}), 400

        if len(typing_samples) < 3:
            return jsonify({'success': False, 'error': 'At least 3 typing samples required'}), 400

        # Check if user already exists
        if username in model.user_profiles:
            return jsonify({'success': False, 'error': 'Username already exists'}), 409

        # Register user with ML model
        profile = model.register_user(username, typing_samples)

        if profile is None:
            return jsonify({'success': False, 'error': 'Failed to process typing patterns'}), 400

        # Store PIN (in production, use proper password hashing!)
        profile['pin'] = pin

        # Save to database
        model.save_profiles(DATABASE_FILE)

        return jsonify({
            'success': True,
            'message': 'User registered successfully',
            'username': username,
            'samples_collected': len(typing_samples)
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/authenticate', methods=['POST'])
def authenticate_user():
    """
    API endpoint to authenticate a user with typing pattern.

    Expected JSON payload:
    {
        "username": "string",
        "pin": "string",
        "typing_data": [{"key": "1", "keydown": 123, "keyup": 150}, ...]
    }

    Returns:
    {
        "success": true/false,
        "decision": "Access Granted" / "OTP Required" / "Access Denied",
        "similarity": 85.5,
        "requires_otp": true/false
    }
    """
    try:
        data = request.get_json()
        username = data.get('username')
        pin = data.get('pin')
        typing_data = data.get('typing_data', [])

        # Validation
        if not username or not pin:
            return jsonify({'success': False, 'error': 'Username and PIN required'}), 400

        # Check if user exists
        if username not in model.user_profiles:
            return jsonify({'success': False, 'error': 'User not found'}), 404

        # Verify PIN first (basic security layer)
        stored_pin = model.user_profiles[username].get('pin')
        if pin != stored_pin:
            return jsonify({
                'success': False,
                'decision': 'Access Denied',
                'reason': 'Invalid PIN',
                'similarity': 0
            }), 401

        # Analyze typing pattern
        result = model.authenticate(username, typing_data)

        # Generate OTP if required
        if result['requires_otp']:
            otp = otp_generator.generate_otp(username)
            result['otp'] = otp  # In demo, we return OTP to frontend
            result['message'] = 'Typing pattern partially matched. OTP verification required.'

        # Store auth state in session
        session['auth_username'] = username
        session['auth_stage'] = result['decision']

        # Add typing statistics
        result['typing_stats'] = analyze_typing_pattern(typing_data)

        return jsonify(result)

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """
    API endpoint to verify OTP.

    Expected JSON payload:
    {
        "username": "string",
        "otp": "1234"
    }
    """
    try:
        data = request.get_json()
        username = data.get('username')
        otp_input = data.get('otp')

        if not username or not otp_input:
            return jsonify({'success': False, 'error': 'Username and OTP required'}), 400

        is_valid = otp_generator.verify_otp(username, otp_input)

        if is_valid:
            otp_generator.clear_otp(username)
            return jsonify({
                'success': True,
                'decision': 'Access Granted',
                'message': 'OTP verified successfully'
            })
        else:
            return jsonify({
                'success': False,
                'decision': 'Access Denied',
                'error': 'Invalid OTP'
            }), 401

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/api/users', methods=['GET'])
def list_users():
    """API endpoint to list registered users (for demo purposes)."""
    users = list(model.user_profiles.keys())
    return jsonify({'users': users, 'count': len(users)})


@app.route('/api/clear-data', methods=['POST'])
def clear_data():
    """Clear all user data (for testing)."""
    model.user_profiles = {}
    model.save_profiles(DATABASE_FILE)
    return jsonify({'success': True, 'message': 'All data cleared'})


if __name__ == '__main__':
    print("=" * 60)
    print("EchoLock: AI-Based Behavioral Biometric Authentication")
    print("=" * 60)
    print("Starting server on http://127.0.0.1:5000")
    print("=" * 60)
    app.run(debug=True, host='127.0.0.1', port=5000)
