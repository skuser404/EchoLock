# 🔐 EchoLock: AI-Based Behavioral Biometric Authentication System

EchoLock is a complete, working demonstration of behavioral biometric authentication using typing pattern analysis. The system uses machine learning (KNN classifier) to analyze keystroke dynamics including dwell time and flight time to verify user identity.

## 📋 Features

- **User Registration**: Create a biometric profile by typing your PIN 5 times
- **Behavioral Analysis**: Captures 9 different typing metrics including:
  - Average dwell time (how long keys are held)
  - Average flight time (time between key releases and presses)
  - Typing consistency (standard deviation)
  - Typing speed (WPM)
- **Smart Authentication**: Three-tier decision system:
  - 🟢 **Access Granted** (Similarity ≥ 80%)
  - 🟡 **OTP Required** (Similarity 50-80%)
  - 🔴 **Access Denied** (Similarity < 50%)
- **Demo OTP System**: Built-in OTP verification for medium-confidence matches
- **Real-time Visualization**: Live typing pattern capture display

## 🛠️ Tech Stack

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Python, Flask
- **Machine Learning**: Scikit-learn (KNN Classifier), NumPy
- **Data Storage**: JSON file-based database

## 📁 Project Structure

```
EchoLock/
│
├── app.py                 # Flask backend API
├── model.py               # ML logic for typing pattern analysis
├── database.json          # User profiles storage
├── requirements.txt       # Python dependencies
│
├── templates/
│   ├── index.html        # Home page
│   ├── register.html     # User registration
│   └── login.html        # Authentication page
│
└── static/
    ├── style.css         # Modern UI styling
    └── script.js         # Typing capture & UI logic
```

## 🚀 Installation & Setup

### Step 1: Install Python Dependencies

```bash
# Navigate to project directory
cd EchoLock

# Install required packages
pip install -r requirements.txt
```

### Step 2: Run the Flask Application

```bash
# Start the server
python app.py
```

You should see:
```
============================================================
EchoLock: AI-Based Behavioral Biometric Authentication
============================================================
Starting server on http://127.0.0.1:5000
============================================================
```

### Step 3: Open in Browser

Navigate to: **http://127.0.0.1:5000**

## 📖 Usage Guide

### 1. Register a New User

1. Click "Register New User"
2. Enter a username
3. Choose a 4-6 digit PIN
4. Type your PIN **5 times** at your natural speed
   - Type consistently - the system learns your rhythm
   - Don't rush or change your typing style
5. Click "Complete Registration"

### 2. Authenticate

1. Click "Login"
2. Enter your username
3. Type your PIN naturally
4. Click "Authenticate"
5. View the result:
   - **Green (80%+ match)**: Immediate access granted
   - **Yellow (50-80% match)**: OTP verification required
   - **Red (<50% match)**: Access denied

### 3. Testing Different Scenarios

**To see "Access Granted":**
- Login immediately after registration
- Type at the same speed and rhythm

**To see "OTP Required":**
- Wait a few minutes before logging in
- Type slightly faster or slower
- Use a different keyboard

**To see "Access Denied":**
- Ask a friend to type your PIN
- Type extremely fast or slow
- Use only one hand

## 🔬 How It Works

### Typing Features Extracted

1. **Dwell Time**: How long each key is pressed
2. **Flight Time**: Time between releasing one key and pressing the next
3. **Consistency**: Standard deviation of timing patterns
4. **Speed**: Words per minute calculation
5. **Pattern**: Min/max timing values

### ML Algorithm

- **Algorithm**: K-Nearest Neighbors (KNN)
- **Similarity Metric**: Weighted Euclidean distance
- **Normalization**: Exponential decay function (0-100 scale)
- **Thresholds**: 
  - High confidence: ≥80%
  - Medium confidence: 50-80%
  - Low confidence: <50%

### Security Model

```
User Input
    ↓
PIN Verification (Knowledge factor)
    ↓
Typing Pattern Analysis (Inherence factor)
    ↓
Similarity Score Calculation
    ↓
Decision:
    ├── High (≥80%) → Access Granted
    ├── Medium (50-80%) → OTP Required
    └── Low (<50%) → Access Denied
```

## 🎯 Demo Script for Presentation

### Introduction (30 seconds)

> "Today I'm demonstrating EchoLock, an AI-powered behavioral biometric authentication system. Unlike traditional passwords that can be stolen or guessed, EchoLock analyzes HOW you type - your unique typing rhythm that acts like a fingerprint."

### Registration Demo (1 minute)

1. Navigate to Register page
2. Enter username: "demo_user"
3. Set PIN: "1234"
4. Type PIN 5 times naturally
5. Explain: "The system captures 9 different metrics from each sample and creates an average profile"

### Successful Authentication (1 minute)

1. Go to Login page
2. Enter same username and PIN
3. Type at normal speed
4. Show result: "Access Granted with 85-95% similarity"
5. Explain: "The AI recognizes my typing pattern and grants immediate access"

### Failed Authentication (1 minute)

1. Ask someone else to try the same PIN
2. OR type with unusual rhythm (very fast/slow)
3. Show result: "Access Denied with <50% similarity"
4. Explain: "Even with the correct PIN, the typing pattern doesn't match"

### OTP Flow (1 minute)

1. Try typing slightly differently (medium speed)
2. Show result: "OTP Required with 60-75% similarity"
3. Display the generated OTP
4. Enter OTP to complete authentication
5. Explain: "Multi-factor authentication adds security when confidence is medium"

### Technical Explanation (1 minute)

> "The system uses scikit-learn's KNN classifier to compare typing patterns. We extract features like dwell time, flight time, and consistency. The similarity score uses Euclidean distance with exponential decay normalization. All data is stored locally in JSON format."

### Conclusion (30 seconds)

> "EchoLock demonstrates how behavioral biometrics can enhance security without additional hardware. It's lightweight, runs entirely on Flask, and showcases practical AI/ML application in cybersecurity."

## 🔧 Customization

### Adjust Sensitivity

Edit `model.py` line 158:
```python
scale = 500  # Increase for more lenient, decrease for stricter
```

### Change Thresholds

Edit `model.py` line 175-176:
```python
threshold_high=80,  # Access granted threshold
threshold_low=50     # OTP required threshold
```

### Modify PIN Length

Edit HTML files, change `maxlength` attribute:
```html
<input maxlength="6">  <!-- Change to desired length -->
```

## 🐛 Troubleshooting

**Issue**: "Module not found" error
- **Solution**: Run `pip install -r requirements.txt`

**Issue**: Port already in use
- **Solution**: Change port in `app.py`: `app.run(port=5001)`

**Issue**: Similarity always low
- **Solution**: Type more consistently during registration. The system needs consistent samples.

**Issue**: Changes not reflecting
- **Solution**: Clear browser cache or hard refresh (Ctrl+F5)

## 📊 Expected Results

| Scenario | Expected Similarity | Result |
|----------|-------------------|--------|
| Same user, same rhythm | 85-100% | Access Granted |
| Same user, tired/different keyboard | 60-80% | OTP Required |
| Different user, same PIN | 20-45% | Access Denied |
| Same user, extremely rushed | 40-60% | Access Denied/OTP |

## 📝 Notes

- This is a **demo system** for educational purposes
- In production, use proper password hashing (bcrypt)
- For real deployment, implement HTTPS and secure session management
- The OTP is displayed on screen for demo purposes only

## 🏆 Hackathon Tips

1. **Practice the demo flow** before presenting
2. **Have a backup user registered** in case of issues
3. **Show both success and failure cases** to demonstrate security
4. **Explain the ML concepts** clearly but concisely
5. **Highlight the real-world applications** (banking, corporate security)

## 📄 License

This project is for educational and demonstration purposes.

---

**Built with ❤️ using Flask, Scikit-learn, and JavaScript**
