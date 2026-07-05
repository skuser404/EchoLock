/**
 * EchoLock: JavaScript for Typing Pattern Capture
 * Handles keystroke dynamics, API communication, and UI updates
 */

// Global state
let currentTypingData = [];  // Stores current typing session
let typingSamples = [];      // Stores all samples for registration
let isRecording = false;     // Recording state

/**
 * Initialize typing capture on an input element
 * @param {string} inputId - ID of the input element
 * @param {Function} onComplete - Callback when typing is complete
 */
function initTypingCapture(inputId, onComplete) {
    const input = document.getElementById(inputId);
    if (!input) return;

    // Clear previous data
    currentTypingData = [];
    isRecording = true;

    // Visual feedback
    input.classList.add('recording');
    updateTypingVisualizer('Start typing your PIN...');

    // Keydown event - when key is pressed
    input.addEventListener('keydown', function(e) {
        if (!isRecording) return;

        // Prevent recording of modifier keys alone
        if (['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Tab'].includes(e.key)) {
            return;
        }

        const timestamp = Date.now();

        // Find if we already have a keydown for this key without keyup
        const existingIndex = currentTypingData.findIndex(
            item => item.key === e.key && !item.keyup
        );

        if (existingIndex === -1) {
            currentTypingData.push({
                key: e.key,
                keydown: timestamp,
                keyCode: e.code
            });
        }

        updateTypingVisualizer('Recording keystrokes... (' + currentTypingData.length + ' keys)');
    });

    // Keyup event - when key is released
    input.addEventListener('keyup', function(e) {
        if (!isRecording) return;

        const timestamp = Date.now();

        // Find the matching keydown event
        const index = currentTypingData.findIndex(
            item => item.key === e.key && !item.keyup
        );

        if (index !== -1) {
            currentTypingData[index].keyup = timestamp;
        }

        // Check if input is complete (Enter key or max length)
        if (e.key === 'Enter' || input.value.length >= input.maxLength) {
            setTimeout(() => {
                isRecording = false;
                input.classList.remove('recording');
                if (onComplete) {
                    onComplete(currentTypingData);
                }
            }, 100);
        }
    });

    // Handle input completion on blur
    input.addEventListener('blur', function() {
        if (isRecording && currentTypingData.length > 0) {
            isRecording = false;
            input.classList.remove('recording');
            if (onComplete) {
                onComplete(currentTypingData);
            }
        }
    });
}

/**
 * Update the typing visualizer display
 */
function updateTypingVisualizer(text) {
    const visualizer = document.getElementById('typingVisualizer');
    if (visualizer) {
        visualizer.textContent = text;
        visualizer.classList.add('active');
        setTimeout(() => visualizer.classList.remove('active'), 300);
    }
}

/**
 * Reset typing data
 */
function resetTypingData() {
    currentTypingData = [];
    isRecording = false;
    const input = document.getElementById('pinInput');
    if (input) {
        input.value = '';
        input.classList.remove('recording');
    }
}

/**
 * Show status message
 */
function showStatus(message, type = 'info') {
    const statusDiv = document.getElementById('statusMessage');
    if (!statusDiv) return;

    statusDiv.className = 'status-message status-' + type;
    statusDiv.innerHTML = message;
    statusDiv.style.display = 'flex';

    // Auto-hide after 5 seconds for success messages
    if (type === 'success') {
        setTimeout(() => {
            statusDiv.style.display = 'none';
        }, 5000);
    }
}

/**
 * Update progress bar
 */
function updateProgress(current, total) {
    const progressFill = document.getElementById('progressFill');
    const progressText = document.getElementById('progressText');

    if (progressFill) {
        const percentage = (current / total) * 100;
        progressFill.style.width = percentage + '%';
    }

    if (progressText) {
        progressText.textContent = `Sample ${current} of ${total}`;
    }

    // Update sample dots
    const dots = document.querySelectorAll('.sample-dot');
    dots.forEach((dot, index) => {
        dot.classList.remove('completed', 'active');
        if (index < current) {
            dot.classList.add('completed');
        } else if (index === current) {
            dot.classList.add('active');
        }
    });
}

/**
 * Register a new user with typing patterns
 */
async function registerUser() {
    const username = document.getElementById('username').value.trim();
    const pin = document.getElementById('pinInput').value.trim();

    if (!username || !pin) {
        showStatus('❌ Please enter both username and PIN', 'error');
        return;
    }

    if (typingSamples.length < 3) {
        showStatus(`❌ Please complete at least 3 typing samples. Current: ${typingSamples.length}`, 'error');
        return;
    }

    showStatus('⏳ Registering user...', 'warning');

    try {
        const response = await fetch('/api/register', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                pin: pin,
                typing_samples: typingSamples
            })
        });

        const data = await response.json();

        if (data.success) {
            showStatus(`✅ ${data.message}! You can now login.`, 'success');
            setTimeout(() => {
                window.location.href = '/login';
            }, 2000);
        } else {
            showStatus(`❌ ${data.error}`, 'error');
        }
    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Authenticate user with typing pattern
 */
async function authenticateUser() {
    const username = document.getElementById('username').value.trim();
    const pin = document.getElementById('pinInput').value.trim();

    if (!username || !pin) {
        showStatus('❌ Please enter both username and PIN', 'error');
        return;
    }

    if (currentTypingData.length === 0) {
        showStatus('❌ Please type your PIN first', 'error');
        return;
    }

    showStatus('⏳ Analyzing typing pattern...', 'warning');

    try {
        const response = await fetch('/api/authenticate', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: username,
                pin: pin,
                typing_data: currentTypingData
            })
        });

        const data = await response.json();
        displayAuthResult(data);

    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Display authentication result
 */
function displayAuthResult(data) {
    const resultDiv = document.getElementById('authResult');
    const resultContainer = document.getElementById('resultContainer');

    if (!resultDiv || !resultContainer) return;

    resultDiv.classList.remove('hidden');

    let icon, titleClass, message;

    if (data.decision === 'Access Granted') {
        icon = '🔓';
        titleClass = 'score-high';
        message = 'Access Granted!';
    } else if (data.decision === 'OTP Required') {
        icon = '🔐';
        titleClass = 'score-medium';
        message = 'Additional Verification Required';
    } else {
        icon = '🔒';
        titleClass = 'score-low';
        message = 'Access Denied';
    }

    let html = `
        <div class="result-container">
            <div class="result-icon">${icon}</div>
            <div class="result-title ${titleClass}">${message}</div>
            <div class="result-score ${titleClass}">${data.similarity}%</div>
            <p>Similarity Score</p>
    `;

    // Show typing stats if available
    if (data.typing_stats) {
        html += `
            <div class="stats-grid">
                <div class="stat-box">
                    <div class="stat-label">Keystrokes</div>
                    <div class="stat-value">${data.typing_stats.keystroke_count}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Avg Dwell</div>
                    <div class="stat-value">${data.typing_stats.avg_dwell_time}ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Avg Flight</div>
                    <div class="stat-value">${data.typing_stats.avg_flight_time}ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Speed</div>
                    <div class="stat-value">${data.typing_stats.typing_speed_wpm} WPM</div>
                </div>
            </div>
        `;
    }

    // Show OTP section if required
    if (data.requires_otp && data.otp) {
        html += `
            <div class="otp-display">
                <div class="otp-label">Your One-Time Password (Demo)</div>
                <div class="otp-code">${data.otp}</div>
            </div>
            <div class="form-group">
                <label>Enter OTP to complete authentication:</label>
                <input type="text" id="otpInput" class="pin-input" maxlength="4" placeholder="0000">
            </div>
            <button class="btn btn-primary" onclick="verifyOTP()">Verify OTP</button>
        `;

        // Store username for OTP verification
        window.currentUser = document.getElementById('username').value.trim();
    }

    html += '</div>';
    resultContainer.innerHTML = html;

    // Update status
    if (data.decision === 'Access Granted') {
        showStatus('✅ Authentication successful!', 'success');
    } else if (data.decision === 'OTP Required') {
        showStatus('⚠️ Please verify with OTP', 'warning');
    } else {
        showStatus('❌ Authentication failed', 'error');
    }
}

/**
 * Verify OTP
 */
async function verifyOTP() {
    const otpInput = document.getElementById('otpInput').value.trim();

    if (!otpInput || otpInput.length !== 4) {
        showStatus('❌ Please enter the 4-digit OTP', 'error');
        return;
    }

    try {
        const response = await fetch('/api/verify-otp', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({
                username: window.currentUser,
                otp: otpInput
            })
        });

        const data = await response.json();

        if (data.success) {
            showStatus('✅ OTP verified! Access granted.', 'success');
            document.getElementById('resultContainer').innerHTML = `
                <div class="result-container">
                    <div class="result-icon">🔓</div>
                    <div class="result-title score-high">Access Granted!</div>
                    <p>OTP verification successful. Welcome!</p>
                </div>
            `;
        } else {
            showStatus('❌ Invalid OTP. Access denied.', 'error');
        }
    } catch (error) {
        showStatus(`❌ Error: ${error.message}`, 'error');
    }
}

/**
 * Collect typing sample for registration
 */
function collectSample() {
    const pinInput = document.getElementById('pinInput');
    const expectedPin = document.getElementById('expectedPin').value;

    if (!pinInput.value) {
        showStatus('❌ Please type the PIN first', 'error');
        return;
    }

    if (pinInput.value !== expectedPin) {
        showStatus('❌ PIN does not match. Please type the same PIN.', 'error');
        pinInput.value = '';
        currentTypingData = [];
        return;
    }

    // Add sample
    typingSamples.push([...currentTypingData]);

    // Update progress
    updateProgress(typingSamples.length, 5);

    // Clear input for next sample
    pinInput.value = '';
    currentTypingData = [];

    showStatus(`✅ Sample ${typingSamples.length} collected!`, 'success');

    // Check if we have enough samples
    if (typingSamples.length >= 5) {
        document.getElementById('submitBtn').classList.remove('hidden');
        showStatus('✅ All samples collected! You can now register.', 'success');
    }
}

/**
 * Initialize registration page
 */
function initRegistration() {
    typingSamples = [];
    updateProgress(0, 5);

    // Initialize typing capture
    initTypingCapture('pinInput', function(data) {
        console.log('Typing captured:', data);
    });
}

/**
 * Initialize login page
 */
function initLogin() {
    currentTypingData = [];

    // Initialize typing capture
    initTypingCapture('pinInput', function(data) {
        console.log('Typing captured:', data);
    });
}

// Auto-initialize based on page
document.addEventListener('DOMContentLoaded', function() {
    if (document.getElementById('registerForm')) {
        initRegistration();
    } else if (document.getElementById('loginForm')) {
        initLogin();
    }
});
