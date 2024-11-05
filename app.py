from flask import Flask, request, jsonify, render_template, session, redirect, url_for, flash
import pyodbc
import base64
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import uuid
import random


app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Ensure to set a secret key for session management

# Configure your database connection here
server = 'DESKTOP-16A7BAU'
database = 'fashion'
username = 'cube_sl'
password = '123'
driver = '{ODBC Driver 17 for SQL Server}'

connection_string = f'DRIVER={driver};SERVER={server};DATABASE={database};UID={username};PWD={password}'

@app.route('/', methods=['GET'])
def welcome():
    # Minimal JSON data with key information
    data = {
        "title": "Welcome to Our Fashion Design Recommendation System",
        "buttons": [
            {"label": "Register", "url": "/register"},
            {"label": "Login", "url": "/user_login"},
            {"label": "Admin Login", "url": "/admin_login"}
        ]
    }
    return jsonify(data)

@app.route('/register', methods=['GET'])
def register():
    # Send minimal JSON data needed for the page
    data = {
        "title": "Enter Your Phone Number",
        "instructions": "Input your phone number to register.",
        "buttons": [
            {"label": "Submit", "action": "/submit"},
            {"label": "Reset", "action": "resetInput()"}
        ],
        "redirect_url": "/gender"
    }
    return jsonify(data)

@app.route('/user_login', methods=['GET', 'POST'])
def user_login():
    if request.method == 'POST':
        data = request.get_json()  # Get JSON data from request
        phone_number = data.get('phone_number')
        gender = data.get('gender')

        if not phone_number or len(phone_number) != 10 or not gender:
            return jsonify({"error": "Invalid phone number or gender!"}), 400

        try:
            conn = pyodbc.connect(connection_string)
            cursor = conn.cursor()

            cursor.execute('SELECT user_id FROM user2 WHERE PhoneNumber = ? AND Gender = ?', 
                           (phone_number, gender))
            result = cursor.fetchone()

            if result:
                user_id = result[0]
                session['user_id'] = user_id

                # Delete any previous videos linked with this user_id
                cursor.execute('DELETE FROM video WHERE user_id = ?', (user_id,))
                conn.commit()

                return jsonify({
                    "message": "Login successful! All existing snapshots have been deleted.",
                    "redirect_url": "/material",
                    "user_id": user_id
                })

            return jsonify({"error": "Invalid phone number or gender!"}), 401

        except Exception as e:
            print(f"Error during login: {str(e)}")
            return jsonify({"error": "An error occurred. Please try again later."}), 500

        finally:
            conn.close()

    # If it's a GET request, return the user ID if logged in
    user_id = session.get('user_id')
    return jsonify({
        "user_id": user_id,
        "message": "User ID found." if user_id else "User ID not found. Please register first."
    })

@app.route('/admin_login', methods=['POST'])
def login():
    data = request.get_json()  # Receive JSON data from the AJAX request
    username = data.get('username')
    password = data.get('password')

    try:
        # Connect to the database and check credentials
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM Admins WHERE username = ? AND password = ?', (username, password))
        account = cursor.fetchone()
    except Exception as e:
        print(f"Database error: {str(e)}")
        return jsonify({"error": "An error occurred. Please try again later."}), 500
    finally:
        conn.close()

    if account:
        # Store user details in session
        session['loggedin'] = True
        session['id'] = account[0]
        session['username'] = account[1]

        return jsonify({
            "message": "Logged in successfully!",
            "redirect_url": "/next_page"  # Adjust according to your route
        }), 200
    else:
        return jsonify({"error": "Incorrect username or password!"}), 401

@app.route('/logout')
def logout():
    session.pop('loggedin', None)
    session.pop('id', None)
    session.pop('username', None)
    flash('You have successfully logged out.', 'success')
    return redirect(url_for('welcome'))


def generate_unique_user_id():
    while True:
        user_id = str(random.randint(100000, 999999))
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) FROM user2 WHERE user_id = ?", (user_id,))
            count = cursor.fetchone()[0]
            if count == 0:
                return user_id


@app.route('/submit', methods=['POST'])
def submitInput():
    data = request.json
    phone_number = data.get('phone_number')

    if not phone_number or len(phone_number) != 10:
        return jsonify({'error': 'Invalid phone number'}), 400

    user_id = generate_unique_user_id()  # Generate a unique user_id

    try:
        with pyodbc.connect(connection_string) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO user2 (PhoneNumber, user_id) VALUES (?, ?)", (phone_number, user_id))
            conn.commit()

        session['phone_number'] = phone_number
        session['user_id'] = user_id  # Store user_id in session
        return jsonify({'message': 'Registration successful', 'user_id': user_id}), 200
    except Exception as e:
        print(f"Error occurred during registration: {str(e)}")  # Add error logging
        return jsonify({'error': str(e)}), 500


@app.route('/gender', methods=['POST'])
def gender():
    try:
        # Receive JSON data from the AJAX request
        data = request.get_json()
        gender = data.get('gender')
        phone_number = data.get('phone_number')

        # Validate inputs
        if not gender:
            return jsonify({'error': 'Gender not provided'}), 400
        if not phone_number or len(phone_number) != 10:
            return jsonify({'error': 'Invalid phone number'}), 400

        # Connect to the database and update gender
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("UPDATE user2 SET gender = ? WHERE PhoneNumber = ?", (gender, phone_number))
        conn.commit()
        cursor.close()
        conn.close()

        return jsonify({'message': 'Gender updated successfully'}), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500

@app.route('/submit_phone', methods=['POST'])
def submit_phone_number():
    data = request.json
    phone_number = data.get('phone_number')
    gender = data.get('gender')

    if not phone_number or len(phone_number) != 10:
        return jsonify({'error': 'Invalid phone number'}), 400

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        cursor.execute("SELECT * FROM user2 WHERE PhoneNumber = ? AND Gender = ?", (phone_number, gender))
        result = cursor.fetchone()

        cursor.close()
        conn.close()

        if result:
            return jsonify({'message': 'Login successful'}), 200
        else:
            return jsonify({'error': 'Invalid phone number or gender'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/material', methods=['GET'])
def material():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401  # Unauthorized

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        # Fetch materials from the MaterialType table
        cursor.execute('SELECT name, description, material_image FROM MaterialType')
        materials = cursor.fetchall()

        # Format materials for JSON response
        material_list = [
            {
                'name': material[0],
                'description': material[1],
                'img': base64.b64encode(material[2]).decode('utf-8') if material[2] else None
            }
            for material in materials
        ]

        conn.close()
        return jsonify(material_list), 200

    except Exception as e:
        return jsonify({'error': f'An error occurred: {str(e)}'}), 500
    
@app.route('/submit_material', methods=['POST'])
def submit_material():
    data = request.json
    material = data.get('material')
    phone_number = session.get('phone_number')

    if not material:
        return jsonify({'error': 'Material not provided'}), 400

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute("UPDATE user2 SET material = ? WHERE PhoneNumber = ?", (material, phone_number))
        conn.commit()
        cursor.close()
        conn.close()
        return jsonify({'message': 'Material updated successfully'}), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/outfitCategory', methods=['GET'])
def outfit_category():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401  # Unauthorized

    # Define outfit categories
    categories = [
        {
            'id': 'upper-body',
            'name': 'Upper-body',
            'img': url_for('static', filename='images/category/upperbody.png')
        },
        {
            'id': 'lower-body',
            'name': 'Lower-body',
            'img': url_for('static', filename='images/category/lowerbody.png')
        },
        {
            'id': 'full-body',
            'name': 'Full-body',
            'img': url_for('static', filename='images/category/fullbody (2).png')
        }
    ]

    return jsonify(categories), 200

# Load the trained model
model = load_model("skin_tone_classifier.h5")

categories = ['Black', 'Brown', 'Dark-brown', 'Olive', 'White']
img_size = 128  # This should match the size used during training


@app.route('/capture_video', methods=['GET'])
def capture_video():
    if 'loggedin' not in session:
        return jsonify({'error': 'Not logged in'}), 401  # Unauthorized

    # Define instructions to display on the right-container
    instructions = [
        "When you're ready, please press the 'Start' button on the screen.",
        "The screen will start capturing the video.",
        "You have to turn around to take a video for better recommendations.",
        "After 30 seconds, you have to press the 'Submit' button."
    ]

    return jsonify({'instructions': instructions}), 200

@app.route('/submit_snapshots', methods=['POST'])
def submit_snapshots():
    data = request.get_json()
    snapshots = data.get('snapshots')
    user_id = session.get('user_id')

    if not user_id:
        return jsonify({'message': 'Failed to retrieve phone number for user'})

    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()

        for snapshot in snapshots:
            img_data = base64.b64decode(snapshot.split(',')[1])
            cursor.execute("INSERT INTO video (user_id, Snapshot) VALUES (?, ?)", user_id, img_data)

        conn.commit()
        # Predict skin tone category based on the last snapshot
        last_snapshot = base64.b64decode(snapshots[-1].split(',')[1])
        img_array = np.asarray(bytearray(last_snapshot), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        # Resize and normalize image
        img_resized = cv2.resize(img, (img_size, img_size))
        img_normalized = img_resized / 255.0
        img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))

        # Perform prediction
        prediction = model.predict(img_reshaped)
        predicted_class = np.argmax(prediction)
        predicted_tone = categories[predicted_class]

        # Save the predicted skin tone in the session
        session['predicted_tone'] = predicted_tone

        cursor.close()
        conn.close()

        return jsonify({'message': 'Snapshots saved successfully'})
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'message': 'Failed to save snapshots'})

import os
import cv2

def load_color_images():
    """Loads color images from the dataset directory."""
    base_path = "static/colors"  # Path to the color dataset
    color_images = {}

    for category in categories:
        category_path = os.path.join(base_path, category)
        color_images[category] = []

        if not os.path.exists(category_path):
            print(f"Warning: Directory does not exist for category '{category}': {category_path}")
            continue

        for file in os.listdir(category_path):
            if file.endswith((".png", ".jpg", ".jpeg")):
                img_path = os.path.join(category_path, file)
                img = cv2.imread(img_path)
                if img is not None:
                    color_images[category].append((file, img))

    return color_images

def recommend_colors_for_skin_tone(predicted_tone):
    """Recommends color images based on the predicted skin tone."""
    color_images = load_color_images()
    recommended_colors = color_images.get(predicted_tone, [])

    return random.sample(recommended_colors, min(10, len(recommended_colors)))  # Changed to 10


def classify_skin_tone_from_base64(base64_image):
    img_data = base64.b64decode(base64_image.split(',')[1])
    img_array = np.asarray(bytearray(img_data), dtype=np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    img_resized = cv2.resize(img, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = np.reshape(img_normalized, (1, img_size, img_size, 3))
    prediction = model.predict(img_reshaped)
    predicted_class = np.argmax(prediction)
    return categories[predicted_class]


@app.route('/get_skin_tone')
def get_skin_tone():
    predicted_tone = session.get('predicted_tone')

    if not predicted_tone:
        return jsonify({'message': 'No skin tone prediction available'}), 400

    recommended_colors = recommend_colors_for_skin_tone(predicted_tone)

    return jsonify({'predicted_tone': predicted_tone, 'recommended_colors': recommended_colors})

@app.route('/outfit', methods=['GET'])
def outfit():
    try:
        # Retrieve the predicted tone from the session
        predicted_tone = session.get('predicted_tone')

        if not predicted_tone:
            return jsonify({'error': 'Predicted skin tone not found. Please capture and submit video first.'}), 404

        # Get the recommended colors based on the predicted tone
        recommended_colors = recommend_colors_for_skin_tone(predicted_tone)

        # Prepare the color data for the response
        color_data = []
        for color_name, color_img in recommended_colors:
            color_name = color_name.split('.')[0]  # Remove file extension
            _, buffer = cv2.imencode('.jpg', color_img)
            img_base64 = base64.b64encode(buffer).decode('utf-8')  # Convert to Base64
            color_data.append({
                'name': color_name,
                'image': f"data:image/jpeg;base64,{img_base64}"
            })

        # Return JSON with predicted tone and recommended colors
        return jsonify({
            'predicted_tone': predicted_tone,
            'recommended_colors': color_data
        }), 200

    except Exception as e:
        print(f"Error in /outfit route: {str(e)}")
        return jsonify({'error': str(e)}), 500

# Define constraints
skin_tones = [ 'black', 'dark brown' , 'brown' , 'olive' , 'white']
colors = ['bright white',  'ruby red',  'bright purple',  'cobalt blue',  'royal blue',  'tomato red',  'denim blue',
          'citrine',  'olive green',  'peach', 'copper', 'russet', 'pumpkin orange', 'taupe', 'silver', 'plum', 'maroon', 'forest green',
          'black', 'mauve', 'pastel lilac', 'amber', 'emerald green', 'burnt orange', 'carmine', 'bordeaux', 'burgundy', 'fuchsia',
          'blush pink', 'sapphire blue', 'coral', 'jade', 'black', 'charcoal', 'mahogany', 'royal blue', 'emerald green', 'silver',
          'forest green', 'champagne', 'seafoam green', 'black', 'periwinkle', 'olive green', 'peach', 'teal', 'rose', 'soft pink', 'crimson',
          'light grey']
learning_rate = 0.1
discount_factor = 0.9
exploration_rate = 1.0
min_exploration_rate = 0.01
exploration_decay = 0.01

# Survey data (Example: Ratings 1-5 for each color per skin tone)
survey_ratings = {
    'brown': [3,	5,	5,	2,	4,	1,	4,	4,	4,	5,
              4,	2,	3,	2,	3,	3,	2,	2,	2,	3,
              4,	5,	3,	2,	2,	2,	3,	3,	4,	3,
              5,	4,	5,	3,	3,	5,	5,	4,	5,	5,
              3,	3,	4,	4,	3,	4,	5,	5,	1,	5,
              3,	4,	5,	3,	4,	4,	1,	5,	3,	3,
              2,	2,	5,	4,	5,	4,	5,	2,	3,	3,
              1,	4,	5,	2,	3,	2,	4,	3,	2,	3,
              4,	3,	2,	5,	3,	4,	3,	5,	1,	3,
              2,	5,	5,	3,	3,	1,	1,	3,	5,	2,
              2,	3,	2,	3,	2,	3,	4,	4,	2,	4,
              5,	3,	3,	2,	5,	4,	4,	5,	4,	3,
              3,	4,	3,	3,	5,	5,	5,	3,	5,	3,
              5,	5,	5,	4,	4,	5,	3,	4,	3,	3,
              5,	2,	3,	4,	5,	2,	3,	3,	4,	4,
              3,	4,	4,	3,	5,	3,	4 ],
    'dark brown': [5,	5,	5,	4,	5,	2,	4,	3,	4,	5,
                   5, 3,	1,	4,	5,	3,	2,	3,	1,	3,
                   3,	4,  3,	4,	5,	4,	5,	5,	5,	5,
                   5,	3,	4,  5,	5,	5,	5,	4,	5,	5,
                   5,	3,	3,	3,  2,	2,	3,	3,	5,	2,
                   2,	5,	5,	5,	5,  3,	3,	4,	3,	3,
                   4,	1,	4,	3,	4,	2,  2,	2,	4,	3,
                   4,	2,	3,	4,	2,	3,	3,  1,	5,	3,
                   3,	1,	3,	2,	4,	3,	2,	2,  3,	1,
                   4,	5,	4,	4,	5,	3,	3,	3,	2,  3,
                   3,	2,	2,	4,	3,	2,	1,	5,	5,	3,
                   4,	2,	5,	2,	3,	4,	1,	2,	3,	4,
                   5, 5,	3,	3,	4,	2,	2,	4,	4,	5,
                   3,	3,  3,	4,	4,	3,	5,	4,	4,	4,
                   3,	4,	4,  3,	4,	2,	5,	3,	3,	5,
                   4,	5,	5,	3,  2,	1],
    'olive': [4,	5,	4,	3,	2,	1,	4,	3,	5,	5,
              4,	3,	2,	2,	3,	3,	4,	4,	4,	5,
              3,	4,	4,	4,	5,	4,	5,	4,	4,	4,
              5,	5,	3,	5,	5,	5,	3,	2,	2,	3,
              4,	5,	4,	5,	5,	4,	3,	4,	5,	5,
              4,	5,	4,	5,	5,	2,	5,	5,	5,	5,
              5,	3,	5,	1,	3,	2,	4,	4,	4,	2,
              4,	5,	4,	3,	5,	5,	3,	3,	5,	2,
              3,	5,	5,	5,	3,	4,	4,	5,	3,	5,
              4,	4,	2,	3,	5,	4,	3,	2,	4,	5,
              2,	4,	4,	4,	3,	1,	5,	5,	4,	4,
              3,	1,	4,	5,	4,	5,	3,	5,	1,	5,
              5,	2,	5,	2,	1,	3,	5,	4,	4,	5,
              5,	5,	3,	3,	4,	3,	4,	4,	4,	5,
              5,	3,	5,	2,	4,	4,	4,	4,	5,	2,
              5,	3 ],
    'black': [3,	5,	4,	3,	5,	5,	4,	2,	3,	5,
              5,	3,	5,	2,	5,	3,  4,	3,	2,	3,
              1,	2,	3,	2,	1,	1,	3,	2,	3,	1,
              3,	2,  2,	5,	5,	5,	5,	5,	4,	5,
              5,	4,	3,	4,	5,	4,	3,	2,  3,	3,
              1,	2,	3,	3,	3,	2,	5,	5,	2,	3,
              3,	2,	5,	4,  1,	3,	3,	3,	2,	2,
              2,	1,	1,	3,	5,	5,	3,	3,	3,	3,
              2,	2,	4,	5,	3,	4,	3,	1,	3,	2,
              2,	1,	5,	2,	5,	3,  2,	4,	3,	2,
              3,	4,	2,	2,	3,	3,	5,	4,	3,	2,
              4,	2,  3,	3,	5,	3,	4,	4,	3,	5,
              1,	1,	3,	2,	3,	1,	3,	5,  5,	5,
              4,	5,	5,	5,	4,	1,	4,	1,	5,	5,
              2,	5,	5,	1,  2,	2,	3,	4,	4,	1,
              3,	2,	5,	5,	4,	4	],
    'white': [5,	5,	3,	5,	3,	5,	4,	4,	5,	5,
              4,	3,	3,	4,	4,	2,	4,	5,	5,	4,
              2,	4,	5,	5,	4,	5,	5,	4,	4,	5,
              5,	5,	5,	4,	4,	5,	3,	3,	4,	4,
              4,	4,	4,	5,	4,	5,	4,	5,	5,	5,
              5,	2,	5,	5,	5,	5,	4,	5,	5,	5,
              5,	2,	4,	5,	5,	3,	2,	1,	5,	5,
              5,	5,	4,	5,	5,	4,	5,	1,	5,	3,
              4,	5,	4,	5,	5,	4,	5,	5,	5,	5,
              4,	4,	4,	1,	5,	5,	4,	5,	5,	5,
              3,	1,	5,	5,	5,	5,	4,	3,	3,	5,
              5,	5,	4,	5,	4,	5,	3,	5,	3,	4,
              5,	3,	5,	5,	4,	5,	3,	2,	4,	5,
              4,	5,	2,	5,	4,	5,	5,	4,	5,	5,
              5,	5,	5,	4,	5,	1,	4,	5,	4,	4,
              5,	3,	3,	3,	4,	1 ]
}

# Convert ratings to rewards (1-5 rating scale to -2 to +2)
def normalize_rating(rating):
    return rating - 3  # Convert 1-5 ratings to -2 to +2

# Initialize Q-table with zeros
q_table = np.zeros((len(skin_tones), len(colors)))

# Pre-train Q-table using survey data
for skin_tone_index, skin_tone in enumerate(skin_tones):
    for color_index, rating in enumerate(survey_ratings[skin_tone]):
        reward = normalize_rating(rating)
        q_value = q_table[skin_tone_index, color_index]

        # Update Q-table based on survey data (initial pre-training)
        q_table[skin_tone_index, color_index] = q_value + learning_rate * reward

print("Q-table after pre-training with survey data:")
print(q_table)


# Helper function to fetch feedback from database
def fetch_latest_feedback():
    query = "SELECT skin_tone, color, size, outfit, overall FROM Feedback WHERE rating IS NOT NULL"
    cursor.execute(query)
    feedback_data = cursor.fetchall()
    return feedback_data

# Normalize feedback rating
def normalize_rating(rating):
    return rating - 3  # Convert 1-5 ratings to -2 to +2

# Function to process feedback and update Q-table
def process_feedback_and_update_qtable():
    feedback_data = fetch_latest_feedback()
    
    for feedback in feedback_data:
        skin_tone, color_rating, size_rating, outfit_rating, overall_rating = feedback
        reward = sum([normalize_rating(r) for r in [color_rating, size_rating, outfit_rating, overall_rating]]) / 4
        
        skin_tone_index = skin_tones.index(skin_tone)
        color_index = colors.index('example_color')  # Update with actual color logic

        q_value = q_table[skin_tone_index, color_index]
        best_future_q = np.max(q_table[skin_tone_index])
        q_table[skin_tone_index, color_index] = q_value + learning_rate * (reward + discount_factor * best_future_q - q_value)





@app.route('/outfits')
def get_outfits():
    try:
        conn = pyodbc.connect(connection_string)
        cursor = conn.cursor()
        cursor.execute('SELECT outfit_id, name, price, outfit_img FROM outfit')
        outfits = cursor.fetchall()
        cursor.close()
        conn.close()

        outfit_list = []
        for outfit in outfits:
            outfit_id = outfit.outfit_id
            name = outfit.name
            price = outfit.price
            outfit_img = base64.b64encode(outfit.outfit_img).decode('utf-8')
            outfit_list.append({'id': outfit_id, 'name': name, 'price': price, 'img': outfit_img})

        return jsonify(outfit_list)

    except Exception as e:
        print(f"Error fetching outfits: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/submit', methods=['POST'])
def submit():
    # Add your logic here for handling the POST request to /submit
    return jsonify({'message': 'Submit endpoint reached successfully'}), 200

@app.route('/submit_feedback', methods=['POST'])
def submit_feedback():
    data = request.json
    skin_tone = data.get('skin_tone')
    color_rating = data.get('color_rating')
    size_rating = data.get('size_rating')
    outfit_rating = data.get('outfit_rating')
    overall_rating = data.get('overall_rating')
    
    insert_query = """
    INSERT INTO Feedback (skin_tone, color, size, outfit, overall)
    VALUES (?, ?, ?, ?, ?)
    """
    cursor.execute(insert_query, skin_tone, color_rating, size_rating, outfit_rating, overall_rating)
    conn.commit()
    
    # Update Q-table with feedback data
    process_feedback_and_update_qtable()
    
    return jsonify({"message": "Feedback submitted and Q-table updated successfully!"})

@app.route('/thank', methods=['GET'])
def thank():
    try:
        # Prepare JSON response content
        response_data = {
            'message': 'Thank you for sharing your valuable feedback with us.',
            'image': '/static/images/feedback/feedback.jpeg',
            'next_url': '/'
        }

        # Return JSON response
        return jsonify(response_data), 200

    except Exception as e:
        print(f"Error in /thank route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/admin_dashboard', methods=['GET'])
def admin_dashboard():
    try:
        if 'loggedin' in session:
            # Prepare JSON response content
            response_data = {
                'message': f"Welcome, {session['username']}!",
                'links': {
                    'manage_material': url_for('manage_material'),
                    'manage_category': url_for('manage_category'),
                    'manage_inventory': url_for('manage_inventory'),
                    'logout': url_for('logout')
                }
            }
            return jsonify(response_data), 200
        else:
            # Redirect if not logged in
            return jsonify({'redirect': url_for('login')}), 302

    except Exception as e:
        print(f"Error in /admin_dashboard route: {str(e)}")
        return jsonify({'error': str(e)}), 500


@app.route('/manage_material', methods=['GET', 'POST'])
def manage_material():
    if 'loggedin' not in session:
        return jsonify({'redirect': url_for('login')}), 302

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    try:
        if request.method == 'POST':
            if 'add' in request.form:
                name = request.form['name']
                description = request.form['description']
                if name and description:
                    cursor.execute(
                        'INSERT INTO MaterialType (name, description) VALUES (?, ?)',
                        (name, description)
                    )
                    conn.commit()
                    flash('Material added successfully!', 'success')
                else:
                    flash('Material name and description cannot be empty.', 'error')

            elif 'update' in request.form:
                material_id = request.form['update_material_id']
                new_name = request.form['update_name']
                new_description = request.form['update_description']
                if material_id and new_name and new_description:
                    cursor.execute(
                        'UPDATE MaterialType SET name = ?, description = ? WHERE material_id = ?',
                        (new_name, new_description, material_id)
                    )
                    conn.commit()
                    flash('Material updated successfully!', 'success')
                else:
                    flash('Please provide all fields to update material.', 'error')

            elif 'drop' in request.form:
                material_id = request.form['drop_material_id']
                cursor.execute(
                    'SELECT COUNT(*) FROM Inventory WHERE category_id = ?', 
                    (material_id,)
                )
                count = cursor.fetchone()[0]
                if count == 0:
                    cursor.execute(
                        'DELETE FROM MaterialType WHERE material_id = ?',
                        (material_id,)
                    )
                    conn.commit()
                    flash('Material dropped successfully!', 'success')
                else:
                    flash('Cannot delete material because it is referenced in the inventory.', 'error')

            return jsonify({'redirect': url_for('manage_material')}), 302

        # Fetch all materials
        cursor.execute('SELECT * FROM MaterialType')
        materials = cursor.fetchall()

        # Format materials into JSON-friendly structure
        material_data = [
            {'material_id': material.material_id, 'name': material.name, 'description': material.description}
            for material in materials
        ]

        return jsonify({'materials': material_data}), 200

    except Exception as e:
        print(f"Error in /manage_material route: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()

@app.route('/manage_inventory', methods=['GET', 'POST'])
def manage_inventory():
    if 'loggedin' not in session:
        return jsonify({'redirect': url_for('login')}), 302

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    try:
        if request.method == 'POST':
            if 'add' in request.form:
                category_id = request.form['category_id']
                quantity = request.form['quantity']
                if category_id and quantity:
                    cursor.execute(
                        'INSERT INTO Inventory (category_id, quantity) VALUES (?, ?)',
                        (category_id, quantity)
                    )
                    conn.commit()
                    flash('Inventory added successfully!', 'success')
                else:
                    flash('Category and quantity cannot be empty.', 'error')

            elif 'update_inventory' in request.form:
                inventory_id = request.form['update_inventory_id']
                new_quantity = request.form['update_quantity']
                if inventory_id and new_quantity:
                    cursor.execute(
                        'UPDATE Inventory SET quantity = ? WHERE inventory_id = ?',
                        (new_quantity, inventory_id)
                    )
                    conn.commit()
                    flash('Inventory updated successfully!', 'success')
                else:
                    flash('Please select an inventory item and enter a new quantity.', 'error')

            elif 'drop_inventory' in request.form:
                inventory_id = request.form['drop_inventory_id']
                if inventory_id:
                    cursor.execute('DELETE FROM Inventory WHERE inventory_id = ?', (inventory_id,))
                    conn.commit()
                    flash('Inventory dropped successfully!', 'success')
                else:
                    flash('Please select an inventory item to drop.', 'error')

            return jsonify({'redirect': url_for('manage_inventory')}), 302

        # Fetch categories for Add Inventory form
        cursor.execute('SELECT category_id, name FROM Category')
        categories = cursor.fetchall()

        # Fetch inventories for Update and Drop Inventory forms
        cursor.execute('''
            SELECT i.inventory_id, c.name AS category_name, i.quantity
            FROM Inventory i
            JOIN Category c ON i.category_id = c.category_id
        ''')
        inventories = cursor.fetchall()

        # Format the data into JSON-friendly structure
        category_data = [
            {'category_id': category.category_id, 'name': category.name}
            for category in categories
        ]
        inventory_data = [
            {'inventory_id': inv.inventory_id, 'category_name': inv.category_name, 'quantity': inv.quantity}
            for inv in inventories
        ]

        return jsonify({'categories': category_data, 'inventories': inventory_data}), 200

    except Exception as e:
        print(f"Error in /manage_inventory route: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()


@app.route('/manage_category', methods=['GET', 'POST'])
def manage_category():
    if 'loggedin' not in session:
        return jsonify({'redirect': url_for('login')}), 302

    conn = pyodbc.connect(connection_string)
    cursor = conn.cursor()

    try:
        if request.method == 'POST':
            if 'add' in request.form:
                name = request.form.get('name')
                description = request.form.get('description')
                if name and description:
                    cursor.execute(
                        'INSERT INTO Category (name, description) VALUES (?, ?)',
                        (name, description)
                    )
                    conn.commit()
                    flash('Category added successfully!', 'success')
                else:
                    flash('Category name and description cannot be empty.', 'error')

            elif 'update' in request.form:
                category_id = request.form.get('update_category_id')
                new_name = request.form.get('update_name')
                new_description = request.form.get('update_description')
                if category_id and new_name and new_description:
                    cursor.execute(
                        'UPDATE Category SET name = ?, description = ? WHERE category_id = ?',
                        (new_name, new_description, category_id)
                    )
                    conn.commit()
                    flash('Category updated successfully!', 'success')
                else:
                    flash('Please provide all details for updating the category.', 'error')

            elif 'drop' in request.form:
                category_id = request.form.get('drop_category_id')
                cursor.execute('SELECT COUNT(*) FROM Inventory WHERE category_id = ?', (category_id,))
                count = cursor.fetchone()[0]
                if count == 0:
                    cursor.execute('DELETE FROM Category WHERE category_id = ?', (category_id,))
                    conn.commit()
                    flash('Category dropped successfully!', 'success')
                else:
                    flash('Cannot delete category because it is referenced in the inventory.', 'error')

            return jsonify({'redirect': url_for('manage_category')}), 302

        # Fetch categories for display
        cursor.execute('SELECT category_id, name, description FROM Category')
        categories = cursor.fetchall()

        # Fetch inventories for reference
        cursor.execute('''
            SELECT i.inventory_id, c.name AS category_name, i.quantity
            FROM Inventory i
            JOIN Category c ON i.category_id = c.category_id
        ''')
        inventories = cursor.fetchall()

        # Format data into JSON-friendly structure
        category_data = [
            {'category_id': cat.category_id, 'name': cat.name, 'description': cat.description}
            for cat in categories
        ]
        inventory_data = [
            {'inventory_id': inv.inventory_id, 'category_name': inv.category_name, 'quantity': inv.quantity}
            for inv in inventories
        ]

        return jsonify({'categories': category_data, 'inventories': inventory_data}), 200

    except Exception as e:
        print(f"Error in /manage_category route: {str(e)}")
        return jsonify({'error': str(e)}), 500

    finally:
        conn.close()

if __name__ == '__main__':
    app.run(debug=True)
