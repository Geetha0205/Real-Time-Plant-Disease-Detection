import streamlit as st
import sqlite3
import uuid
import time
import cv2
import numpy as np
from PIL import Image
import hashlib
from ultralytics import YOLO

# === DB setup ===
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    
    # Check if table exists
    c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='users'")
    table_exists = c.fetchone()
    
    if not table_exists:
        # Create new table with all columns
        c.execute('''CREATE TABLE users 
                    (id TEXT PRIMARY KEY, 
                     username TEXT UNIQUE, 
                     password TEXT,
                     email TEXT)''')
        conn.commit()
    else:
        # Check if email column exists
        c.execute("PRAGMA table_info(users)")
        columns = [column[1] for column in c.fetchall()]
        if 'email' not in columns:
            c.execute("ALTER TABLE users ADD COLUMN email TEXT")
            conn.commit()
    
    conn.close()

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

def register_user(username, email, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    try:
        user_id = str(uuid.uuid4())
        hashed_password = hash_password(password)
        c.execute("INSERT INTO users (id, username, password, email) VALUES (?, ?, ?, ?)", 
                 (user_id, username, hashed_password, email))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False
    finally:
        conn.close()

def authenticate_user(username, password):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = hash_password(password)
    c.execute("SELECT * FROM users WHERE username = ? AND password = ?", 
             (username, hashed_password))
    user = c.fetchone()
    conn.close()
    return user is not None

# Initialize database with proper schema
init_db()

# === Model Setup ===
# Load YOLO model properly
CLASS_NAMES = ["Healthy", "leafy-minner", "downy mildow"]  # Explicitly define class names
try:
    model = YOLO("best.pt")  # This should be your trained YOLOv8 model
    if hasattr(model, "names"):
        CLASS_NAMES = [model.names[i] for i in sorted(model.names.keys())]  # Use model names if available
except Exception as e:
    st.error(f"Failed to load model: {e}")
    st.stop()

# === App State ===
if 'page' not in st.session_state:
    st.session_state.page = 'index'
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'username' not in st.session_state:
    st.session_state.username = ''

def navigate_to(page):
    st.session_state.page = page

def ensure_rgb(image):
    """Convert image to RGB format if needed"""
    if isinstance(image, np.ndarray):
        if image.shape[2] == 4:  # RGBA
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        elif image.shape[2] == 1:  # Grayscale
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        else:  # Assume BGR
            return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:  # PIL Image
        return image.convert('RGB')

def process_image(image):
    try:
        if isinstance(image, Image.Image):
            image = ensure_rgb(image)
            img_np = np.array(image)
        else:
            img_np = ensure_rgb(image)

        results = model(img_np)  # Perform inference
        result = results[0]
        annotated_img = img_np.copy()

        # Draw bounding boxes and labels
        if result.boxes is not None and len(result.boxes) > 0:
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cls_id = int(box.cls[0].cpu().numpy())
                confidence = float(box.conf[0].cpu().numpy())
                label = CLASS_NAMES[cls_id]

                # Draw bounding box and label
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(annotated_img, f"{label} ({confidence:.2%})", 
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            return annotated_img, label, confidence

        # Fallback to classification if no boxes
        elif hasattr(result, 'probs') and result.probs is not None:
            cls_id = int(result.probs.top1)
            confidence = float(result.probs.top1conf)
            label = CLASS_NAMES[cls_id]

            cv2.putText(annotated_img, f"{label} ({confidence:.2%})", 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            return annotated_img, label, confidence

        return annotated_img, "No Detection", 0.0

    except Exception as e:
        st.error(f"Error processing image: {e}")
        return np.array(image), "Error", 0.0


# === Sidebar ===
def render_sidebar():
    with st.sidebar:
        st.markdown("<h2 style='text-align: center;'>Plant Disease Detector</h2>", unsafe_allow_html=True)
        st.markdown(f"**Classes:** {', '.join(CLASS_NAMES)}")  # Display classes in sidebar
        if st.button("🏠 Home"): navigate_to('index')
        if st.button("ℹ️ About"): navigate_to('about')
        
        if not st.session_state.logged_in:
            if st.button("📝 Register"): navigate_to('register')
            if st.button("🔑 Login"): navigate_to('login')
        else:
            st.markdown(f"**Logged in as:** {st.session_state.username}")
            if st.button("🖥️ Dashboard"): navigate_to('home')
            if st.button("🔍 Prediction"): navigate_to('prediction')
            if st.button("🚪 Logout"):
                st.session_state.logged_in = False
                st.session_state.username = ''
                navigate_to('index')

render_sidebar()

# === Page Views ===
if st.session_state.page == 'index':
    st.title("🌿 Welcome to RealTime Plant Disease Detection Using AI")
    st.write("Use the sidebar to login/register and test predictions with images or webcam.")
    st.markdown(f"**Supported Classes:** {', '.join(CLASS_NAMES)}")
    
    if st.session_state.logged_in:
        st.success(f"You are currently logged in as {st.session_state.username}")
        if st.button("Go to Dashboard"):
            navigate_to('home')

elif st.session_state.page == 'about':
    st.title("ℹ️ About the Project")
    st.write("""
    This project demonstrates real-time plant disease detection using a YOLOv8 model deployed with Streamlit.
    
    **Features:**
    - User authentication (register/login)
    - Image upload for disease detection
    - Real-time webcam detection
    - Confidence scoring for predictions
    
    **Supported Classes:**
    """)
    for class_name in CLASS_NAMES:
        st.write(f"- {class_name}")

elif st.session_state.page == 'register':
    st.title("📝 User Registration")
    
    with st.form("registration_form"):
        username = st.text_input("Username", help="Choose a unique username")
        email = st.text_input("Email", help="Your email address")
        password = st.text_input("Password", type="password", help="Create a strong password")
        c_password = st.text_input("Confirm Password", type="password", help="Re-enter your password")
        
        submitted = st.form_submit_button("Register")
        if submitted:
            if not username or not email or not password:
                st.error("Please fill in all fields")
            elif password != c_password:
                st.error("Passwords do not match")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters")
            else:
                if register_user(username, email, password):
                    st.success("Registration successful! Please login.")
                    time.sleep(1)
                    navigate_to('login')
                else:
                    st.error("Username already exists. Please choose another.")

elif st.session_state.page == 'login':
    st.title("🔑 User Login")
    
    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        submitted = st.form_submit_button("Login")
        if submitted:
            if authenticate_user(username, password):
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success("Login successful!")
                time.sleep(1)
                navigate_to('home')
            else:
                st.error("Invalid username or password")

elif st.session_state.page == 'home':
    if not st.session_state.logged_in:
        st.error("Please login to access this page")
        navigate_to('login')
    else:
        st.title(f"🖥️ Welcome {st.session_state.username}")
        st.write("""
        **Dashboard Features:**
        - Upload plant images for disease detection
        - Use your webcam for real-time detection
        - View detection history (coming soon)
        
        **Supported Classes:**
        """)
        for class_name in CLASS_NAMES:
            st.write(f"- {class_name}")
        
        if st.button("Go to Prediction"):
            navigate_to('prediction')

elif st.session_state.page == 'prediction':
    if not st.session_state.logged_in:
        st.error("Please login to use the prediction feature.")
        navigate_to('login')
    else:
        st.title("🔍 Plant Disease Prediction")
        st.write(f"Welcome, {st.session_state.username}! Upload an image or use your webcam for real-time detection.")
        st.markdown(f"**Supported Classes:** {', '.join(CLASS_NAMES)}")
        
        # Tab interface for different prediction methods
        tab1, tab2 = st.tabs(["📁 Upload Image", "🎥 Live Webcam"])
        
        with tab1:
            st.subheader("Image Upload")
            uploaded_files = st.file_uploader("Choose images (jpg/png)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)
            
            if uploaded_files:
                # Display images in a grid
                num_images = len(uploaded_files)
                cols = st.columns(3)  # 3 columns for the grid
                
                for i, uploaded_file in enumerate(uploaded_files):
                    try:
                        image = Image.open(uploaded_file)
                        
                        with st.spinner(f"Analyzing image {i+1}/{num_images}..."):
                            annotated_img, prediction, confidence = process_image(image)
                        
                        if annotated_img is not None:
                            # Display in grid (3 columns)
                            with cols[i % 3]:
                                st.image(annotated_img, use_column_width=True)
                                if prediction == "Healthy":
                                    st.success(f"{prediction} ({confidence:.2%})")
                                else:
                                    st.warning(f"{prediction} ({confidence:.2%})")
                    except Exception as e:
                        st.error(f"Error processing image {uploaded_file.name}: {e}")
        
        with tab2:
            st.subheader("Live Webcam Detection")
            st.warning("Note: Webcam feature may not work on all browsers. For best results, use Chrome.")
            
            run_camera = st.checkbox("Start Webcam", key="webcam_checkbox")
            FRAME_WINDOW = st.empty()
            camera = cv2.VideoCapture(0)
            
            # Add a placeholder for captured images grid
            st.subheader("Captured Frames")
            captured_images_placeholder = st.empty()
            captured_images = []
            
            stop_button = st.button("Stop Webcam", key="stop_webcam")
            capture_button = st.button("Capture Frame", key="capture_frame")

            while run_camera and not stop_button:
                ret, frame = camera.read()
                if not ret:
                    st.warning("Failed to access webcam. Please check permissions.")
                    break

                annotated_img, prediction, confidence = process_image(frame)

                # Convert BGR to RGB for display
                display_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                FRAME_WINDOW.image(display_img)

                if capture_button:
                    captured_images.append((display_img, prediction, confidence))
                    capture_button = False
                    st.rerun()

                time.sleep(0.1)

            
            # Display captured images in a grid
            if captured_images:
                num_images = len(captured_images)
                cols = st.columns(3)  # 3 columns for the grid
                
                with captured_images_placeholder.container():
                    st.write("## Captured Results")
                    for i, (img, pred, conf) in enumerate(captured_images):
                        with cols[i % 3]:
                            st.image(img, use_column_width=True)
                            if pred == "Healthy":
                                st.success(f"{pred} ({conf:.2%})")
                            else:
                                st.warning(f"{pred} ({conf:.2%})")
            
            camera.release()
            if stop_button:
                run_camera = False
                st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<style>
.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #f1f1f1;
    color: black;
    text-align: center;
    padding: 10px;
}
</style>
<div class="footer">
    <p>Plant Disease Detection System © 2023</p>
</div>
""", unsafe_allow_html=True)