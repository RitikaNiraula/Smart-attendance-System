import cv2
import os
import pandas as pd
import numpy as np
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import datetime  # Import the datetime module
import dlib
# Load Haarcascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# List of reference images with names and roll numbers (Ensure paths are correct)
reference_images = [
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT049\PUR078BCT049.jpeg", "name": "MD. Astfar alarm", "roll": "PUR078BCT049"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT051\PUR078BCT051.jpg", "name": "Milan Pokharel", "roll": "PUR078BCT051"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT052\PUR078BCT052.jpg", "name": "Mithun Yadav", "roll": "PUR078BCT052"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT053\PUR078BCT053.jpg", "name": "Nigma Yadav", "roll": "PUR078BCT053"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT055\PUR078BCT055.jpg", "name": "Pooja Rana", "roll": "PUR078BCT055"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT057\PUR078BCT057.jpg", "name": "Priyanka Mishara", "roll": "PUR078BCT057"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT061\PUR078BCT061.jpg", "name": "Rajesh Pandy", "roll": "PUR078BCT061"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT062\PUR078BCT062.jpg", "name": "Ram Chandra Ghimire", "roll": "PUR078BCT062"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT065\PUR078BCT065.jpg", "name": "Ravi Pandy", "roll": "PUR078BCT065"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT068\PUR078BCT068.jpg", "name": "Ritesh Sahani", "roll": "PUR078BCT068"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT069\PUR078BCT069.jpg", "name": "Ritika Niraula", "roll": "PUR078BCT069"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT072\PUR078BCT072.jpg", "name": "Sagar", "roll": "PUR078BCT072"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT077\PUR078BCT077.jpg", "name": "Sandhya Shrestha", "roll": "PUR078BCT077"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT081\PUR078BCT081.jpg", "name": "Sukavant Chaudhary", "roll": "PUR078BCT081"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT082\PUR078BCT082.jpg", "name": "Shyam Krishna Yadav", "roll": "PUR078BCT082"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT083\PUR078BCT083.jpg", "name": "Sneha Patel", "roll": "PUR078BCT083"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT084\PUR078BCT084.jpg", "name": "Sonu Gupta", "roll": "PUR078BCT084"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT085\PUR078BCT085.jpg", "name": "Sony Kumari Chaudhary", "roll": "PUR078BCT085"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT086\PUR078BCT086.jpg", "name": "Spandan Guragain", "roll": "PUR078BCT086"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT088\PUR078BCT088.jpg", "name": "Sudesh Subedi", "roll": "PUR078BCT088"},
    {"image": r"C:\Users\HP\Desktop\hello\deepface\Images\PUR078BCT096\PUR078BCT096.jpg", "name": "Yamaraj khadka", "roll": "PUR078BCT096"},
]

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, 800)  # width
cap.set(4, 520)  # height

# Preload Reference Face Embeddings using VGG-Face Model
preloaded_faces = {}
for ref_data in reference_images:
    try:
        abs_path = os.path.abspath(ref_data["image"])
        if os.path.exists(abs_path):
            # Using VGG-Face model to represent the face and generate embeddings
            ref_embedding = DeepFace.represent(abs_path, model_name="VGG-Face", enforce_detection=False,anti_spoofing=True)
            preloaded_faces[ref_data["name"]] = {"embedding": ref_embedding[0]['embedding'], "roll": ref_data["roll"]}
            print(f"Preloaded face embedding for {ref_data['name']} ({ref_data['roll']})")
            print(f"Reference embedding for {ref_data['name']}: {ref_embedding[0]['embedding'][:10]}...")  # Show only first 10 values
    except Exception as e:
        print(f"Error loading reference image {ref_data['image']}: {e}")
    


similarity_threshold = 0.60

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces using Haar Cascade
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))

    # Get current date and time
    current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Iterate through each detected face
    for (x, y, w, h) in faces:
        # Draw rectangle around detected face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract and preprocess face
        detected_face = frame[y:y + h, x:x + w]
        detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
        detected_face = cv2.resize(detected_face, (112, 112))  # Smaller size for speed

        # Generate embedding for the live face using VGG-Face
        live_embedding = DeepFace.represent(detected_face, model_name="VGG-Face", enforce_detection=False)
        live_embedding = live_embedding[0]['embedding']

        # Normalize the live face embedding
        live_embedding = np.array(live_embedding) / np.linalg.norm(live_embedding)
        print(f"Live face embedding: {live_embedding[:10]}...")  # Show first 10 values for inspection

        # Compare with each preloaded face embedding
        best_match_name = "Unknown"
        best_match_score = -1  # Start with a very low score for cosine similarity
        best_match_roll = "Unknown"  # Initialize roll number

        for name, ref_data in preloaded_faces.items():
            ref_embedding = np.array(ref_data['embedding']) / np.linalg.norm(ref_data['embedding'])
            similarity_score = cosine_similarity([live_embedding], [ref_embedding])[0][0]
            print(f"Comparing live face with {name} (Roll: {ref_data['roll']}), Similarity score: {similarity_score}")
            
            if similarity_score > best_match_score:
                best_match_name = name
                best_match_score = similarity_score
                best_match_roll = ref_data["roll"]  # Save corresponding roll number

        if best_match_score >= similarity_threshold:
           text_color = (0, 255, 0)  # Green color (you can adjust this if you prefer a different color)
           cv2.putText(frame, f"Name: {best_match_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
           cv2.putText(frame, f"Roll: {best_match_roll}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

           cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
        
            # If no match or below threshold, display "Unknown"
            cv2.putText(frame, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
    cv2.imshow("Live Face Recognition", frame)

    # Exit on 'q' press
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()