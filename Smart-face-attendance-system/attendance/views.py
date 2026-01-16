from django.shortcuts import render,redirect
from django.contrib.auth.models import User  
from django.contrib import messages
from django.contrib.auth import authenticate, login
from attendance import models
from django.http import JsonResponse
from .models import Student,Attendance,Password
from django.core.exceptions import ObjectDoesNotExist
from django.http import HttpResponse
import openpyxl
from datetime import datetime
import cv2
import os 
from deepface import DeepFace 
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from django.utils import timezone
from django.utils.timezone import now
import numpy as np
import pickle
import json
from PIL import Image
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
from datetime import datetime
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.utils import timezone
from django.shortcuts import render, redirect
from django.conf import settings




# Create your views here.
def Home(request):
    return render(request,'index1.html')
def Forgot(request):
    if request.method == "POST":
        email = request.POST.get("email")
        if email:
            # Save the email in the PasswordResetRequest model
            Password.objects.create(email=email)

    # Render the form again (same page) after POST request
    return render(request, 'forgot.html')




def SignUp(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        email = request.POST.get('email')
        contact = request.POST.get('contact')  
        password = request.POST.get('password')
        print(username,email,contact,password)

        if User.objects.filter(email=email).exists():
            messages.error(request, "Email already exists.")
            return render (request ,'sign.html')

        # Create the user and save to the database
        my_user = User.objects.create_user(username,email,password)
        my_user.save()

        
        return redirect('Home')  # Redirect to login page

    return render(request, 'sign.html')  



def Login(request):
    if request.method == 'POST':
        username = request.POST.get('username')
        password = request.POST.get('password')

        # Authenticate user using Django's built-in method
        user = authenticate(request, username=username, password=password)

        if user is not None:
            # Log in the user
            login(request, user)

            return redirect('Home')  # Redirect to home page
        else:
            messages.error(request,"Invalid username or password .")
            return redirect('Login')  # Redirect to login page

    return render(request, 'login.html')  # Render login page if not POST

def Studentview(request):
    message=None
    if request.method == "POST":
        if 'saveBtn' in request.POST:
            student_id = request.POST['studentId']

            
            if Student.objects.filter(student_id=student_id).exists():
                messages.error(request, "Student ID already exists.")
                return render(request, 'index.html')
            # Create new student
            student = Student(
                department=request.POST['department'],
                year=request.POST['year'],
                semester=request.POST['semester'],
                student_id=request.POST['studentId'],
                student_name=request.POST['studentName'],
                class_group=request.POST['classGroup'],
                teacher_name=request.POST['teacherName'],
                photo=request.FILES.get('photo')  
            )
            if Student.objects.filter(student_id=student_id).exists():
              messages.error(request, "Student Id already exists.")
              return render (request ,'index.html')
            student.save()
            message = "Student information saved successfully!"
        
        elif 'updateBtn' in request.POST:
            # Update existing student
            try:
                student = Student.objects.get(student_id=request.POST['studentId'])
                student.department = request.POST['department']
                student.year = request.POST['year']
                student.semester = request.POST['semester']
                student.student_name = request.POST['studentName']
                student.class_group = request.POST['classGroup']
                
                student.teacher_name = request.POST['teacherName']
                if 'photo' in request.FILES:
                    student.photo = request.FILES['photo']
                student.save()
                message = "Student information updated successfully!"
            except ObjectDoesNotExist:
                message = "Student not found!"
        
        elif 'deleteBtn' in request.POST:
            # Delete student
            try:
                student = Student.objects.get(student_id=request.POST['studentId'])
                student.delete()
                message = "Student information deleted successfully!"
            except ObjectDoesNotExist:
                message = "Student not found!"

        return render(request, 'index.html', {'message': message})

        

    
    students = Student.objects.all()
    return render(request, 'index.html', {'students': students})

    return render(request,'index.html')


def generate_excel_sheet(request):
    
    
    today_date = now().date()

    # Query the Attendance model for records on this date
    attendance_records = Attendance.objects.filter(date=today_date).order_by('student_id')  # Sorting by student_id

    # Prepare a list to store attendance data
    data = []
    for record in attendance_records:
        data.append({
            'Student ID': record.student_id,
            'Student Name': record.student_name,
            'Morning Attendance': record.morning_attendance,
            'Afternoon Attendance': record.afternoon_attendance,
            'Attendance Status': record.attendance_status,
            'Date': str(record.date,)  # Make sure date is being added correctly
        })

    # Create a pandas DataFrame from the list
    df = pd.DataFrame(data)

    # Create an HTTP response with Excel content type
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    response['Content-Disposition'] = 'attachment; filename="attendance_{}.xlsx"'.format(today_date)

    # Create an Excel writer object with the response as the destination
    with pd.ExcelWriter(response, engine='openpyxl') as writer:
        # Write the DataFrame to the Excel file
        df.to_excel(writer, index=False, sheet_name='Attendance')

        # Get the workbook and sheet objects
        workbook = writer.book
        sheet = workbook['Attendance']

        # Adjust column widths (you can adjust these values to your needs)
        column_widths = {
            'A': 15,  # Student ID
            'B': 25,  # Student Name
            'C': 20,  # Date
            'D': 20,  # Morning Attendance
            'E': 20,  # Afternoon Attendance
            'F': 20,  # Attendance Status
        }

        # Apply the column widths
        for col, width in column_widths.items():
            sheet.column_dimensions[col].width = width

        # Optionally, you can auto-adjust column width based on the content
        for col in sheet.columns:
            max_length = 0
            column = col[0].column_letter  # Get the column name
            for cell in col:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(cell.value)
                except:
                    pass
            adjusted_width = (max_length + 2)  # Add some padding
            sheet.column_dimensions[column].width = adjusted_width

    return response


# def Face_Detector(request):
 
 
#     attendance={}
#     face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# # List of reference images with names and roll numbers (Ensure paths are correct)
#     reference_images = [
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT049\PUR078BCT049.jpeg", "name": "MD. Astafar Alam", "student_id": "PUR078BCT049"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT051\PUR078BCT051.jpg", "name": "Milan Pokharel", "student_id": "PUR078BCT051"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT052\PUR078BCT052.jpg", "name": "Mithun Yadav", "student_id": "PUR078BCT052"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT053\PUR078BCT053.jpg", "name": "Nigam Yadav", "student_id": "PUR078BCT053"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT055\PUR078BCT055.jpg", "name": "Pooja Rana", "student_id": "PUR078BCT055"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT057\PUR078BCT057.jpg", "name": "Priyanka Mishara", "student_id": "PUR078BCT057"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT061\PUR078BCT061.jpg", "name": "Rajesh Pandey", "student_id": "PUR078BCT061"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT062\PUR078BCT062.jpg", "name": "Ram Chandra Ghimire", "student_id": "PUR078BCT062"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT065\PUR078BCT065.jpg", "name": "Ravi Pandit", "student_id": "PUR078BCT065"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT068\PUR078BCT068.jpg", "name": "Ritesh Sahani", "student_id": "PUR078BCT068"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT069\PUR078BCT069.jpg", "name": "Ritika Niraula", "student_id": "PUR078BCT069"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT072\PUR078BCT072.jpg", "name": "Sagar", "student_id": "PUR078BCT072"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT077\PUR078BCT077.jpg", "name": "Sandhya Shrestha", "student_id": "PUR078BCT077"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT081\PUR078BCT081.jpg", "name": "Sukavant Chaudhary", "student_id": "PUR078BCT081"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT082\PUR078BCT082.jpg", "name": "Shyam Krishna Yadav", "student_id": "PUR078BCT082"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT083\PUR078BCT083.jpg", "name": "Sneha Patel", "student_id": "PUR078BCT083"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT084\PUR078BCT084.jpg", "name": "Sonu Gupta", "student_id": "PUR078BCT084"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT085\PUR078BCT085.jpg", "name": "Sony Kumari Chaudhary", "student_id": "PUR078BCT085"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT086\PUR078BCT086.jpg", "name": "Spandan Guragain", "student_id": "PUR078BCT086"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT088\PUR078BCT088.jpg", "name": "Sudesh Subedi", "student_id": "PUR078BCT088"},
#     {"image": r"C:\Users\HP\Desktop\Face\Images\PUR078BCT096\PUR078BCT096.jpg", "name": "Yamraj khadka", "student_id": "PUR078BCT096"},
#    ]

# # Initialize webcam
#     cap = cv2.VideoCapture(0)
#     cap.set(3, 800)  # width
#     cap.set(4, 520)  # height

# # Preload Reference Face Embeddings using VGG-Face Model
#     preloaded_faces = {}
#     for ref_data in reference_images:
#        try:
#          abs_path = os.path.abspath(ref_data["image"])
#          if os.path.exists(abs_path):
#             # Using Facenet model to represent the face and generate embeddings
#             ref_embedding = DeepFace.represent(abs_path, model_name="Facenet512", enforce_detection=False)
#             preloaded_faces[ref_data["name"]] = {"embedding": ref_embedding[0]['embedding'], "student_id": ref_data["student_id"]}
#             print(f"Preloaded face embedding for {ref_data['name']} ({ref_data['student_id']})")
#             print(f"Reference embedding for {ref_data['name']}: {ref_embedding[0]['embedding'][:10]}...")  # Show only first 10 values
#        except Exception as e:
#             print(f"Error loading reference image {ref_data['image']}: {e}")



#     while True:
#         ret, frame = cap.read()
#         if not ret:
#           print("Failed to capture frame.")
#           break

#     # Convert to grayscale
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#     # Detect faces using Haar Cascade
#         faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=6, minSize=(50, 50))


#         current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        

#     # Iterate through each detected face
#         for (x, y, w, h) in faces:
#         # Draw rectangle around detected face
#             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

#         # Extract and preprocess face
#             detected_face = frame[y:y + h, x:x + w]
#             detected_face = cv2.cvtColor(detected_face, cv2.COLOR_BGR2RGB)
#             detected_face = cv2.resize(detected_face, (112, 112))  # Smaller size for speed

#         # Generate embedding for the live face using facenet
#             live_embedding = DeepFace.represent(detected_face, model_name="Facenet512", enforce_detection=False)
#             live_embedding = live_embedding[0]['embedding']

#         # Normalize the live face embedding
#             live_embedding = np.array(live_embedding) / np.linalg.norm(live_embedding)
#             print(f"Live face embedding: {live_embedding[:10]}...")  # Show first 10 values for inspection

#         # Compare with each preloaded face embedding
#             best_match_name = "Unknown"
#             best_match_score = -1  # Start with a very low score for cosine similarity
#             best_match_student_id = "Unknown"  # Initialize roll number

#             for name, ref_data in preloaded_faces.items():
#                ref_embedding = np.array(ref_data['embedding']) / np.linalg.norm(ref_data['embedding'])
#                similarity_score = cosine_similarity([live_embedding], [ref_embedding])[0][0]
#                print(f"Comparing live face with {name} (Student_Id: {ref_data['student_id']}), Similarity score: {similarity_score}")
            
#                if similarity_score > best_match_score:
#                    best_match_name = name
#                    best_match_score = similarity_score
#                    best_match_student_id = ref_data["student_id"]  # Save corresponding roll number

#         # If a match is found, display name, roll number, and current time
#             text_color = (0, 255, 0)  # Green color (you can adjust this if you prefer a different color)
#             cv2.putText(frame, f"Name: {best_match_name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, text_color, 2)
#             cv2.putText(frame, f"Student_Id: {best_match_student_id}", (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

#             cv2.putText(frame, f"Time: {current_time}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
#             if best_match_name != "Unknown":
#                 if best_match_student_id not in attendance:
#                     attendance[best_match_student_id] = {"morning": False, "afternoon": False}

#                 current_hour = datetime.now().hour
#                 today = timezone.now().date()
#                 existing_record = Attendance.objects.filter(student_id=best_match_student_id, date=timezone.now().date()).first()
                
#                 if existing_record:
#                   attendance[best_match_student_id]["morning"] = existing_record.morning_attendance
#                   attendance[best_match_student_id]["afternoon"] = existing_record.afternoon_attendance

#                 # Mark attendance for the morning session (before 11 AM)
#                 if current_hour < 11 and not attendance[best_match_student_id]["morning"]:
#                     attendance[best_match_student_id]["morning"] = True

#                 # Mark attendance for the afternoon session (after 3 PM)
#                 elif current_hour >= 14 and not attendance[best_match_student_id]["afternoon"]:
#                     attendance[best_match_student_id]["afternoon"] = True

#                 # Mark attendance as 1 or 0.5 based on session completion
#                 attendance_status = 0.5
#                 if attendance[best_match_student_id]["morning"] and attendance[best_match_student_id]["afternoon"]:
#                     attendance_status = 1

#                 if current_hour < 11 or current_hour >= 14: 
#                  try:
#                     # Check if attendance already exists for this student on the same date
#                     # existing_record = Attendance.objects.filter(student_id=best_match_student_id, date=timezone.now().date()).first()
#                     # if existing_record:
#                     #     # Update existing attendance record
#                     #     existing_record.morning_attendance = attendance[best_match_student_id]["morning"]
#                     #     existing_record.afternoon_attendance = attendance[best_match_student_id]["afternoon"]
#                     #     existing_record.attendance_status = attendance_status
#                     #     existing_record.save()


#                     if existing_record:
#                 # Only update fields that are False, avoiding overwriting previous attendance
#                         if not existing_record.morning_attendance and attendance[best_match_student_id]["morning"]:
#                           existing_record.morning_attendance = True

#                         if not existing_record.afternoon_attendance and attendance[best_match_student_id]["afternoon"]:
#                           existing_record.afternoon_attendance = True

#                 # Update attendance status accordingly
#                         if existing_record.morning_attendance and existing_record.afternoon_attendance:
#                           existing_record.attendance_status = 1
#                         else:
#                            existing_record.attendance_status = 0.5

#                         existing_record.save()
#                     else:
#                         # Create a new attendance record
#                         Attendance.objects.create(
#                             student_id=best_match_student_id,
#                             student_name=best_match_name,
#                             morning_attendance=attendance[best_match_student_id]["morning"],
#                             afternoon_attendance=attendance[best_match_student_id]["afternoon"],
#                             attendance_status=attendance_status,
#                             date=timezone.now().date(),
#                         )
#                  except Exception as e:
#                     print(f"Error saving attendance to database: {e}")
#         cv2.imshow("Live Face Recognition", frame)

#     # Exit on 'q' press
#         key = cv2.waitKey(1) & 0xFF
#         if key == ord('q'):
#            break

# # Release resources
#     cap.release()
#     cv2.destroyAllWindows()

#     return render(request,'index1.html')

import cv2
import numpy as np
import pickle
import json
from PIL import Image
from numpy import asarray, expand_dims
from keras_facenet import FaceNet
from datetime import datetime
from django.http import StreamingHttpResponse
from django.shortcuts import render

# Global flag for the webcam capture loop
is_capture_running = True

def generate_frames(request):
    global is_capture_running
    HaarCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    MyFaceNet = FaceNet()
    
    file_path = os.path.join(settings.BASE_DIR,'attendance', 'data.pkl')

    with open(file_path, "rb") as myfile:
      database = pickle.load(myfile)

    file_path = os.path.join(settings.BASE_DIR, 'attendance', 'students.json')

    with open(file_path, "r") as json_file:
        student_names = json.load(json_file)

    # Function to calculate cosine similarity
    def cosine_similarity(a, b):
        a = a.flatten()
        b = b.flatten()
        return np.dot(a, b.T) / (np.linalg.norm(a) * np.linalg.norm(b))

    SIMILARITY_THRESHOLD = 0.7  # Adjusted for better accuracy
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return "Webcam not found."

    while is_capture_running:
        success, frame = cap.read()
        if not success:
            break

        # Detect faces (tuned minNeighbors for better accuracy)
        faces = HaarCascade.detectMultiScale(frame, scaleFactor=1.1, minNeighbors=6, minSize=(100, 100))

        if len(faces) == 0:
            print("No faces detected.")

        for (x1, y1, width, height) in faces:
            if y1 < 50:  # Ignore faces too close to the top
                continue

            face = frame[y1:y1 + height, x1:x1 + width]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = Image.fromarray(face).resize((160, 160))
            face_array = expand_dims(asarray(face), axis=0)
            signature = MyFaceNet.embeddings(face_array)

            # Normalize the signature embedding
            signature = signature / np.linalg.norm(signature)

            best_match = "Unknown"
            highest_similarity = -1  

            for student_id, embedding in database.items():
                embedding = embedding / np.linalg.norm(embedding)  # Normalize database embedding
                similarity = cosine_similarity(embedding, signature)
                
                print(f"Comparing {student_id} - Similarity: {similarity:.4f}")  # Debugging
                
                if similarity > highest_similarity:
                    highest_similarity = similarity
                    best_match = student_id

            # Display results
            student_name = student_names.get(best_match, "Unknown")
            current_time = datetime.now().strftime("%H:%M:%S")
            current_date = datetime.now().strftime("%Y-%m-%d")

            # Set color based on recognition
            if highest_similarity >= SIMILARITY_THRESHOLD:
                print(f"Recognized Student: {student_name} (Student ID: {best_match})")
                cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height),(0,255,0) , 2)
                cv2.putText(frame, f"Student ID: {best_match}", (x1, y1 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0), 2)
                cv2.putText(frame, f"Name: {student_name}", (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
            

            # else:
            #     best_match = "Unknown"
            #     student_name = "Unknown"
            #     print("No matching student found, marking as 'Unknown'")
            #     cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 0, 255), 2)
            #     cv2.putText(frame, f"Unknown Person", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            current_time = datetime.now()
            current_hour = current_time.hour
            # Draw bounding box and text
            cv2.putText(frame, f"Date: {current_date}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, f"Time: {current_time}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            current_date = timezone.now().date()
            existing_record = Attendance.objects.filter(student_id=best_match, date=current_date).first()

            if existing_record:
                    print(f"Existing record found. Morning: {existing_record.morning_attendance}, Afternoon: {existing_record.afternoon_attendance}")
            else:
                    print("No existing record found.")

                # Attendance marking
            if not existing_record:
                    # Create a new attendance record for the current date
                    Attendance.objects.create(
                        student_id=best_match,
                        student_name=student_name,
                        morning_attendance="Present" if current_hour < 11 else "Absent",
                        afternoon_attendance="Present" if current_hour >= 14 else "Absent",
                        attendance_status=1 if current_hour < 11 and current_hour >= 14 else 0.5,
                        date=current_date,
                    )
            else:
                    # Update attendance based on time for the current date
                    updated = False
                    if current_hour < 11 and existing_record.morning_attendance != "Present":
                        existing_record.morning_attendance = "Present"
                        updated = True
                    if current_hour >= 14 and existing_record.afternoon_attendance != "Present":
                        print("Updating afternoon attendance to 'Present'")  # Debugging: Print when updating afternoon attendance
                        existing_record.afternoon_attendance = "Present"
                        updated = True

                    # Update attendance status
                    if (existing_record.morning_attendance == "Present" and existing_record.afternoon_attendance == "Present"):
                        existing_record.attendance_status = 1
                    elif existing_record.morning_attendance == "Present" or existing_record.afternoon_attendance == "Present":
                        existing_record.attendance_status = 0.5
                    else:
                        existing_record.attendance_status = 0

                    if updated:
                        print(f"Saving updated record for {best_match}")
                        existing_record.save()

        # Convert frame to JPEG and yield it
        _, jpeg = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n\r\n')

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def stop_video_capture():
    global is_capture_running
    is_capture_running = False

# View to handle face detection and streaming response
def Face_Detector(request):
    return StreamingHttpResponse(generate_frames(request), content_type='multipart/x-mixed-replace; boundary=frame')

# View to render face detector page
def face_detector_view(request):
    return render(request, 'face_detection.html')
