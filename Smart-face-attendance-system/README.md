SMART ATTENDANCE SYSTEM USING DEEP LEARNING MODEL

A Django-based attendance system that uses OpenCV for real-time face detection and recognition to automatically mark student attendance.

## Features

- Automatic face detection and recognition  
- Records morning, afternoon, and total attendance  
- Date-wise attendance tracking  
- Simple web interface for managing attendance  

## Technologies

- Python 3.10  
- Django 5.1.6  
- OpenCV for face detection  
- SQLite database  

## Setup

1. Clone the repository  
2. Create and activate a virtual environment  
3. Install dependencies:  
   ```bash
   pip install -r requirements.txt

4.Run migrations:
1.python manage.py makemigrations
2.python manage.py migrate

5.Start the development server:
python manage.py runserver

6.Open your browser and visit:
http://127.0.0.1:8000/

## Usage

1.Navigate to the web interface to view and manage attendance records.

2.The system will automatically detect and recognize faces from the webcam or uploaded images.

3.Attendance is recorded separately for morning and afternoon sessions and the total attendance is calculated automatically.

4.Use the interface to check attendance by date and student.
