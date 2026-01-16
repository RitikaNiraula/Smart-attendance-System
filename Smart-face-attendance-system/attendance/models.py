from django.db import models
import datetime

# Create your models here.

# c
class Student(models.Model):
    department_choices = [
        ('Computer', 'Computer'),
        
        
    ]

   
    

    year_choices = [
        ('2024-2025', '2024-2025'),
        ('2025-2026', '2025-2026'),
    ]

    semester_choices = [
        ('Semester-1', 'Semester-1'),
        ('Semester-2', 'Semester-2'),
        ('Semester-3', 'Semester-3'),
        ('Semester-4', 'Semester-4'),
        ('Semester-5', 'Semester-5'),
        ('Semester-6', 'Semester-6'),
    ]

    class_group_choices = [
        ('AB', 'AB'),
        ('CD', 'CD'),
    ]

    department = models.CharField(max_length=50, choices=department_choices,default=True,null=True)
   
    year = models.CharField(max_length=10, choices=year_choices,default=True,null=True)
    semester = models.CharField(max_length=20, choices=semester_choices,default=True,null=True)
    student_id = models.CharField(max_length=50, unique=True,default=True,null=True)
    student_name = models.CharField(max_length=100,default=True,null=True)
    class_group = models.CharField(max_length=10, choices=class_group_choices,default=True,null=True)
    teacher_name = models.CharField(max_length=100,default=True,null=True)
    photo = models.ImageField(upload_to='student_photos/', blank=True, null=True) 
    def __str__(self):
        return self.student_name




# Model to store attendance information
# class Attendance(models.Model):
#     student_id = models.CharField(max_length=50,default="")
#     student_name = models.CharField(max_length=100,default="")
#     date = models.DateField(auto_now_add=True)
#     morning_attendance = models.BooleanField(default=False)
#     afternoon_attendance = models.BooleanField(default=False)
#     attendance_status = models.FloatField(default=0.0)

#     def __str__(self):
#         return f"{self.student_name} ({self.student_id}) - {self.date}"



class Attendance(models.Model):
    student_id = models.CharField(max_length=50, default="")
    student_name = models.CharField(max_length=100, default="")
    date = models.DateField(auto_now_add=True)
    
    morning_attendance = models.CharField(
        max_length=10, 
        choices=[("Present", "Present"), ("Absent", "Absent")], 
        default="Absent"
    )
    
    afternoon_attendance = models.CharField(
        max_length=10, 
        choices=[("Present", "Present"), ("Absent", "Absent")], 
        default="Absent"
    )

    attendance_status = models.FloatField(default=0.0)  # 1 for full, 0.5 for half-day


    def __str__(self):
        return f"{self.student_name} ({self.student_id}) - {self.date}"


class Password(models.Model):
    email = models.EmailField(unique=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Password reset request for {self.email}"