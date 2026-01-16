"""
URL configuration for attendance_system project.

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/5.1/topics/http/urls/
Examples:
Function views
    1. Add an import:  from my_app import views
    2. Add a URL to urlpatterns:  path('', views.home, name='home')
Class-based views
    1. Add an import:  from other_app.views import Home
    2. Add a URL to urlpatterns:  path('', Home.as_view(), name='home')
Including another URLconf
    1. Import the include() function: from django.urls import include, path
    2. Add a URL to urlpatterns:  path('blog/', include('blog.urls'))
"""
from django.contrib import admin
from django.urls import path
from django.conf import settings
from attendance import views
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('home/',views.Home,name='Home'),
    path('Face_Detector/',views.Face_Detector,name='Face_Detector'),
    path('face_detector/', views.face_detector_view, name='face_detector'),
    path('',views.Login,name='Login'),
    path('Sign/',views.SignUp,name='SignUp'),
    path('Student/',views.Studentview,name='Studentview'),
    path('Excel/',views.generate_excel_sheet,name='generate_excel_sheet'),
    path('Forgot/',views.Forgot,name='Forgot'),
    
   
]+ static(settings.STATIC_URL, document_root=settings.STATIC_ROOT)+static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)


