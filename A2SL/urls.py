"""A2SL URL Configuration

The `urlpatterns` list routes URLs to views. For more information please see:
    https://docs.djangoproject.com/en/3.0/topics/http/urls/
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
from . import views

urlpatterns = [
    path('admin/', admin.site.urls),
    path('about/',views.about_view, name='about'),
    path('video_feed', views.video_feed, name='video_feed'),
    path('sign-to-text/',views.sign_to_text, name='sign-to-text'),
    path('text-to-sign/',views.text_to_sign, name='text-to-sign'),
    path('subtitles/',views.subtitles, name='subtitles'),

    path('',views.home_view,name='home'),
]
