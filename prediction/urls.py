from django.urls import path
from .views import (welcome_page , page2)
app_name = "prediction"

urlpatterns =[
    path ('',welcome_page,name = 'welcome_page'),
    path('view', page2 ,name='page2'),

]