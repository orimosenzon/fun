# Create your views here.
from django.http import HttpResponse
from django.shortcuts import render_to_response
from .models import myline

def home(request):
    return render_to_response("chk1/home.html", {'lines': myline.objects.all()} ) 

def safta(request):
    return HttpResponse("shalom savta, ma nishma?") 



