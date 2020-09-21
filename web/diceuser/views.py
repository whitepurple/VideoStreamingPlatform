from django.shortcuts import render, redirect
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.views import LoginView, LogoutView
from .models import DiceUser

# Create your views here.

class SignupForm(UserCreationForm):
    class Meta(UserCreationForm.Meta):
        model = DiceUser
        fields = ['username',]

def signup(request):
    errstring = ""
    if request.method == "POST":
        filled_form = SignupForm(request.POST)
        if filled_form.is_valid():
            filled_form.save()
            return redirect('login')
        else:
            errstring = "회원가입에 실패하였습니다."
    return render(request, 'register.html', {'err':errstring})

class MyLoginView(LoginView):
    template_name = 'login.html'
