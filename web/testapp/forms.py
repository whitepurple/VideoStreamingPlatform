from django import forms

from .models import Face

class VideoForm(forms.ModelForm):
    videofile = forms.FileField(
        label = False,
        widget=forms.FileInput(
            attrs={
                'class': 'circle',
            }
        ),
    )
    class Meta:
        model= Face
        fields= ["videofile"]
