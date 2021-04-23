from rest_framework import serializers
from .models import Digits
import base64
import uuid
from django.core.files.base import ContentFile

class Base64ImageField(serializers.ImageField):
    '''
    To decode Base64 image to .png image
    '''
    def to_internal_value(self, data):
        '''
        Overriden method to do the decoding
        '''
        #https://stackoverflow.com/questions/39576174/save-base64-image-in-django-file-field
        #https://stackoverflow.com/questions/54648773/django-rest-framework-how-to-import-image-as-jpeg-and-save-it-as-base-64-using
        _format, str_img = data.split(';base64')
        decoded_file = base64.b64decode(str_img)
        fname = f"{str(uuid.uuid4())[:10]}.png"
        data = ContentFile(decoded_file, name=fname)
        return super().to_internal_value(data)

class DigitsSerializer(serializers.ModelSerializer):
    image = Base64ImageField()
    class Meta:
        model = Digits;
        fields = ('id', 'image', 'result1', 'score1', 'result2', 'score2')
