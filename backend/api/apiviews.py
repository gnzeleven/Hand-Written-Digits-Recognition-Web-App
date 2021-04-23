from rest_framework import viewsets
from .models import Digits
from .serializers import DigitsSerializer

class DigitsViewSet(viewsets.ModelViewSet):
    serializer_class = DigitsSerializer
    queryset = Digits.objects.all()
