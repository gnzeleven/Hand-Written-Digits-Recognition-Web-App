from .apiviews import DigitsViewSet
from rest_framework.routers import DefaultRouter

router = DefaultRouter()
router.register(r'digits', DigitsViewSet)
urlpatterns = router.urls
