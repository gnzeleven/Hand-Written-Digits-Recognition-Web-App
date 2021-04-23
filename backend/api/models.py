from django.db import models
import torch
import os
import sys
sys.path.append("..")
from pathlib import Path
from PIL import Image
from classifier.predict import predict
from classifier.train import load_model

# Load the pretrained MNIST classifer
BASE_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = os.path.join(
                os.path.join(
                    os.path.join(BASE_DIR, 'classifier'),
                    'model'),
                'mnist_bcnn.pth')
IMAGE_PATH = os.path.join(
                os.path.join(BASE_DIR, 'src'),
                'media')
net = load_model()
net.load_state_dict(torch.load(MODEL_PATH))

# Create your models here.

class Digits(models.Model):
    image = models.ImageField(upload_to='images')
    result1 = models.CharField(max_length=2, blank=True)
    score1 = models.CharField(max_length=25, blank=True)
    result2 = models.CharField(max_length=2, blank=True)
    score2 = models.CharField(max_length=25, blank=True)
    updated = models.DateTimeField(auto_now=True)
    created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return str(self.id)

    def save(self, *args, **kwargs):
        '''
        override the save method to do the prediction
        before saving to the databsas
        '''
        #super().save(self.image)
        LOAD_IMAGE_PATH = os.path.join(IMAGE_PATH, str(self.image))
        image = Image.open(self.image)
        preds, scores = predict(image, net)
        self.result1 = preds[0]
        self.score1 = round(scores[0], 3)
        self.result2 = preds[1]
        self.score2 = round(scores[1], 3)
        return super().save(*args,**kwargs)
