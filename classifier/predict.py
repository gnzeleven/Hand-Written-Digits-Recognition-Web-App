import torch
from PIL import Image
import numpy as np
from .train import load_model

def predict(image, model=None):
    '''
    Function to predict given a single image
    :params image: PIL Image object
    :return prediction: predicted output - an integer from 0 to 9
    :return score: probability of the prediction - float ranges 0.0 to 1.0
    '''
    # Load pretrained model
    MODEL_PATH = './model/mnist_bcnn.pth'
    if model == None:
        model = load_model()
        model.load_state_dict(torch.load(MODEL_PATH))

    # Preprocess the image to tensor of shape(1, 1, 28, 28)
    image = image.resize((28,28))
    img_np = np.array(image)
    img_np = img_np[:,:,1]
    img_np = img_np.reshape(1, -1, 28, 28)
    img_torch = torch.from_numpy(img_np)

    # Predict
    out = model(img_torch.float())

    # Top two predictions and their probability
    scores_t, preds_t = torch.topk(out, 2, largest=True, sorted=True)
    preds = []
    scores = []
    for i in range(2):
        preds.append(int(preds_t[0][i]))
        scores.append(float(torch.exp(scores_t[0][i])))

    return preds, scores

if __name__ == '__main__':
    image_path = "./sample/digit.jpg"
    image = Image.open(image_path)
    pred, score = predict(image)
    print("predicted: {}, score: {}".format(pred[0], score[0]))
