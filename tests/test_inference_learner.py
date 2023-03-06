import unittest
from fastinference.inference import InferenceLearner
from fastai.vision.learner import vision_learner
from fastai.data.external import untar_data, URLs
from fastai.vision.data import get_image_files, ImageDataLoaders
from fastai.vision.augment import Resize
from fastai.metrics import error_rate

class TestInferenceLearner(unittest.TestCase):
    def setUp(self):
        path = untar_data(URLs.PETS)/'images'
        self.images = get_image_files(path)
        dataloaders = ImageDataLoaders.from_name_func(
            path, get_image_files(path), valid_pct=0.2, seed=42,
            label_func=lambda x: x[0].isupper(), item_tfms=Resize(224)
        )
        self.learn = vision_learner(dataloaders, arch='resnet34', metrics=error_rate)
    
    def test_predict(self):
        inference_learner = InferenceLearner(self.learn)
        probabilities, decoded_outputs = inference_learner.predict(self.images[0])
        self.assertEqual(probabilities.shape, (2,))
        self.assertEqual(decoded_outputs.shape, (1,))
    
    def test_predict_with_input(self):
        inference_learner = InferenceLearner(self.learn)
        input, probabilities, decoded_outputs = inference_learner.predict(self.images[0], with_input=True)
        self.assertEqual(input.shape, (3, 224, 224))
        self.assertEqual(probabilities.shape, (2,))
        self.assertEqual(decoded_outputs.shape, (1,))

