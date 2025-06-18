import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image, UnidentifiedImageError
import requests
from io import BytesIO
from requests.exceptions import Timeout, RequestException
import warnings
import time

class FashionFeatureExtractor:
    def __init__(self, device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.preprocess = self._build_preprocess()

    def _load_model(self):
        model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove final classification layer
        model.eval().to(self.device)
        return model

    def _build_preprocess(self):
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def _safe_download(self, url, retries=3, timeout=10):
        for attempt in range(retries):
            try:
                headers = {"User-Agent": "Mozilla/5.0"}
                response = requests.get(url, headers=headers, timeout=timeout)
                response.raise_for_status()
                return response.content
            except Timeout:
                warnings.warn(f"[Timeout] Attempt {attempt + 1}/{retries} for URL: {url}")
            except RequestException as e:
                warnings.warn(f"[HTTP Error] {url}: {e}")
                break
            time.sleep(1)
        return None

    def extract_features_from_url(self, image_url):
        try:
            content = self._safe_download(image_url)
            if content is None:
                return None

            img = Image.open(BytesIO(content)).convert('RGB')
            return self._extract_tensor_features(img)

        except UnidentifiedImageError:
            warnings.warn(f"[Invalid Image] Cannot identify image file from URL: {image_url}")
        except Exception as e:
            warnings.warn(f"[Exception] Failed to extract features from URL: {image_url}. Error: {e}")
        return None

    def extract_features_from_path(self, image_path):
        try:
            img = Image.open(image_path).convert('RGB')
            return self._extract_tensor_features(img)

        except UnidentifiedImageError:
            warnings.warn(f"[Invalid Image] Cannot identify image file: {image_path}")
        except Exception as e:
            warnings.warn(f"[Exception] Failed to extract features from path {image_path}: {e}")
        return None

    def _extract_tensor_features(self, img):
        img_tensor = self.preprocess(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model(img_tensor)
        return features.squeeze().cpu().numpy()







