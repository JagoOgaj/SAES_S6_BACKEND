import base64
from typing import Self
from backend.app.core.const.enum import ENUM_MODELS_TYPE, ENUM_MODELS_ENV, ENUM_CLASSES
from backend.app.exeptions import ModelTypeNotFoundError
import numpy as np
import tensorflow as tf
import os
from dotenv import load_dotenv
from PIL import Image
import face_recognition
import io
import torch
import torchvision.transforms as transforms
from PIL import Image
from facenet_pytorch import MTCNN
from ultralytics import YOLO

load_dotenv()


class Service_MODEL:

    _model = None

    _ethnieModel: dict = {}

    def __init__(self: Self, typeModel: str) -> None:
        if typeModel not in (
            [
                ENUM_MODELS_TYPE.GENDER_AND_AGE_SCRATCH.value,
                ENUM_MODELS_TYPE.GENDER_SCRATCH.value,
                ENUM_MODELS_TYPE.AGE_SCRATCH.value,
                ENUM_MODELS_TYPE.GENDER_AND_AGE_TRANSFER.value,
                ENUM_MODELS_TYPE.ETHNIE_AGE_GENDER_TRANSFER.value,
            ]
        ):
            raise ModelTypeNotFoundError(typeModel)
        self._typeModel = typeModel

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.map_location = torch.device(self.device)
        self.mtcnn = MTCNN(
            image_size=224, margin=20, keep_all=False, device=self.device
        )

        match typeModel:
            case ENUM_MODELS_TYPE.GENDER_AND_AGE_SCRATCH.value:
                self._model = tf.keras.models.load_model(
                    str(os.environ.get(ENUM_MODELS_ENV.PATH_GENDER_AGE_MODEL.value))
                )
            case ENUM_MODELS_TYPE.GENDER_SCRATCH.value:
                self.initGenderScratch()
            case ENUM_MODELS_TYPE.AGE_SCRATCH.value:
                self._model = tf.keras.models.load_model(
                    str(os.environ.get(ENUM_MODELS_ENV.PATH_AGE_MODEL.value))
                )
            case ENUM_MODELS_TYPE.GENDER_AND_AGE_TRANSFER.value:
                self._model = tf.keras.models.load_model(
                    str(
                        os.environ.get(ENUM_MODELS_ENV.PATH_GENDER_AGE_FINETUNING.value)
                    )
                )
            case ENUM_MODELS_TYPE.ETHNIE_AGE_GENDER_TRANSFER.value:
                self.model_yolo = YOLO(
                    os.environ.get(ENUM_MODELS_ENV.PATH_YOLO.value)
                ).to(self.device)
                self._ethnieModel.update(self.initEthnieModels())

    def handle_prediction(self: Self, imageFile) -> str:
        if self._typeModel == ENUM_MODELS_TYPE.ETHNIE_AGE_GENDER_TRANSFER.value:
            return self.get_prediction_with_ethnicity(imageFile)
        else:
            return self.get_prediction(imageFile)

    def predict_gender_yolo(self, face_image):
        results = self.model_yolo(face_image)

        CONFIDENCE_THRESHOLD = 0.4

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = box.conf[0].item()

                if confidence < CONFIDENCE_THRESHOLD:
                    continue

                return (
                    ENUM_CLASSES.CLASS_NAMES_GENDER.value.get(class_id, "Inconnu"),
                    confidence,
                )

        return "Inconnu", 0.0

    def initGenderScratch(self) -> None:
        self._model = tf.keras.models.load_model(
                str(os.environ.get(ENUM_MODELS_ENV.PATH_GENDER_MODEL.value))
        )

    def get_prediction(self: Self, imageFile) -> str:
        cropped_faces = self.loadImageFile(imageFile)
        if isinstance(cropped_faces, str):
            return cropped_faces
        gender_results = []
        age_results = []

        for face_image in cropped_faces:
            match self._typeModel:
                case ENUM_MODELS_TYPE.GENDER_SCRATCH.value:
                    gender, _ = self.predict_gender(face_image)
                    gender_results.append(gender)
                case ENUM_MODELS_TYPE.AGE_SCRATCH.value:
                    age, _ = self.predict_age(face_image)
                    age_results.append(age)
                case ENUM_MODELS_TYPE.GENDER_AND_AGE_SCRATCH.value:
                    gender, age = self.predict_gender_age(face_image)
                    gender_results.append(gender)
                    age_results.append(age)
                case ENUM_MODELS_TYPE.GENDER_AND_AGE_TRANSFER.value:
                    gender, age = self.predict_gender_age_transfer(face_image)
                    gender_results.append(gender)
                    age_results.append(age)
                case _:
                    raise ModelTypeNotFoundError(self._typeModel)

        num_faces = len(cropped_faces)
        avg_age = int(np.mean(age_results)) if age_results else None

        if num_faces == 1:
            message = "J'ai détecté un visage sur cette image.\n"
        else:
            message = f"J'ai trouvé {num_faces} visages sur cette image.\n"

        if gender_results:
            if num_faces == 1:
                message += f"La personne semble être un(e) {gender_results[0]}.\n"
            else:
                message += (
                    "Voici les genres détectés : " + ", ".join(gender_results) + ".\n"
                )

        if avg_age is not None:
            if num_faces == 1 :    
                message += (
                    "Voici la tranche d'âge que j'ai trouvée : \n"
                    + self.categorize_age(age_results[0])
                    + ".\n"
                )
            else :
                message += (
                    "Voici les tranches d'âge que j'ai trouvées : \n" +
                    ", ".join(map(self.categorize_age, age_results))
                    + ".\n"
                )          
                message += (
                    f"L'âge moyen des visages détectés est d'environ {avg_age} ans.\n"
                )  

        if len(gender_results) > 1:
            message += (
                f"Répartition des genres : {self.get_gender_average(gender_results)}.\n"
            )

        return message

    def categorize_age(self, age):
        if age <= 10:
            return "0 - 10 ans"
        elif age <= 20:
            return "11 - 20 ans"
        elif age <= 30:
            return "21 - 30 ans"
        elif age <= 40:
            return "31 - 40 ans"
        elif age <= 50:
            return "41 - 50 ans"
        elif age <= 60:
            return "51 - 60 ans"
        elif age <= 70:
            return "61 - 70 ans"
        elif age <= 80:
            return "71 - 80 ans"
        elif age <= 90:
            return "81 - 90 ans"
        elif age <= 100:
            return "91 - 100 ans"
        else:
            return "101 - 120 ans"

    def get_prediction_with_ethnicity(self: Self, imageFile) -> str:
        cropped_faces = self.loadImageFile(imageFile)

        if isinstance(cropped_faces, str):
            return cropped_faces

        age_model = self._ethnieModel["age_model"]
        ethnicity_model = self._ethnieModel["ethnicity_model"]

        age_transform, ethnicity_transform = self.getFeaturesTransfor()

        gender_results, age_results, ethnicity_results = [], [], []

        for face in cropped_faces:
            face_pil = face.convert("RGB")

            face_age = age_transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                age_pred = round(age_model(face_age).item())
            age_results.append(age_pred)

            gender_pred, _ = self.predict_gender_yolo(face_pil)
            if gender_pred == "Inconnu":
                self.initGenderScratch()
                gender_pred, _ = self.predict_gender(face)
            gender_results.append(gender_pred)

            face_ethnicity = ethnicity_transform(face_pil).unsqueeze(0).to(self.device)
            with torch.no_grad():
                ethnicity_outputs = ethnicity_model(face_ethnicity)
                _, ethnicity_class = torch.max(ethnicity_outputs, 1)
            ethnicity_pred = ENUM_CLASSES.ETHNICITY.value[ethnicity_class.item()]
            ethnicity_results.append(ethnicity_pred)

        num_faces = len(cropped_faces)
        avg_age = int(np.mean(age_results)) if age_results else None

        if num_faces == 1:
            message = "J'ai détecté un visage sur cette image.\n"
        else:
            message = f"J'ai trouvé {num_faces} visages sur cette image.\n"

        if gender_results:
            if num_faces == 1:
                message += f"La personne semble être un(e) {gender_results[0]}.\n"
            else:
                message += (
                    "Voici les genres détectés : " + ", ".join(gender_results) + ".\n"
                )

       
        if avg_age is not None:
            if num_faces == 1 :    
                message += (
                    "Voici la tranche d'âge que j'ai trouvée : \n"
                    + self.categorize_age(age_results[0])
                    + ".\n"
                )
            else :
                message += (
                    "Voici les tranches d'âge que j'ai trouvées : \n" +
                    ", ".join(map(self.categorize_age, age_results))
                    + ".\n"
                )          
                message += (
                    f"L'âge moyen des visages détectés est d'environ {avg_age} ans.\n"
                )  

        if ethnicity_results:
            if num_faces == 1:
                message += f"Elle semble être d'origine {ethnicity_results[0]}.\n"
            else:
                message += (
                    "Origines détectées : " + ", ".join(ethnicity_results) + ".\n"
                )

        if len(gender_results) > 1:
            message += (
                f"Répartition des genres : {self.get_gender_average(gender_results)}.\n"
            )

        return message

    def loadImageFile(self, imageFile) -> list:

        image = Image.open(io.BytesIO(imageFile.read()))

        face_locations = self.detect_faces(image)

        if not face_locations:
            return "Je n'ai détecté aucun visage sur cette image. Assurez-vous qu'il est bien visible et réessayez !"

        return self.crop_faces(image, face_locations)

    def get_gender_average(self, gender_results):
        female_count = gender_results.count("Femme")
        male_count = gender_results.count("Homme")
        total = female_count + male_count
        if total == 0:
            return "Indéfini"
        female_percentage = (female_count / total) * 100
        return (
            f"Femme : {female_percentage:.2f}%, Homme : {100 - female_percentage:.2f}%"
        )

    def ensure_8bit(self, image):
        image_array = np.array(image)
        if image_array.dtype != np.uint8:
            image_array = (image_array / (image_array.max() / 255)).astype(np.uint8)
            image = Image.fromarray(image_array, mode=image.mode)

        if image.mode not in ["RGB", "L"]:
            image = image.convert("RGB")

        return image

    def detect_faces(self, image):
        image = self.ensure_8bit(image)
        image_array = np.array(image)
        face_locations = face_recognition.face_locations(image_array)
        return face_locations

    def crop_faces(
        self, image, face_locations, target_size=(200, 200), margin_ratio=0.1
    ):
        cropped_faces = []
        img_width, img_height = image.size

        if len(face_locations) == 1:
            top, right, bottom, left = face_locations[0]
            face_width = right - left
            face_height = bottom - top
            if img_width == 200 and img_height == 200:
                if 120 <= face_width <= 160 and 120 <= face_height <= 160:
                    return [image]

        for _, (top, right, bottom, left) in enumerate(face_locations):
            face_width = right - left
            face_height = bottom - top

            margin_width = int(face_width * margin_ratio)
            margin_height = int(face_height * margin_ratio)

            new_left = max(0, left - margin_width)
            new_top = max(0, top - margin_height)
            new_right = min(img_width, right + margin_width)
            new_bottom = min(img_height, bottom + margin_height)

            face_image = image.crop((new_left, new_top, new_right, new_bottom))
            face_image = face_image.resize(target_size, Image.Resampling.LANCZOS)

            cropped_faces.append(face_image)

        return cropped_faces

    def preprocess_gender(self, face_image):
        image_gray = face_image.convert("L").resize((100, 100))
        image_array = np.array(image_gray) / 255.0
        return np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

    def preprocess_age(self, face_image):
        image_gray = face_image.convert("L").resize((128, 128))
        image_array = np.array(image_gray) / 255.0
        return np.expand_dims(np.expand_dims(image_array, axis=0), axis=-1)

    def preprocess_gender_age(self, face_image):
        image_gray = face_image.convert("RGB").resize((200, 200))
        image_array = np.array(image_gray) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array

    def preprocess_gender_age_transfer(self, face_image):
        image = face_image.resize((180, 180))
        if image.mode == "RGBA":
            image = image.convert("RGB")
        return np.array(image)

    def predict_gender(self, face_image):
        processed_image = self.preprocess_gender(face_image)
        prediction = self._model.predict(processed_image)
        return "Homme" if prediction[0][0] > 0.5 else "Femme", float(prediction[0][0])

    def predict_age(self, face_image):
        processed_image = self.preprocess_age(face_image)
        prediction = self._model.predict(processed_image)
        return int(round(prediction[0][0] * 116)), float(prediction[0][0])

    def predict_gender_age(self, face_image):
        processed_image = self.preprocess_gender_age(face_image)
        gender_age_prediction = self._model.predict(processed_image)
        predicted_gender = ENUM_CLASSES.CLASS_NAMES_GENDER.value[
            round(gender_age_prediction[0][0][0])
        ]
        predicted_age = round(gender_age_prediction[1][0][0])
        return predicted_gender, predicted_age

    def predict_gender_age_transfer(self, face_image):
        image = self.preprocess_gender_age_transfer(face_image)
        prediction = self._model.predict(np.expand_dims(image, axis=0))
        age, gender = round(prediction[0][0][0]), round(prediction[1][0][0])
        return ENUM_CLASSES.CLASS_NAMES_GENDER.value.get(gender), age

    def getFeaturesTransfor(self) -> tuple:

        age_transform = ethnicity_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        return age_transform, ethnicity_transform

    def load_ckpt_weights(self, model, ckpt_path, device="cpu"):
        checkpoint = torch.load(ckpt_path, map_location=torch.device(device))

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
        new_state_dict = {}
        for key, value in state_dict.items():
            new_key = key.replace("model.", "")
            new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)
        model.to(device)
        model.eval()
        return model

    def initAgeModel(self):
        from timm import create_model

        age_model = create_model("tf_efficientnetv2_m", pretrained=False, num_classes=1)
        age_model.load_state_dict(
            torch.load(
                os.environ.get(ENUM_MODELS_ENV.PATH_AGE_ETHNIE_MODEL.value),
                map_location=self.map_location,
            )
        )
        age_model = age_model.to(self.map_location)
        age_model.eval()
        return age_model

    def initEthnieModel(self):
        from timm import create_model

        num_classes = 5
        model = create_model(
            "tf_efficientnetv2_m", pretrained=True, num_classes=num_classes
        )
        model.load_state_dict(
            torch.load(
                os.environ.get(ENUM_MODELS_ENV.PATH_ETHNIE_MODEL.value),
                map_location=self.map_location,
            )
        )
        model = model.to(self.map_location)
        model.eval()
        return model

    def initEthnieModels(self) -> dict:

        age_model = self.initAgeModel()

        ethnicity_model = self.initEthnieModel()

        return {"age_model": age_model, "ethnicity_model": ethnicity_model}
