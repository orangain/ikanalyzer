import os

import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.data import DataLoader
from PIL import Image


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


def create_model(num_classes):
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    model = model.to(device)
    return model


def train_model(dataset_path, save_path, num_epochs=10):
    dataset = datasets.ImageFolder(dataset_path, transform=transform)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = create_model(num_classes=len(dataset.classes))

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader):.4f}")

    checkpoint = {
        "model_state_dict": model.state_dict(),
        "class_names": dataset.classes,
    }
    torch.save(checkpoint, save_path)


class Classifier:
    def __init__(self, model_path):
        # モデルとクラス情報を読み込み
        checkpoint = torch.load(model_path, map_location=device)
        self.class_names = checkpoint["class_names"]

        self.model = create_model(num_classes=len(self.class_names))
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()

    def predict(self, image: cv2.typing.MatLike) -> str:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(image_rgb)
        img_tensor = transform(img).unsqueeze(0).to(device)

        topk = 3
        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)  # スコアを確率にする
            topk_probs, topk_indices = torch.topk(probs, k=topk, dim=1)

        results = []
        for i in range(topk):
            label = self.class_names[topk_indices[0, i].item()]
            score = topk_probs[0, i].item()
            results.append((label, score))

            # print(outputs)
            # _, preds = torch.max(outputs, 1)

        # for label, score in results:
        #     print(f"Predicted: {label}, Score: {score:.4f}")

        return results[0][0]


if __name__ == "__main__":
    import sys

    image_path = sys.argv[1]
    classifier = Classifier(
        model_path=os.path.join("classifiers", "weapon_classifier.pth")
    )
    image = cv2.imread(image_path)
    result = classifier.predict(image)
    print(f"Predicted: {result}")
