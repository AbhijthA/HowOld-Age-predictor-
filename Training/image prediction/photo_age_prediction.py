"""
Age prediction from photo using trained ResNet model.
"""

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import argparse

# Original age classes
AGE_CLASSES = ["0;2", "3;9", "10;19", "20;29", "30;39", "40;49", "50;59", "60;69", "70;120"]

def load_model(model_path, device):
    """Load trained ResNet model"""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, 9)  # 9 classes
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model

def get_transform():
    """Get image preprocessing transform"""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def predict_age_from_photo(image_path, model_path):
    """Predict age from photo file"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = load_model(model_path, device)
    transform = get_transform()
    
    # Load and process image
    img = cv2.imread(image_path)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("No faces detected in the image")
        return
    
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face
        face = img[y:y+h, x:x+w]
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        
        # Predict
        input_tensor = transform(pil_img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(input_tensor)
            probs = torch.softmax(outputs, dim=1)
            pred_idx = torch.argmax(outputs, dim=1).item()
            confidence = probs[0][pred_idx].item()
        
        age_group = AGE_CLASSES[pred_idx]
        print(f"Face {i+1}: Age group {age_group} (confidence: {confidence:.3f})")
        
        # Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        text = f"{age_group} ({confidence:.2f})"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
    
    # Save result
    output_path = image_path.replace('.', '_predicted.')
    cv2.imwrite(output_path, img)
    print(f"Result saved to: {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    parser.add_argument('--model', type=str, 
                       default=r"C:\Users\abhij\Desktop\cognizant hackathon\model\resnet_age_best.pth",
                       help='Path to model file')
    
    args = parser.parse_args()
    predict_age_from_photo(args.image, args.model)