# Live age prediction using webcam and trained ResNet model

import cv2
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np

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

def predict_age(model, face_img, transform, device):
    """Predict age from face image"""
    if face_img.size == 0:
        return "Unknown", 0.0
    
    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(face_rgb)
    
    # Preprocess
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(input_tensor)
        probs = torch.softmax(outputs, dim=1)
        pred_idx = torch.argmax(outputs, dim=1).item()
        confidence = probs[0][pred_idx].item()
    
    return AGE_CLASSES[pred_idx], confidence

def predict_on_captured_image(model, image_path, transform, device):
    """Predict age on captured still image"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load captured image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Could not load image: {image_path}")
        return
    
    # Detect faces
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    if len(faces) == 0:
        print("No faces detected in captured image")
        return
    
    print(f"\nAge predictions for captured image:")
    # Process each face
    for i, (x, y, w, h) in enumerate(faces):
        # Extract face
        face = img[y:y+h, x:x+w]
        
        # Predict age
        age_group, confidence = predict_age(model, face, transform, device)
        
        print(f"Face {i+1}: {age_group} (confidence: {confidence:.3f})")
        
        # Draw on image
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        text = f"{age_group} ({confidence:.2f})"
        cv2.putText(img, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save and show result
    result_path = image_path.replace('.jpg', '_predicted.jpg')
    cv2.imwrite(result_path, img)
    print(f"Result saved as: {result_path}")
    
    # Display result
    cv2.imshow('Captured Image Prediction', img)
    print("Press any key to close the prediction window")
    cv2.waitKey(0)
    cv2.destroyWindow('Captured Image Prediction')

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = r"C:\Users\abhij\Desktop\cognizant hackathon\model\resnet_age_best.pth"
    
    # Load model
    model = load_model(model_path, device)
    transform = get_transform()
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Start webcam
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 'c' to capture and predict, 's' to save photo")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Detect faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        
        # Process each face
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]    # Extract face
            
            age_group, confidence = predict_age(model, face, transform, device)    # Predict age
            
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)    # Draw rectangle and text
            text = f"{age_group} ({confidence:.2f})"
            cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        cv2.imshow('Age Prediction', frame)    # Show frame
        
        # Handle keys
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            # Capture and predict
            capture_path = 'captured_image.jpg'
            cv2.imwrite(capture_path, frame)
            print(f"\nImage captured: {capture_path}")
            predict_on_captured_image(model, capture_path, transform, device)
        elif key == ord('s'):
            # Save photo
            cv2.imwrite('saved_photo.jpg', frame)
            print("Photo saved as saved_photo.jpg")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
