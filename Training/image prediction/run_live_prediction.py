# Runner script for live age prediction

import subprocess
import os

def main():
    print("Age Prediction Options:")
    print("1. Live webcam prediction")
    print("2. Photo prediction")
    
    choice = input("Enter choice (1 or 2): ").strip()
    
    if choice == "1":
        print("Starting live webcam prediction...")
        print("Press 'q' to quit, 's' to save photo")
        subprocess.run(["python", "live_age_prediction.py"])
    
    elif choice == "2":
        image_path = input("Enter path to image file: ").strip()
        if os.path.exists(image_path):
            subprocess.run(["python", "photo_age_prediction.py", "--image", image_path])
        else:
            print(f"Image file not found at {image_path}")
    
    else:
        print("Invalid choice")

if __name__ == '__main__':
    main()
