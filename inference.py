import os
import cv2
import time
import pickle
import face_recognition
from collections import Counter
import tkinter as tk
from tkinter import filedialog

# Function to open a file dialog for image selection
def choose_image():

    # Create a hidden Tkinter window
    root = tk.Tk()
    root.withdraw()  

    # Open a file dialog to choose an image and return the selected file path
    file_path = filedialog.askopenfilename(title="Choose an image")
    return file_path

# Function to recognize faces in the selected image
def recognize_faces():

    ti = time.time()
    print('[INFO] Loading encodings...')

    # Check if the encodings file exists
    if not os.path.exists('encodings.pickle'):
        return None, "Error: Encodings file not found."

    # Load the precomputed facial encodings from the encodings.pickle file
    data = pickle.loads(open('encodings.pickle', 'rb').read())

    # Use the choose_image function to select an image
    image_path = choose_image()

    if not image_path:
        return None, "No image selected or image not found."

    # Read the selected image and convert it to RGB format (required for face_recognition)
    image = cv2.imread(image_path)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    print('[INFO] Recognizing faces...')

    # Detect face locations in the selected image
    boxes = face_recognition.face_locations(rgb, model='cnn')

    # Compute face encodings for each detected face in the image
    encodings = face_recognition.face_encodings(rgb, boxes)

    names = []

    # Compare the computed encodings with known encodings to identify individuals
    for encoding in encodings:

        votes = face_recognition.compare_faces(data['encodings'], encoding)

        if True in votes:

            # Find the most common name (voted) among the recognized faces
            matches = [name for name, vote in zip(data['names'], votes) if vote]
            name = Counter(matches).most_common()[0][0]
        else:
            name = 'Unknown'
        names.append(name)  

    # Print the recognized names with title case formatting
    print([' '.join([e.title() for e in name.split('_')]) for name in names])

    # Draw rectangles around detected faces and label them with names
    for ((top, right, bottom, left), name) in zip(boxes, names):
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

    # Define the output path for the annotated image
    output_path = image_path.rsplit('.', 1)[0] + '_output.jpg'
    cv2.imwrite(output_path, image)

    return output_path, None

if __name__ == "__main__":
    output_path, error_message = recognize_faces()

    if error_message:
        print(error_message)
    else:
        print(f"Output image saved to: {output_path}")
