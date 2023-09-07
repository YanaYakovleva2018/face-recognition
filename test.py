from imutils import paths
import face_recognition
import cv2
import os
import pickle
import time
from collections import Counter

# Record the start time for measuring execution time
ti = time.time()
print('[INFO] creating facial embeddings...')

# Try to load precomputed facial encodings from a pickle file
try:
    data = pickle.loads(open('encodings.pickle', 'rb').read()) 
except FileNotFoundError:

    # If the pickle file doesn't exist, compute facial encodings from images
    knownEncodings, knownNames = [], []

    # List all image file paths in the 'dataset' directory
    imagePaths = list(paths.list_images('dataset')) 

    # Loop through each image path and process it
    for (i, imagePath) in enumerate(imagePaths):
        print('{}/{}'.format(i + 1, len(imagePaths)), end=', ')

        # Read the image and extract the person's name from the file path
        image, name = cv2.imread(imagePath), imagePath.split(os.path.sep)[-2]

        # Convert the image to RGB format (required by face_recognition)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations in the image
        boxes = face_recognition.face_locations(rgb, model='cnn')

        # Compute face encodings for each detected face in the image
        for encoding in face_recognition.face_encodings(rgb, boxes):
            knownEncodings.append(encoding)
            knownNames.append(name)

    # Store the computed encodings and associated names in a dictionary
    data = {'encodings': knownEncodings, 'names': knownNames}

    # Save the dictionary to a pickle file for future use
    with open('encodings.pickle', 'wb') as f:
        f.write(pickle.dumps(data))

print('Done! \n[INFO] recognizing faces in images...')

# List all image file paths in the 'image_test' directory
imagePaths = list(paths.list_images('image_test'))

# Loop through each image path and perform facial recognition
for (_, imagePath) in enumerate(imagePaths):

    if '_output' not in imagePath:

        # Read the image and convert it to RGB format
        image = cv2.imread(imagePath)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Detect face locations in the image
        boxes = face_recognition.face_locations(rgb, model='cnn')

        # Compute face encodings for each detected face in the image
        encodings = face_recognition.face_encodings(rgb, boxes)
        names = []

        # Compare the computed encodings with known encodings to identify individuals
        for encoding in encodings:
            votes = face_recognition.compare_faces(data['encodings'], encoding)
            if True in votes:

                # Find the most common name (voted) among the recognized faces
                names.append(Counter([name for name, vote in list(zip(data['names'], votes)) if vote]).most_common()[0][0])
            else:

                # If no match is found, label the face as 'Unknown'
                names.append('Unknown')
                
        # Draw rectangles around detected faces and label them with names
        for ((top, right, bottom, left), name) in zip(boxes, names):
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
            y = top - 15 if top - 15 > 15 else top + 15
            cv2.putText(image, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)
        
        #output_path = imagePath.rsplit('.', 1)[0] + '_output.jpg'
        #cv2.imwrite(output_path, image)

# Print a message indicating that facial recognition is done and display the execution time
print('Done! \nTime taken: {:.1f} minutes'.format((time.time() - ti) / 60))
