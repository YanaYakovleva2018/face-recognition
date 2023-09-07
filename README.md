This GitHub repository contains code for performing face recognition using a pre-trained feature encoder. 
The code consists of two main scripts, test.py and inference.py, each serving a specific purpose.
## Overview
`test.py` is a Python script that performs face recognition on a set of test images using the [face_recognition library](https://github.com/ageitgey/face_recognition), OpenCV, and Python. 
It uses precomputed facial embeddings to recognize individuals in the test images. 
The code detects and annotates recognized faces in the images, ultimately creating a visual output with identified names.

`inference.py` is a simple graphical user interface (GUI) application for recognizing faces in an image using the face_recognition library, OpenCV, and Python. 
The application allows users to select an image file, and it automatically detects and recognizes faces in the chosen image. 
The recognized faces are then annotated with names and saved as an output image.

1. Make sure you have the necessary libraries installed. You can install them using `pip`:
```bash
pip install face_recognition opencv-python
```
2. Place your known dataset images in the 'dataset' directory, organizing them into subdirectories by name.
3. Run the script:
```bash
python test.py
```
4. Optionally, provide the path to the 'image_test' directory containing test images when prompted.
5. The script will process the images, recognize faces, annotate them, and save the annotated images (if chosen).
6. Run the script
```bash
python inference.py
```
7. Use the graphical interface to select an image file.
8. The script will process the selected image, recognize faces, annotate them, and save the annotated image.

![test4_output](https://github.com/YanaYakovleva2018/face-recognition/assets/40498328/80f8f91b-16e2-402b-a5ae-2da3e93a9c56)

