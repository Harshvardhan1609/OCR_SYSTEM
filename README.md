# OCR PROJECT ![GitHub license](https://img.shields.io/badge/license-MIT-blue.svg)

### NOTES LINK : CLICK_HERE(https://drive.google.com/file/d/1vGqFZzCO2330xXFtFHXBHQtQ6xHq-Bul/view?usp=sharing)

# WHAT DO YOU MEAN BY OCR ?

- Ocr stands for optical character recognition .
- It is an technology that recognizes text within an digital image .
- It is commonly used in order to recognize text in the scanned  documents and images .
- Using ocr can convert physical paper document into the accessible electronic version of it .

## STEPS INVOLVED IN OCR

- Image acquisition  = process of scanning paper document .
- Pre-processing  = Converting scanned image into gray level image .
- Feature extraction = Process of converting preprocessed image into single characters .
- Classification = It is the process of converting feature extraction into feature vector .
- Post-processing = It is process of converting classification into classified characters.
- After post processing the paper document text is converted into classified text .

# OCR USING EASY OCR

## GOAL

- We will be using python and EasyOCR in order to make an text based detections.
- Then we will be going to visualize results using OpenCV.

## STEPS

- For this project we will be using an jupiter notebook .
- So for doing this project we will follow some of the steps :

### INSTALLING AND IMPORTING REQUIRED DEPENDENCIES

- PYTORCH
    - It is an tensor flow library .
    - It is used for the deep learning applications using GPU's and CPU's .
    - It is an open source library for python and developed by the facebook AI research team .
    - It can be stored using the pip command available on the wesite which is :
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled.png)
    
    ```python
    pip3 install torch torchvision torchaudio
    ```
    

- EASYOCR
    - It is an python package which converts the image into the text .
    - This is available in 70+ different languages including english , hindi , chinese etc .
    - Easy OCR is created by  Jaided AI company .
    - For installing easy OCR consider the following pip command .
    - link : [https://github.com/JaidedAI/EasyOCR](https://github.com/JaidedAI/EasyOCR)
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%201.png)
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%202.png)
    
    ```python
    pip install easyocr
    ```
    

### IMPORTING THE REQUIRED LIBRARIES USED IN THE OCR

- EASYOCR
    - It is used to convert the images into the text .
- CV2
    - This is special library of python designed to solve computer vision problems .
    - imread() method loads an image from the specified file .
    - If the image cannot be read then it will return an empty matrix .
    - All the documentation and further information is given here : 
    [https://pypi.org/project/opencv-python/](https://pypi.org/project/opencv-python/)
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%203.png)
    
- MATPLOT LIBRARY
    - This python library is an amazing visualization library is used for plotting arrays in 2D .
    - It is an multi-platform data visualization library build on NumPy arrays .
    - It provides an object oriented API for embidding plots into application using genral purpose GUI toolkits like Tkinter , wxPython , Qt .
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%204.png)
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%205.png)
    
    - From here you can download the cheat sheets of matplot library .
    - link : [https://github.com/matplotlib/cheatsheets#cheatsheets](https://github.com/matplotlib/cheatsheets#cheatsheets)

- NUMPY
    - It stands for numerical python .
    - It consists of multi-dimensional array objects and a collection of routines for processing those arrays .
    - Using NumPy mathematical and logical operations on arrays can be performed .
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%206.png)
    

### READING IMAGES

- So first we will choose an image to work upon here we have choosen img1.jpg .
- And then we will assign it an variable .
- So now we will use the easyocr in order to start working upon the ocr process .
- so the code for that is given below :
    - Here the en is an representation for the english language and you can change it according to language purposes .
    - gpu = False it is used to tell the complier that our computer does not have an gpu and if you have and want to use then you can write the true in it .

```python
readed_image = easyocr.Reading(['en'],gpu=False)
readed_image = reader.readtext(IMG_PATH)
readed_image
```

- So using the above code we are getting the ocr results as the given below :
    - The image we have used .
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%207.png)
    
    - The output we are getting right now is :
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%208.png)
    
    - So your output can be readed as :
    
    ![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%209.png)
    

### PLOTTING THE RESULTS AND USING OPEN CV

- So for plotting the results we will use open-cv .
- So we require top left corner and bottom right corner value .
- In this case it is [18,18] , [293,145] .
- So Open cv takes the values in the form of tuples , so we will given them the values in the top left and bottom right corner and then we are also given the text to them  and assigning and font to it .
- So this is the following code for the following .

```python
top_left = tuple(readed_image[0][0][0]) #Taken top_left cordinates from the text extraction done by easy_ocr
bottom_right = tuple(readed_image[0][0][2]) #Taken bottom_left coordinates from the text extraction done by easy_ocr
text = readed_image[0][1] #Taken image text  done by easy ocr 
font = cv2.FONT_HERSHEY_SIMPLEX #from open cv font is choosen for the text.
img = cv2.imread(IMG_PATH) #By this we are using the open cv in order to read the image file
img = cv2.rectangle(img,top_left,bottom_right,(0,255,0), 5) #By this we are providing an rectangle overlay to the image
img = cv2.putText(img,text,top_left,font,.5,(255,255,255),2,cv2.LINE_AA) #By this we are writing the text for the image
plt.imshow(img)
plt.show
```

![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%2010.png)

![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%2011.png)

- The above image we are getting after doing all the functionalities .

### NOW IF YOU ARE HAVING THE IMAGES WITH MULTIPLE LINES

- So if you are having the multiple lines then you can use the looping method in order to loop through the entire process again .
- Image before processing :

![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%2012.png)

- Then the code for that is given below :

```python
img = cv2.imread(IMG_PATH)
for detection in readed_image:
    top_left = tuple(int(val) for val in detection[0][0])
    bottom_right = tuple(int(val) for val in detection[0][2])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img , top_left , bottom_right ,(0,255,0),5)
    img  = cv2.putText(img,text,top_left,font,.3,(255,256,255),1,cv2.LINE_AA)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()
```

- In the first part of it we are getting an image .
- Then we are looping till all the lines are completed .
- then we are assigning the top_left coordinate from the output array .
- Similarly we are assigning values to all the coordinates and text by using the output array .
- Then we are using the open cv libnrary in order to get the rectangular overlay .
- Then we are again using cv library in order to place our text with the desires thickness and size .
- Then we are using plotting library to plot the image and show the condition of the image .
- Image after processing

![Untitled](OCR%20PROJECT%205f144c78ea54451a98b53f12293c12e5/Untitled%2013.png)

## ENTIRE CODE OF THE PROJECT

- Please change the image path for the use .

```python
!pip install torch torchvision torchaudio
!pip install easyocr

import easyocr
import cv2
from matplotlib import pyplot as plt
import numpy as np

#Reading images

IMG_PATH = 'img2.jpg' #Here we have assigned the image into the variable

reader = easyocr.Reader(['en'],gpu=False) #initializing the easyocr engine with english language and gpu as false 
readed_image = reader.readtext(IMG_PATH) #now using the engine to read the text from image.
readed_image #evantually the output

# USING THE PLOTTING LIBRARY TO VISUALIZE THE OUTPUT WE ARE GETTING FROM THE IMAGE

top_left = tuple(readed_image[0][0][0]) #Taken top_left cordinates from the text extraction done by easy_ocr
bottom_right = tuple(readed_image[0][0][2]) #Taken bottom_left coordinates from the text extraction done by easy_ocr
text = readed_image[0][1] #Taken image text  done by easy ocr 
font = cv2.FONT_HERSHEY_SIMPLEX #from open cv font is choosen for the text.

# VISULIZING THE IMAGE USING THE OPEN CV

img = cv2.imread(IMG_PATH) #By this we are using the open cv in order to read the image file
img = cv2.rectangle(img,top_left,bottom_right,(0,255,0), 5) #By this we are providing an rectangle overlay to the image
img = cv2.putText(img,text,top_left,font,.5,(255,255,255),2,cv2.LINE_AA) #By this we are writing the text for the image
plt.imshow(img)
plt.show

# VISUALIZING THE IMAGE WITH MULTIPLE LINES

img = cv2.imread(IMG_PATH)
for detection in readed_image:
    top_left = tuple(int(val) for val in detection[0][0])
    bottom_right = tuple(int(val) for val in detection[0][2])
    text = detection[1]
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.rectangle(img , top_left , bottom_right ,(0,255,0),5)
    img  = cv2.putText(img,text,top_left,font,.3,(255,256,255),1,cv2.LINE_AA)

plt.figure(figsize=(10,10))
plt.imshow(img)
plt.show()
```
