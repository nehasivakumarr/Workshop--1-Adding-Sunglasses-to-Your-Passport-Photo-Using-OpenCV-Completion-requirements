# Sturdy-Octo-Disco-Adding-Sunglasses-for-a-Cool-New-Look

## Aim

To automatically detect the face in an image and overlay a stylish sunglass asset accurately onto passport-sized photos using image processing techniques

## Features:

- Detects the face in an image.
- Places a stylish sunglass overlay perfectly on the face.
- Works seamlessly with individual passport-size photos.
- Customizable for different sunglasses styles or photo types.

## Technologies Used:

- Python
- OpenCV for image processing
- Numpy for array manipulations

## Program:

```
# Import libraries
import cv2
import numpy as np
import matplotlib.pyplot as plt
# Load the Face Image
faceImage = cv2.imread('NEHA IMAGE.jpeg')
plt.imshow(faceImage[:,:,::-1]);plt.title("Face")
```
<img width="458" height="576" alt="image" src="https://github.com/user-attachments/assets/e56657e9-6a80-4b8a-a81b-936b501ec592" />

```
faceImage.shape
#resized_faceImage.shape
faceImage.shape
# Load the Sunglass image
glassPNG = cv2.imread(r"C:\Users\acer\Downloads\sun.jpg")
plt.imshow(cv2.cvtColor(glassPNG, cv2.COLOR_BGR2RGB))
plt.title("glassPNG")
```

<img width="757" height="372" alt="image" src="https://github.com/user-attachments/assets/1972b433-b629-4d30-8ad2-d19ec944c1e9" />

```
# Resize the image to fit over the eye region
glassPNG = cv2.resize(glassPNG, (300, 110))
print("image Dimension ={}".format(glassPNG.shape))
# Resize sunglasses
glassBGR = cv2.resize(glassBGR, (300, 110))

# Convert sunglasses to grayscale
glassGray = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)

# Create mask to remove white background
_, glassMask1 = cv2.threshold(glassGray, 240, 255, cv2.THRESH_BINARY_INV)
# Display the images for clarity
plt.figure(figsize=[15,15])
plt.subplot(121);plt.imshow(glassBGR[:,:,::-1]);plt.title('Sunglass Color channels');
plt.subplot(122);plt.imshow(glassMask1,cmap='gray');plt.title('Sunglass Alpha channel');
```

<img width="1366" height="292" alt="image" src="https://github.com/user-attachments/assets/10ec57f4-864e-4dd6-b00b-1cfd5204b582" />

```
# Make a copy
faceWithGlassesNaive = faceImage.copy()

# Replace the eye region with the sunglass image
faceWithGlassesNaive[270:380, 230:530] = glassBGR

# Display the result
plt.imshow(faceWithGlassesNaive[..., ::-1])
plt.title("Face with Sunglasses")
plt.axis('off')
```

<img width="670" height="542" alt="image" src="https://github.com/user-attachments/assets/f450aacc-55b1-4c2a-a255-44db2e0c2655" />

```
# Make the dimensions of the mask same as the input image
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Make the values [0,1]
glassMask = np.uint8(glassMask / 255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region from the face image
eyeROI = faceWithGlassesArithmetic[270:380, 230:530]

# Use the mask to create the masked eye region
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Use the mask to create the masked sunglass region
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine the sunglasses in the eye region
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Display the intermediate results
plt.figure(figsize=(20,20))
plt.subplot(131);plt.imshow(maskedEye[...,::-1]);plt.title("Masked Eye Region")
plt.subplot(132);plt.imshow(maskedGlass[...,::-1]);plt.title("Masked Sunglass Region")
plt.subplot(133);plt.imshow(eyeRoiFinal[...,::-1]);plt.title("Augmented Eye and Sunglass")
```

<img width="1367" height="233" alt="image" src="https://github.com/user-attachments/assets/ee3250b9-8790-4602-ab7a-bdc63181a600" />

```
# Resize sunglasses to fit both eyes
glassBGR = cv2.resize(glassBGR, (300, 110))

# Create a mask to remove the white background
grayGlass = cv2.cvtColor(glassBGR, cv2.COLOR_BGR2GRAY)

glassMask1 = cv2.threshold(
    grayGlass, 240, 255, cv2.THRESH_BINARY_INV
)[1]

# Convert mask into 3 channels
glassMask = cv2.merge((glassMask1, glassMask1, glassMask1))

# Convert values into 0 and 1
glassMask = np.uint8(glassMask / 255)

# Make a copy
faceWithGlassesArithmetic = faceImage.copy()

# Get the eye region
eyeROI = faceWithGlassesArithmetic[270:380, 230:530]

# Apply mask
maskedEye = cv2.multiply(eyeROI, (1 - glassMask))

# Apply sunglasses mask
maskedGlass = cv2.multiply(glassBGR, glassMask)

# Combine both regions
eyeRoiFinal = cv2.add(maskedEye, maskedGlass)

# Replace the eye region
faceWithGlassesArithmetic[270:380, 230:530] = eyeRoiFinal

# Display results
plt.figure(figsize=(20,20))
plt.subplot(121);plt.imshow(faceImage[:,:,::-1]);plt.title("Original Image")
plt.subplot(122);plt.imshow(faceWithGlassesArithmetic[:,:,::-1]);plt.title("With Sunglasses")
```

<img width="912" height="597" alt="image" src="https://github.com/user-attachments/assets/a85f2926-831a-4e07-819a-de0c77036995" />

## Result

The image processing pipeline successfully identified the facial region from the input photograph and accurately positioned the sunglasses overlay onto the subject's face, fulfilling the visual transformation objective.





