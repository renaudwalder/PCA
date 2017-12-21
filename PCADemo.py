import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import cv2

## Linearize an image
def lin_image(base_image):
    (H,W,channel) = np.shape(base_image)    # Get the Shape
    lin_image = base_image.reshape(H*W,channel).squeeze()  # linearize
    return lin_image

## Unlinearize an pre-linearize image
def unlin_image(in_image,H,W):
    (L,channel) = np.shape(in_image)   # Get the Shape
    # Verify the posssible size of the image
    if L != H*W:
        raise ValueError('The length of the image does not correspond to the Height and width input')
    # unlinearize
    base_image = in_image.reshape(H,W,channel).squeeze()
    return base_image

## Compute the PCA of the image
def PCAImage(image,n_components=1):
    # Get the size of the image
    (H, W, ch) = np.shape(image)

    # Create PCA Object from scikit-learn library
    sklearn_pca = PCA(n_components=n_components)

    # Apply the PCA to the linearize image
    lin_pca = sklearn_pca.fit_transform(lin_image(image))

    # transform from linearize image to normal image
    out_pca = unlin_image(lin_pca, H, W)

    # casting the output image to uint8 format (0-255)
    out_pca = np.uint8((out_pca - np.min(out_pca)) * 255 / (np.max(out_pca) - np.min(out_pca)))
    return out_pca


## Use the camera to print the computation example
def VideoDemo() :
    cap = cv2.VideoCapture(0)
    while(True):
        for i in range(60):
            # Capture frame-by-frame
            ret, frame = cap.read()

            # Resize image for real time processing
            frame = cv2.resize(frame,None,fx=0.5,fy=0.5)

            # Compute PCA
            image_pca = PCAImage(frame, n_components=1)

            # Display the different image each 20 frames
            if i < 20:
                cv2.imshow('image',frame)   # first 20 frames original image
            else :
                if i>40:
                    cv2.imshow('image',image_pca) #last 20 frames the PCA grayscale image
                else :
                    # the 20 frames in the middle the classique grayscale transformation
                    cv2.imshow('image',cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))

            # stop when user press 'q' ( cv2.WaitKey(1) is non blocking)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyAllWindows()
                return

## print the example of the PCA computation
def imageExample():
    #Read the Image and compute the grayscale transform
    input_image =cv2.imread('color-balls.jpg')
    gray_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
    pca_image = PCAImage(input_image, n_components=1)

    # Show the tree image to compare
    plt.figure()
    plt.subplot(2,2,1)
    plt.title('Input image')
    plt.imshow(input_image)

    plt.subplot(2,2,2)
    plt.title('Gray image')
    plt.imshow(gray_image, cmap='gray')

    plt.subplot(2,2,4)
    plt.title('PCA image')
    plt.imshow(pca_image, cmap='gray')
    plt.show()

## main program
if __name__ == '__main__':
    imageExample()
    VideoDemo()

#tutu = cv2.imread('color-balls.jpg')

# When everything done, release the capture
