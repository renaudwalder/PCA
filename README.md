# PCA
PCA  (Principal Component Analysis) implementation for multidimensional image to grayscale on python with opencv and scikit-learn Edit
Add topics
## Intro
PCA is used in machine learning to reduced the dimension of some data.

The reduction of the data is made by using the eigen vector of the correlation matrix and by finding some data which are correleted to other data

The goal of this little code is to apply this to an image and transform a multidimensional image (RGB) to a grayscale image

## Processus
- Linearize the image _from (H,W,number_of_channel) to (H*W,number_of_channel)_. 
- The linearize image is then the input of the PCA algorithme _from (H*W,number_of_channel) to (H*W,n_components)_
- Unlinearize the image _from (H*W,n_components) to (H,W,n_components)_. 

## Example
![Example Image](https://github.com/renaudwalder/PCA/blob/master/DemoBall.png)
