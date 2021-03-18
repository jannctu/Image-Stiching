# Image-Stiching
## Basic Image Stiching 

This repository contains implementation automated panorama stitching, which is combines a collection of photos in to a wide angle panorama using feature matching.

## Algorithm
- Feature Extraction
SIFT (Scale-invariant feature transform)
- Feature Matching
KNN (k-Nearest Neighbors)
- Homography Estimation
- RANSAC

## Dependencies

Requires 
- OpenCV

## Input
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A001.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A002.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A003.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A004.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A005.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A006.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A007.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A008.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/input/A009.jpg)

## Output
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A001_A002.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A001_A002_A003.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A004_A005.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A004_A005_A006.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A007_A008.jpg)
![image](https://github.com/jannctu/Image-Stiching/blob/main/output/A007_A008_A009.jpg)

## License
MIT

**Free Software, Hell Yeah!**
