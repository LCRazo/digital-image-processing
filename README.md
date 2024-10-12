# Lung CT Image Processing

## Project Description

This project aims to perform various image processing tasks on a lung CT image. Tasks include loading and displaying the image, computing the 2D FFT, down-sampling the image, interpolating in both frequency and spatial domains, and computing errors between the original and interpolated images.

## Table of Contents

- [Installation](#installation)
- [Features](#Features)
- [Technologies](#technologies)
- [Credits](#credits)
- [License](#license)

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/lung-ct-image-processing.git

   pip 24.0 (python 3.11)
   pip install opencv-python
   
## Features
![alt text](https://raw.githubusercontent.com/LCRazo/digital-image-processing/refs/heads/main/downsample.png)
fig.1 Shows the effects of reducing the resolution of a CT scan which leads to fewer pixels and the loss of fine details 

![alt text](https://raw.githubusercontent.com/LCRazo/digital-image-processing/refs/heads/main/magnitudeSpectrum.png)
fig.2 Represents the amplitude of different frequency components. After shifting, the low frequencies (main features of the image) are centered providing a clearer view of the distribution 

![alt text](https://raw.githubusercontent.com/LCRazo/digital-image-processing/refs/heads/main/phases.png)
fig.3 Shifting helps visualize phase distribution around the image

## Technologies
