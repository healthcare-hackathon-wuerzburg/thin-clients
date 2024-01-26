# Capillary AI based Imaging

## Problem Statement

AI-based analysis of capillary microscopic imaging.

In rheumatology, the diagnostics process is often challenging and integrates different methods.
Some rheumatic diseases directly affect small blood vessels and can be
diagnosed by non-invasive imaging in the nail fold.

The examiner has to distinguish different types of blood vessel structures as well as the total vessel density.
Therefore, semi-quantitative analysis of the different capillary malformations is important. Manual inspection of
capillary microscopy images is a time-consuming diagnostic procedure. An AI-based software could help
the examiner by analyzing the capillary density as well as detect relevant pathologic abnormalities. It can, in a later
step, offer a diagnosis to the examiner based on the previous results.

This project was developed as part of a 2 day effort at Healthcare Hackathon Wuerzburg from Jan 25th-26th 2024.

## Idea

We have developed an AI based on Python and Pytorch, which can detect abnormalities given a microscopic image
of capillaries.

The following abnormalities can be detected at the moment:

- Microhemorrhages,
- Giant capillaries
- Bushy capillaries

The AI can also detect capillaries of low density.

Currently, the AI has an accuracy of X %.

The low accuracy is because of the lack of training data, in total, less than 150 images were available to train the AI.

In the future, as more training data becomes available (hopefully), the accuracy will increase dramatically.

## Getting Started

This section describes the requirements of the project and how to get started.

### Requirements

List all requirements here

- Python
- PyTorch
- TorchVision
- tqdm
- Pillow
- Pandas

### Installation

Describe, how to install the project, like:

1) clone the repository via `git clone `
2) install requirements via `pip3 install -r /path/to/requirements.txt`
3) run the project and test your own microscopic images of capillaries.

## Project Overview

Describe how the project is structured.

Describe the architecture and the main components (if necessary) and the interaction between these components.

The project is structured into 3 main parts:

- Front End/User Interface, which loads the images into the program and presents results of the AI analysis to the
  users.
  We used PySimpleGUI to create a graphical user interface which allows users to test their individual microscopy
  images (.jpg) for anomalies.
  The images are evaluated by our previously trained network.
- Back End/AI, the actual network computing, whether an AI image shows anomalies.
  We used PyTorch, an open-source machine learning framework to build, train and validate our network, as well as test
  user images.
- Data including training images (not in in this repo)
  We had a number of images with various sizes and different magnifications. These images are stored in three different
  folders in the training, and validation directories, respectively. 'all' contains all images that were provided. '
  detail' contains a subset of images that have a higher magnification. 'overview' contains images that show a bigger
  part of the finger.
  The images are labelled in a file called 'labels.csv'. The first columns contains an "ID" (the name of the image).
  The other four columns contain information regarding the appearance of the following patterns:
  'microhemorrhages', 'giant_capillaries', 'bushy_capillaries' and 'low_density'. A value of '1' in the column means,
  that a medical professional has identified the pattern in that image, whereas a value of '0' shows the absence of said pattern.


## Useful links:

- [HealthCare Hackathon Wuerzburg 25-26.01.2024](https://www.healthcare-hackathon.info/hhwuerzburg)

