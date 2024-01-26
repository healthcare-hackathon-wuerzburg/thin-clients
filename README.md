# Capillary AI based Imaging

## Problem Statement

AI-based analysis of capillary microscopic imaging.

In rheumatology, the diagnos'c process is oZen challenging and integrates different types of
diagnostics.
Some rheumatic diseases directly affect the small blood vessels and can be
diagnosed by non-invasive imaging of these blood vessels in the nail fold.

The examer has to distinguish different types of blood vessel structures as well as the total vessel density.
Thereby, semi-quantitative analysis of the different capillary malformations is important. These make the
capillary microscopy a time-consuming diagnostic procedure. An AI-based software could help
the examer by analysis the capillary density as well as mark relevant pathologic abnormalities
and perhaps can offer a suspicious diagnosis to the examer.

This project was developed as part of a 2 day effort at Healthcare Hackathon Wuerzburg from Jan 25th-26th 2024.

## Idea

We have therefore developed an AI based on Python and Pytorch, which can detect abnormalities given a microscopic image of capillaries.

The following abnormalities can be detected at the moment:

- Microhemorrhages,
- Giant capillaries
- Bushy capillaries

The AI can also detect capillaries of normal density.

Currently, the AI has an accuracy of X %.

The low accuracy is because of the lack of training data, in total, less than 100 images were available to train the AI.

In the future, as more training data (hopefully) becomes available, the accuracy will increase dramatically.

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
- Front End/User Interface, which loads the images into the program and presents results of AI analysis to the users.
- Back End/AI, the actual net computing whether an AI image shows abnormalities
- data including training images

## Useful links:

- [HealthCare Hackathon Wuerzburg 25-26.01.2024](https://www.healthcare-hackathon.info/hhwuerzburg)

