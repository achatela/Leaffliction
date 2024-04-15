# Leaffliction Computer Vision

## Overview
Leaffliction is a comprehensive Python project focused on applying computer vision techniques to the identification and analysis of plant diseases. The project leverages image processing and machine learning to analyze plant leaves, perform data augmentations, extract meaningful features, and ultimately classify plant diseases with high accuracy.

## Features
- **Data Analysis**: Automatically analyzes images within a directory to extract and display data, including pie charts and bar charts based on plant types.
- **Data Augmentation**: Balances the dataset by applying various transformations like flipping, rotating, skewing, shearing, cropping, and distorting images.
- **Image Transformations**: Implements multiple image transformations to assist in feature extraction, including Gaussian blur, masks, ROI objects, and color histograms.
- **Disease Classification**: Trains a model to recognize and classify different types of plant diseases and validates the accuracy of the model.

## Installation

Clone the repository and navigate to the project directory:
```bash
git clone https://github.com/yourusername/leaffliction.git
cd leaffliction
```

Install the dataset and setup the environment:
```bash
wget https://cdn.intra.42.fr/document/document/17547/leaves.zip
unzip leaves.zip
python -m venv venv
source venv/bin/activate # To exit the virtual environment, run `deactivate`
pip install -r requirements.txt
```

## Usage

Use the following command to display the graphs and data analysis:
```bash
python3 Distribution.py images/
```

To see the data augmentation in action, run the following command:
```bash
python3 Augmentation.py images/Apple_Black_rot/image\ \(101\).JPG
```

To see the image transformations, run the following command:
```bash
python3 Transformations.py images/Apple_Black_rot/image\ \(101\).JPG
```