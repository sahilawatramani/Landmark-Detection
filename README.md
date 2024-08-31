# Landmark Classification Project

This project is aimed at classifying images of landmarks using a deep learning model built with Keras and VGG19. The dataset contains images of various landmarks, each labeled with a unique identifier.

## Table of Contents
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Evaluation](#evaluation)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## Installation

### Prerequisites
- Python 3.x
- Required Python packages:
  - numpy
  - pandas
  - keras
  - tensorflow
  - opencv-python
  - matplotlib
  - scikit-learn
  - pillow

### Installing
Clone the repository and install the required packages:

```bash
git clone https://github.com/your-username/landmark-classification.git
cd landmark-classification
pip install -r requirements.txt
```

## Usage

1. Prepare your dataset and place the `train.csv` file in the root directory.
2. Run the script to preprocess the data and train the model.

```bash
python train.py
```

## Dataset

The dataset should include a `train.csv` file containing the image file names and their corresponding landmark IDs. Images should be organized in folders following the structure derived from their filenames.

- `train.csv` - CSV file with two columns: `fname` (image file name) and `landmark_id` (class label).
- Images should be in subdirectories named based on the first three characters of their filenames.

## Model Architecture

The model is based on the VGG19 architecture with some modifications:

- Batch Normalization after specific layers
- Dropout layers for regularization
- Dense layer at the end for classification

## Training

The training script `train.py` preprocesses the images, encodes the labels, and trains the model. The model is trained using the RMSprop optimizer and the sparse categorical crossentropy loss function.

### Training Parameters

- Batch size: 16
- Epochs: 1 (can be adjusted)
- Learning rate: 0.0001
- Data augmentation is applied to training images.

### Training Script

```python
python train.py
```

## Evaluation

The model is evaluated on a validation set, and the accuracy and loss are plotted for both training and validation sets. Misclassified images are stored for further analysis.

### Evaluation Script

```python
python evaluate.py
```

## Results

The results of the training and evaluation process, including accuracy and loss plots, are displayed using matplotlib.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request with your changes.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
