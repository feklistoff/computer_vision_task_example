# Computer Vision Task Example

This project implements an object detection model to identify specific icons in images. 

It uses a Faster R-CNN architecture with ResNet50 backbone for detecting and classifying icons (iron, wood, brick) in images.


## Project Structure

```
computer_vision_task_example/
├── data/
│   └── .gitkeep
├── model/
│   └── .gitkeep
├── notebooks/
│   └── exploration.ipynb
├── artifacts/
│   ├── backgrounds/
│   ├── icons/
│   └── test/
├── src/
│   ├── __init__.py
│   ├── dataset.py
│   ├── model.py
│   ├── inference.py
│   ├── generate_data.py
│   └── utils.py
├── .gitignore
├── README.md
└── requirements.txt
```

## Installation

1. Create a virtual environment:

```bash
python -m venv cv_env
source cv_env/bin/activate
```

2. Install dependencies:

```bash
pip install --upgrade pip
pip install uv
uv pip install -r requirements.txt
```

## Usage

### 1. **Generate Synthetic Data**

Generate synthetic images by overlaying icons onto background images.

```bash
python src/generate_data.py --num 20
```
- `--num`: Number of images to generate per label.

### 2. **Train the Model**

Train the Faster R-CNN model using the generated synthetic data.

```bash
python src/train.py --data_dir data --batch_size 4 --num_epochs 20 --lr 0.0005 --device cpu
```

- `--data_dir`: Directory containing data.csv.
- `--batch_size`: Batch size for training.
- `--num_epochs`: Number of training epochs.
- `--lr`: Learning rate.
- `--device`: Device to run training on (cuda, mps, or cpu).

### 3. **Evaluate the Model**

Evaluate the trained model on the validation dataset.

```bash
python src/train.py --device cpu --evaluate --model_path model/fasterrcnn_final.pth
```

- `--evaluate`: Flag to run evaluation instead of training.
- `--model_path`: Path to the trained model checkpoint.
- `--device`: Device to run evaluation on (cuda, mps, or cpu).

### 4. **Check Inference**

Use the trained model to run inference on new images. For the training we used synthetic data, so it is a great idea to check inference using the real world data.

```bash
python src/inference.py --image artifacts/test/test.png --model_path model/fasterrcnn_final.pth --device cpu --inference_thresh 0.2
```

Or we can use any image from the generated data

```bash
python src/inference.py --image data/synth_data/some_image.png --model_path model/fasterrcnn_final.pth --device cpu --inference_thresh 0.2
```

- `--image`: Path to the input image.
- `--model_path`: Path to the trained model checkpoint.
- `--device`: Device to run inference on (cuda, mps, or cpu).
- `--inference_thresh`: Threshold to use for filtering.
