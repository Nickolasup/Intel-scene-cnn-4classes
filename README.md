# Intel Scene CNN (4 classes)

This project trains a Convolutional Neural Network (CNN) in TensorFlow/Keras to classify images into 4 classes (buildings, forest, glacier, mountain). The notebook covers the full workflow: data preprocessing with ImageDataGenerator, model training and metric visualization (accuracy/loss), testing on individual images with the predicted label written directly onto the image (OpenCV), and evaluation on the test set using evaluate(), accuracy_score, a confusion matrix, and a classification report. It also includes saving and reloading the trained model (JSON + weights) and preparing multiple network versions to compare performance. This repository is still a work in progress and is being actively updated, so it does not represent the final version of the project yet.

## What is included
- Data preprocessing with ImageDataGenerator
- CNN training (accuracy/loss plots)
- Model evaluation:
  - `evaluate()` score
  - `accuracy_score`
  - Confusion matrix + heatmap
  - Classification report
- Single image prediction + writing prediction text into the image (OpenCV `putText`)
- Saving/loading the model (JSON + weights)

## Files in this repository (my outputs)
- `Zadanie2test.ipynb` – main notebook (training, evaluation, prediction)
- `Model_accuracy.jpg` – accuracy curve
- `Model_loss.jpg` – loss curve
- `model_plot.png` – model plot
- `model_layered.png` – visualkeras model view
- `glacier_with_signature.jpg` – example image with prediction text overlay
- `neo_v1.json` + `neo_v1_weights.hdf5` – saved model (architecture + weights)

## Dataset (not included in repo)
Dataset is too large for GitHub, link - https://www.kaggle.com/code/evgenyzorin/intel-image-classification.


Expected folder structure after download:
seg_train/seg_train/
buildings/
forest/
glacier/
mountain/

seg_test/seg_test/
buildings/
forest/
glacier/
mountain/

In the notebook, update paths in `flow_from_directory()` to point to your `seg_train/seg_train` and `seg_test/seg_test`.

## How to run
1) Install dependencies:
```bash
pip install -r requirements.txt
Open and run the notebook:

Zadanie2test.ipynb
```

## Notes

test_set should use shuffle=False for correct confusion matrix and report.

## 5) requirements.txt

```txt
tensorflow==2.10.1
keras==2.10.0
numpy
matplotlib
scikit-learn
seaborn
opencv-python
pillow
visualkeras
pydot
graphviz


