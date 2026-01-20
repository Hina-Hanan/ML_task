# Single Pixel Coordinate Prediction

**Author:** Hina Hanan  
**Date:** 20-Jan-2026  

---

## Project Overview

This project uses **Deep Learning** to predict the coordinates of a single bright pixel in a 50x50 grayscale image.  

- Each image has **all pixels set to 0** except **one pixel with value 255**, placed randomly.  
- The task is to predict the pixel’s `(x, y)` coordinates using a **Convolutional Neural Network (CNN)**.  
- The project is implemented in a **single Jupyter notebook** for clarity and easy submission.

---

## Notebook Contents

The notebook contains:

1. **Problem Statement & Approach**  
2. **Dataset Generation** – Synthetic 50x50 images with a single bright pixel  
3. **CNN Model** – Built and trained on the synthetic dataset  
4. **Training & Validation** – Loss and MAE plots  
5. **Evaluation** – Test set evaluation  
6. **Visualization** – Predicted vs ground truth coordinates  
7. **Conclusion** – Summary of results  

---

## Dataset

- Synthetic dataset of **10,000 images** generated inside the notebook.  
- Each image: 50x50 pixels, one bright pixel (value 255).  
- Labels: `(x, y)` coordinates of the bright pixel.  
- Images are normalized to 0–1 before training.

---

## Model

- **Type:** Convolutional Neural Network (CNN)  
- **Architecture:** Conv2D → Conv2D → Flatten → Dense → Dense(2)  
- **Loss Function:** Mean Squared Error (MSE)  
- **Optimizer:** Adam  
- **Output:** `(x, y)` coordinates  

---

## How to Run

1. Clone the repository
2. Install dependencies:
   - pip install -r requirements.txt
3. Open the notebook
4. Run all cells to generate the dataset, train the model, and visualize predictions.

## Results

After training the Convolutional Neural Network on the synthetically generated dataset, the model successfully learned to predict the coordinates of the single bright pixel in a 50x50 grayscale image.

### Quantitative Results
- The **training and validation loss** consistently decreased over epochs, indicating stable learning.
- The **Mean Absolute Error (MAE)** on the test set was low, showing that predicted coordinates were very close to the ground truth.
- The model generalized well to unseen images, confirming that it did not simply memorize pixel positions.

### Qualitative Results
- Visual inspection of test images shows that:
  - The **predicted coordinates (red X)** closely overlap with the **true coordinates (green dot)**.
  - The model correctly identifies the location of the bright pixel across different positions in the image.

### Key Observations
- A simple CNN is sufficient for this spatial regression task.
- Synthetic data generation works effectively for problems with clearly defined rules.
- The approach prioritizes clarity and correctness over unnecessary architectural complexity.

Overall, the model meets the problem requirements and performs the task as expected.


## References

1. TensorFlow Documentation – Keras API  
   https://keras.io  
2. TensorFlow Regression Tutorial  
   https://www.tensorflow.org/tutorials/keras/regression  



