# Diabetic retinopathy detection

Diabetic Retinopathy is typically diagnosed by identifying key retinal lesions such as microaneurysms, hemorrhages, hard exudates, soft exudates, and neovascularization.
The system follows the same approach:

1. **Detect and segment retinal lesions** using a U-Net model.
2. **Measure lesion severity** by calculating pixel-level areas.
3. **Predict DR Stage (0–4)** using a Random Forest classifier trained on lesion features.

This pipeline ensures transparent and explainable predictions, as the DR stage is derived from clinically recognized lesions.

---

# **Phase 1: Lesion Detection (Semantic Segmentation)**

### Dataset Used

**IDRiD — Part A: Segmentation Dataset**

* 54 Training Images
* 27 Testing Images
* Pixel-level annotation masks for:

  * Microaneurysms
  * Haemorrhages
  * Hard Exudates
  * Soft Exudates
  * Optic Disc

These masks were used as ground truth during U-Net training.

---

### Model Description (U-Net)

The U-Net architecture is used for pixel-wise segmentation. Key features include:

* Encoder-decoder CNN structure
* Skip connections to preserve fine spatial details
* Bilinear interpolation for upsampling
* Sigmoid activation for binary mask prediction
* **Saved Model:** `unet.pth`

---

# **Phase 2: DR Stage Prediction (Lesion-Based Classification)**

### Dataset Used

**IDRiD — Part B: Disease Grading Dataset**

* 413 retinal images
* Labels provided for:

  * Retinopathy Grade (0–4)
  * Risk of Macular Edema

Only the DR grade is used for training.

---

### Method

The lesion pixel counts extracted from Phase 1 are combined into a single feature vector:

```
[MA_pixels, HE_pixels, EX_pixels, SE_pixels, OD_pixels]
```

These features are mapped to DR stages using a **Random Forest Classifier**, trained on the IDRiD B grading labels.

* **Saved Classification Model:** `dr_stage_model.pkl`

---

#Output 

* Predicted DR Stage (0–4)
* Feature-based explanation (lesion presence and severity)
* Works for both IDRiD images and external fundus images

---

# File Structure

```
📁 Project Root
│
├── Dataset
│     ├── Disease_Grading_CNN
│     └── Segmentation_for_Unet
│            
│
├── Models
|     |__ dr_stage_model.pkl
│     └── unet.pth
│      
└── pipeline
      ├── dr_stage.ipynb
      │      DR stage classifier training + prediction script.
      └── correct_dataset.ipynb
             Training + testing pipeline for U-Net segmentation.
```

---

# How to Run the Project

## **1. Lesion Detection (Segmentation)**

### Open the Folder - Pipeline

```
run correct_dataset.ipynb 

```

### Run inference on your test image

Use the provided inference cell to generate:

* Lesion masks
* Pixel counts
* Presence/absence summary

Output masks will be shown for all five lesion types.

---

## **2. DR Stage Prediction**

### Step 1: Open the DR stage file and run it

```
cd pipeline/dr_stage.ipynb
run dr_stage.ipynb

```

### Run inference on your test image

first we will pass the image from the U-Net model to get:

```
MA_pixels
HE_pixels
EX_pixels
SE_pixels
OD_pixels
```

### Predict DR stage

```python
features = [[MA, HE, EX, SE, OD]]
stage = clf.predict(features)
print(stage)
```

This will return a value between **0 to 4**.

---
