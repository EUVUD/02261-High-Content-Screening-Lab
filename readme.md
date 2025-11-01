# High Content Screening Analysis Pipeline

Pipeline for quantifying cellular phenotypes in high content screening (HCS) imagery using classical computer vision and tabular machine learning. The workflow segments nuclei, extracts handcrafted descriptors, and trains Random Forest classifiers to predict treatment, group, and assay day labels. Results include feature-importance reports and confusion matrices to support exploratory analysis and slide-deck ready storytelling.

- Built with OpenCV, scikit-image, scikit-learn, and seaborn
- Automates both raw-image and segmented-image experiments for fair comparison
- Saves misclassified exemplars to speed up root-cause analysis and presentation prep
- Feature-importance exports make it easy to communicate biological signal contributors

## Project Structure
- `predictor.py` main experiment runner: loads images, engineers features, trains and evaluates Random Forest models.
- `imgSeg.py` standalone script for generating segmented overlays that highlight cell boundaries.
- `quickfix.py` utility stub used during prototyping; safe to ignore for standard runs.
- `train_data/` expected location for raw DAPI and TRANS image pairs (not version-controlled).
- `seg_data/` auto-generated segmented images used for the segmentation-based treatment model.
- `results/` stores confusion matrices, feature-importance CSVs/plots, and misclassified image copies.

## Getting Started
1. Install Python 3.10+ and create a virtual environment (recommended).
2. Install dependencies:
	 ```bash
	 pip install opencv-python numpy scikit-image scikit-learn seaborn matplotlib pandas
	 ```
3. Download the HCS image set (shared separately) and place it in a local `train_data/` directory at the project root.

### Data Notes
- Training uses data from groups 2, 4, 5, 9, 10, and 11; group 7 is excluded due to inconsistent formatting.
- File `G2_10x_C4_F2_T1_DAPI.jpg` was removed because its paired TRANS image is missing.
- Large image assets are intentionally excluded from source control.

## Running the Pipeline
```bash
python predictor.py
```

What happens:
- Loads raw images, extracts morphology, intensity, and object-level features.
- Trains three Random Forest models (Treatment, Group, Day) on raw-image features.
- Optionally reuses segmented overlays from `seg_data/` to rerun the Treatment classifier with segmentation-informed inputs.
- Persists diagnostics to `results/`, including:
	- `*_confusion_matrix.png`
	- `*_feature_importance.csv` and matching bar plots
	- `misclassified/` folder with annotated filenames for quick review

If `seg_data/` is empty, uncomment the segmentation loop near the bottom of `predictor.py` or run `imgSeg.py` directly to generate overlays before the segmentation experiment.

## Highlights
- Designed an end-to-end HCS analytics workflow built on reproducible Python tooling.
- Engineered custom feature extractors that capture nuclear morphology, texture, and intensity signals.
- Operationalized model evaluation with automated reporting, feature-importance ranking, and misclassification auditing.
- Demonstrated ability to communicate model insights to cross-functional teams through ready-to-share artifacts.