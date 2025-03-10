# Awright

**1.	Introduction**
Diabetic Retinopathy (DR) is a serious eye disease that can lead to blindness if not diagnosed early. This project focuses on classifying retinal images into five severity levels using deep learning techniques. The model is developed using transfer learning with InceptionV3, leveraging pre-trained weights to improve classification accuracy.

**2.	Project Workflow**
1.	Setup and Dependencies
o	Libraries used: PyTorch, Torchvision, NumPy, Pandas, Matplotlib, PIL, TQDM, KaggleHub.
o	GPU support is enabled for faster training.
2.	Dataset Overview
o	Dataset consists of retinal images categorized into five severity levels.
o	Data is split into Training, Validation, and Test sets.
o	Stored in directories:
	train/
	validation/
	test/

**3.	Data Preprocessing**
o	Training Data Augmentation: Random resizing, cropping, flipping, rotation, color jittering.
o	Validation Data Transformation: Resizing, center cropping, normalization.
o	Data is loaded using ImageFolder and PyTorch DataLoader.

**4.	Model Development**
o	Pre-trained InceptionV3 model is used with a modified final fully connected (FC) layer for five classes.
o	Early layers are frozen, and deeper layers (Mixed_6, Mixed_7, FC) are fine-tuned.
o	Training is optimized using cross-entropy loss, dropout, and weight decay.

**5.	Training Strategy**
o	Learning rate: 0.00005 with momentum.
o	Early stopping and model checkpointing implemented.
o	Best model is selected based on validation accuracy.

**6.	Evaluation & Results**
o	Model is tested on unseen data with the following results:
	Precision: 0.5680
	Recall: 0.5469
	F1-Score: 0.5378
o	A confusion matrix is generated for performance assessment.

**7.	Grad-CAM Visualization**
o	Grad-CAM is implemented to highlight important regions in images that influence the model’s decision.

**8.	Image Classification**
o	predict_image function is used for classifying individual images.
o	Outputs class index and corresponding severity level.

**Final Accuracy**
The final test accuracy achieved is 54.23%, indicating areas for further improvements such as:
•	Hyperparameter tuning
•	Additional data augmentation
•	Fine-tuning deeper layers

**How to Use**
1.	Install dependencies using pip install torch torchvision numpy pandas matplotlib pillow tqdm kagglehub.
2.	Download dataset from Kaggle.
3.	Train the model using train.py.
4.	Evaluate performance using evaluate_model.py.
5.	Classify individual images with predict_image.py.

