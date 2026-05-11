## Retinal-Disease-Classification
Deep learning-based retinal disease classification using CNN and transfer learning
This project classifies retinal diseases from fundus images using CNN and Transfer Learning models such as VGG16, DenseNet201, MobileNetV2, EfficientNetB0, ViT Small, and a custom Eye Disease Classifier.


## Features
- Compares performance of multiple pretrained models
- Implements Grad-CAM visualization for explainability
- Includes accuracy/loss comparison graphs
- Supports efficient training using TensorFlow/Keras


## Models Compared
- EfficientNet-B0  
- MobileNet-V2  
- VGG16  
- DenseNet201  
- ViT Small  
- Custom Eye Disease Classifier



## Results Summary
|        Model            | Training Accuracy | Validation Accuracy | Remarks         |
|-------------------------|-------------------|---------------------|-----------------|
| MobileNet-V2            | ~99%              | ~91%                | Best overall    |
| DenseNet201             | ~99%              | ~90%                | Very stable     |
| VGG16                   | ~98%              | ~88%                | Strong baseline |
| Eye Disease Classifier  | ~95%              | ~85%                | Performs well   |
| EfficientNet-B0         | Unstable          | Low                 | Gradient issues |
| ViT Small               | Poor              |Poor                 | Did not converge|

---

##  Technologies Used
- Python  
- TensorFlow / Keras  
- NumPy, Pandas  
- Matplotlib, Seaborn  
- OpenCV  
- scikit-learn

## Dataset used
https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification/code

## Author
Uroosha Usman
MSc Computer Science, Lucknow University
