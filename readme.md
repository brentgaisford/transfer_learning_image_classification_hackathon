## Tools / Libraries used
Seaborn
Numpy
Keras / Tensorflow
Matplotlib

## Dataset
- [Intel Image Classification](https://www.kaggle.com/puneet6060/intel-image-classification)
Extract this into local data/ (data/seg_pred, data/seg_test, data/seg_train)

### Problem Statement
This project was completed as part of a team of two working on a one day hackathon. Given only six hours of working time, can we build an accurate convolutional neural network to classify images into six categories - buildings, forest, glacier, mountain, sea, and street?

### The Data
The provided dataset included 14,000 images for training (in six folders, presorted by category), 3,000 images for testing (again, presorted by category), and 7,000 images for predicting (although the Kaggle competition to which those predictions could be submitted is no longer running, so the prediction data was not used).

For our purposes, only the training and test datasets were used.

### Methods

First, we used Keras to import the training and testing images via directory (maintaining the existing folder structure to maintain correct image categorization). Then we built and trained a custom convolutional neural network to predict which of those six categories unseen images belonged to. That custom model performed at 74% accuracy (vs a baseline of 17%).

Then, we used a transfer learning approach. We implemented the EfficientNetV2L model - a publicly available, trained deep learning model for general image classification. We customized the model with two new layers to allow it solve our particular classification problem, while leaving the majority of the model unchanged. That transfer learning model performed at 90% accuracy, and was selected as our final model.

Finally, we created a confusion matrix to look at which categories the model was most accurate at predicting. Unsurprisingly, it was hardest to differentiate between mountains vs glaciers and streets vs buildings, but even those categories the model was accurate more than 80% of the time.

### Conclusions and results

Convolutional neural networks (CNN) are extremely powerful tools for automated image categorization. However, to get the most out of a CNN requires complex architecture, extremely deep models with very large numbers of parameters, and huge datasets for effective training. However, the availability of pre-trained models within Keras make it much easier to customize a model for a smaller dataset, and we highly recommend transfer learning for anyone working on an image classification problem with a small dataset.

As an additional note, for most real world applications it is both more useful and more accurate to train a model to recognize particular features (e.g., does this photo contain a mountain) rather than putting images into a single bucket. In the case of this problem, many images of glaciers and mountains contained both things - and a more broadly useful image recognition model would be built to recognize that those images contained both features.
