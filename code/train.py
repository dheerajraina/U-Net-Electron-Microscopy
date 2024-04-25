from dataloader import DatasetGenerator
from sklearn.model_selection import train_test_split
import numpy as np
from model import unet
import matplotlib.pyplot as plt

datagenerator = DatasetGenerator(
    mask_directory="../datasets/training_groundtruth.tif", image_directory="../datasets/training.tif", generate_data_from_scratch=False)

imageDataset, maskDataset = datagenerator.get_dataset()


imageDataset = np.array(imageDataset)
maskDataset = np.array(maskDataset)

X_train, y_train, X_val, y_val, X_test, y_test = datagenerator.split_dataset(
    0.1, 0.1, imageDataset, maskDataset)


# --------------------------- Plot a few examples---------------------------
for i in range(0, 2):
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(np.reshape(X_train[i], (256, 256)), cmap='gray')
    plt.subplot(122)
    plt.imshow(np.reshape(y_train[i], (256, 256)), cmap='gray')
    plt.show()

# -----------Model Creation------------------
unet = unet()

model_filename, model_callbacks = unet.get_callbacks()

model = unet.create_model(imgs_shape=(
    imageDataset.shape[1], imageDataset.shape[2], imageDataset.shape[3]))

# ---------------MODEL TRAINING-------------------
history = model.fit(X_train, y_train, batch_size=16,
                    epochs=50, validation_data=(X_val, y_val), callbacks=model_callbacks)


#--------------PLOT HISTORY------------------------------
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(50)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


""" --------------------------Testing Accuracy----------------"""
_, acc = model.evaluate(X_test, y_test)
print(f"Accuracy = {acc}")  # Turns out to be 0.844


# General accuracy isn't the proper way to test the accuracy of our segmentation task
# Using IOU
y_pred = model.predict(X_test)
y_pred_with_threshold = y_pred > 0.5

intersection = np.logical_and(y_test, y_pred)
union = np.logical_or(y_test, y_pred_with_threshold)
iou_score = np.sum(intersection) / np.sum(union)
print("IoU socre is: ", iou_score)

# IOU score turns out to be 1.0 for our model
