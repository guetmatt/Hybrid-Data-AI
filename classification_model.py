# https://www.tensorflow.org/tutorials/images/classification

#%%
# Imports
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
import pathlib
import joblib
from joblib import dump, load

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential


# %%
# load dataset
data_dir = pathlib.Path("./data/exp1_training_real/")
batch_size = 32
img_height = 112
img_width = 112

train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    color_mode="grayscale",
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=0.2,
    subset="validation",
    color_mode="grayscale",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

class_names = train_ds.class_names



# %%
# visualize the data
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis("off")


# %%
# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


#%%
# Create the model
# - 3 convolution blocks (Conv2D) with a max pooling layer in each
# - 1 fully-connected layer with 128 units on top of it (Dense)
# - activated by relu-function
num_classes = len(class_names)

model = Sequential([
    # data normalization
    layers.Rescaling(1./255, input_shape=(img_height, img_width, 1)),
    layers.Conv2D(16, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding="same", activation="relu"),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dense(num_classes)
])


#%%
# Compile the model
model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])


#%%
# Model summary
model.summary()



#%%
# Train the model
epochs = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)


# %%
# Visualize training results
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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



# Methods againts Overfitting
# - Data Augmentation (to enlarge training data set)
# - dropout (randomly dropping out a number of output units, i.e. setting their activation to zero)

#%%
# Data augmentation
# data_augmentation = keras.Sequential([
#     layers.RandomFlip("horiztonal",
#                       input_shape=(img_height,
#                                    img_width,
#                                    1)),
#     layers.RandomRotation(0.1),
#     layers.RandomZoom(0.1),
#     ])

# # visualize augmented data
# plt.figure(figsize=(10, 10))
# for images, _ in train_ds.take(1):
#     for i in range(9):
#         augmented_images = data_augmentation(images)
#         ax = plt.subplot(3, 3, i+1)
#         plt.imshow(augmented_images[0].numpy().astype("uint8"))
#         plt.axis("off")


# #%%
# # New model with data augmentation and dropout
# model = Sequential([
#   data_augmentation,
#   layers.Rescaling(1./255),
#   layers.Conv2D(16, 3, padding="same", activation="relu"),
#   layers.MaxPooling2D(),
#   layers.Conv2D(32, 3, padding="same", activation="relu"),
#   layers.MaxPooling2D(),
#   layers.Conv2D(64, 3, padding="same", activation="relu"),
#   layers.MaxPooling2D(),
#   layers.Dropout(0.2),
#   layers.Flatten(),
#   layers.Dense(128, activation="relu"),
#   layers.Dense(num_classes, name="outputs")
# ])


# #%%
# # Compile new model
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])


# #%%
# # Summary
# model.summary()

# #%%
# # Train new model
# epochs = 15
# history = model.fit(
#   train_ds,
#   validation_data=test_ds,
#   epochs=epochs
# )



# # %%
# # Visualize new training results
# acc = history.history["accuracy"]
# val_acc = history.history["val_accuracy"]

# loss = history.history["loss"]
# val_loss = history.history["val_loss"]

# epochs_range = range(epochs)

# plt.figure(figsize=(8, 8))
# plt.subplot(1, 2, 1)
# plt.plot(epochs_range, acc, label="Training Accuracy")
# plt.plot(epochs_range, val_acc, label="Validation Accuracy")
# plt.legend(loc="lower right")
# plt.title("Training and Validation Accuracy")

# plt.subplot(1, 2, 2)
# plt.plot(epochs_range, loss, label="Training Loss")
# plt.plot(epochs_range, val_loss, label="Validation Loss")
# plt.legend(loc="upper right")
# plt.title("Training and Validation Loss")
# plt.show()




# PREDCIT ON NEW DATA / TEST DATA

# %%
# lod test dataset
data_dir = pathlib.Path("./data/exp1_test")
batch_size = 32
img_height = 112
img_width = 112

test_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    labels="inferred",
    color_mode="grayscale",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

class_names = test_ds.class_names



# %%
# Extract images and labels from test dataset
# store them as numpy arrays
test_images = list()
test_labels = list()
for image_batch, label_batch in test_ds:
    for i in range(image_batch.shape[0]):
        image = image_batch[i]
        label = label_batch[i]
        test_images.append(image)
        test_labels.append(label)
test_images = np.array(test_images)
test_labels = np.array(test_labels)


#%%
# test trained model on test data
# and print results
results = model.evaluate(x=test_ds)
print(f"Test loss: {round(results[0], 4)}, Test Accuracy: {round(results[1], 4)}")

results2 = model.evaluate(x=test_images, y=test_labels)
print(f"Test loss: {round(results2[0], 4)}, Test Accuracy: {round(results2[1], 4)}")


#%%
test_predictions = model.predict(x=test_images)
predicted_classes = list()
for index, prediction in enumerate(test_predictions):
    scores = tf.nn.softmax(prediction)
    predicted_class = np.argmax(scores)
    predicted_classes.append(predicted_class)

count_ovr=0
count_right=0
count_wrong=0
for index, pred in enumerate(predicted_classes):
    count_ovr += 1
    if test_labels[index] == pred:
        count_right += 1
    else:
        count_wrong += 1

print(f"Overall: {count_ovr}")
print(f"right: {count_right}")
print(f"wrong: {count_wrong}")
print(f"Accuracy:{round(count_right/count_ovr, 4)}")

confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=predicted_classes)
print(f"Confusion Matrix:\n{class_names}\n{confusion_matrix}")


# %%
# set filename for model
filename = "model_trained_firsttest.sav"

# %%
# Save trained model
joblib.dump(model, filename)

#%%
# load model from disk
model = joblib.load(filename)


# %%
