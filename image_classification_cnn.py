# based on: https://www.tensorflow.org/tutorials/images/classification


# %%

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
from tensorflow.keras.callbacks import EarlyStopping



def load_training_data(filepath, img_height=112, img_width=112,
              batch_size=32, validation_split=0.2):
    
    data_dir = pathlib.Path(filepath)

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

    # configure dataset for performance    
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, class_names



def visualize_data(dataset, class_names):

    plt.figure(figsize=(10, 10))
    for images, labels in dataset.take(1):
        for i in range(9):
            ax = plt.subplot(3, 3, i+1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")

    return None



def configure_model(dataset, class_names,
                    img_height=112, img_width=112):

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

    model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

    model.summary()

    return model



def train_model(train_dataset, val_dataset, model,
                epochs=20, visualize_training=True):
    
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[early_stopping]
        )

    if visualize_training:
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss = history.history['loss']
        val_loss = history.history['val_loss']

        epochs_range = range(len(history.history["accuracy"]))

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

    return model, history



def configure_model_with_augmentation(dataset, class_names,
                                      visualize_augmented_data=True,
                                      img_height=112, img_width=112):
    # Data Augmentation and Dropout

    # Data augmentation
    data_augmentation = keras.Sequential([
        layers.RandomFlip("horiztonal",
                          input_shape=(img_height,
                                       img_width,
                                       1)),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
        ])

    # visualize augmented data
    if visualize_augmented_data:
        plt.figure(figsize=(10, 10))
        for images, _ in dataset.take(1):
            for i in range(9):
                augmented_images = data_augmentation(images)
                ax = plt.subplot(3, 3, i+1)
                plt.imshow(augmented_images[0].numpy().astype("uint8"))
                plt.axis("off")

    num_classes = len(class_names)
    model = Sequential([
        data_augmentation,
        layers.Rescaling(1./255),
        layers.Conv2D(16, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, padding="same", activation="relu"),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, name="outputs")
        ])

    model.compile(optimizer="adam",
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=["accuracy"])

    model.summary()

    return model



def load_test_data(filepath, img_height=112, img_width=112, batch_size=32):
    
    data_dir = pathlib.Path(filepath)

    test_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels="inferred",
        color_mode="grayscale",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size
        )

    return test_ds
    


def predict_testdata(test_dataset, model, use_predict=True):
    
    results = model.evaluate(x=test_dataset)
    print(f"Overall test loss: {round(results[0], 4)}\nOverall test accuracy: {round(results[1], 4)}\n")

    if use_predict:
        test_images = list()
        test_labels = list()
        for image_batch, label_batch in test_dataset:
            for i in range(image_batch.shape[0]):
                image = image_batch[i]
                label = label_batch[i]
                test_images.append(image)
                test_labels.append(label)
        test_images = np.array(test_images)
        test_labels = np.array(test_labels)

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

        print(f"Total test images: {count_ovr}")
        print(f"Correct predictions: {count_right}")
        print(f"False Predictions: {count_wrong}")
        print(f"Accuracy:{round(count_right/count_ovr, 4)}")

        confusion_matrix = tf.math.confusion_matrix(labels=test_labels, predictions=predicted_classes)
        print(f"\nConfusion Matrix:\n{test_dataset.class_names}\n{confusion_matrix}")

    return results    



def save_model(model, filepath):

    filepath = pathlib.Path(filepath)
    joblib.dump(model, filepath)

    return None



def load_model(filepath):

    filepath = pathlib.Path(filepath)
    model = joblib.load(filepath)

    return model


# %%
if __name__ == "__main__":

    # Parameters for size of images
    # if changed, pass them to functions
    img_height = 112
    img_width = 112
    batch_size = 32
    epochs=30

    # load additional synthetic images
    syn_train_data_filepath = "./data/exp2_training_synthetic_selfmade"
    syn_train_ds, syn_val_ds, class_names = load_training_data(syn_train_data_filepath, img_height=img_height, img_width=img_width)
    
    syn_train_data_filepath2 = "./data/exp2_training_synthetic"
    syn_train_ds2, syn_val_ds2, class_names = load_training_data(syn_train_data_filepath2, img_height=img_height, img_width=img_width)
    
    
    # load data
    train_data_filepath = "./data/exp2_training_real/"
    test_data_filepath = "./data/exp2_test"
    train_ds, val_ds, class_names = load_training_data(train_data_filepath, img_height=img_height, img_width=img_width)
    test_ds = load_test_data(test_data_filepath, img_height=img_height, img_width=img_width)

    # merge real and synthetic datasets
    train_ds = train_ds.concatenate(syn_train_ds)
    val_ds = val_ds.concatenate(syn_val_ds)
    train_ds = train_ds.concatenate(syn_train_ds2)
    val_ds = val_ds.concatenate(syn_val_ds2)
    
    # visualize training data
    visualize_data(train_ds, class_names)
    visualize_data(test_ds, class_names)

    # configure and train model
    # model = configure_model(train_ds, class_names, img_height=img_height, img_width=img_width)
    # model, history = train_model(train_ds, val_ds, model, epochs=epochs)

    # configure and train model with data augmentation techniques
    model = configure_model_with_augmentation(train_ds, class_names, img_height=img_height, img_width=img_width)
    model, history = train_model(train_ds, val_ds, model, epochs=epochs)

    # use trained model on testdata
    results = predict_testdata(test_ds, model)

    # # save model to disk with file ending '.sav'
    model_path = "./models/model_exp2.4.sav"
    save_model(model, model_path)

    # model = load_model("./models/model_test.sav")
    # results = predict_testdata(test_ds, model)

# %%
