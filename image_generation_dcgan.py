# imports
import tensorflow as tf
import glob
import imageio
import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
from tensorflow.keras import layers
import time
import pathlib
import joblib
from joblib import dump, load
from IPython import display



# create a generator model
def make_generator_model():
    model = tf.keras.Sequential()
    model.add(layers.Dense(28*28*256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((28, 28, 256)))
    assert model.output_shape == (None, 28, 28, 256) # None is the batch size

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    assert model.output_shape == (None, 28, 28, 128)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    assert model.output_shape == (None, 56, 56, 64)
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    assert model.output_shape == (None, 112, 112, 1)

    return model



# create a discriminator model
# --> a cnn-based image classifier
def make_discriminator_model():
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(56, (5, 5), strides=(2, 2), padding='same', input_shape=[112, 112, 1]))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Conv2D(112, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    model.add(layers.Dropout(0.3))

    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model



# discriminator loss
# to quantify discriminators ability
# to distinguish real images from fake images
def discriminator_loss(real_output, fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss



# generator loss
# to quantify generators ability
# generate realistic-looking images
def generator_loss(fake_output):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(tf.ones_like(fake_output), fake_output)



# define a step in training the
# discriminator and generator model
@tf.function
def train_step(images, generator, discriminator,
               generator_optimizer, discriminator_optimizer,
               batch_size, noise_dim):
    noise = tf.random.normal([batch_size, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



# train discriminator and generator model
def train(dataset, epochs, generator, discriminator,
          generator_optimizer, discriminator_optimizer,
          batch_size, noise_dim, seed):
  for epoch in range(epochs):
    print("EPOCH:", epoch)
    start = time.time()

    for image_batch in dataset:
      print("IMAGE BATCH")
      train_step(image_batch, generator, discriminator,
                 generator_optimizer, discriminator_optimizer,
                 batch_size, noise_dim)

    # save images during training
    print("DISPLAY")
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # save model as checkpoint
    checkpoint_dir = './training_checkpoints2'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")

    # save model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# Generate and save images
def generate_and_save_images(model, epoch, test_input):
  predictions = model(test_input, training=False)
  print(predictions.shape[0])

  for i in range(predictions.shape[0]):
     plt.figure(figsize=[1.12, 1.12])
     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
     plt.axis("off")
     plt.savefig(f"image_at_epoch{epoch}_{i}_train.png")
     plt.close()


# save model to  disk
def save_model(model, filepath):

    filepath = pathlib.Path(filepath)
    joblib.dump(model, filepath)

    return None


# load model from disk
def load_model(filepath):

    filepath = pathlib.Path(filepath)
    model = joblib.load(filepath)

    return model



if __name__ == "__main__":
  
  # load training data (car images)
  data_dir = pathlib.Path("./dataset_cars_bikes/car")
  batch_size = 200
  img_height = 112
  img_width = 112

  # define  the training parameters
  EPOCHS = 200
  noise_dim = 100
  num_examples_to_generate = 16
  seed = tf.random.normal([num_examples_to_generate, noise_dim])

  dataset = tf.keras.utils.image_dataset_from_directory(
      data_dir,
      labels="inferred",
      color_mode="grayscale",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size
      )
  
  
  # create numpy array of car images
  images = np.empty((0, 112, 112, 1))
  for image, label in dataset.take(10):
     for i in range(1):
      nparray = image.numpy()
      images = np.concatenate((images, nparray))
    
  # normalize the images to range [-1, 1]
  images = images.reshape(images.shape[0], 112, 112, 1).astype('float32')
  images = (images - 127.5) / 127.5


  # batch and shuffle the data
  BUFFER_SIZE = 60000
  BATCH_SIZE = 256
  dataset = tf.data.Dataset.from_tensor_slices(images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

  # optimizers
  generator_optimizer = tf.keras.optimizers.Adam(1e-4)
  discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

  # create generator and discriminator model
  generator = make_generator_model()
  discriminator = make_discriminator_model()

  # create checkpoint for training models during training
  checkpoint_dir = './training_checkpoints'
  checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
  checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                   discriminator_optimizer=discriminator_optimizer,
                                   generator=generator, discriminator=discriminator)
  
  # train discriminator and generator model
  train(dataset, EPOCHS, generator, discriminator,
        generator_optimizer, discriminator_optimizer,
        batch_size, noise_dim, seed)

  #%%
  # restore the latest checkpoint from training on car images
  # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

  # %%
  # save model to disk with file ending '.sav'
  model_path = "./training_checkpoints/generator_cars.sav"
  save_model(generator, model_path)
  model_path = "./training_checkpoints/discriminator_cars.sav"
  save_model(discriminator, model_path)

  # load model from disk
  # generator = load_model("./training_checkpoints/generator_cars.sav")
  # discriminator = load_model("./training_checkpoints/discriminator_cars.sav")
