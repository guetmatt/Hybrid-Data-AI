# Walkthrough of tutorial for implementing a DCGAN
# see: https://www.tensorflow.org/tutorials/generative/dcgan#next_steps
# 07.08.2024

# 13.08.2024
# adjusted to train with subsets of 'dataset_cars_bikes'

# %%
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
from IPython import display


# %%
# Load and prepare the dataset
#(train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()

bike_dir = pathlib.Path("./dataset_cars_bikes/bike")
car_dir = pathlib.Path("./dataset_cars_bikes/car")
batch_size = 200
img_height = 112
img_width = 112

bike_dataset = tf.keras.utils.image_dataset_from_directory(
    bike_dir,
    labels="inferred",
    color_mode="grayscale",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )
print(bike_dataset)

car_dataset = tf.keras.utils.image_dataset_from_directory(
    car_dir,
    labels="inferred",
    color_mode="grayscale",
    seed=123,
    image_size=(img_height, img_width),
    batch_size=batch_size
    )

#print(f"Class names bike: {bike_dataset.class_names}")
#print(f"Class names car: {car_dataset.class_names}")

#%%
# initiate empty numpy array
bike_images = np.empty((0, 112, 112, 1))
car_images = np.empty((0, 112, 112, 1))

# create arrays of car/bike images
for images, labels in bike_dataset.take(10):
    nparray = images.numpy()
    bike_images = np.concatenate((bike_images, nparray))

for images, labels in car_dataset.take(10):
   for i in range(1):
      nparray = images.numpy()
      car_images = np.concatenate((car_images, nparray))
    

#%%
# Normalize the images to [-1, 1]
bike_images = bike_images.reshape(bike_images.shape[0], 112, 112, 1).astype('float32')
bike_images = (bike_images - 127.5) / 127.5

car_images = car_images.reshape(car_images.shape[0], 112, 112, 1).astype('float32')
car_images = (car_images - 127.5) / 127.5


#%%
# batch and shuffle the data
BUFFER_SIZE = 60000
BATCH_SIZE = 256
bike_dataset = tf.data.Dataset.from_tensor_slices(bike_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
car_dataset = tf.data.Dataset.from_tensor_slices(car_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)


# %%
# Create the models

# the generator
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

#%%
# use the (untrained) generator to create an image
generator = make_generator_model()
noise = tf.random.normal([1, 100])
generated_image = generator(noise, training=False)
plt.imshow(generated_image[0, :, :, 0], cmap='gray')



# %%
# the discriminator - a cnn-based image classifier
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

# %%
# use the (untrained) disciriminator to classifiy the generated images as real or fake
# -- positive values = real, negative values = fake
discriminator = make_discriminator_model()
decision = discriminator(generated_image)
print(decision)


# %%
# Define the loss and optimizers

# define loss functions and optimizers for both models

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

# discriminator loss
# --This method quantifies how well the discriminator is able to distinguish
# -- real images from fakes. It compares the discriminator's predictions on
# -- real images to an array of 1s, and the discriminator's predictions on
# -- fake (generated) images to an array of 0s.
def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


# generator loss
# -- the generator's loss quantifies how well it was able to trick
# -- the discriminator. Intuitively, if the generator is performing well,
# -- the discriminator will calssify the fake images as real (or 1).
# -- Here, compare the disciriminators decisions on the generated images
# -- to an array of 1s.
def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

# optimizers
generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)



# %%
# save checkpoints
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)



# %%
# Define the training loop
EPOCHS = 15
noise_dim = 100
num_examples_to_generate = 16

# You will reuse this seed overtime (so it's easier)
# to visualize progress in the animated GIF)
seed = tf.random.normal([num_examples_to_generate, noise_dim])

"""
The training loop begins with generator receiving a random seed as input.
That seed is used to produce an image.
The discriminator is then used to classify real images (drawn from the training set)
and fakes images (produced by the generator).
The loss is calculated for each of these models,
and the gradients are used to update the generator and discriminator.
"""

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

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


def train(dataset, epochs):
  for epoch in range(epochs):
    print("EPOCH:", epoch)
    start = time.time()

    for image_batch in dataset:
      print("IMAGE BATCH")
      train_step(image_batch)

    # Produce images for the GIF as you go
    print("DISPLAY")
    display.clear_output(wait=True)
    generate_and_save_images(generator,
                             epoch + 1,
                             seed)

    # Save the model every 15 epochs
    if (epoch + 1) % 15 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time for epoch {} is {} sec'.format(epoch + 1, time.time()-start))

  # Generate after the final epoch
  display.clear_output(wait=True)
  generate_and_save_images(generator,
                           epochs,
                           seed)


# Generate and save images
def generate_and_save_images(model, epoch, test_input):
  # Notice `training` is set to False.
  # This is so all layers run in inference mode (batchnorm).
  predictions = model(test_input, training=False)
  print(predictions.shape[0])

  for i in range(predictions.shape[0]):
     plt.figure(figsize=[1.12, 1.12])
     plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
     plt.axis("off")
     plt.savefig(f"image_at_epoch{epoch}_{i}_train.png")
     plt.close()

  #fig = plt.figure(figsize=(4, 4))
  #fig = plt.figure()
  # predictions.shape[0] = 16
  #for i in range(predictions.shape[0]):
     #plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap="gray")
     #plt.imshow(predictions[i, :, :, :] *127.5 + 127.5, cmap="gray")
     #plt.axis('off')
     #plt.show()
     #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
     #plt.savefig(f'image_at_epoch_{epoch}_{i}.png')

  #for i in range(predictions.shape[0]):
   #   plt.subplot(4, 4, i+1)
    #  plt.imshow(predictions[i, :, :, 0] * 127.5 + 127.5, cmap='gray')
     # plt.axis('off')

  #plt.savefig('image_at_epoch_{:04d}.png'.format(epoch))
  #plt.show()




# %%
# train the model
"""
Call the train() method defined above to train the generator
and discriminator simultaneously.
Note, training GANs can be tricky.
It's important that the generator and discriminator do not overpower each other
(e.g., that they train at a similar rate).
"""
train(bike_dataset, EPOCHS)


# %%
# restore the latest checkpoint
# from training on car images
checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))