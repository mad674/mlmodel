# Import necessary libraries
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
import cv2
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define dataset directories in Google Drive
sketch_images_dir = '/content/drive/MyDrive/SGan/searring'
color_images_dir = '/content/drive/MyDrive/SGan/earring'

# Load and preprocess the dataset
def load_image(image_file):
    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (1024, 1024))
    image = (image / 127.5) - 1  # Normalize to [-1, 1]
    return tf.cast(image, tf.float32)

def load_data(sketch_images_dir, color_images_dir):
    sketch_images = []
    color_images = []

    sketch_image_files = sorted(os.listdir(sketch_images_dir))
    color_image_files = sorted(os.listdir(color_images_dir))
    assert len(sketch_image_files) == len(color_image_files), "Mismatch in number of sketch and color images"

    for sketch_file, color_file in zip(sketch_image_files, color_image_files):
        sketch_image = load_image(os.path.join(sketch_images_dir, sketch_file))
        color_image = load_image(os.path.join(color_images_dir, color_file))
        sketch_images.append(sketch_image)
        color_images.append(color_image)

    return tf.data.Dataset.from_tensor_slices((sketch_images, color_images))


# # Residual Block
def residual_block(x, filters, size=3):
    initializer = tf.random_normal_initializer(0., 0.02)
    shortcut = x
    # Adjust shortcut dimensions if necessary
    if x.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, 1, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = layers.Conv2D(filters, size, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, size, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x

# # Downsampling Block
def downsample(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2D(filters, size, strides=2, padding='same',
                             kernel_initializer=initializer, use_bias=False))
    if apply_batchnorm:
        result.add(layers.BatchNormalization())
    result.add(layers.LeakyReLU())
    return result

# Upsampling Block
def upsample(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0., 0.02)
    result = tf.keras.Sequential()
    result.add(layers.Conv2DTranspose(filters, size, strides=2, padding='same',
                                      kernel_initializer=initializer, use_bias=False))
    result.add(layers.BatchNormalization())
    if apply_dropout:
        result.add(layers.Dropout(0.5))
    result.add(layers.ReLU())
    return result

# Generator Model
def Generator():
    inputs = tf.keras.layers.Input(shape=[1024, 1024, 3])

    # Downsampling layers
    down_stack = [
        downsample(64, 4, apply_batchnorm=False),
        downsample(128, 4),
        downsample(256, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
        downsample(512, 4),
    ]

    # Upsampling layers
    up_stack = [
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4, apply_dropout=True),
        upsample(512, 4),
        upsample(256, 4),
        upsample(128, 4),
        upsample(64, 4),
    ]

    initializer = tf.random_normal_initializer(0., 0.02)
    last = layers.Conv2DTranspose(3, 4, strides=2, padding='same',
                                  kernel_initializer=initializer, activation='tanh')

    x = inputs
    skips = []

    # Downsampling with residual blocks
    for down in down_stack:
        x = down(x)
        x = residual_block(x, down.layers[0].filters)  # Ensure filters match
        skips.append(x)

    skips = reversed(skips[:-1])

    # Upsampling with residual blocks
    for up, skip in zip(up_stack, skips):
        x = up(x)
        if x.shape[1] != skip.shape[1] or x.shape[2] != skip.shape[2]:
            skip = tf.image.resize(skip, [x.shape[1], x.shape[2]])  # Align spatial dimensions
        x = layers.Concatenate()([x, skip])
        x = residual_block(x, up.layers[0].filters)

    x = last(x)
    return tf.keras.Model(inputs=inputs, outputs=x)

# Discriminator model (PatchGAN for 1024x1024 images)
def Discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)

    # Input images (1024x1024)
    inp = layers.Input(shape=[1024, 1024, 3], name='input_image')
    tar = layers.Input(shape=[1024, 1024, 3], name='target_image')

    # Concatenate input and target images
    x = layers.concatenate([inp, tar])  # Shape: (1024, 1024, 6)

    # Downsample layers for 1024x1024 input
    down1 = downsample(64, 4, False)(x)         # 512x512x64
    down2 = downsample(128, 4)(down1)           # 256x256x128
    down3 = downsample(256, 4)(down2)           # 128x128x256
    down4 = downsample(512, 4)(down3)           # 64x64x512
    down5 = downsample(512, 4)(down4)           # 32x32x512
    down6 = downsample(512, 4)(down5)           # 16x16x512
    down7=downsample(512,4)(down6)
    # Comment out or modify the final downsample layers to avoid small dimensions
    zero_pad1 = layers.ZeroPadding2D()(down7)   # Adjust this if you reach a very small size
    conv = layers.Conv2D(512, 4, strides=1, kernel_initializer=initializer, use_bias=False)(zero_pad1)
    batchnorm1 = layers.BatchNormalization()(conv)
    leaky_relu = layers.LeakyReLU()(batchnorm1)

    zero_pad2 = layers.ZeroPadding2D()(leaky_relu)  # Padding to keep dimensions manageable
    last = layers.Conv2D(1, 4, strides=1, kernel_initializer=initializer)(zero_pad2)  # Output layer

    return tf.keras.Model(inputs=[inp, tar], outputs=last)


# Loss functions
LAMBDA = 100
loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def generator_loss(disc_generated_output, gen_output, target):
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    l1_loss = tf.reduce_mean(tf.abs(target - gen_output))
    total_gen_loss = gan_loss + (LAMBDA * l1_loss)
    return total_gen_loss, gan_loss, l1_loss

def discriminator_loss(disc_real_output, disc_generated_output):
    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)
    total_disc_loss = real_loss + generated_loss
    return total_disc_loss

# Optimizers
generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Training step
@tf.function
def train_step(input_image, target):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)
        gen_output = tf.image.resize(gen_output, (1024, 1024))  # Ensure 1024x1024 output
        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    return gen_total_loss, disc_loss


# Image generation for visualization
def generate_images(model, test_input, target):
    prediction = model(test_input, training=False)
    plt.figure(figsize=(15, 15))
    display_list = [test_input[0], target[0], prediction[0]]
    title = ['Input Image', 'Target Image', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow((display_list[i] * 0.5 + 0.5))
        plt.axis('off')
    plt.show()
generator = Generator()
discriminator = Discriminator()

model_path = '/content/drive/MyDrive/SGan/ear20.h5'
ep = 20
if os.path.exists(model_path):
    print("Loading pre-trained generator model...")
    generator = tf.keras.models.load_model(model_path)
    # Optionally, set the starting epoch for continuation
   # Set to the last completed epoch + 1 if continuing from that point
else:
    print("No pre-trained model found. Starting fresh training...")

# Training function
def train(dataset, epochs):
    for epoch in range(epochs):
        for input_image, target in dataset:
            gen_loss, disc_loss = train_step(input_image, target)
        print(f"Epoch {epoch+1}/{epochs}, Generator Loss: {gen_loss.numpy()}, Discriminator Loss: {disc_loss.numpy()}")
        example_input, example_target = next(iter(dataset))
        generate_images(generator, example_input, example_target)
        if (epoch + 1) % 5 == 0:
            save_path = f'/content/drive/MyDrive/SGan/ear{epoch+1+ep}.h5'
            generator.save(save_path)
            print(f"Model saved at epoch {epoch+1+ep} to {save_path}")

# Load dataset
dataset = load_data(sketch_images_dir, color_images_dir).shuffle(70).batch(1)

# Train the model
train(dataset, epochs=100000)
