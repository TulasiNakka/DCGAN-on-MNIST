# Nakka, Tulasi
# 1001_928_971
# 2023_11_12
# Assignment_04_01

import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
# imported the keras layers library
from keras.layers import*
def plot_images(generated_images, n_rows=1, n_cols=10):
    """
    Plot the images in a 1x10 grid
    :param generated_images:
    :return:
    """
    f, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    ax = ax.flatten()
    for i in range(n_rows*n_cols):
        ax[i].imshow(generated_images[i, :, :], cmap='gray')
        ax[i].axis('off')
    return f, ax

class GenerateSamplesCallback(tf.keras.callbacks.Callback):
    """
    Callback to generate images from the generator model at the end of each epoch
    Uses the same noise vector to generate images at each epoch, so that the images can be compared over time
    """
    def __init__(self, generator, noise):
        self.generator = generator
        self.noise = noise

    def on_epoch_end(self, epoch, logs=None):
        if not os.path.exists("generated_images"):
            os.mkdir("generated_images")
        generated_images = self.generator(self.noise, training=False)
        generated_images = generated_images.numpy()
        generated_images = generated_images*127.5 + 127.5
        generated_images = generated_images.reshape((10, 28, 28))
        # plot images using matplotlib
        plot_images(generated_images)
        plt.savefig(os.path.join("generated_images", f"generated_images_{epoch}.png"))
        # close the plot to free up memory
        plt.close()

def build_discriminator():
    
    model = tf.keras.models.Sequential()
    
    # your code here
    # 1. Conv2D layer with 16 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    model.add(Conv2D(16, (5, 5), (2, 2), padding = 'same', input_shape = (28, 28, 1)))
    
    #2. LeakyReLU activation function (default parameters)
    model.add(LeakyReLU())

    #3. Dropout layer with a rate of 0.3.
    model.add(Dropout(rate = 0.3))

    #4. Conv2D layer with 32 filters, kernel size of (5, 5), strides of (2, 2), and padding set to 'same'.
    model.add(Conv2D(32, (5, 5), (2, 2), padding = 'same'))

    #5. LeakyReLU activation function (default parameters)
    model.add(LeakyReLU())

    #6. Dropout layer with a rate of 0.3.
    model.add(Dropout(rate = 0.3))

    #7. Flatten layer to convert the feature maps into a 1D array.
    model.add(Flatten())

    #8. Dense layer with 1 output neuron.
    model.add(Dense(1))

    return model

def build_generator():
    
    model = tf.keras.models.Sequential()
    # your code here

    #1. Dense layer with 7 * 7 * 8 (392) neurons and no bias, input shape of (100,).
    model.add(Dense(7*7*8, input_shape = (100,), use_bias = False))
    
    #2. Batch normalization layer, default params
    model.add(BatchNormalization())

    #3. LeakyReLU activation function with default params
    model.add(LeakyReLU())

    #4. Reshape layer to convert the 1D array into a 3D feature map with a shape of (7, 7, 8).
    model.add(Reshape(target_shape = (7, 7, 8)))

    #5. Conv2DTranspose (deconvolution) layer with 8 filters, kernel size of (5, 5), strides of (1, 1)
    model.add(Conv2DTranspose(8, (5, 5), (1, 1), padding = 'same', use_bias = False))

    #6. Batch normalization layer.
    model.add(BatchNormalization())

    #7. LeakyReLU activation function with default params
    model.add(LeakyReLU())

    #8. Conv2DTranspose (deconvolution) layer with 16 filters, kernel size of (5, 5), strides of (2, 2)
    model.add(Conv2DTranspose(16, (5, 5), (2, 2), padding = 'same', use_bias = False))

    #9. Batch normalization layer.
    model.add(BatchNormalization())

    #10. LeakyReLU activation function with default params
    model.add(LeakyReLU())

    #11. Conv2DTranspose (deconvolution) layer with 1 filter, kernel size of (5, 5), strides of (2, 2), with tanh activation included
    model.add(Conv2DTranspose(1, (5, 5), (2, 2), padding = 'same', activation = 'tanh',use_bias = False))

    return model

class DCGAN(tf.keras.Model):
    def __init__(self, discriminator, generator):
        super(DCGAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(DCGAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    # discriminator loss
    def discriminator_loss(self,real_output, fake_output):
        real_loss = self.loss_fn(tf.ones_like(real_output), real_output)
        fake_loss = self.loss_fn(tf.zeros_like(fake_output), fake_output)
        # need to multiply with 0.5 to balance the training set between generator and discriminator
        d_loss = 0.5 * (real_loss + fake_loss)
        return d_loss

    # generator loss
    def generator_loss(self,fake_output):
        fake_loss =  tf.ones_like(fake_output)
        g_loss = self.loss_fn(fake_loss,fake_output)
        return g_loss
    
    def train_step(self, data):
        
        batch_size = tf.shape(data)[0]
        noise_dim=100
        # TRAINING CODE START HERE
        # Referred the code from given tutorial
        

        noise = tf.random.uniform([batch_size, noise_dim])
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_images = self.generator(noise, training=True)
            real_output = self.discriminator(data, training=True)
            fake_output = self.discriminator(generated_images, training=True)
            g_loss = self.generator_loss(fake_output)
            d_loss = self.discriminator_loss(real_output, fake_output)
        
        gradients_of_generator = gen_tape.gradient(g_loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(d_loss, self.discriminator.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.discriminator.trainable_variables))

        # TRAINING CODE END HERE
        return {"d_loss": d_loss, "g_loss": g_loss}



def train_dcgan_mnist():
    tf.keras.utils.set_random_seed(5368)
    # load mnist
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # the images are in the range [0, 255], we need to rescale them to [-1, 1]
    x_train = (x_train - 127.5) / 127.5
    x_train = x_train[..., tf.newaxis].astype(np.float32)

    # plot 10 random images
    example_images = x_train[:10]*127.5 + 127.5
    plot_images(example_images)

    plt.savefig("real_images.png")


    # build the discriminator and the generator
    discriminator = build_discriminator()
    generator = build_generator()


    # build the DCGAN
    dcgan = DCGAN(discriminator=discriminator, generator=generator)

    # compile the DCGAN
    dcgan.compile(d_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  g_optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                  loss_fn=tf.keras.losses.BinaryCrossentropy(from_logits=True))

    callbacks = [GenerateSamplesCallback(generator, tf.random.uniform([10, 100]))]
    # train the DCGAN
    dcgan.fit(x_train, epochs=50, batch_size=64, callbacks=callbacks, shuffle=True)

    # generate images
    noise = tf.random.uniform([16, 100])
    generated_images = generator(noise, training=False)
    plot_images(generated_images*127.5 + 127.5, 4, 4)
    plt.savefig("generated_images.png")

    generator.save('generator.h5')


if __name__ == "__main__":
    train_dcgan_mnist()
