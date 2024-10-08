#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

class WGAN_clip(keras.Model):
    """WGAN class for Wasserstein GAN with gradient clipping"""
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005):
        """ Init function for the WGAN class """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        
        self.learning_rate = learning_rate
        self.beta_1 = .5  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary
        
        # define the generator loss and optimizer:
        self.generator.loss = self.generator_loss
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss = self.discriminator_loss
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    # generator of real samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of fake samples of size batch_size
    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
        
    # overloading train_step()    
    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # Get a real sample
                real_samples = self.get_real_sample()
                real_output = self.discriminator(real_samples, training=True)
                
                # Get a fake sample
                fake_samples = self.get_fake_sample(training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                
                # Compute the loss of the discriminator on real and fake samples
                discr_loss = self.discriminator_loss(real_output, fake_output)

            # Apply gradient descent once to the discriminator
            disc_gradients = disc_tape.gradient(discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

            # Clip the weights of the discriminator between -1 and 1
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as gen_tape:
            # Get a fake sample
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            
            # Compute the loss of the generator on this sample
            gen_loss = self.generator_loss(fake_output)

        # Apply gradient descent to the generator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {"discr_loss": discr_loss, "gen_loss": gen_loss}

# Example usage:
# generator = ...  # Define your generator model
# discriminator = ...  # Define your discriminator model
# latent_generator = ...  # Define your latent generator model
# real_examples = ...  # Load your real examples
# wgan = WGAN_clip(generator, discriminator, latent_generator, real_examples)
# real_samples = ...  # Load your real samples
# wgan.train_step(real_samples)