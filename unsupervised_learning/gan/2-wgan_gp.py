#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
import tensorflow as tf

class WGAN_GP(keras.Model):
    
    def __init__(self, generator, discriminator, latent_generator, real_examples, batch_size=200, disc_iter=2, learning_rate=.005, lambda_gp=10):
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        
        self.learning_rate = learning_rate
        self.beta_1 = .3  # standard value, but can be changed if necessary
        self.beta_2 = .9  # standard value, but can be changed if necessary
        
        self.lambda_gp = lambda_gp  # <---- New !
        self.dims = self.real_examples.shape  # <---- New !
        self.len_dims = tf.size(self.dims)  # <---- New !
        self.axis = tf.range(1, self.len_dims, delta=1, dtype='int32')  # <---- New !
        self.scal_shape = self.dims.as_list()  # <---- New !
        self.scal_shape[0] = self.batch_size  # <---- New !
        for i in range(1, self.len_dims):  # <---- New !
            self.scal_shape[i] = 1  # <---- New !
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)  # <---- New !
        
        # define the generator loss and optimizer:
        self.generator.loss = self.generator_loss  # <---- to be filled in                 
        self.generator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.generator.compile(optimizer=self.generator.optimizer, loss=self.generator.loss)
        
        # define the discriminator loss and optimizer:
        self.discriminator.loss = self.discriminator_loss  # <---- to be filled in  
        self.discriminator.optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate, beta_1=self.beta_1, beta_2=self.beta_2)
        self.discriminator.compile(optimizer=self.discriminator.optimizer, loss=self.discriminator.loss)

    def generator_loss(self, fake_output):
        return -tf.reduce_mean(fake_output)

    def discriminator_loss(self, real_output, fake_output):
        return tf.reduce_mean(fake_output) - tf.reduce_mean(real_output)

    def get_fake_sample(self, size=None, training=False):
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)
    
    def get_interpolated_sample(self, real_sample, fake_sample):
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample
    
    def gradient_penalty(self, interpolated_sample):
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)
    
    def train_step(self, useless_argument):
        for _ in range(self.disc_iter):
            with tf.GradientTape() as disc_tape:
                # Get a real sample
                real_samples = self.get_real_sample()
                real_output = self.discriminator(real_samples, training=True)
                
                # Get a fake sample
                fake_samples = self.get_fake_sample(training=True)
                fake_output = self.discriminator(fake_samples, training=True)
                
                # Get the interpolated sample
                interpolated_samples = self.get_interpolated_sample(real_samples, fake_samples)
                
                # Compute the old loss of the discriminator on real and fake samples
                discr_loss = self.discriminator_loss(real_output, fake_output)
                
                # Compute the gradient penalty
                gp = self.gradient_penalty(interpolated_samples)
                
                # Compute the sum new_discr_loss = discr_loss + self.lambda_gp * gp
                total_discr_loss = discr_loss + self.lambda_gp * gp

            # Apply gradient descent with respect to new_discr_loss once to the discriminator
            disc_gradients = disc_tape.gradient(total_discr_loss, self.discriminator.trainable_variables)
            self.discriminator.optimizer.apply_gradients(zip(disc_gradients, self.discriminator.trainable_variables))

        with tf.GradientTape() as gen_tape:
            # Get a fake sample
            fake_samples = self.get_fake_sample(training=True)
            fake_output = self.discriminator(fake_samples, training=True)
            
            # Compute the loss of the generator on this sample
            gen_loss = self.generator_loss(fake_output)

        # Apply gradient descent to the generator
        gen_gradients = gen_tape.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(zip(gen_gradients, self.generator.trainable_variables))
        
        return {"discr_loss": discr_loss, "gen_loss": gen_loss, "gp": gp}

# Example usage:
# generator = ...  # Define your generator model
# discriminator = ...  # Define your discriminator model
# latent_generator = ...  # Define your latent generator model
# real_examples = ...  # Load your real examples
# wgan_gp = WGAN_GP(generator, discriminator, latent_generator, real_examples)
# real_samples = ...  # Load your real samples
# wgan_gp.train_step(real_samples)