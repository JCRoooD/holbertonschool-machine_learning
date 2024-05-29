#!/usr/bin/env python3
""" based on 8-train.py"""
import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, save_best=False,
                filepath=None, verbose=True, shuffle=False):
    """to also train the model using learning rate decay:"""
    callbacks = []
    if validation_data and early_stopping:
        early_stop = K.callbacks.EarlyStopping(monitor='val_loss',
                                               patience=patience)
        callbacks.append(early_stop)
    if validation_data and learning_rate_decay:
        def scheduler(epoch):
            """scheduler function"""
            return alpha / (1 + decay_rate * epoch)
        lr_decay = K.callbacks.LearningRateScheduler(scheduler, verbose=1)
        callbacks.append(lr_decay)
    if save_best:
        save = K.callbacks.ModelCheckpoint(filepath, save_best_only=True)
        callbacks.append(save)
    return network.fit(data, labels, batch_size=batch_size, epochs=epochs,
                       verbose=verbose, shuffle=shuffle,
                       validation_data=validation_data,
                       callbacks=callbacks)
