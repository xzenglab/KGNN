# -*- coding: utf-8 -*-

import os
from keras.callbacks import *

from config import ModelConfig
from callbacks import SWA


class BaseModel(object):
    def __init__(self, config: ModelConfig):
        self.config = config
        self.callbacks = []
        self.model = self.build()

    def add_model_checkpoint(self):
        self.callbacks.append(ModelCheckpoint(
            filepath=os.path.join(self.config.checkpoint_dir,
                                  '{}.hdf5'.format(self.config.exp_name)),
            monitor=self.config.checkpoint_monitor,
            save_best_only=self.config.checkpoint_save_best_only,
            save_weights_only=self.config.checkpoint_save_weights_only,
            mode=self.config.checkpoint_save_weights_mode,
            verbose=self.config.checkpoint_verbose
        ))
        print('Logging Info - Callback Added: ModelCheckPoint...')

    def add_early_stopping(self):
        self.callbacks.append(EarlyStopping(
            monitor=self.config.early_stopping_monitor,
            mode=self.config.early_stopping_mode,
            patience=self.config.early_stopping_patience,
            verbose=self.config.early_stopping_verbose
        ))
        print('Logging Info - Callback Added: EarlyStopping...')

    def add_swa(self, swa_start: int=5):
        self.callbacks.append(SWA(self.build(), self.config.checkpoint_dir, self.config.exp_name,
                                  swa_start=swa_start))
        print('Logging Info - Callback Added: SWA with constant lr...')

    def init_callbacks(self):
        if 'modelcheckpoint' in self.config.callbacks_to_add:
            self.add_model_checkpoint()
        if 'earlystopping' in self.config.callbacks_to_add:
            self.add_early_stopping()
        if 'swa' in self.config.callbacks_to_add:
            self.add_swa(swa_start=self.config.swa_start)

    def build(self):
        raise NotImplementedError

    def fit(self, x_train, y_train, x_valid, y_valid):
        raise NotImplementedError

    def predict(self, x):
        raise NotImplementedError

    def score(self, x, y):
        raise NotImplementedError

    def load_weights(self, filename: str):
        self.model.load_weights(filename)

    def load_model(self, filename: str):
        # we only save model's weight instead of the whole model
        self.model.load_weights(filename)

    def load_best_model(self):
        print('Logging Info - Loading model checkpoint: %s.hdf5' % self.config.exp_name)
        self.load_model(os.path.join(self.config.checkpoint_dir, f'{self.config.exp_name}.hdf5'))
        print('Logging Info - Model loaded')

    def load_swa_model(self):
        print(f'Logging Info - Loading SWA model checkpoint: {self.config.exp_name}_swa.hdf5')
        self.load_model(os.path.join(self.config.checkpoint_dir,
                                     f'{self.config.exp_name}_swa.hdf5'))
        print('Logging Info - SWA Model loaded')

    def summary(self):
        self.model.summary()
