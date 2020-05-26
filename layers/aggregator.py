# -*- coding: utf-8 -*-

from keras.engine.topology import Layer
from keras import backend as K

# class AvgAggregator(Layer):
#     def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
#                  **kwargs):
#         super(AvgAggregator, self).__init__(**kwargs)
#         if activation == 'relu':
#             self.activation = K.relu
#         elif activation == 'tanh':
#             self.activation = K.tanh
#         else:
#             raise ValueError(f'`activation` not understood: {activation}')
#         self.initializer = initializer
#         self.regularizer = regularizer
#     def build(self, input_shape):
#         ent_embed_dim = input_shape[0][-1]
#         self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
#                                  initializer=self.initializer, regularizer=self.regularizer)
#         self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
#         super(SumAggregator, self).build(input_shape) 



class SumAggregator(Layer):
    def __init__(self, activation: str ='relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(SumAggregator, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        self.w = self.add_weight(name=self.name+'_w', shape=(ent_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name+'_b', shape=(ent_embed_dim,), initializer='zeros')
        super(SumAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot((entity + neighbor), self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class ConcatAggregator(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(ConcatAggregator, self).__init__(**kwargs)
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(ent_embed_dim+neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(ConcatAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot(K.concatenate([entity, neighbor]), self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]


class NeighAggregator(Layer):
    def __init__(self, activation: str = 'relu', initializer='glorot_normal', regularizer=None,
                 **kwargs):
        super(NeighAggregator, self).__init__()
        if activation == 'relu':
            self.activation = K.relu
        elif activation == 'tanh':
            self.activation = K.tanh
        else:
            raise ValueError(f'`activation` not understood: {activation}')
        self.initializer = initializer
        self.regularizer = regularizer

    def build(self, input_shape):
        ent_embed_dim = input_shape[0][-1]
        neighbor_embed_dim = input_shape[1][-1]
        self.w = self.add_weight(name=self.name + '_w',
                                 shape=(neighbor_embed_dim, ent_embed_dim),
                                 initializer=self.initializer, regularizer=self.regularizer)
        self.b = self.add_weight(name=self.name + '_b', shape=(ent_embed_dim,),
                                 initializer='zeros')
        super(NeighAggregator, self).build(input_shape)

    def call(self, inputs, **kwargs):
        entity, neighbor = inputs
        return self.activation(K.dot(neighbor, self.w) + self.b)

    def compute_output_shape(self, input_shape):
        return input_shape[0]
