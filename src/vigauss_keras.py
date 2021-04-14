import tensorflow as tf
import tensorflow_probability as tfp
from keras.layers import Layer
from keras import activations, initializers
from keras import backend as K


class DenseVIGAUSS(Layer):
    def __init__(self,
                 units,
                 kl_weight,
                 num_samples_per_epoch=2,
                 activation=None,
                 prior_mu=0.,
                 prior_sigma=1., **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.num_samples=num_samples_per_epoch
        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

    def build(self, input_shape):
        self.kernel_mu = self.add_weight(name='kernel_mu',
                                         shape=(input_shape[1], self.units),
                                         initializer=initializers.RandomNormal(mean=self.prior_mu,stddev=self.prior_sigma),
                                         trainable=True)
        self.bias_mu = self.add_weight(name='bias_mu',
                                       shape=(self.units,),
                                       initializer=initializers.RandomNormal(mean=self.prior_mu,stddev=self.prior_sigma),
                                       trainable=True)
        self.kernel_rho = self.add_weight(name='kernel_rho',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.constant(0.0),
                                          trainable=True)
        self.bias_rho = self.add_weight(name='bias_rho',
                                        shape=(self.units,),
                                        initializer=initializers.constant(0.0),
                                        trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        kernel_sigma = tf.math.softplus(self.kernel_rho)
        bias_sigma = tf.math.softplus(self.bias_rho)

        for current_sample in range(self.num_samples):
            kernel = self.kernel_mu + kernel_sigma * tf.random.normal(self.kernel_mu.shape)
            bias = self.bias_mu + bias_sigma * tf.random.normal(self.bias_mu.shape)

            kernel_kl_loss=self.kl_loss(kernel, self.kernel_mu, kernel_sigma)
            bias_kl_loss=self.kl_loss(bias, self.bias_mu, bias_sigma)

            if current_sample==0:
                kernel_list=tf.reshape(kernel,[1,kernel.shape[0],kernel.shape[1]])
                bias_list=tf.reshape(bias,[1,bias.shape[0]])
                kernel_kl_loss_sum=kernel_kl_loss
                bias_kl_loss_sum=bias_kl_loss

            else:
                kernel_list=tf.concat([kernel_list,tf.reshape(kernel,[1,kernel.shape[0],kernel.shape[1]])],axis=0) 
                bias_list=tf.concat([bias_list,tf.reshape(bias,[1,bias.shape[0]])],axis=0) 
                kernel_kl_loss_sum+=kernel_kl_loss
                bias_kl_loss_sum+=bias_kl_loss
                


        self.add_loss(1/self.num_samples*kernel_kl_loss_sum+1/self.num_samples*bias_kl_loss_sum) #DEBUG

        kernel_mean=tf.reduce_mean(kernel_list,axis=0)
        bias_mean=tf.reduce_mean(bias_list,axis=0)


        return self.activation(K.dot(inputs, kernel_mean) + bias_mean)

    def kl_loss(self, w, mu, sigma):
        variational_dist = tfp.distributions.Normal(mu, sigma)
        return self.kl_weight * K.sum(variational_dist.log_prob(w) - self.log_prior_prob(w))

    def log_prior_prob(self, w):
        prior_dist = tfp.distributions.Normal(self.prior_mu, self.prior_sigma)
        return prior_dist.log_prob(w) 