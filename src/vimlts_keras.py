import tensorflow as tf
import tensorflow_probability as tfp
import sys
from keras.layers import Layer
from keras import activations, initializers
from keras import backend as K
sys.path.append('../src')
import vimlts_utils_keras as VIMLTS_utils



class DenseVIMLTS(Layer):
    

    def __init__(self,
                 units,
                 kl_weight,
                 activation=None,
                 prior_mu=0.,
                 prior_sigma=1.,
                 init_gauss_like=True,
                 using_f3=True, 
                 **kwargs):
        self.units = units
        self.kl_weight = kl_weight
        self.activation = activations.get(activation)
        self.prior_mu = prior_mu
        self.prior_sigma = prior_sigma
        self.kernel_m=10
        self.bias_m=10
        self.kernel_beta_dist=VIMLTS_utils.init_beta_dist(self.kernel_m)
        self.bias_beta_dist=VIMLTS_utils.init_beta_dist(self.bias_m)
        self.epsilon=tf.constant(0.001)
        self.init_gauss=init_gauss_like
        self.using_f3=using_f3

        super().__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return input_shape[0], self.units

 
    def build(self, input_shape):
        """
        Initialization of the trainable variational parameters, for x (independent of #units) and for bias
        """

        # Kernel
        self.kernel_tilde_a = self.add_weight(name='kernel_tilde_a',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=0.2 ,stddev=0.2),
                                          trainable=True)
        self.kernel_b = self.add_weight(name='kernel_b',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=-0.5 if self.init_gauss else 0.,stddev=0.2),
                                          trainable=True)

        self.kernel_start_theta = self.add_weight(name='kernel_start_theta',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=-6. if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_1 = self.add_weight(name='kernel_delta_theta_1',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=3. if self.init_gauss else 2.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_2 = self.add_weight(name='kernel_delta_theta_2',
                                          shape=(input_shape[1], self.units), 
                                          initializer=initializers.RandomNormal(mean=-1. if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_3 = self.add_weight(name='kernel_delta_theta_3',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=1.1 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_4 = self.add_weight(name='kernel_delta_theta_4',
                                          shape=(input_shape[1], self.units), 
                                          initializer=initializers.RandomNormal(mean=-1.2 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_5 = self.add_weight(name='kernel_delta_theta_5',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=1.3 if self.init_gauss else 8.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_6 = self.add_weight(name='kernel_delta_theta_6',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=-1.4 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_7 = self.add_weight(name='kernel_delta_theta_7',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=-1.5 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_8 = self.add_weight(name='kernel_delta_theta_8',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=-1.6 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.kernel_delta_theta_9 = self.add_weight(name='kernel_delta_theta_9',
                                          shape=(input_shape[1], self.units),
                                          initializer=initializers.RandomNormal(mean=4. if self.init_gauss else 3.,stddev=0.2),
                                          trainable=True)

 
        self.kernel_tilde_alpha = self.add_weight(name='kernel_tilde_alpha',
                                        shape=(input_shape[1], self.units),
                                        initializer=initializers.RandomNormal(mean=1.,stddev=(0.2 if self.using_f3 else 0.)),
                                        trainable=True)
        self.kernel_beta = self.add_weight(name='kernel_beta',
                                        shape=(input_shape[1], self.units),
                                        initializer=initializers.RandomNormal(mean=0.,stddev=(0.2 if self.using_f3 else 0.)),
                                        trainable=True)

        
        # Bias
        self.bias_tilde_a = self.add_weight(name='bias_tilde_a',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=0.2,stddev=0.2),
                                          trainable=True)
        self.bias_b = self.add_weight(name='bias_b',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-0.5 if self.init_gauss else 0.,stddev=0.2),
                                          trainable=True)
        
        self.bias_start_theta = self.add_weight(name='bias_start_theta',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-6. if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_1 = self.add_weight(name='bias_delta_theta_1',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=3. if self.init_gauss else 2.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_2 = self.add_weight(name='bias_delta_theta_2',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-1. if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_3 = self.add_weight(name='bias_delta_theta_3',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=1.1 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_4 = self.add_weight(name='bias_delta_theta_4',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-1.2 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_5 = self.add_weight(name='bias_delta_theta_5',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=1.3 if self.init_gauss else 8.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_6 = self.add_weight(name='bias_delta_theta_6',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-1.4 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_7 = self.add_weight(name='bias_delta_theta_7',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=-1.5 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_8 = self.add_weight(name='bias_delta_theta_8',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=1.6 if self.init_gauss else -6.,stddev=0.2),
                                          trainable=True)
        self.bias_delta_theta_9 = self.add_weight(name='bias_delta_theta_9',
                                          shape=(self.units,),
                                          initializer=initializers.RandomNormal(mean=4. if self.init_gauss else 3.,stddev=0.2),
                                          trainable=True)
        
        self.bias_tilde_alpha = self.add_weight(name='bias_tilde_alpha',
                                        shape=(self.units,),
                                        initializer=initializers.RandomNormal(mean=.2,stddev=(0.2 if self.using_f3 else 0.)),
                                        trainable=(True if self.using_f3 else False))
        self.bias_beta = self.add_weight(name='bias_beta',
                                        shape=(self.units,),
                                        initializer=initializers.RandomNormal(mean=0.,stddev=(0.2 if self.using_f3 else 0.)),
                                        trainable=(True if self.using_f3 else False))
        super().build(input_shape)


    def call(self, inputs, **kwargs):
        """Calculation of the forward direction of the network by
        -drawing parameters
        -Calculation loss KL
        -Calculation output of the layer
        """
             
        # 1.) Conversion of trainable variational parameters to parameters of distribution (x)
        kernel_a = VIMLTS_utils.to_a(self.kernel_tilde_a)
        kernel_theta = VIMLTS_utils.to_theta([self.kernel_start_theta,
                                        self.kernel_delta_theta_1,
                                        self.kernel_delta_theta_2,
                                        self.kernel_delta_theta_3,
                                        self.kernel_delta_theta_4,
                                        self.kernel_delta_theta_5,
                                        self.kernel_delta_theta_6,
                                        self.kernel_delta_theta_7,
                                        self.kernel_delta_theta_8,
                                        self.kernel_delta_theta_9])
        print(kernel_theta)
        kernel_theta=tf.transpose(kernel_theta, perm=[1, 2, 0])
        kernel_alpha = VIMLTS_utils.to_alpha(self.kernel_tilde_alpha)

        # 2.) Conversion of trainable variational parameters to parameters of distribution (bias)
        bias_a = VIMLTS_utils.to_a(self.bias_tilde_a)
        bias_theta = VIMLTS_utils.to_theta([self.bias_start_theta,
                                        self.bias_delta_theta_1,
                                        self.bias_delta_theta_2,
                                        self.bias_delta_theta_3,
                                        self.bias_delta_theta_4,
                                        self.bias_delta_theta_5,
                                        self.bias_delta_theta_6,
                                        self.bias_delta_theta_7,
                                        self.bias_delta_theta_8,
                                        self.bias_delta_theta_9])
        bias_theta=tf.transpose(bias_theta, perm=[1, 0])
        bias_alpha = VIMLTS_utils.to_alpha(self.bias_tilde_alpha)


        # 3.) Sampling from variational dist by reparametrization trick - prototype version (x)
        kernel_z_sample=tf.random.normal(self.kernel_tilde_a.shape)
        kernel_w_sample=VIMLTS_utils.kernel_h_z2w(z=kernel_z_sample,
                                        a=kernel_a, 
                                        b=self.kernel_b, 
                                        theta=kernel_theta, 
                                        alpha=kernel_alpha, 
                                        beta=self.kernel_beta, 
                                        beta_dist=self.kernel_beta_dist)


        # 4.) Sampling from variational dist by reparametrization trick - prototype version (bias)
        bias_z_sample=tf.random.normal(self.bias_tilde_a.shape)
        bias_w_sample=VIMLTS_utils.bias_h_z2w(z=bias_z_sample,
                                        a=bias_a, 
                                        b=self.bias_b, 
                                        theta=bias_theta, 
                                        alpha=bias_alpha, 
                                        beta=self.bias_beta, 
                                        beta_dist=self.bias_beta_dist)

        # 5.) Calculation l_kl by calling function with passing the sample and the variational parameters (x and bias)
        kernel_kl_loss=self.calc_kernel_kl_loss(kernel_w_sample=kernel_w_sample,
                            kernel_z_sample=kernel_z_sample,
                            kernel_a=kernel_a,
                            kernel_theta=kernel_theta,
                            kernel_alpha=kernel_alpha)

        bias_kl_loss=self.calc_bias_kl_loss(bias_w_sample=bias_w_sample,
                            bias_z_sample=bias_z_sample,
                            bias_a=bias_a,
                            bias_theta=bias_theta,
                            bias_alpha=bias_alpha)

        self.add_loss(kernel_kl_loss+bias_kl_loss)

        # 6.) Calculation output of the layer
        return self.activation(K.dot(inputs, kernel_w_sample)+ bias_w_sample)

    def calc_kernel_kl_loss(self,kernel_w_sample,kernel_z_sample,kernel_a,kernel_theta,kernel_alpha):
        kernel_z_epsilon=kernel_z_sample+self.epsilon
        kernel_q,w_check=VIMLTS_utils.kernel_eval_variational_dist(z=kernel_z_sample, 
                                                            z_epsilon=kernel_z_epsilon, 
                                                            a=kernel_a, 
                                                            b=self.kernel_b, 
                                                            theta=kernel_theta, 
                                                            alpha=kernel_alpha, 
                                                            beta=self.kernel_beta, 
                                                            beta_dist=self.kernel_beta_dist)
        kernel_kl_loss=self.kl_loss(kernel_q,kernel_w_sample)
        return kernel_kl_loss

    def calc_bias_kl_loss(self,bias_w_sample,bias_z_sample,bias_a,bias_theta,bias_alpha):
        bias_z_epsilon=bias_z_sample+self.epsilon
        bias_q,w_check=VIMLTS_utils.bias_eval_variational_dist(z=bias_z_sample, 
                                                            z_epsilon=bias_z_epsilon, 
                                                            a=bias_a, 
                                                            b=self.bias_b, 
                                                            theta=bias_theta, 
                                                            alpha=bias_alpha, 
                                                            beta=self.bias_beta, 
                                                            beta_dist=self.bias_beta_dist)
        bias_kl_loss=self.kl_loss(bias_q,bias_w_sample)
        return bias_kl_loss


    def kl_loss(self, q, w_sample):
        # Calculate KL divergence between variational dist and prior
        return self.kl_weight * K.sum(K.log(q) - self.log_prior_prob(w_sample))


    def log_prior_prob(self, w):
        # Prior
        prior_dist = tfp.distributions.Normal(self.prior_mu, self.prior_sigma)
        return prior_dist.log_prob(w) 


# class DenseVIMLTS(Layer):
    

#     def __init__(self,
#                  units,
#                  kl_weight,
#                  num_samples_per_epoch=2,
#                  activation=None,
#                  prior_mu=0.,
#                  prior_sigma=1.,
#                  init_gauss_like=True,
#                  using_f3=True, 
#                  **kwargs):
#         self.units = units
#         self.kl_weight = kl_weight
#         self.activation = activations.get(activation)
#         self.prior_mu = prior_mu
#         self.prior_sigma = prior_sigma
#         self.kernel_m=10
#         self.bias_m=10
#         self.kernel_beta_dist=VIMLTS_utils.init_beta_dist(self.kernel_m)
#         self.bias_beta_dist=VIMLTS_utils.init_beta_dist(self.bias_m)
#         self.epsilon=tf.constant(0.001)
#         self.init_gauss=init_gauss_like
#         self.num_samples=num_samples_per_epoch
#         self.using_f3=using_f3

#         super().__init__(**kwargs)

#     def compute_output_shape(self, input_shape):
#         return input_shape[0], self.units, self.num_samples

 
#     def build(self, input_shape):
#         """
#         Initialization of the trainable variational parameters, for x (independent of #units) and for bias
#         """

#         # Kernel
#         self.kernel_tilde_a = self.add_weight(name='kernel_tilde_a',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=0.2 ,stddev=0.2),
#                                           trainable=True)
#         self.kernel_b = self.add_weight(name='kernel_b',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-0.5 if self.init_gauss else 0.,stddev=0.2),
#                                           trainable=True)

#         self.kernel_start_theta = self.add_weight(name='kernel_start_theta',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-6. if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_1 = self.add_weight(name='kernel_delta_theta_1',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=3. if self.init_gauss else 2.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_2 = self.add_weight(name='kernel_delta_theta_2',
#                                           shape=(input_shape[1], self.units, self.num_samples), 
#                                           initializer=initializers.RandomNormal(mean=-1. if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_3 = self.add_weight(name='kernel_delta_theta_3',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=1.1 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_4 = self.add_weight(name='kernel_delta_theta_4',
#                                           shape=(input_shape[1], self.units, self.num_samples), 
#                                           initializer=initializers.RandomNormal(mean=-1.2 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_5 = self.add_weight(name='kernel_delta_theta_5',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=1.3 if self.init_gauss else 8.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_6 = self.add_weight(name='kernel_delta_theta_6',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.4 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_7 = self.add_weight(name='kernel_delta_theta_7',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.5 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_8 = self.add_weight(name='kernel_delta_theta_8',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.6 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.kernel_delta_theta_9 = self.add_weight(name='kernel_delta_theta_9',
#                                           shape=(input_shape[1], self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=4. if self.init_gauss else 3.,stddev=0.2),
#                                           trainable=True)

 
#         self.kernel_tilde_alpha = self.add_weight(name='kernel_tilde_alpha',
#                                         shape=(input_shape[1], self.units, self.num_samples),
#                                         initializer=initializers.RandomNormal(mean=1.,stddev=(0.2 if self.using_f3 else 0.)),
#                                         trainable=True)
#         self.kernel_beta = self.add_weight(name='kernel_beta',
#                                         shape=(input_shape[1], self.units, self.num_samples),
#                                         initializer=initializers.RandomNormal(mean=0.,stddev=(0.2 if self.using_f3 else 0.)),
#                                         trainable=True)

        
#         # Bias
#         self.bias_tilde_a = self.add_weight(name='bias_tilde_a',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=0.2,stddev=0.2),
#                                           trainable=True)
#         self.bias_b = self.add_weight(name='bias_b',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-0.5 if self.init_gauss else 0.,stddev=0.2),
#                                           trainable=True)
        
#         self.bias_start_theta = self.add_weight(name='bias_start_theta',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-6. if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_1 = self.add_weight(name='bias_delta_theta_1',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=3. if self.init_gauss else 2.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_2 = self.add_weight(name='bias_delta_theta_2',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1. if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_3 = self.add_weight(name='bias_delta_theta_3',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=1.1 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_4 = self.add_weight(name='bias_delta_theta_4',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.2 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_5 = self.add_weight(name='bias_delta_theta_5',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=1.3 if self.init_gauss else 8.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_6 = self.add_weight(name='bias_delta_theta_6',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.4 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_7 = self.add_weight(name='bias_delta_theta_7',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=-1.5 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_8 = self.add_weight(name='bias_delta_theta_8',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=1.6 if self.init_gauss else -6.,stddev=0.2),
#                                           trainable=True)
#         self.bias_delta_theta_9 = self.add_weight(name='bias_delta_theta_9',
#                                           shape=(self.units, self.num_samples),
#                                           initializer=initializers.RandomNormal(mean=4. if self.init_gauss else 3.,stddev=0.2),
#                                           trainable=True)
        
#         self.bias_tilde_alpha = self.add_weight(name='bias_tilde_alpha',
#                                         shape=(self.units, self.num_samples),
#                                         initializer=initializers.RandomNormal(mean=.2,stddev=(0.2 if self.using_f3 else 0.)),
#                                         trainable=(True if self.using_f3 else False))
#         self.bias_beta = self.add_weight(name='bias_beta',
#                                         shape=(self.units, self.num_samples),
#                                         initializer=initializers.RandomNormal(mean=0.,stddev=(0.2 if self.using_f3 else 0.)),
#                                         trainable=(True if self.using_f3 else False))
#         super().build(input_shape)


#     def call(self, inputs, **kwargs):
#         """Calculation of the forward direction of the network by
#         -drawing parameters
#         -Calculation loss KL
#         -Calculation output of the layer
#         """
             
#         # 1.) Conversion of trainable variational parameters to parameters of distribution (x)
#         kernel_a = VIMLTS_utils.to_a(self.kernel_tilde_a)
#         kernel_theta = VIMLTS_utils.to_theta([self.kernel_start_theta,
#                                         self.kernel_delta_theta_1,
#                                         self.kernel_delta_theta_2,
#                                         self.kernel_delta_theta_3,
#                                         self.kernel_delta_theta_4,
#                                         self.kernel_delta_theta_5,
#                                         self.kernel_delta_theta_6,
#                                         self.kernel_delta_theta_7,
#                                         self.kernel_delta_theta_8,
#                                         self.kernel_delta_theta_9])
#         print(kernel_theta)
#         kernel_theta=tf.transpose(kernel_theta, perm=[1, 2, 0])
#         kernel_alpha = VIMLTS_utils.to_alpha(self.kernel_tilde_alpha)

#         # 3.) Conversion of trainable variational parameters to parameters of distribution (bias)
#         bias_a = VIMLTS_utils.to_a(self.bias_tilde_a)
#         bias_theta = VIMLTS_utils.to_theta([self.bias_start_theta,
#                                         self.bias_delta_theta_1,
#                                         self.bias_delta_theta_2,
#                                         self.bias_delta_theta_3,
#                                         self.bias_delta_theta_4,
#                                         self.bias_delta_theta_5,
#                                         self.bias_delta_theta_6,
#                                         self.bias_delta_theta_7,
#                                         self.bias_delta_theta_8,
#                                         self.bias_delta_theta_9])
#         bias_theta=tf.transpose(bias_theta, perm=[1, 0])
#         bias_alpha = VIMLTS_utils.to_alpha(self.bias_tilde_alpha)


#         # 2.) Sampling from variational dist by reparametrization trick - prototype version (x)
#         for current_sample in range(self.num_samples):

#             kernel_z_sample=tf.random.normal(self.kernel_tilde_a.shape)
#             kernel_w_sample=VIMLTS_utils.kernel_h_z2w(z=kernel_z_sample,
#                                             a=kernel_a, 
#                                             b=self.kernel_b, 
#                                             theta=kernel_theta, 
#                                             alpha=kernel_alpha, 
#                                             beta=self.kernel_beta, 
#                                             beta_dist=self.kernel_beta_dist)

#             if current_sample==0:
#                 kernel_w_sample_list=tf.reshape(kernel_w_sample,[1,kernel_w_sample.shape[0],kernel_w_sample.shape[1]])
#             else:
#                 kernel_w_sample_list=tf.concat([kernel_w_sample_list,tf.reshape(kernel_w_sample,[1,kernel_w_sample.shape[0],kernel_w_sample.shape[1]])],axis=0)


#             # 4.) Sampling from variational dist by reparametrization trick - prototype version (bias)
#             bias_z_sample=tf.random.normal(self.bias_tilde_a.shape)
#             bias_w_sample=VIMLTS_utils.bias_h_z2w(z=bias_z_sample,
#                                             a=bias_a, 
#                                             b=self.bias_b, 
#                                             theta=bias_theta, 
#                                             alpha=bias_alpha, 
#                                             beta=self.bias_beta, 
#                                             beta_dist=self.bias_beta_dist)
#             if current_sample==0:
#                 bias_w_sample_list=tf.reshape(bias_w_sample,[1,bias_w_sample.shape[0]])
#             else:
#                 bias_w_sample_list=tf.concat([bias_w_sample_list,tf.reshape(bias_w_sample,[1,bias_w_sample.shape[0]])],axis=0)


#             # 5.) Calculation l_kl by calling function with passing the sample and the variational parameters (x and bias)

#             kernel_kl_loss=self.calc_kernel_kl_loss(kernel_w_sample=kernel_w_sample,
#                                 kernel_z_sample=kernel_z_sample,
#                                 kernel_a=kernel_a,
#                                 kernel_theta=kernel_theta,
#                                 kernel_alpha=kernel_alpha)
#             if current_sample==0:
#                 kernel_kl_loss_sum=kernel_kl_loss
#             else:
#                 kernel_kl_loss_sum+=kernel_kl_loss


#             bias_kl_loss=self.calc_bias_kl_loss(bias_w_sample=bias_w_sample,
#                                 bias_z_sample=bias_z_sample,
#                                 bias_a=bias_a,
#                                 bias_theta=bias_theta,
#                                 bias_alpha=bias_alpha)
#             if current_sample==0:
#                 bias_kl_loss_sum=bias_kl_loss
#             else:
#                 bias_kl_loss_sum+=bias_kl_loss


#         self.add_loss(1/self.num_samples*kernel_kl_loss_sum+1/self.num_samples*bias_kl_loss_sum)

#         # kernel_mean=tf.reduce_mean(kernel_w_sample_list,axis=0)
#         # bias_mean=tf.reduce_mean(bias_w_sample_list,axis=0)

#         # 6.) Calculation output of the layer
#         return self.activation(K.dot(inputs, kernel_mean)+ bias_mean)

#     def calc_kernel_kl_loss(self,kernel_w_sample,kernel_z_sample,kernel_a,kernel_theta,kernel_alpha):
#         kernel_z_epsilon=kernel_z_sample+self.epsilon
#         kernel_q,w_check=VIMLTS_utils.kernel_eval_variational_dist(z=kernel_z_sample, 
#                                                             z_epsilon=kernel_z_epsilon, 
#                                                             a=kernel_a, 
#                                                             b=self.kernel_b, 
#                                                             theta=kernel_theta, 
#                                                             alpha=kernel_alpha, 
#                                                             beta=self.kernel_beta, 
#                                                             beta_dist=self.kernel_beta_dist)
#         kernel_kl_loss=self.kl_loss(kernel_q,kernel_w_sample)
#         return kernel_kl_loss

#     def calc_bias_kl_loss(self,bias_w_sample,bias_z_sample,bias_a,bias_theta,bias_alpha):
#         bias_z_epsilon=bias_z_sample+self.epsilon
#         bias_q,w_check=VIMLTS_utils.bias_eval_variational_dist(z=bias_z_sample, 
#                                                             z_epsilon=bias_z_epsilon, 
#                                                             a=bias_a, 
#                                                             b=self.bias_b, 
#                                                             theta=bias_theta, 
#                                                             alpha=bias_alpha, 
#                                                             beta=self.bias_beta, 
#                                                             beta_dist=self.bias_beta_dist)
#         bias_kl_loss=self.kl_loss(bias_q,bias_w_sample)
#         return bias_kl_loss


#     def kl_loss(self, q, w_sample):
#         # Calculate KL divergence between variational dist and prior
#         return self.kl_weight * K.sum(K.log(q) - self.log_prior_prob(w_sample))


#     def log_prior_prob(self, w):
#         # Prior
#         prior_dist = tfp.distributions.Normal(self.prior_mu, self.prior_sigma)
#         return prior_dist.log_prob(w) 