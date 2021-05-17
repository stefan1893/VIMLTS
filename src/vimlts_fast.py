import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp
from tensorflow.keras import initializers


class VimltsLinear(tf.keras.layers.Layer):
    def __init__(self,
                 units: int,
                 activation: tf.keras.activations = tf.keras.activations.linear,
                 num_samples: int = 10,
                 kernel_init_alpha_w: initializers = initializers.Constant(1.),
                 kernel_init_beta_w: initializers = initializers.Constant(0.),
                 kernel_init_alpha_z: initializers = initializers.Constant(1.),
                 kernel_init_beta_z: initializers = initializers.Constant(0.),
                 kernel_init_thetas: list = [initializers.RandomNormal(mean=0., stddev=1.),
                                             initializers.RandomNormal(mean=0., stddev=1.)],
                 bias_init_alpha_w: initializers = None,
                 bias_init_beta_w: initializers = None,
                 bias_init_alpha_z: initializers = None,
                 bias_init_beta_z: initializers = None,
                 bias_init_thetas: list = None,
                 prior_dist: object = tfd.Normal(loc=0., scale=1.)) -> object:
        """

        Args:
            units: number of output neurons
            activation: activation function
            num_samples: Number of samples to approximate KL and expected(NLL)
            kernel_init_alpha_w: initializer for scale parameter alpha_w
            kernel_init_beta_w: initializer for shift parameter beta_w
            kernel_init_alpha_z: initializer for scale parameter alpha_z
            kernel_init_beta_z: initializer for shift parameter beta_z
            kernel_init_thetas: initializer for the coefficient of the Bernstein polynomial for kernel parameter. Number of initializers defines the degree M.
            bias_init_alpha_w: initializer for scale parameter alpha_w (If any of the bias initializers is None, no bias parameter are used)
            bias_init_beta_w: initializer for shift parameter beta_w
            bias_init_alpha_z: initializer for scale parameter alpha_z
            bias_init_beta_z: initializer for shift parameter beta_z
            bias_init_thetas: initializer for the coefficient of the Bernstein polynomial for bias parameter. Number of initializers defines the degree M.
            prior_dist: prior distribution p(\theta)
        """
        super().__init__()
        self.units_ = units
        self.activation_ = activation
        self.num_samples_ = num_samples
        # initializers for parameters
        self.k_alpha_w_ = kernel_init_alpha_w
        self.k_beta_w_ = kernel_init_beta_w
        self.k_alpha_z_ = kernel_init_alpha_z
        self.k_beta_z_ = kernel_init_beta_z
        self.k_thetas_ = kernel_init_thetas
        self.k_beta_dist = self.init_beta_dist(len(kernel_init_thetas))

        if (bias_init_thetas is None) or \
                (bias_init_beta_z is None) or \
                (bias_init_alpha_z is None) or \
                (bias_init_beta_w is None) or \
                (bias_init_alpha_w is None):
            self.use_bias = False
        else:
            self.use_bias = True
            self.b_alpha_w_ = bias_init_alpha_w
            self.b_beta_w_ = bias_init_beta_w
            self.b_alpha_z_ = bias_init_alpha_z
            self.b_beta_z_ = bias_init_beta_z
            self.b_thetas_ = bias_init_thetas
            self.b_beta_dist = self.init_beta_dist(len(bias_init_thetas))

        self.prior_dist_ = prior_dist
        self.z_dist_ = None
        self.alpha_w = None
        self.beta_w = None
        self.alpha_z = None
        self.beta_z = None
        self.theta_prime = None
        self.beta_dist = None

    @staticmethod
    def init_beta_dist(M):
        in1 = []
        in2 = []
        for i in range(1, M + 1):
            in1.append(i)
            in2.append(M - i + 1)
        # print("Koeffizienten beta_dist:")
        # print(f'in1 = {in1}')
        # print(f'in2 = {in2}')
        return tfd.Beta(in1, in2)

    def build(self, input_shape):
        """
        Initialization of the trainable variational parameters, for x (independent of #units) and for bias
        """

        # Kernel
        shape = (input_shape[1], self.units_)
        self.z_dist_ = tfd.Normal(loc=tf.zeros(shape),
                                  scale=tf.ones(shape))
        self.k_alpha_w = self.add_weight(name='k_alpha_w',
                                         shape=shape,
                                         initializer=self.k_alpha_w_,
                                         trainable=True)
        self.k_beta_w = self.add_weight(name='k_beta_w',
                                        shape=shape,
                                        initializer=self.k_beta_w_,
                                        trainable=True)
        self.k_alpha_z = self.add_weight(name='k_alpha_z',
                                         shape=shape,
                                         initializer=self.k_alpha_z_,
                                         trainable=True)
        self.k_beta_z = self.add_weight(name='k_beta_z',
                                        shape=shape,
                                        initializer=self.k_beta_z_,
                                        trainable=True)
        theta_prime = tf.stack([i(shape=shape) for i in self.k_thetas_], axis=-1)
        self.k_theta_prime = tf.Variable(initial_value=theta_prime, trainable=True)

        if self.use_bias:
            shape = (self.units_,)
            self.b_z_dist_ = tfd.Normal(loc=tf.zeros(shape),
                                        scale=tf.ones(shape))
            self.b_alpha_w = self.add_weight(name='b_alpha_w',
                                             shape=shape,
                                             initializer=self.b_alpha_w_,
                                             trainable=True)
            self.b_beta_w = self.add_weight(name='b_beta_w',
                                            shape=shape,
                                            initializer=self.b_beta_w_,
                                            trainable=True)
            self.b_alpha_z = self.add_weight(name='b_alpha_z',
                                             shape=shape,
                                             initializer=self.b_alpha_z_,
                                             trainable=True)
            self.b_beta_z = self.add_weight(name='b_beta_z',
                                            shape=shape,
                                            initializer=self.b_beta_z_,
                                            trainable=True)
            b_theta_prime = tf.stack([i(shape=shape) for i in self.b_thetas_], axis=-1)
            self.b_theta_prime = tf.Variable(initial_value=b_theta_prime, trainable=True)
        super().build(input_shape)

    def activate_bias_transformation(self):
        self.alpha_w = self.b_alpha_w
        self.beta_w = self.b_beta_w
        self.alpha_z = self.b_alpha_z
        self.beta_z = self.b_beta_z
        self.theta_prime = self.b_theta_prime
        self.beta_dist = self.b_beta_dist

    def activate_kernel_transformation(self):
        self.alpha_w = self.k_alpha_w
        self.beta_w = self.k_beta_w
        self.alpha_z = self.k_alpha_z
        self.beta_z = self.k_beta_z
        self.theta_prime = self.k_theta_prime
        self.beta_dist = self.k_beta_dist

    def f_1(self, z):
        """

        :param z: [#samples x #input x #output]
        :return: [#samples x #input x #output]
        """
        z_ = tf.math.multiply(tf.math.softplus(self.alpha_z), z) - self.beta_z
        return tf.math.sigmoid(z_)

    def f_2(self, z_):
        """

        :param z_: [#samples x #input x #output]
        :return:
        """
        theta_p = self.theta_prime  # [#input x #output x M]
        theta_p = tf.concat((theta_p[..., 0:1], tf.math.softplus(theta_p[..., 1:])), axis=-1)

        n = theta_p.shape[-1]
        tf.ones((n * (n + 1) // 2))
        m_triangle = tfp.math.fill_triangular(tf.ones(n * (n + 1) // 2), upper=True)

        theta = theta_p @ m_triangle
        fIm = self.beta_dist.prob(z_[..., None])  # to broadcast beta dist [#samples x #input x #output x M]
        return tf.math.reduce_mean(fIm * theta, axis=-1)
        # return z_

    def f_3(self, z_w):
        """

        :type z_w: object
        :return: shape [#sample x #input x #output]
        """
        return tf.math.multiply(tf.math.softplus(self.alpha_w), z_w) - self.beta_w

    def get_w_dist(self, num=1000):
        with tf.GradientTape() as tape:
            zz = tf.dtypes.cast(tf.reshape(tf.linspace(-6, 6, num), shape=(-1, 1, 1)), tf.float32)
            tape.watch(zz)
            self.activate_kernel_transformation()
            w = self.f_3(self.f_2(self.f_1(zz)))
            dw_dz = tape.gradient(w, zz)
        # tf.reduce_prod(w.shape[1:]) -> undo gradiant adding because of zz broadcasting
        dw_dz /= tf.cast(tf.reduce_prod(w.shape[1:]), dtype=tf.float32)
        log_p_z = self.z_dist_.log_prob(zz)
        log_q_w = log_p_z - tf.math.log(tf.math.abs(dw_dz))
        return tf.math.exp(log_q_w).numpy(), w.numpy()

    def call(self, inputs, **kwargs):
        """

        :param inputs: [#batch x #input]
        :param kwargs:
        :return: [#samples x #batch x #output]
        """
        with tf.GradientTape() as tape:
            z = self.z_dist_.sample(self.num_samples_)
            tape.watch(z)
            self.activate_kernel_transformation()
            w = self.f_3(self.f_2(self.f_1(z)))
            dw_dz = tape.gradient(w, z)

        with tf.GradientTape() as tape:
            if self.use_bias:
                zb = self.b_z_dist_.sample(self.num_samples_)
                tape.watch(zb)
                self.activate_bias_transformation()
                b = self.f_3(self.f_2(self.f_1(zb)))
                db_dz = tape.gradient(b, zb)

        # inputs (batch, in); w (sample, in ,out)
        if self.use_bias:
            out = self.activation_(inputs @ w + b[:, None, :])
        else:
            out = self.activation_(inputs @ w)

        # compute kl divergence
        # change of variable ==> p(w) = p(z)/|dw/dz|
        log_p_z = self.z_dist_.log_prob(z)
        # log rules ==> log(p(w)) = log(p(z)) - log(|dw/dz|)
        log_q_w = log_p_z - tf.math.log(tf.math.abs(dw_dz))
        if isinstance(self.activation_, tfp.bijectors.Bijector):
            print("YES activation is Bijector")
            log_p_out = self.prior_dist_.log_prob(out)
            log_q_out = log_q_w + self.activation_.inverse_log_det_jacobian(out, event_ndims=0)
            # kl = tf.reduce_sum(tf.reduce_mean(log_q_out, 0)) - tf.reduce_sum(tf.reduce_mean(log_p_out, 0))
            kl = tf.reduce_sum(tf.reduce_mean(log_q_out, (0, 1))) - tf.reduce_sum(tf.reduce_mean(log_p_out, (0, 1)))
        else:
            log_p_w = self.prior_dist_.log_prob(w)
            kl = tf.reduce_sum(tf.reduce_mean(log_q_w, 0)) - tf.reduce_sum(tf.reduce_mean(log_p_w, 0))
        if self.use_bias:
            # compute kl divergence for bias term
            # change of variable ==> p(w) = p(z)/|dw/dz|
            b_log_p_z = self.b_z_dist_.log_prob(zb)
            # log rules ==> log(p(w)) = log(p(z)) - log(|dw/dz|)
            b_log_q_w = b_log_p_z - tf.math.log(tf.math.abs(db_dz))
            b_log_p_w = self.prior_dist_.log_prob(b)
            kl += tf.reduce_sum(tf.reduce_mean(b_log_q_w, 0)) - tf.reduce_sum(tf.reduce_mean(b_log_p_w, 0))
        self.add_loss(kl)
        # tf.print("KL: ", kl)
        return out
