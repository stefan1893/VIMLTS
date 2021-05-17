import tensorflow as tf
from tensorflow_probability import distributions as tfd
import tensorflow_probability as tfp


class VimltsLinear(tf.keras.layers.Layer):
    def __init__(self,
                 units,
                 activation,
                 alpha_w=1.,
                 beta_w=0.,
                 alpha_z=1.,
                 beta_z=0.,
                 theta=[],
                 num_samples=10,
                 prior_dist=tfd.Normal(loc=0., scale=1.)):
        super().__init__()
        self.alpha_w_init = alpha_w
        self.beta_w_init = beta_w
        self.alpha_z_init = alpha_z
        self.beta_z_init = beta_z
        self.theta_init = theta
        self.units_ = units
        self.num_samples_ = num_samples
        self.prior_dist_ = prior_dist
        self.beta_dist = self.init_beta_dist(len(theta))
        self.z_dist_ = None
        self.activation_ = activation

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
        self.z_dist_ = tfd.Normal(loc=tf.zeros((input_shape[1], self.units_)),
                                  scale=tf.ones((input_shape[1], self.units_)))
        self.alpha_w = self.add_weight(name='alpha_w',
                                       shape=(input_shape[1], self.units_),
                                       initializer=tf.constant_initializer(self.alpha_w_init),
                                       trainable=True)
        self.beta_w = self.add_weight(name='beta_w',
                                      shape=(input_shape[1], self.units_),
                                      initializer=tf.constant_initializer(self.beta_w_init),
                                      trainable=True)
        self.alpha_z = self.add_weight(name='alpha_z',
                                       shape=(input_shape[1], self.units_),
                                       initializer=tf.constant_initializer(self.alpha_z_init),
                                       trainable=True)
        self.beta_z = self.add_weight(name='beta_z',
                                      shape=(input_shape[1], self.units_),
                                      initializer=tf.constant_initializer(self.beta_z_init),
                                      trainable=True)


        # theta_prime = tf.tile(self.theta_init, [input_shape[1] * self.units_])
        # theta_prime = tf.reshape(theta_prime, (input_shape[1], self.units_, len(self.theta_init)))
        shape = (input_shape[1], self.units_)
        theta_prime = tf.stack([i(shape=shape) for i in theta_initializer], axis=2)
        self.theta_prime = tf.Variable(initial_value=theta_prime, trainable=True)
        super().build(input_shape)

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
            w = self.f_3(self.f_2(self.f_1(z)))
            dw_dz = tape.gradient(w, z)

        # inputs (batch, in); w (sample, in ,out)
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
            kl = tf.reduce_sum(tf.reduce_mean(log_q_out, (0,1))) - tf.reduce_sum(tf.reduce_mean(log_p_out, (0,1)))
        else:
            log_p_w = self.prior_dist_.log_prob(w)
            kl = tf.reduce_sum(tf.reduce_mean(log_q_w, 0)) - tf.reduce_sum(tf.reduce_mean(log_p_w, 0))
        self.add_loss(kl)
        # tf.print("KL: ", kl)
        return out
