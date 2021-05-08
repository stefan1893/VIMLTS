import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions



def init_beta_dist(M):
    in1 = []
    in2 = []
    for i in range(1,M+1):
        in1.append(i)
        in2.append(M-i+1)

    return tfd.Beta(in1,in2)
    
def init_beta_dist_dash(M):
    M=M-1
    in1 = []
    in2 = []
    for i in range(1,M+1):
        in1.append(i)
        in2.append(M-i+1)

    return tfd.Beta(in1,in2)

def eval_h_MLT(z,theta,beta_dist):
    if (z.shape==()):
        zI=z
        fIm=beta_dist.prob(zI)
        return tf.math.reduce_mean(fIm*theta) 
    else:
        zI=tf.reshape(z,[-1,1])
        zI=tf.cast(zI, tf.float32)
        fIm=beta_dist.prob(zI)
        return tf.math.reduce_mean(fIm*theta,axis=1) 

def eval_h_MLT_dash(z, theta, beta_dist_dash):
    len_koeff=theta.shape[0]
    zI=tf.reshape(z,[-1,1])
    zI=tf.cast(zI, tf.float32)

    by=beta_dist_dash.prob(zI)
    d_Theta=theta[1:len_koeff]-theta[0:(len_koeff-1)]

    bern_dash=tf.reduce_sum(by*d_Theta,axis=1)
    return bern_dash

def h_z2w(z, a, b, theta, alpha, beta, beta_dist):
    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)
    h_MLT=eval_h_MLT(z=z_sig, theta=theta, beta_dist=beta_dist)
    w=alpha*h_MLT-beta
    return w

def h_w2z_black_box_inverse(w_to_inverse, a, b, theta, alpha, beta, beta_dist):
    z_optimized = tf.Variable(0.)

    a_not_trainable=a.numpy()
    b_not_trainable=b.numpy()
    theta_not_trainable=theta.numpy()
    alpha_not_trainable=alpha.numpy()
    beta_not_trainable=beta.numpy()

    loss_fn = lambda: (h_z2w(z=z_optimized,a=a_not_trainable,b=b_not_trainable,theta=theta_not_trainable,alpha=alpha_not_trainable,beta=beta_not_trainable,beta_dist=beta_dist) - w_to_inverse )**2
    tfp.math.minimize(loss_fn,
                    num_steps=30, 
                    optimizer=tf.optimizers.Adam(learning_rate=0.1))
    return z_optimized

def h_w2z_fake_inverse_taylor(w_to_inverse, a, b, theta, alpha, beta, beta_dist, beta_dist_dash):
    m_plus_1=theta.shape[0]

    z=h_w2z_black_box_inverse(w_to_inverse=w_to_inverse,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)

    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)

    taylor0=beta_dist.prob(z_sig)
    taylor1=theta.shape[0]*(0-beta_dist_dash.prob(z_sig)[0:1])
    for i in range(theta.shape[0]-2):
        taylor1=tf.concat((taylor1,(theta.shape[0]*(beta_dist_dash.prob(z_sig)[i:i+1]-beta_dist_dash.prob(z_sig)[i+1:i+2]))),axis=0)
    taylor1=tf.concat((taylor1,(theta.shape[0]*(beta_dist_dash.prob(z_sig)[theta.shape[0]-2:theta.shape[0]-1]))),axis=0)

    z_sig_fake=(((w_to_inverse+beta)/alpha)*m_plus_1-tf.reduce_sum(taylor0*theta))/tf.reduce_sum(taylor1*theta)+z_sig

    arg_log=1/z_sig_fake-1
    z_fake=(-tf.math.log(arg_log)+b)/a

    return z_fake

# Method using derivation
def eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist, beta_dist_dash):
    fz=tfd.Normal(loc=0,scale=1).prob(z)
    with tf.GradientTape() as tape:
        tape.watch([z])
        w=h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
        dw_dz = tape.gradient(w, z)
    h_w2z_dash = 1.0 / dw_dz
    q=fz*tf.math.abs(h_w2z_dash)
    return q,w

# Method using epsilon
# def eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist, beta_dist_dash):
#     fz=tfd.Normal(loc=0,scale=1).prob(z)
#     w=h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     w_epsilon=h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)
#     q=fz*tf.math.abs(h_w2z_dash)
#     return q,w 

def to_a(a_tunable):
    return tf.math.softplus(a_tunable[0:1])

def to_theta(theta_tunable):
    theta=theta_tunable[0:1]
    for i in range(np.shape(theta_tunable)[0]-1):
        theta=tf.concat((theta,(theta[i:i+1]+tf.math.softplus(theta_tunable[i+1:i+2]))),axis=0)
    return theta

def to_alpha(alpha_tunable):
    return tf.math.softplus(alpha_tunable[0:1])






class VIMLTS:
    def __init__(self, m):
        self.m=m
        self.num_params=self.m+4
        self.beta_dist=init_beta_dist(self.m)
        self.beta_dist_dash=init_beta_dist_dash(self.m)
        
    def update_lambda_param(self, lambda_update):
        self.a_tilde=lambda_update[0:1]
        self.b=lambda_update[1:2]
        self.theta_delta=lambda_update[2:self.num_params-2]
        self.alpha_tilde=lambda_update[self.num_params-2:self.num_params-1]
        self.beta=lambda_update[self.num_params-1:self.num_params]

    def get_target_dist(self):
        zz=tf.Variable(np.linspace(-6,6,1000),dtype='float32')
        epsilon=tf.constant(0.001)
        z_epsilon=tf.Variable(zz+epsilon)
        q_dist,ww=eval_variational_dist(z=zz,z_epsilon=z_epsilon,a=to_a(self.a_tilde), b=self.b, theta=to_theta(self.theta_delta), alpha=to_alpha(self.alpha_tilde), beta=self.beta, beta_dist=self.beta_dist, beta_dist_dash=self.beta_dist_dash)
        return q_dist,ww
    
    def get_target_dist_for_z(self,z):
        epsilon=tf.constant(0.001)
        z_epsilon=tf.Variable(z+epsilon)
        q_dist,ww=eval_variational_dist(z=z,z_epsilon=z_epsilon,a=to_a(self.a_tilde), b=self.b, theta=to_theta(self.theta_delta), alpha=to_alpha(self.alpha_tilde), beta=self.beta, beta_dist=self.beta_dist, beta_dist_dash=self.beta_dist_dash)
        return q_dist

    def get_sample_w(self):
        z_sample = tfd.Normal(loc=0., scale=1.).sample()
        return z_sample,h_z2w(z=z_sample, a=to_a(self.a_tilde), b=self.b, theta=to_theta(self.theta_delta), alpha=to_alpha(self.alpha_tilde), beta=self.beta, beta_dist=self.beta_dist)

    def get_h_mlt(self,overValues):
        return eval_h_MLT(z=overValues,theta=to_theta(self.theta_delta),beta_dist=self.beta_dist)

    def get_h_mlt_dash(self,overValues):
        return eval_h_MLT_dash(z=overValues,theta=to_theta(self.theta_delta),beta_dist_dash=self.beta_dist_dash)

    ##################################################### Debug functions ####################################################
    def get_beta(self,x):
        return self.beta_dist.prob(x)

    def get_beta_dash(self,x):
        return self.beta_dist_dash.prob(x)

    def get_param(self):
        param_array=to_a(self.a_tilde).numpy()
        param_array=np.concatenate((param_array,self.b.numpy()),axis=0)
        param_array=np.concatenate((param_array,to_theta(self.theta_delta).numpy()),axis=0)
        param_array=np.concatenate((param_array,to_alpha(self.alpha_tilde).numpy()),axis=0)
        param_array=np.concatenate((param_array,self.beta.numpy()),axis=0)
        return param_array

    def test_transformation(self,w_test):
        z_test=h_w2z_fake_inverse_taylor(w_to_inverse=w_test, a=to_a(self.a_tilde), b=self.b, theta=to_theta(self.theta_delta), alpha=to_alpha(self.alpha_tilde), beta=self.beta, beta_dist=self.beta_dist, beta_dist_dash=self.beta_dist_dash)
        print("ztest:",z_test)
        return h_z2w(z=z_test, a=to_a(self.a_tilde), b=self.b, theta=to_theta(self.theta_delta), alpha=to_alpha(self.alpha_tilde), beta=self.beta, beta_dist=self.beta_dist)
    
    def print_param(self):
        print("self.num_params:\t",self.num_params)
        print("self.a_tilde:\t\t",self.a_tilde.numpy())
        print("a:\t\t\t",to_a(self.a_tilde).numpy())
        print("self.b:\t\t\t",self.b.numpy())
        print("self.theta_delta:\t",self.theta_delta.numpy())
        print("theta:\t\t\t",to_theta(self.theta_delta).numpy())
        print("self.alpha_tilde:\t",self.alpha_tilde.numpy())
        print("alpha:\t\t\t",to_alpha(self.alpha_tilde).numpy())
        print("self.beta:\t\t",self.beta.numpy())
