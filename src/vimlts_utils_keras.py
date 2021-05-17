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

def kernel_eval_h_MLT(z,theta,beta_dist):
    my_result=[]
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            my_result=tf.concat((my_result,beta_dist.prob(z[i][j])),axis=0)
            
    fIm=tf.reshape(my_result,[theta.shape[0],theta.shape[1],-1])
    #fIm=beta_dist.prob(z)
    my_return=tf.math.reduce_mean(fIm*theta,axis=2)
    return my_return

def bias_eval_h_MLT(z,theta,beta_dist):
    my_result=[]
    for i in range(z.shape[0]):
        my_result=tf.concat((my_result,beta_dist.prob(z[i])),axis=0)
    fIm=tf.reshape(my_result,[theta.shape[0],theta.shape[1]])
    #fIm=beta_dist.prob(z)
    my_return=tf.math.reduce_mean(fIm*theta,axis=1)
    return my_return

def eval_h_MLT(z,theta,beta_dist):
    z=tf.reshape(z,[-1,1])
    fIm=beta_dist.prob(z)
    return tf.math.reduce_mean(fIm*theta,axis=1)

def kernel_h_z2w(z, a, b, theta, alpha, beta, beta_dist):
    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)
    h_MLT=kernel_eval_h_MLT(z=z_sig, theta=theta, beta_dist=beta_dist)
    w=alpha*h_MLT-beta 
    return w

def bias_h_z2w(z, a, b, theta, alpha, beta, beta_dist):
    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)
    h_MLT=bias_eval_h_MLT(z=z_sig, theta=theta, beta_dist=beta_dist)
    w=alpha*h_MLT-beta 
    return w

def h_z2w(z, a, b, theta, alpha, beta, beta_dist):
    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)
    h_MLT=eval_h_MLT(z=z_sig, theta=theta, beta_dist=beta_dist)
    w=alpha*h_MLT-beta 
    return w


# Method using derivation
def kernel_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):
    fz=tfd.Normal(loc=0,scale=1).prob(z)
    with tf.GradientTape() as tape:
        tape.watch([z])
        w=kernel_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
        dw_dz = tape.gradient(w, z)
    h_w2z_dash = 1.0 / dw_dz
    q=fz*tf.math.abs(h_w2z_dash)
    return q,w 

# Method using epsilon
# def kernel_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):#, beta_dist_dash):
#     fz=tfd.Normal(loc=0,scale=1).prob(z) 
#     w=kernel_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     w_epsilon=kernel_h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)
#     q=fz*tf.math.abs(h_w2z_dash)
#     return q,w 

# Method using derivation
def bias_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):
    fz=tfd.Normal(loc=0,scale=1).prob(z)
    with tf.GradientTape() as tape:
        tape.watch([z])
        w=bias_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
        dw_dz = tape.gradient(w, z)
    h_w2z_dash = 1.0 / dw_dz
    q=fz*tf.math.abs(h_w2z_dash)
    return q,w 

# # Method using epsilon
# def bias_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):
#     fz=tfd.Normal(loc=0,scale=1).prob(z) 
#     w=bias_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     w_epsilon=bias_h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
#     h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)
#     q=fz*tf.math.abs(h_w2z_dash)
#     return q,w 

# Method using derivation
def eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):
    fz=tfd.Normal(loc=0,scale=1).prob(z)
    with tf.GradientTape() as tape:
        tape.watch([z])
        w=h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
        dw_dz = tape.gradient(w, z)
    h_w2z_dash = 1.0 / dw_dz
    q=fz*tf.math.abs(h_w2z_dash)
    return q,w

# Method using epsilon
# def eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):#, beta_dist_dash):
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


def get_target_dist(lambdas):
    zz=tf.Variable(np.linspace(-6,6,1000),dtype='float32')

    epsilon=tf.constant(0.001)
    beta_dist=init_beta_dist(10)
    z_epsilon=tf.Variable(zz+epsilon)

    a=to_a(lambdas[0:1])
    theta=to_theta(lambdas[2:12])
    alpha=to_alpha(lambdas[12:13])

    q_dist,ww=eval_variational_dist(z=zz,z_epsilon=z_epsilon,a=a, b=lambdas[1:2], theta=theta, alpha=alpha, beta=lambdas[13:14], beta_dist=beta_dist)
    return q_dist,ww