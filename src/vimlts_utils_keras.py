import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd=tfp.distributions

#import vi_mlt_dist_utils


def init_beta_dist(M):
    in1 = []
    in2 = []
    for i in range(1,M+1):
        in1.append(i)
        in2.append(M-i+1)
    # print("Koeffizienten beta_dist:")
    # print(f'in1 = {in1}')
    # print(f'in2 = {in2}')
    return tfd.Beta(in1,in2)
    
def init_beta_dist_dash(M): # TODO: Noch zu Überprüfen, ob Koeffizienten richtig angelegt werden
    M=M-1
    in1 = []
    in2 = []
    for i in range(1,M+1):
        in1.append(i)
        in2.append(M-i+1)
    # print("Koeffizienten beta_dist:")
    # print(f'in1 = {in1}')
    # print(f'in2 = {in2}')
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

# def eval_h_MLT_dash(z, theta, beta_dist_dash):
#     # Hier noch abfrage das wI zwischen 0 und 1
#     #w=tf.clip_by_value(w,1E-5,1.0-1E-5)
#     len_koeff=theta.shape[0]
#     zI=tf.reshape(z,[-1,1])
#     zI=tf.cast(zI, tf.float32)

#     by=beta_dist_dash.prob(zI)
#     d_Theta=theta[1:len_koeff]-theta[0:(len_koeff-1)]

#     bern_dash=tf.reduce_sum(by*d_Theta,axis=1)
#     return bern_dash

@tf.function
def kernel_h_z2w(z, a, b, theta, alpha, beta, beta_dist):
    z_tilde=a*z-b
    z_sig=tf.math.sigmoid(z_tilde)
    h_MLT=kernel_eval_h_MLT(z=z_sig, theta=theta, beta_dist=beta_dist)
    w=alpha*h_MLT-beta 
    return w

@tf.function
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

# def h_w2z_black_box_inverse(w_to_inverse, a, b, theta, alpha, beta, beta_dist):
#     z_optimized = tf.Variable(0.)
#     # a_not_trainable=tf.Variable(a,trainable=False)
#     # b_not_trainable=tf.Variable(b,trainable=False)
#     # theta_not_trainable=tf.Variable(theta,trainable=False)
#     # alpha_not_trainable=tf.Variable(alpha,trainable=False)
#     # beta_not_trainable=tf.Variable(beta,trainable=False)
#     a_not_trainable=a.numpy()
#     b_not_trainable=b.numpy()
#     theta_not_trainable=theta.numpy()
#     alpha_not_trainable=alpha.numpy()
#     beta_not_trainable=beta.numpy()
#     loss_fn = lambda: (h_z2w(z=z_optimized,a=a_not_trainable,b=b_not_trainable,theta=theta_not_trainable,alpha=alpha_not_trainable,beta=beta_not_trainable,beta_dist=beta_dist) - w_to_inverse )**2
#     tfp.math.minimize(loss_fn,
#                     num_steps=30, 
#                     optimizer=tf.optimizers.Adam(learning_rate=0.1))
#     return z_optimized

# def h_w2z_fake_inverse_taylor(w_to_inverse, a, b, theta, alpha, beta, beta_dist, beta_dist_dash):
#     #Variablen
#     m_plus_1=theta.shape[0]

#     z=h_w2z_black_box_inverse(w_to_inverse=w_to_inverse,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)

#     z_tilde=a*z-b
#     z_sig=tf.math.sigmoid(z_tilde)

#     # taylor10=M*(0-betaDistDash.prob(w)[0])
#     # taylor11=M*(betaDistDash.prob(w)[0]-betaDistDash.prob(w)[1])
#     # taylor12=M*(betaDistDash.prob(w)[1]-betaDistDash.prob(w)[2])
#     # taylor13=M*(betaDistDash.prob(w)[2]-0)
#     taylor0=beta_dist.prob(z_sig)

#     taylor1=theta.shape[0]*(0-beta_dist_dash.prob(z_sig)[0:1])
#     for i in range(theta.shape[0]-2):
#         taylor1=tf.concat((taylor1,(theta.shape[0]*(beta_dist_dash.prob(z_sig)[i:i+1]-beta_dist_dash.prob(z_sig)[i+1:i+2]))),axis=0)
#     taylor1=tf.concat((taylor1,(theta.shape[0]*(beta_dist_dash.prob(z_sig)[theta.shape[0]-2:theta.shape[0]-1]))),axis=0)


#     z_sig_fake=(((w_to_inverse+beta)/alpha)*m_plus_1-tf.reduce_sum(taylor0*theta))/tf.reduce_sum(taylor1*theta)+z_sig

#     # # Umkehrfunktion Sigmoid und f1
#     arg_log=1/z_sig_fake-1
#     z_fake=(-tf.math.log(arg_log)+b)/a
#     #w_fake=(-tf.math.log(1/w_sig_fake-1)+b)/a

#     return z_fake

@tf.function
def kernel_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):#, beta_dist_dash):
    #z=h_w2z_fake_inverse_taylor(w_to_inverse=w, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    fz=tfd.Normal(loc=0,scale=1).prob(z)  #Evt besser für Laufzeit, wenn hier direkt ein z Bereich benutzt wird,
                                                                                                #Bzw. schaue Dir Trick von Beate und Oliver an

    w=kernel_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
    w_epsilon=kernel_h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)

    #z_epsilon=h_w2z_fake_inverse_taylor(w_to_inverse=w_epsilon, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)

    q=fz*tf.math.abs(h_w2z_dash)
    return q,w 

@tf.function
def bias_eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):#, beta_dist_dash):
    #z=h_w2z_fake_inverse_taylor(w_to_inverse=w, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    fz=tfd.Normal(loc=0,scale=1).prob(z)  #Evt besser für Laufzeit, wenn hier direkt ein z Bereich benutzt wird,
                                                                                                #Bzw. schaue Dir Trick von Beate und Oliver an

    w=bias_h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
    w_epsilon=bias_h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)

    #z_epsilon=h_w2z_fake_inverse_taylor(w_to_inverse=w_epsilon, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)

    q=fz*tf.math.abs(h_w2z_dash)
    return q,w 

def eval_variational_dist(z, z_epsilon, a, b, theta, alpha, beta, beta_dist):#, beta_dist_dash):
    #z=h_w2z_fake_inverse_taylor(w_to_inverse=w, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    fz=tfd.Normal(loc=0,scale=1).prob(z)  #Evt besser für Laufzeit, wenn hier direkt ein z Bereich benutzt wird,
                                                                                                #Bzw. schaue Dir Trick von Beate und Oliver an

    w=h_z2w(z=z,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)
    w_epsilon=h_z2w(z=z_epsilon,a=a,b=b,theta=theta,alpha=alpha,beta=beta,beta_dist=beta_dist)

    #z_epsilon=h_w2z_fake_inverse_taylor(w_to_inverse=w_epsilon, a=a, b=b, theta=theta, alpha=alpha, beta=beta, beta_dist=beta_dist, beta_dist_dash=beta_dist_dash)
    h_w2z_dash=(z_epsilon-z)/(w_epsilon-w)

    q=fz*tf.math.abs(h_w2z_dash)
    return q,w 

def to_a(a_tunable):
    #return tf.Variable(1.)
    return tf.math.softplus(a_tunable[0:1])
    #return tf.math.sigmoid(tf.math.softplus(a_tunable[0:1]))

def to_theta(theta_tunable):
    theta=theta_tunable[0:1]
    for i in range(np.shape(theta_tunable)[0]-1):
        theta=tf.concat((theta,(theta[i:i+1]+tf.math.softplus(theta_tunable[i+1:i+2]))),axis=0)
    return theta

def to_alpha(alpha_tunable):
    #return tf.math.sigmoid(tf.math.softplus(alpha_tunable[0:1]))
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