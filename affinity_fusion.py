# Deep Multimodal Subspace Clustering Networks
# https://arxiv.org/abs/1804.06498
# Mahdi Abavisani
# mahdi.abavisani@rutgers.edu
# Built upon https://github.com/panji1990/Deep-subspace-clustering-networks

import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers
import scipy.io as sio
from scipy.sparse.linalg import svds
from sklearn import cluster
from sklearn.metrics.cluster import normalized_mutual_info_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from munkres import Munkres
import argparse
from datetime import datetime




class ConvAE(object):
    def __init__(self, n_input, kernel_size, n_hidden, reg_constant1 = 1.0, re_constant2 = 1.0, batch_size = 100, reg = None, \
                denoise = False, model_path = None, restore_path = None, \
                logs_path = './logs', num_modalities=2):
        self.n_input = n_input
        self.kernel_size = kernel_size
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.reg = reg
        self.model_path = model_path
        self.restore_path = restore_path
        self.iter = 0
        self.num_modalities =num_modalities
        weights = self._initialize_weights()
        self.x={}
        
        #input required to be fed
        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])

        self.learning_rate = tf.placeholder(tf.float32, [],
                                        name='learningRate')
        

        
        if denoise == False:
            x_input = self.x
            latents, shape = self.encoder(x_input,weights,self.num_modalities)

        Coef = weights['Coef']
        Coef = Coef - tf.diag(tf.diag_part(Coef))
        self.Coef = Coef
        z={}
        z_c={}
        latent_c={}
        for i in range(0, self.num_modalities):
            modality = str(i)
            z[modality] = tf.reshape(latents[modality], [batch_size, -1])
            z_c[modality] = tf.matmul(Coef,z[modality])
            latent_c[modality] = tf.reshape(z_c[modality], tf.shape(latents[modality]))


        self.z = z
        self.z_c =z_c

        self.x_r = self.decoder(latent_c, weights, self.num_modalities, shape)

        # l_2 reconstruction loss

        self.reconst_cost_x =  0.6*tf.reduce_sum(tf.pow(tf.subtract(self.x['0'], self.x_r['0']), 2.0))
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.reconst_cost_x = self.reconst_cost_x +  0.1*tf.reduce_sum(tf.pow(tf.subtract(self.x[modality],
                                                                                               self.x_r[modality]), 2.0))



        tf.summary.scalar("recons_loss", self.reconst_cost_x)
                
        self.reg_losses = tf.reduce_sum(tf.pow(self.Coef,2.0))
        
        tf.summary.scalar("reg_loss", reg_constant1 * self.reg_losses )


        self.selfexpress_losses =  0.3*tf.reduce_sum(tf.pow(tf.subtract(self.z['0'], self.z_c['0']), 2.0))
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.selfexpress_losses = self.selfexpress_losses +  0.05*tf.reduce_sum(tf.pow(tf.subtract(self.z[modality],
                                                                                               self.z_c[modality]), 2.0))



        tf.summary.scalar("selfexpress_loss", re_constant2 * self.selfexpress_losses )
        
        self.loss = self.reconst_cost_x + reg_constant1 * self.reg_losses + re_constant2 * self.selfexpress_losses
        
        self.merged_summary_op = tf.summary.merge_all()
        self.optimizer = tf.train.AdamOptimizer(learning_rate = self.learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        
        self.init = tf.global_variables_initializer()
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tfconfig)
        self.sess.run(self.init)
        self.saver = tf.train.Saver([v for v in tf.trainable_variables() if not (v.name.startswith("Coef"))])
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())

    def _initialize_weights(self):
        all_weights = dict()

        for i in range(0, self.num_modalities):
            modality = str(i)
            with tf.variable_scope(modality):
                print modality
                all_weights[modality + '_enc_w0'] = tf.get_variable(modality + "_enc_w0",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                           self.n_hidden[0]],
                                                                    initializer=layers.xavier_initializer_conv2d())

                all_weights['enc1_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality + '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality + '_enc_w1'] = tf.get_variable(modality + "_enc_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality + '_enc_w2'] = tf.get_variable(modality + "_enc_w2",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[2]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

                all_weights[modality + '_dec_w0'] = tf.get_variable(modality + "_dec1_w0",
                                                                    shape=[self.kernel_size[2], self.kernel_size[2],
                                                                           self.n_hidden[1],
                                                                           self.n_hidden[3]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b0'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality + '_dec_w1'] = tf.get_variable(modality + "_dec1_w1",
                                                                    shape=[self.kernel_size[1], self.kernel_size[1],
                                                                           self.n_hidden[0],
                                                                           self.n_hidden[1]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b1'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality + '_dec_w2'] = tf.get_variable(modality + "_dec1_w2",
                                                                    shape=[self.kernel_size[0], self.kernel_size[0], 1,
                                                                           self.n_hidden[0]],
                                                                    initializer=layers.xavier_initializer_conv2d())
                all_weights[modality + '_dec_b2'] = tf.Variable(tf.zeros([1], dtype=tf.float32))

        all_weights['enc_w3'] = tf.get_variable("enc_w3",
                                                shape=[self.kernel_size[3], self.kernel_size[3],
                                                       self.n_hidden[2],
                                                       self.n_hidden[3]],
                                                initializer=layers.xavier_initializer_conv2d())
        all_weights['enc_b3'] = tf.Variable(tf.zeros([self.n_hidden[3]], dtype=tf.float32))

        all_weights['Coef'] = tf.Variable(1.0e-4 * tf.ones([self.batch_size, self.batch_size], tf.float32), name='Coef')

        return all_weights

    # Building the encoder
    def encoder(self,X, weights,num_modalities):
        shapes = []
        latents={}
        # Encoder Hidden layer with relu activation #1
        shapes.append(X['0'].get_shape().as_list())
        for i in range(0,num_modalities):
            modality = str(i)
            layer1 = tf.nn.bias_add(tf.nn.conv2d(X[modality], weights[modality+ '_enc_w0'], strides=[1,2,2,1],padding='SAME'),weights[modality+ '_enc_b0'])
            layer1 = tf.nn.relu(layer1)
            layer2 = tf.nn.bias_add(tf.nn.conv2d(layer1, weights[modality+ '_enc_w1'], strides=[1,1,1,1],padding='SAME'),weights[modality+ '_enc_b1'])
            layer2 = tf.nn.relu(layer2)
            layer3 = tf.nn.bias_add(tf.nn.conv2d(layer2, weights[modality+ '_enc_w2'], strides=[1,2,2,1],padding='SAME'),weights[modality+ '_enc_b2'])
            layer3 = tf.nn.relu(layer3)
            latents[modality] = layer3
            print(layer3.shape)
            if (i==0):
                shapes.append(layer1.get_shape().as_list())
                shapes.append(layer2.get_shape().as_list())
                shapes.append(layer3.get_shape().as_list())



        return  latents, shapes
    # Building the decoder
    def decoder(self,z, weights,num_modalities, shapes):
        recons={}
        # Encoder Hidden layer with relu activation #1
        for i in range(0,num_modalities):
            modality = str(i)
            shape_de1 = shapes[2]
            layer1 = tf.add(tf.nn.conv2d_transpose(z[modality], weights[modality+'_dec_w0'], tf.stack([tf.shape(self.x['0'])[0],shape_de1[1],shape_de1[2],shape_de1[3]]),\
             strides=[1,2,2,1],padding='SAME'),weights[modality+'_dec_b0'])
            layer1 = tf.nn.relu(layer1)
            shape_de2 = shapes[1]
            layer2 = tf.add(tf.nn.conv2d_transpose(layer1, weights[modality+'_dec_w1'], tf.stack([tf.shape(self.x['0'])[0],shape_de2[1],shape_de2[2],shape_de2[3]]),\
             strides=[1,1,1,1],padding='SAME'),weights[modality+'_dec_b1'])
            layer2 = tf.nn.relu(layer2)
            shape_de3= shapes[0]
            layer3 = tf.add(tf.nn.conv2d_transpose(layer2, weights[modality+'_dec_w2'], tf.stack([tf.shape(self.x['0'])[0],shape_de3[1],shape_de3[2],shape_de3[3]]),\
             strides=[1,2,2,1],padding='SAME'),weights[modality+'_dec_b2'])
            layer3 = tf.nn.relu(layer3)
            recons[modality] = layer3


        return recons

    def partial_fit(self, X ,lr):
        feed_dict={}
        feed_dict[self.learning_rate]= lr
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _, Coef = self.sess.run(
                (self.reconst_cost_x, self.merged_summary_op, self.optimizer, self.Coef), feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost,Coef
    
    def initlization(self):
        self.sess.run(self.init)
    
    def reconstruct(self,X):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        return self.sess.run(self.x_r, feed_dict = feed_dict)
    
    def transform(self, X):
        feed_dict = {}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        return self.sess.run(self.z, feed_dict = feed_dict)

    def save_model(self):
        save_path = self.saver.save(self.sess,self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")
        
def best_map(L1,L2):
    #L1 should be the groundtruth labels and L2 should be the clustering labels we got
    Label1 = np.unique(L1)
    nClass1 = len(Label1)
    Label2 = np.unique(L2)

    nClass2 = len(Label2)
    nClass = np.maximum(nClass1,nClass2)
    G = np.zeros((nClass,nClass))
    for i in range(nClass1):
        ind_cla1 = L1 == Label1[i]
        ind_cla1 = ind_cla1.astype(float)
        for j in range(nClass2):
            ind_cla2 = L2 == Label2[j]
            ind_cla2 = ind_cla2.astype(float)
            G[i,j] = np.sum(ind_cla2 * ind_cla1)
    m = Munkres()
    index = m.compute(-G.T)
    index = np.array(index)
    c = index[:,1]
    newL2 = np.zeros(L2.shape)
    for i in range(nClass2):
        newL2[L2 == Label2[i]] = Label1[c[i]]
    return newL2   

def thrC(C,ro):
    if ro < 1:
        N = C.shape[1]
        Cp = np.zeros((N,N))
        S = np.abs(np.sort(-np.abs(C),axis=0))
        Ind = np.argsort(-np.abs(C),axis=0)
        for i in range(N):
            cL1 = np.sum(S[:,i]).astype(float)
            stop = False
            csum = 0
            t = 0
            while(stop == False):
                csum = csum + S[t,i]
                if csum > ro*cL1:
                    stop = True
                    Cp[Ind[0:t+1,i],i] = C[Ind[0:t+1,i],i]
                t = t + 1
    else:
        Cp = C

    return Cp

def build_aff(C):
    N = C.shape[0]
    Cabs = np.abs(C)
    ind = np.argsort(-Cabs,0)
    for i in range(N):
        Cabs[:,i]= Cabs[:,i] / (Cabs[ind[0,i],i] + 1e-6)
    Cksym = Cabs + Cabs.T;
    return Cksym

def post_proC(C, K, d=6, alpha=8):
    # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
    C = 0.5*(C + C.T)
    r = d*K + 1
    U, S, _ = svds(C,r,v0 = np.ones(C.shape[0]))
    U = U[:,::-1]    
    S = np.sqrt(S[::-1])
    S = np.diag(S)    
    U = U.dot(S)    
    U = normalize(U, norm='l2', axis = 1)       
    Z = U.dot(U.T)
    Z = Z * (Z>0)    
    L = np.abs(Z ** alpha)
    L = L/L.max()   
    L = 0.5 * (L + L.T)    
    spectral = cluster.SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',assign_labels='discretize')
    spectral.fit(L)
    grp = spectral.fit_predict(L) + 1
    return grp, L

def err_rate(gt_s, s):
    c_x = best_map(gt_s,s)
    err_x = np.sum(gt_s[:] != c_x[:])
    nmi = normalized_mutual_info_score(gt_s[:], c_x[:])
    ari = adjusted_rand_score(gt_s[:], c_x[:])
    missrate = err_x.astype(float) / (gt_s.shape[0])
    return missrate ,nmi, ari

def build_laplacian(C):
    C = 0.5 * (np.abs(C) + np.abs(C.T))
    W = np.sum(C,axis=0)         
    W = np.diag(1.0/W)
    L = W.dot(C)    
    return L

def logit(msg):
    f1 = open('./corss_validation/'+str(datetime.now())+'.txt','w+')
    f1.write(msg)
    f1.close()
    return

        
def test_face(Img, Label, CAE, num_class,num_modalities):
    
    alpha = max(0.4 - (num_class-1)/10 * 0.1, 0.1)

    for j in range(0,num_modalities):
        modality=str(j)
        Img[modality] = np.array(Img[modality])
        Img[modality] = Img[modality].astype(float)


    label = np.array(Label[:])
    label = label - label.min() + 1
    label = np.squeeze(label)


    CAE.initlization()
    CAE.restore() # restore from pre-trained model

    max_step =1000#500 + num_class*25# 100+num_class*20
    display_step = 100
    lr = 1.0e-3
    # fine-tune network
    epoch = 0
    while epoch < max_step:
        epoch = epoch + 1
        cost, Coef = CAE.partial_fit(Img, lr)#

        if epoch % display_step == 0:
            print "epoch: %.1d" % epoch, "cost: %.8f" % (cost/float(batch_size))
            Coef = thrC(Coef,alpha)
            y_x, _ = post_proC(Coef, label.max())
            missrate_x,nmi,ari = err_rate(label, y_x)
            acc = 1 - missrate_x
            print "accuracy: %.4f" % acc, "NMI: %.4f" % nmi,  "ARI: %.4f" % ari


    print("%d subjects:" % num_class)    
    print("ACC: %.4f%%" % (acc*100))
    print("NMI: %.4f%%" % (nmi*100))
    print("ARI: %.4f%%" % (ari*100))

    return acc, Coef,nmi,ari
        
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='YaleB', help='path of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=150000, help='# of epoch')
    parser.add_argument('--model', dest='model', default='model-102030-32x32-yaleb',
                        help='name of the model to be saved')

    args = parser.parse_args()
    # load face images and labels
    datapath = './Data/'+ args.mat + '.mat'
    data = sio.loadmat(datapath)

    num_modalities1 = data['num_modalities']
    print num_modalities1
    Img = {}
    X={}
    I=[]
    for i in range(0, num_modalities1):
        I = []
        modality = str(i)
        img = data['modality_'+modality]
        for i in range(img.shape[1]):
            temp = np.reshape(img[:, i], [32, 32])
            I.append(temp)
        Img [modality] = np.transpose(np.array(I),[0,2,1])
        Img[modality] = np.expand_dims(Img[modality][:], 3)
        print(Img[modality].shape)



    Label = data['Label']
    Label = np.array(Label)


    # face image clustering
    n_input = [32,32]
    kernel_size = [5,3,3,3]
    n_hidden = [10, 20, 30, 30]
    
    all_subjects = Label.max()

    
    avg = []
    med = []
    
    iter_loop = 0
    num_class = all_subjects
    batch_size = Img['0'].shape[0]
    print batch_size
    reg1 = 1.0
    reg2 = 1.0 * 10 ** (num_class / 10.0 - 3.0)

    model_path = './models_DSC/' + args.model + '.ckpt'
    restore_path = './models/' + args.model + '.ckpt'
    logs_path = './logs'
    tf.reset_default_graph()
    CAE = ConvAE(n_input=n_input, n_hidden=n_hidden, reg_constant1=reg1, re_constant2=reg2, \
                 kernel_size=kernel_size, batch_size=batch_size, model_path=model_path, restore_path=restore_path, logs_path=logs_path,num_modalities=num_modalities1)

    ACC, C,NMI,ARI = test_face(Img, Label, CAE, num_class,num_modalities1)

    result_path = './models_DSC/results_' + args.model + '.mat'
    sio.savemat(result_path, dict(C=C, ACC=ACC, NMI=NMI,ARI=ARI))

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
