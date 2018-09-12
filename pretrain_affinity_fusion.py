# Deep Multimodal Subspace Clustering Networks
# https://arxiv.org/abs/1804.06498
# Mahdi Abavisani
# mahdi.abavisani@rutgers.edu
# Built upon https://github.com/panji1990/Deep-subspace-clustering-networks
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.contrib import layers
import scipy.io as sio
import argparse




def next_batch(data_, _index_in_epoch ,batch_size ,num_modalities, _epochs_completed):
    _num_examples = data_['0'].shape[0]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
        # Finished epoch
        _epochs_completed += 1
        # Shuffle the data
        perm = np.arange(_num_examples)
        np.random.shuffle(perm)
        for i in range(0,num_modalities):
            data_[str(i)] = data_[str(i)][perm]

        start = 0
        _index_in_epoch = batch_size
        assert batch_size <= _num_examples
    end = _index_in_epoch
    data={}
    for i in range(0, num_modalities):
        data[str(i)] = data_[str(i)][start:end]
    return data, _index_in_epoch, _epochs_completed

class ConvAE(object):
    def __init__(self, n_input, kernel_size,n_hidden, num_modalities=2,learning_rate = 1e-3, batch_size = 256,\
        reg = None, denoise = False ,model_path = None,restore_path = None, logs_path = './logs'):
        #n_hidden is a arrary contains the number of neurals on every layer
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.reg = reg,
        self.model_path = model_path
        self.restore_path = restore_path
        self.kernel_size = kernel_size
        self.batch_size = batch_size
        self.iter = 0
        self.num_modalities =num_modalities
        weights = self._initialize_weights()
        self.x={}

        # model

        for i in range(0, self.num_modalities):
            modality = str(i)
            self.x[modality] = tf.placeholder(tf.float32, [None, self.n_input[0], self.n_input[1], 1])

        if denoise == False:
            x_input = self.x
            latents, shape = self.encoder(x_input,weights,self.num_modalities)

        Coef = weights['Coef']
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
        self.x_r = self.decoder(latent_c, weights, self.num_modalities, shape)

        self.saver = tf.train.Saver()

        # cost for reconstruction
        self.reconst_cost_x = 0.6* tf.reduce_sum(tf.pow(tf.subtract(self.x_r['0'], self.x['0']), 2.0))
        for i in range(1, self.num_modalities):
            modality = str(i)
            self.reconst_cost_x = self.reconst_cost_x +  0.1*tf.reduce_sum(tf.pow(tf.subtract(self.x_r[modality], self.x[modality]), 2.0))


        self.cost = self.reconst_cost_x
        # l_2 loss
        tf.summary.scalar("l2_loss", self.cost)
        
        self.merged_summary_op = tf.summary.merge_all()        
        
        self.loss = self.cost

        self.optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(self.loss) #GradientDescentOptimizer #AdamOptimizer
        tf.set_random_seed(1234)

        init = tf.global_variables_initializer()

        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        self.sess = tf.InteractiveSession(config=tfconfig)
        self.sess.run(init)
        self.summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
        t_vars = tf.trainable_variables()
        for var in t_vars:
            print(var.name)
            print(var.shape)

    def _initialize_weights(self):
        all_weights = dict()

        for i in range(0,self.num_modalities):
            modality = str(i)
            with tf.variable_scope(modality):
                print modality
                all_weights[modality+ '_enc_w0'] = tf.get_variable(modality+"_enc_w0",
                                                         shape=[self.kernel_size[0], self.kernel_size[0], 1, self.n_hidden[0]],
                                                         initializer=layers.xavier_initializer_conv2d())

                all_weights['enc1_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))


                all_weights[modality+ '_enc_b0'] = tf.Variable(tf.zeros([self.n_hidden[0]], dtype=tf.float32))

                all_weights[modality+ '_enc_w1'] = tf.get_variable(modality+"_enc_w1",
                                                         shape=[self.kernel_size[1], self.kernel_size[1], self.n_hidden[0],
                                                                self.n_hidden[1]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality+ '_enc_b1'] = tf.Variable(tf.zeros([self.n_hidden[1]], dtype=tf.float32))

                all_weights[modality+ '_enc_w2'] = tf.get_variable(modality+"_enc_w2",
                                                         shape=[self.kernel_size[2], self.kernel_size[2], self.n_hidden[1],
                                                                self.n_hidden[2]],
                                                         initializer=layers.xavier_initializer_conv2d())
                all_weights[modality+ '_enc_b2'] = tf.Variable(tf.zeros([self.n_hidden[2]], dtype=tf.float32))

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
                                                shape=[self.kernel_size[3], self.kernel_size[3],self.n_hidden[2],
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


        return latents, shapes
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

    def partial_fit(self, X):
        feed_dict={}
        for i in range(0, len(X)):
            feed_dict[self.x[str(i)]] = X[str(i)]
        cost, summary, _ = self.sess.run((self.cost, self.merged_summary_op, self.optimizer), feed_dict=feed_dict)
        self.summary_writer.add_summary(summary, self.iter)
        self.iter = self.iter + 1
        return cost

    def initlization(self):
        self.sess.run(self.init)

    def reconstruct(self, X):
        return self.sess.run((self.x_r), feed_dict={self.x: X})

    def transform(self, X):
        return self.sess.run(self.z, feed_dict={self.x: X})

    def save_model(self):
        save_path = self.saver.save(self.sess, self.model_path)
        print ("model saved in file: %s" % save_path)

    def restore(self):
        self.saver.restore(self.sess, self.restore_path)
        print ("model restored")

def ae_feature_clustering(CAE, X,args ):
    CAE.restore()

    #eng = matlab.engine.start_matlab()
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/SSC_ADMM_v1.1',nargout=0)
    #eng.addpath(r'/home/pan/workspace-eclipse/deep-subspace-clustering/EDSC_release',nargout=0)

    Z = CAE.transform(X)
    AEpath=  './Data/AE_'+ args.mat + '.mat'
    sio.savemat(AEpath, dict(Z =  np.transpose(Z,[1,0])) )

    return

def train_face(Img, CAE, n_input, batch_size,num_modalities,max_epochs):
    it = 0
    display_step = 300
    save_step = 900
    _index_in_epoch = 0
    _epochs= 0

    #CAE.restore()
    # train the network
    while it<max_epochs:
        batch, _index_in_epoch, _epochs =  next_batch(Img, _index_in_epoch , batch_size,num_modalities, _epochs)
        for i in range(0,num_modalities):
            batch[str(i)] = np.reshape(batch[str(i)],[batch_size,n_input[0],n_input[1],1])
        cost = CAE.partial_fit(batch)
        it = it +1
        avg_cost = cost/(batch_size)
        if it % display_step == 0:
            print ("epoch: %.1d" % _epochs)
            print  ("cost: %.8f" % avg_cost)
        if it % save_step == 0:
            CAE.save_model()
    return

def test_face(Img, CAE, n_input):

    batch_x_test = Img[200:300,:]
    batch_x_test= np.reshape(batch_x_test,[100,n_input[0],n_input[1],1])
    CAE.restore()
    x_re = CAE.reconstruct(batch_x_test)

    plt.figure(figsize=(8,12))
    for i in range(5):
        plt.subplot(5,2,2*i+1)
        plt.imshow(batch_x_test[i,:,:,0], vmin=0, vmax=255, cmap="gray") #
        plt.title("Test input")
        plt.colorbar()
        plt.subplot(5, 2, 2*i + 2)
        plt.imshow(x_re[i,:,:,0], vmin=0, vmax=255, cmap="gray")
        plt.title("Reconstruction")
        plt.colorbar()
        plt.tight_layout()
    plt.show()
    return

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--mat', dest='mat', default='YaleB', help='path of the dataset')
    parser.add_argument('--epoch', dest='epoch', type=int, default=150000, help='# of epoch')
    parser.add_argument('--model', dest='model', default='multimodal',
                        help='name of the model to be saved')

    args = parser.parse_args()
    datapath = './Data/'+ args.mat + '.mat'
    data = sio.loadmat(datapath)
    num_modalities1 =  data['num_modalities']
    print num_modalities1
    Img = {}
    X={}

    for i in range(0, num_modalities1):
        I = []
        modality = str(i)
        img = data['modality_'+modality]
        for i in range(img.shape[1]):
            temp = np.reshape(img[:, i], [32, 32])
            I.append(temp)
        Img [modality] = np.transpose(np.array(I),[0,2,1]) #TODO: might need adding expand_dims
        print(Img[modality].shape)

    Label = data['Label']
    Label = np.array(Label)


    n_input = [32,32]
    kernel_size = [5,3,3,3]
    n_hidden = [10, 20, 30, 30]
    batch_size = Img['0'].shape[0]
    print batch_size
    lr = 1.0e-3 # learning rate
    model_path = './models/'+ args.model + '.ckpt'
    CAE = ConvAE(n_input = n_input, n_hidden = n_hidden, learning_rate = lr, kernel_size = kernel_size,
                 batch_size = batch_size, model_path = model_path, restore_path = model_path, num_modalities=num_modalities1)
    #test_face(Img, CAE, n_input)
    train_face(Img, CAE, n_input, batch_size,num_modalities1,args.epoch)

    #for i in range(0, num_modalities):
    #    modality = str(i)
    #    X[modality] = np.reshape(Img[modality], [Img.shape[0],n_input[0],n_input[1],1])

    #ae_feature_clustering(CAE, X,args)

    
    
    
    
    
    
    
    
    
    
    
    
