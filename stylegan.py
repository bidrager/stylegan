import tensorflow as tf
import numpy as np
import cv2
import copy
import os


class AdaInstanceNormalization(tf.keras.layers.Layer):
    def __init__(self,epsilon=1e-4,):
        super(AdaInstanceNormalization, self).__init__()
        self.epsilon=epsilon
    def call(self, inputs,training=None, **kwargs): #[x,beta,gamma]
        x=inputs[0]
        beta = inputs[1]
        gamma = inputs[2]
        mean=tf.keras.backend.mean(x,axis=[1,2],keepdims=True)
        stddev=tf.keras.backend.std(x,axis=[1,2],keepdims=True)+self.epsilon
        normed=(x-mean)/stddev
        return normed*gamma+beta


# 本例图片大小为256*256
latent_size=256
im_size=256


# style z
def noise(batch_size):
    return np.random.normal(0.,1.,size=[batch_size,latent_size])
# noise
def noiseImage(batch_size):
    return np.random.normal(0.,1.,size=[batch_size,im_size,im_size,1])

def normalizer(arr):
    return (arr-np.mean(arr))/(np.std(arr)+1e-7)

# 读数据
class DataGenerator():
    def __init__(self):
        men='E:/pycharm_new/data/cycle_gan_data/man2woman/men/'
        women='E:/pycharm_new/data/cycle_gan_data/man2woman/women/'
        men_images=os.listdir(men)
        women_images=os.listdir(women)
        self.data=[]
        for img in men_images:
            self.data.append(men+img)
        for img in women_images:
            self.data.append(women+img)


    def get_batch(self,batch_size):
        batch_data=[]
        samples=np.random.choice(self.data,batch_size,replace=False)
        for img in samples:
            image=cv2.imread(img)
            image=image/127.5-1
            batch_data.append(image)
        batch_data=np.array(batch_data,dtype=np.float32)
        return batch_data
datagenerator=DataGenerator()

# WGAN-GP的梯度惩罚项
def gradient_penalty(discrimanator,samples,weights):
    with tf.GradientTape() as tape:
        samples=tf.constant(samples,dtype=tf.float32)
        tape.watch(samples)
        d_output=discrimanator(samples)
    gradients=tape.gradient(d_output,samples)
    gradients_sqr=tf.square(gradients)
    gradient_penalty=tf.reduce_sum(gradients_sqr,axis=np.arange(1,len(gradients_sqr.shape)))
    gradient_penalty=tf.sqrt(gradient_penalty)-1
    return  tf.reduce_mean(tf.square(gradient_penalty))*weights


def g_block(inp,style,noise,fil,u=True):
    b=tf.keras.layers.Dense(fil)(style)
    b=tf.keras.layers.Reshape([1,1,fil])(b)
    g=tf.keras.layers.Dense(fil)(style)
    g=tf.keras.layers.Reshape([1,1,fil])(g)
    n=tf.keras.layers.Conv2D(filters=fil,kernel_size=1,padding='same',kernel_initializer='he_normal')(noise)
    if u:
        out=tf.keras.layers.UpSampling2D(interpolation='bilinear')(inp)
        out=tf.keras.layers.Conv2D(filters=fil,kernel_size=3,padding='same',kernel_initializer='he_normal')(out)
    else:
        out=tf.keras.layers.Activation('linear')(inp)
    out = tf.keras.layers.Add()([out, n])
    out = tf.keras.layers.LeakyReLU(0.1)(out)
    out=AdaInstanceNormalization()([out,b,g])
    b = tf.keras.layers.Dense(fil)(style)
    b = tf.keras.layers.Reshape([1, 1, fil])(b)
    g = tf.keras.layers.Dense(fil)(style)
    g = tf.keras.layers.Reshape([1,1,fil])(g)
    n = tf.keras.layers.Conv2D(filters=fil, kernel_size=1, padding='same', kernel_initializer='he_normal')(noise)
    out=tf.keras.layers.Conv2D(filters=fil,kernel_size=3,padding='same',kernel_initializer='he_normal')(out)

    out = tf.keras.layers.Add()([out, n])
    out = tf.keras.layers.LeakyReLU(0.1)(out)

    out = AdaInstanceNormalization()([out, b, g])
    return out

def d_block(inp,fil,p=True):
    route2=tf.keras.layers.Conv2D(filters=fil,kernel_size=3,padding='same',kernel_initializer='he_normal')(inp)
    route2=tf.keras.layers.LeakyReLU(0.1)(route2)
    if p:
        route2=tf.keras.layers.AveragePooling2D()(route2)
    route2=tf.keras.layers.Conv2D(filters=fil,kernel_size=3,padding='same',kernel_initializer='he_normal')(route2)
    out=tf.keras.layers.LeakyReLU(0.1)(route2)

    return out

class Gan():
    def __init__(self,batch_size,lr=0.00001):
        self.D=None
        self.G=None
        self.batch_size=batch_size
        self.LR = lr
        self.ones = tf.ones((self.batch_size, 1), dtype=tf.float32)
        self.weight=2
        self.discriminator()
        self.generator()
        # exponential_decay = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=self.LR, decay_steps=20000, decay_rate=0.95)
        self.g_opt=tf.keras.optimizers.Adam(self.LR)
        self.d_opt = tf.keras.optimizers.Adam(self.LR)

    def discriminator(self):
        if self.D:
            return self.D
        inp=tf.keras.layers.Input(shape=[im_size,im_size,3])
        x = d_block(inp,64)
        x = d_block(x, 128)
        x = d_block(x, 128)
        x = d_block(x, 256)
        x = d_block(x, 256)
        x = d_block(x, 512)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Dense(512)(x)
        x = tf.keras.layers.Activation('relu')(x)
        x =tf.keras.layers.Dropout(0.2)(x)
        x = tf.keras.layers.Dense(1)(x)
        self.D = tf.keras.Model(inputs=inp, outputs=x)

    def generator(self):

        if self.G:
            return self.G

        # 本例前半部分的全连接用了2层，方便训练  ，论文用8层
        inp_s=tf.keras.layers.Input(shape=[latent_size])
        sty=tf.keras.layers.Dense(latent_size,kernel_initializer='he_normal')(inp_s)
        sty=tf.keras.layers.LeakyReLU(0.1)(sty)
        sty = tf.keras.layers.Dense(latent_size, kernel_initializer='he_normal')(sty)
        sty = tf.keras.layers.LeakyReLU(0.1)(sty)
        sty = tf.keras.layers.Dense(latent_size, kernel_initializer='he_normal')(sty)
        sty = tf.keras.layers.LeakyReLU(0.1)(sty)

        inp_n=tf.keras.layers.Input(shape=[im_size,im_size,1])
        noi=[tf.keras.layers.Activation('linear')(inp_n)]
        curr_size=im_size
        while curr_size!=4:
            curr_size=int(curr_size/2)
            noi.append(tf.keras.layers.Cropping2D(int(curr_size/2))(noi[-1]))

        # 本例图像为256*256   synthesis 网络也做了缩减
        inp=tf.keras.layers.Input(shape=[1])
        x=tf.keras.layers.Dense(4*4*512,kernel_initializer='he_normal')(inp)
        x=tf.keras.layers.Reshape([4,4,512])(x)
        x = g_block(x, sty, noi.pop(), 512,u=False)
        x = g_block(x, sty, noi.pop(), 256)
        x = g_block(x, sty, noi.pop(), 256)
        x = g_block(x, sty, noi.pop(), 128)
        x = g_block(x, sty, noi.pop(), 128)
        x = g_block(x, sty, noi.pop(), 64)
        x = g_block(x, sty, noi.pop(), 32)
        x=tf.keras.layers.Conv2D(filters=3,kernel_size=1,padding='same',activation='tanh')(x)
        self.G=tf.keras.Model(inputs=[inp_s,inp_n,inp],outputs=x)

    def train_step(self,n):
        with tf.GradientTape() as g_tape,tf.GradientTape() as d_tape:

            # loss 使用WANG_GP
            generated_iamges=self.G([noise(self.batch_size),noiseImage(self.batch_size),self.ones])
            df=self.D(generated_iamges)
            real_img=datagenerator.get_batch(self.batch_size)
            dr=self.D(real_img)
            loss_real = tf.reduce_mean(dr)
            loss_fake =  tf.reduce_mean(df)
            gradient_penalty_loss_f=gradient_penalty(self.D,generated_iamges,self.weight)
            gradient_penalty_loss_r=gradient_penalty(self.D,real_img,self.weight)
            gradient_penalty_loss=(gradient_penalty_loss_f+gradient_penalty_loss_r)/2

            d_loss = -(loss_real-loss_fake-gradient_penalty_loss)
            g_loss=-loss_fake
            if n%100==0:
                print(d_loss.numpy(),g_loss.numpy(),gradient_penalty_loss_r.numpy(),gradient_penalty_loss_f.numpy(),n)

        gradients_of_generator = g_tape.gradient(g_loss, self.G.trainable_variables)
        gradients_of_discriminator = d_tape.gradient(d_loss, self.D.trainable_variables)
        gradients_of_generator,_=tf.clip_by_global_norm(gradients_of_generator,2)
        gradients_of_discriminator,_ = tf.clip_by_global_norm(gradients_of_discriminator, 2)
        self.g_opt.apply_gradients(zip(gradients_of_generator, self.G.trainable_variables))
        self.d_opt.apply_gradients(zip(gradients_of_discriminator, self.D.trainable_variables))


def train(starts,steps,gan):
    for step in range(starts,steps):
        gan.train_step(step)
        if step%20000==0 and step>1:

            gan.G.save_weights('model/generator_%d.h5'%step)
            gan.D.save_weights('model/discrimator_%d.h5'%step)
            generated_iamges = gan.G([noise(gan.batch_size), noiseImage(gan.batch_size), gan.ones])

            for i in range(gan.batch_size):
                image=(generated_iamges[i]+1)*127.5
                cv2.imwrite('GIMAGE/G_step%d_%d.jpg'%(step,i),np.array(image))

def load_model(gan,step):
    gan.G.load_weights('model/generator_%d.h5'%step)
    gan.D.load_weights('model/discrimator_%d.h5' % step)
    print('load weight success')
    return gan

def predict():
    batch_size=1
    gan = Gan(batch_size)
    load_model(gan,1400000)
    a = noise(batch_size)
    for j in range(10):
        b=copy.deepcopy(a)
        b[0][128]=0.1*(j+1)-0.5
        generated_iamges = gan.G([b, noiseImage(batch_size), gan.ones]).numpy()
        image = (generated_iamges[0] + 1) * 127.5
        cv2.imwrite('result/predict%d.jpg'%j, np.array(image,dtype=np.int))

# def main():
#     batch_size=4
#     gan=Gan(batch_size)
#     load_model(gan,0)
#     train(0,1000001,gan)

if __name__ == '__main__':
    predict()























