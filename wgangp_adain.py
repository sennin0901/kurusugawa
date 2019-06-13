import tensorflow as tf
tf.enable_eager_execution()
import keras as keras
from keras.engine.topology import Layer
import keras.layers as layers
import keras.models as models

from keras.datasets import cifar10


nz = 128
height = 32
width = 32
ch = 3
batch_size = 128
epochs = 30
n_fmaps = 256
penalty_weight = 10
lr = 0.0001
beta1 = 0
beta2 = 0.9


class AdaIN(Layer):

    def __init__(self, **kwargs):
        super(AdaIN, self).__init__(**kwargs)

    def call(self, x, latents, epsilon=1e-8):
        ys = layers.Dense(x.shape[-1], kernel_initializer='random_normal')(latents)
        yb = layers.Dense(x.shape[-1], kernel_initializer='random_normal')(latents)

        # instance normalization
        x -= tf.reduce_mean(x, axis=[2, 3], keepdims=True)
        epsilon = tf.constant(epsilon)
        x *= tf.rsqrt(tf.reduce_mean(tf.square(x), axis=[2, 3], keepdims=True) + epsilon)

        return x * ys + yb

    def compute_output_shape(self, input_shape):
        return input_shape


class MappingNet(keras.Model):

    def __init__(self):
        super(MappingNet, self).__init__(name='map_net')
        self.dense1 = layers.Dense(nz, kernel_initializer='random_normal')
        self.dense2 = layers.Dense(nz, kernel_initializer='random_normal')
        self.activation = layers.LeakyReLU(0.2)

    def call(self, inputs):
        x = self.dense1(inputs)
        x = self.activation(x)
        x = self.dense2(x)
        outputs = self.activation(x)
        return outputs


class SynthesisNetwork(keras.Model):

    def __init__(self):
        super(SynthesisNetwork, self).__init__(name='synthesis_net')

        initial_value = tf.random_normal((None, 4, 4, n_fmaps), dtype=tf.float32)
        self.start_variable = tf.Variable(initial_value=initial_value)

        self.adain1 = AdaIN()
        self.conv1 = layers.Conv2D(n_fmaps, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain2 = AdaIN()

        self.conv2 = layers.Conv2D(n_fmaps / 2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain3 = AdaIN()
        self.conv3 = layers.Conv2D(n_fmaps / 2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain4 = AdaIN()

        self.conv4 = layers.Conv2D(n_fmaps / 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain5 = AdaIN()
        self.conv5 = layers.Conv2D(n_fmaps / 4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain6 = AdaIN()

        self.conv6 = layers.Conv2D(ch, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain7 = AdaIN()
        self.conv6 = layers.Conv2D(ch, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')
        self.adain8 = AdaIN()

        self.activation = layers.LeakyReLU(0.2)
        self.upsampling = layers.UpSampling2D()


    def call(self, latents):
        x = self.adain1(self.start_variable, latents)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.adain2(x, latents)

        x = self.upsampling(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.adain3(x)




def build_synthesis_net():
    latents = layers.Input(shape=(nz,))
    initial_value = tf.random_normal((None, 4, 4, n_fmaps), dtype=tf.float32)

    start_const = tf.Variable(initial_value, dtype=tf.float32)
    x = AdaIN(start_const, latents)
    x = layers.Conv2D(n_fmaps, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(n_fmaps/2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)
    x = layers.Conv2D(n_fmaps/2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(n_fmaps/4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)
    x = layers.Conv2D(n_fmaps/4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)

    x = layers.UpSampling2D()(x)
    x = layers.Conv2D(ch, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = AdaIN(x, latents)
    x = layers.Conv2D(ch, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    outputs = AdaIN(x, latents)

    model = models.Model(latents, outputs)
    return model


def build_discriminator():
    inputs = layers.Input(shape=(height, width, ch))
    x = layers.Conv2D(n_fmaps/4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(inputs)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(n_fmaps/4, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(n_fmaps/2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(n_fmaps/2, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(n_fmaps, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(n_fmaps, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.AveragePooling2D()(x)

    x = layers.Conv2D(n_fmaps, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)
    x = layers.Conv2D(1, (3, 3), strides=(1, 1), padding='same', kernel_initializer='random_normal')(x)
    x = layers.LeakyReLU(0.2)(x)

    x = layers.Flatten()(x)
    outputs = layers.Dense(1, kernel_initializer='random_normal')(x)

    model = models.Model(inputs, outputs)
    return model


if __name__ == "__main__":
    (x_train, _), (x_test, _) = cifar10.load_data()
    n_train = x_train.shape[0]
    batch_num = n_train // batch_size + 1
    net_map = build_mapping_net()
    generator = build_synthesis_net()
    discriminator = build_discriminator()
    generator_optim = generator_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)
    discriminator_optim = tf.train.AdamOptimizer(learning_rate=lr, beta1=beta1, beta2=beta2)

    for epoch in range(epochs):
        for batch in range(batch_num):
            start = batch * batch_size
            end = (batch+1) * batch_size if (end <= n_train) else n_train
            real_images = x_train[start:end]

            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape, tf.GradientTape() as mixed_tape:
                noise = tf.random_normal((real_images.shape[0], nz))
                generated_images = generator(noise, training=True)

                real_output = discriminator(real_images, training=True)
                generated_output = discriminator(generated_images, training=True)
                epsilon = tf.random_normal((real_images.shape[0], 1, 1, 1))
                mixed_images = epsilon * real_images + (1-epsilon) * generated_images
                mixed_output = discriminator(mixed_images, training=True)

                # original critic loss
                real_loss = tf.reduce_mean(real_output)
                fake_loss = tf.reduce_mean(generated_output)

                # gradient penalty
                grad_mixed = mixed_tape.gradient(mixed_output, mixed_images)
                grad_norm = tf.sqrt(tf.reduce_sum(tf.square(grad_mixed), axis=(1, 2, 3)))
                gradient_penalty = tf.reduce_mean(tf.square(grad_norm - 1))

                gen_loss = -1 * tf.reduce_mean(generated_output)
                disc_loss = fake_loss - real_loss + penalty_weight*gradient_penalty

            gradients_of_generator = gen_tape.gradient(gen_loss, generator.variables)
            gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.variables)

            generator_optim.apply_gradients(zip(gradients_of_generator, generator.variables))
            discriminator_optim.apply_gradients(zip(gradients_of_discriminator, discriminator.variables))

            if batch % 50 == 0:
                print(f'epochs[{epoch+1}/{epochs}] batchs[{batch}/{batch_num} g_loss: {gen_loss}, d_loss: {disc_loss}')
