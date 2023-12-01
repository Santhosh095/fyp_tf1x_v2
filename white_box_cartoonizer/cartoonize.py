import os
import cv2
import numpy as np
import tensorflow as tf

import network
import guided_filter

class Catoonify:
    def __init__(self, weights_directory, use_gpu):
        if not os.path.exists(weights_directory):
            raise FileNotFoundError("Weights Directory not found, check path")
        self.load_model(weights_directory, use_gpu)
        print("Weights successfully loaded")
    
    def resize_crop(self, image):
        height, width, channels = np.shape(image)
        if min(height, width) > 720:
            if height > width:
                height, width = int(720 * height / width), 720
            else:
                height, width = 720, int(720 * width / height)
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        height, width = (height // 8) * 8, (width // 8) * 8
        image = image[:height, :width, :]
        return image

    def load_model(self, weights_directory, use_gpu):
        try:
            tf.compat.v1.disable_eager_execution()
        except:
            None

        tf.compat.v1.reset_default_graph()

        self.input_photo = tf.compat.v1.placeholder(tf.compat.v1.float32, [1, None, None, 3], name='input_image')
        network_out = network.unet_generator(self.input_photo)
        self.final_out = guided_filter.guided_filter(self.input_photo, network_out, 1, 5e-3)

        all_variables = tf.compat.v1.trainable_variables()
        generator_variables = [var for var in all_variables if 'generator' in var.name]
        saver = tf.compat.v1.train.Saver(var_list=generator_variables)
        
        if use_gpu:
            gpu_options = tf.compat.v1.GPUOptions(allow_growth=True)
            device_count = {'GPU':1}
        else:
            gpu_options = None
            device_count = {'GPU':0}
        
        config = tf.compat.v1.ConfigProto(gpu_options=gpu_options, device_count=device_count)
        
        self.sess = tf.compat.v1.Session(config=config)

        self.sess.run(tf.compat.v1.global_variables_initializer())
        saver.restore(self.sess, tf.compat.v1.train.latest_checkpoint(weights_directory))

    def infer(self, image):
        image = self.resize_crop(image)
        batch_image = image.astype(np.float32)/127.5 - 1
        batch_image = np.expand_dims(batch_image, axis=0)
        
        output = self.sess.run(self.final_out, feed_dict={self.input_photo: batch_image})
        
        output = (np.squeeze(output)+1)*127.5
        output = np.clip(output, 0, 255).astype(np.uint8)
        return output
    
if __name__ == '__main__':
    pass