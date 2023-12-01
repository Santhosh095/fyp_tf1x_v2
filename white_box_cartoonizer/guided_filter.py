import tensorflow as tf
import numpy as np

def tf_box_filter(image_tensor, radius):
    kernel_size = int(2 * radius + 1)
    channel = image_tensor.get_shape().as_list()[-1]
    weight = 1 / (kernel_size ** 2)
    box_kernel = weight *np.ones((kernel_size, kernel_size, channel, 1))
    box_kernel = np.array(box_kernel).astype(np.float32)
    output = tf.compat.v1.nn.depthwise_conv2d(image_tensor, box_kernel, [1, 1, 1, 1], 'SAME')
    return output

def guided_filter(input_x, guidance_y, radius, epsilon=1e-2):
    x_shape = tf.compat.v1.shape(input_x)
    N = tf_box_filter(tf.compat.v1.ones((1, x_shape[1], x_shape[2], 1), dtype=input_x.dtype), radius)

    mean_x = tf_box_filter(input_x, radius) / N
    mean_y = tf_box_filter(guidance_y, radius) / N
    covariance_xy = tf_box_filter(input_x * guidance_y, radius) / N - mean_x * mean_y
    variance_x  = tf_box_filter(input_x ** 2, radius) / N - mean_x * mean_x

    A = covariance_xy / (variance_x + epsilon)
    b = mean_y - A * mean_x

    mean_A = tf_box_filter(A, radius) / N
    mean_b = tf_box_filter(b, radius) / N

    output = tf.compat.v1.add(mean_A * input_x, mean_b, name='final_add')
    return output

def fast_guided_filter(lr_x, lr_y, hr_x, r=1, epsilon=1e-8):      
    lr_x_shape = tf.compat.v1.shape(lr_x)
    hr_x_shape = tf.compat.v1.shape(hr_x) 
    N = tf_box_filter(tf.compat.v1.ones((1, lr_x_shape[1], lr_x_shape[2], 1), dtype=lr_x.dtype), r)

    mean_x = tf_box_filter(lr_x, r) / N
    mean_y = tf_box_filter(lr_y, r) / N
    covariance_xy = tf_box_filter(lr_x * lr_y, r) / N - mean_x * mean_y
    variance_x  = tf_box_filter(lr_x * lr_x, r) / N - mean_x * mean_x

    A = covariance_xy / (variance_x + epsilon)
    b = mean_y - A * mean_x

    mean_A = tf.compat.v1.image.resize_images(A, hr_x_shape[1: 3])
    mean_b = tf.compat.v1.image.resize_images(b, hr_x_shape[1: 3])

    output = mean_A * hr_x + mean_b
    return output

if __name__ == '__main__':
    pass