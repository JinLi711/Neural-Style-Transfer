from keras.preprocessing.image import load_img, img_to_array
import numpy as np
from keras.applications import vgg19
from keras import backend as K


# -----------------------------------------------------------------------
# Get prelimary attributes.
# -----------------------------------------------------------------------


def get_img_paths(target, style):
    target = 'images/target/' + target
    style = 'images/style/' + style
    return target, style


def get_width_height(target_image_path): 
    """
    Get dimensions of the generated picture.
    
    :param target_image_path: Path of the target (reference) image
    :type  target_image_path: str
    :returns: height, width
    :rtype:   (int, int)
    """
    
    width, height = load_img(target_image_path).size
    img_height = 400
    img_width = int(width * img_height / height)
    return img_height, img_width


# -----------------------------------------------------------------------
# Convert from images to tensors and visa versa.
# -----------------------------------------------------------------------


def preprocess_image(image_path, img_height, img_width):
    """
    Open, resize, and format pictures into tensors.
    
    :param image_path: path of the image
    :type  image_path: str
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    """
    
    img = load_img(image_path, target_size=(img_height, img_width))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = vgg19.preprocess_input(img)
    return img


def deprocess_image(x):
    ''' 
    This reverses a transformation done by vgg19.preprocess_input.
    Basically converts tensors back to images.
    '''
    
    # Remove mean pixel to zero center
    x[:, :, 0] += 103.939
    x[:, :, 1] += 116.779
    x[:, :, 2] += 123.68
    
    # Convert images from BGR to RGB
    x = x[:, :, ::-1]
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# -----------------------------------------------------------------------
# Create the model from VGG19 to train on.
# -----------------------------------------------------------------------


def create_model(target_path, style_path, img_height, img_width):
    """
    Creates the model to train on.

    :param target_path: The path of the target image. 
    :type  target_path: str
    :param style_path: The path of the style image
    :type  style_path: str
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :returns: (Pretrained VGG19 model, Tensorflow placeholder for generated image)
    :rtype:   (keras.engine.training.Model, tensorflow.python.framework.ops.Tensor)
    """

    # note that these two images do not change
    target_image = K.constant(
        preprocess_image(target_path, img_height, img_width)
    )
    style_reference_image = K.constant(
        preprocess_image(style_path, img_height, img_width)
    )

    # placeholder for generated image
    combination_image = K.placeholder(
        (1, img_height, img_width, 3)
    )

    # combine into single branch
    input_tensor = K.concatenate(
        [target_image,
         style_reference_image,
         combination_image],
        axis=0
    )

    # build VGG19 network with the three images as input
    model = vgg19.VGG19(
        input_tensor=input_tensor,
        weights='imagenet',
        include_top=False
    )
    return model, combination_image


# -----------------------------------------------------------------------
# Compute the loss functions
# -----------------------------------------------------------------------


def content_loss(base, combination):
    """
    Compute the content loss.

    :param base: The tensor representing a layer of the base 
    :type  base: tensorflow.python.framework.ops.Tensor
    :param combination: The tensor representing layer of the combination of the target and the style
    :type  combination: tensorflow.python.framework.ops.Tensor
    :returns: Scaler of the content loss
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    return K.sum(K.square(combination - base))


def gram_matrix(x):
    """
    Computes gram matrix of an input matrix.

    :param x: The tensor representing the layer
    :type  x: tensorflow.python.framework.ops.Tensor
    :returns: the inner product of the feature maps of a layer
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    features = K.batch_flatten(K.permute_dimensions(x, (2, 0, 1)))
    gram = K.dot(features, K.transpose(features))
    return gram


def style_loss(style, combination, img_height, img_width):
    """
    Compute the style loss for one layer

    :param style: The tensor representing gram matrix of the style
    :type  style: tensorflow.python.framework.ops.Tensor
    :param combination: The tensor representing gram matrix of the combination
    :type  combination: tensorflow.python.framework.ops.Tensor
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :returns: Scaler of the style loss
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    S = gram_matrix(style)
    C = gram_matrix(combination)
    channels = 3
    size = img_height * img_width
    return K.sum(K.square(S - C)) / (4. * (channels ** 2) * (size ** 2))


def total_variation_loss(x, img_height, img_width):
    """
    This operates on the pixels of the generated image.
    Sort of like a regularization loss to make sure the image isn't overly pixelated.

    :param x: The tensor representing the generated image
    :type  x: tensorflow.python.framework.ops.Tensor
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :returns: Scaler of the style loss
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    a = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, 1:, :img_width - 1, :])
    b = K.square(
        x[:, :img_height - 1, :img_width - 1, :] -
        x[:, :img_height - 1, 1:, :])
    return K.sum(K.pow(a + b, 1.25))


def find_loss(
        model,
        img_height,
        img_width,
        combination_image,
        total_variation_weight=1e-4,
        style_weight=1.,
        content_weight=0.025):
    """
    Calculate total loss by a weighted average of the three above.

    :param model: Pretrained VGG19 model
    :type  model: keras.engine.training.Model
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :param combination_image: Tensorflow placeholder for generated image
    :type  combination_image: tensorflow.python.framework.ops.Tensor
    :param total_variation_weight: Weight of the total variation loss
    :type  total_variation_weight: float
    :param style_weight: Weight of the style loss
    :type  style_weight: float
    :param content_weight: Weight of the content loss
    :type  content_weight: float
    :returns: placeholder for the total loss
    :rtype:   tensorflow.python.framework.ops.Tensor
    """

    # maps layer names to activation tensors
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])
    content_layer = 'block5_conv2'
    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # add content loss
    loss = K.variable(0.)
    layer_features = outputs_dict[content_layer]
    target_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss += content_weight * content_loss(
        target_image_features,
        combination_features
    )

    # adds style loss component for each layer
    for layer_name in style_layers:
        layer_features = outputs_dict[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(
            style_reference_features,
            combination_features,
            img_height,
            img_width
        )
        loss += (style_weight / len(style_layers)) * sl

    # add variational loss
    loss += total_variation_weight * total_variation_loss(
        combination_image,
        img_height,
        img_width
    )

    return loss


# -----------------------------------------------------------------------
# Gradient descent to mimimize loss.
# -----------------------------------------------------------------------


class Evaluator(object):
    """
    This is here because scipy.optimize requires seperate functions for loss and gradients,
    which would be inefficient to compute.
    This Evaluator allows use to compute loss and gradients in one pass.

    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :param fetch_loss_and_grads: Function for getting the current loss and gradient
    :type  fetch_loss_and_grads: keras.backend.tensorflow_backend.Function
    """

    def __init__(self, img_height, img_width, fetch_loss_and_grads):
        self.loss_value = None
        self.grads_values = None
        self.img_height = img_height
        self.img_width = img_width
        self.fetch_loss_and_grads = fetch_loss_and_grads

    def loss(self, x):
        assert self.loss_value is None
        x = x.reshape((1, self.img_height, self.img_width, 3))
        outs = self.fetch_loss_and_grads([x])
        loss_value = outs[0]
        grad_values = outs[1].flatten().astype('float64')
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values


def train_and_generate_img(
        target_file,
        style_file,
        img_height,
        img_width,
        evaluator,
        iterations=20):
    """
    :param target_file: Name of target image
    :type  target_file: str
    :param style_file: Name of style image
    :type  style_file: str
    :param img_height: height of image
    :type  img_height: int
    :param img_width: width of image
    :type  img_width: int
    :param evaluator: class for producing loss and gradients
    :type  evaluator: class
    :param iterations: Number of training iterations
    :type  iterations: int
    """

    from scipy.optimize import fmin_l_bfgs_b
    from scipy.misc import imsave
    import time

    result_prefix = 'images/generated/' + \
        target_file.split('.')[0] + '_' + \
        style_file.split('.')[0]

    target_image_path, style_reference_image_path = get_img_paths(
        target_file, style_file)
    x = preprocess_image(target_image_path, img_height, img_width)
    # scipy.optimize.fmin_l_bfgs_b can only process flat vectors
    x = x.flatten()

    for i in range(iterations):
        print('Start of iteration', i)
        start_time = time.time()

        # Runs L-BFGS optimization over the pixels of the generated image
        # to minimize the neural style loss.
        x, min_val, info = fmin_l_bfgs_b(
            evaluator.loss,
            x,
            fprime=evaluator.grads,
            maxfun=20
        )
        print('Current loss value:', min_val)
        img = x.copy().reshape((img_height, img_width, 3))
        img = deprocess_image(img)
        fname = result_prefix + '_at_iteration_%d.png' % i
        imsave(fname, img)
        print('Image saved as', fname)
        end_time = time.time()
        print('Iteration %d completed in %ds' % (i, end_time - start_time))


# -----------------------------------------------------------------------
# Pipeline to combine the process above and generate image
# -----------------------------------------------------------------------


def generate_image(
        target_file,
        style_file,
        total_variation_weight=1e-4,
        style_weight=1.,
        content_weight=0.025,
        iterations=20):
    """
    Produces a style transfered image given a base image and a style image.

    :param target_file: Name of target image
    :type  target_file: str
    :param style_file: Name of style image
    :type  style_file: str
    :param total_variation_weight: Weight of the total variation loss
    :type  total_variation_weight: float
    :param style_weight: Weight of the style loss
    :type  style_weight: float
    :param content_weight: Weight of the content loss
    :type  content_weight: float
    :param iterations: Number of training iterations
    :type  iterations: int
    """

    target_image_path, style_reference_image_path = get_img_paths(
        target_file, style_file)
    img_height, img_width = get_width_height(target_image_path)
    model, combination_image = create_model(
        target_image_path,
        style_reference_image_path,
        img_height, img_width
    )
    model.summary()
    loss = find_loss(model, img_height, img_width, combination_image)

    # get gradients of generated image with regard to the loss
    grads = K.gradients(loss, combination_image)[0]
    # gets the value of the current loss and current gradients
    fetch_loss_and_grads = K.function([combination_image], [loss, grads])

    evaluator = Evaluator(img_height, img_width, fetch_loss_and_grads)
    train_and_generate_img(
        target_file,
        style_file,
        img_height,
        img_width,
        evaluator,
        iterations=iterations
    )