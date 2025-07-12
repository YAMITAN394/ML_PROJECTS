# Step 1: Import required libraries
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.applications import vgg19
from google.colab import files

# Step 2: Upload your content and style images
uploaded = files.upload()  # Upload your content and style image
from PIL import Image

# Step 3: Load and preprocess images
def load_img(path_to_img, max_dim=512):
    img = tf.io.read_file(path_to_img)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (512, 512))

    shape = tf.cast(tf.shape(img)[:-1], tf.float32)
    long_dim = max(shape)
    scale = max_dim / long_dim

    new_shape = tf.cast(shape * scale, tf.int32)
    img = tf.image.resize(img, new_shape)
    img = img[tf.newaxis, :]
    return img

def imshow(image, title=None):
    if len(image.shape) > 3:
        image = tf.squeeze(image, axis=0)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

# Replace with your uploaded filenames
content_path = 'content (5).jpeg'
style_path = 'style ().jpeg'

content_image = load_img(content_path)
style_image = load_img(style_path)

imshow(content_image, title='Content Image')
imshow(style_image, title='Style Image')

# Step 4: Preprocessing
def preprocess_img(img):
    img = preprocess_input(img * 255.0)
    return img

preprocessed_content = preprocess_img(content_image)
preprocessed_style = preprocess_img(style_image)

# Step 5: Define feature extraction model
content_layers = ['block5_conv2']
style_layers = [
    'block1_conv1',
    'block2_conv1',
    'block3_conv1',
    'block4_conv1',
    'block5_conv1'
]
num_style_layers = len(style_layers)

def vgg_layers(layer_names):
    vgg = vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = Model([vgg.input], outputs)
    return model

extractor = vgg_layers(style_layers + content_layers)

# Step 6: Feature extraction
style_targets = extractor(preprocessed_style)[:num_style_layers]
content_targets = extractor(preprocessed_content)[num_style_layers:]

# Step 7: Define loss functions
def gram_matrix(tensor):
    result = tf.linalg.einsum('bijc,bijd->bcd', tensor, tensor)
    input_shape = tf.shape(tensor)
    num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
    return result / num_locations

def get_content_loss(base_content, target_content):
    return tf.reduce_mean(tf.square(base_content - target_content))

def get_style_loss(base_style, target_style):
    gram_base = gram_matrix(base_style)
    gram_target = gram_matrix(target_style)
    return tf.reduce_mean(tf.square(gram_base - gram_target))

# Step 8: Define training step
content_weight = 1e4
style_weight = 1e-2

generated_image = tf.Variable(content_image, dtype=tf.float32)
optimizer = tf.optimizers.Adam(learning_rate=0.02)

@tf.function()
def train_step(generated_image, content_targets, style_targets, extractor, optimizer):
    with tf.GradientTape() as tape:
        outputs = extractor(generated_image)
        style_outputs = outputs[:num_style_layers]
        content_outputs = outputs[num_style_layers:]

        style_loss = tf.add_n([
            get_style_loss(style_outputs[i], style_targets[i])
            for i in range(num_style_layers)
        ])
        content_loss = get_content_loss(content_outputs[0], content_targets[0])
        total_loss = content_weight * content_loss + style_weight * style_loss

    grad = tape.gradient(total_loss, generated_image)
    optimizer.apply_gradients([(grad, generated_image)])
    generated_image.assign(tf.clip_by_value(generated_image, 0.0, 1.0))

# Step 9: Run the style transfer training loop
epochs = 10
steps_per_epoch = 50

for n in range(epochs):
    for m in range(steps_per_epoch):
        train_step(generated_image, content_targets, style_targets, extractor, optimizer)
        print(f"Epoch {n+1}, step {m+1}/{steps_per_epoch}")
    print(f"Epoch {n+1} completed")

# Step 10: Deprocess and display final image
def deprocess_img(processed_img):
    x = processed_img.numpy()
    x = x.squeeze()
    x = x.clip(0, 1)
    return x

plt.imshow(deprocess_img(generated_image))
plt.title("Styled Image")
plt.axis('off')
plt.show()



final_img = deprocess_img(generated_image)
img = Image.fromarray(np.uint8(final_img * 255))
img.save("stylized_output.jpg")

from google.colab import files
files.download("stylized_output.jpg")

