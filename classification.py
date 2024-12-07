import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import matplotlib.pyplot as plt

# Define the dataset path
dataset_path = "/content/drive/MyDrive/SGan/train"  # Update with your actual dataset path

# Load the training dataset
train_dataset_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,  # Use 20% of data for validation
    subset="training",
    seed=123,
    image_size=(224, 224),  # Resize images to 224x224
    batch_size=32
)

# Load the validation dataset
validation_dataset_raw = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224, 224),
    batch_size=32
)

# Extract class names
class_names = train_dataset_raw.class_names
print("Class Names:", class_names)

# Normalize pixel values
normalization_layer = tf.keras.layers.Rescaling(1./255)

# Apply normalization to training and validation datasets
train_dataset = train_dataset_raw.map(lambda x, y: (normalization_layer(x), y))
validation_dataset = validation_dataset_raw.map(lambda x, y: (normalization_layer(x), y))

# Optimize dataset performance
train_dataset = train_dataset.cache().shuffle(1000).prefetch(buffer_size=tf.data.AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=tf.data.AUTOTUNE)

# Load MobileNetV2 base model
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)

# Freeze the base model layers
base_model.trainable = False

# Build the model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(class_names), activation='softmax')  # Adjust for your number of classes
])

# Compile the model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# Set up ModelCheckpoint to save the model only at every 10 epochs
checkpoint_callback = ModelCheckpoint(
    '/content/drive/MyDrive/SGan/ntepoch{epoch:02d}.h5',
    save_weights_only=False,  # Save the full model
    save_freq=10 * len(train_dataset),  # Save every 10 epochs (adjust based on batch size and dataset)
    verbose=1
)

# Create a custom callback to predict after each epoch
class PredictAfterEpoch(Callback):
    def __init__(self, test_image_path, class_names):
        super().__init__()
        self.test_image_path = test_image_path
        self.class_names = class_names

    def on_epoch_end(self, epoch, logs=None):
        # Load and preprocess test image
        img = load_img(self.test_image_path, target_size=(224, 224))  # Resize image to match input size
        img_array = img_to_array(img) / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict the class
        predictions = self.model.predict(img_array)
        predicted_class_idx = np.argmax(predictions[0])

        # Map predicted index to class name
        predicted_class_name = self.class_names[predicted_class_idx]
        print(f"Epoch {epoch+1}: Predicted Class: {predicted_class_name}")

# Path to a test image for prediction (adjust path as necessary)
test_image_path = "/content/drive/MyDrive/SGan/train/bangle/12.png"  # Update with your test image path

# Initialize the callback
predict_callback = PredictAfterEpoch(test_image_path, class_names)

# Train the model with ModelCheckpoint and custom prediction callback
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=50,  # Number of epochs
    callbacks=[checkpoint_callback, predict_callback]  # Include both callbacks
)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# Load a test image for prediction
image_path = "/content/drive/MyDrive/SGan/train/bangle/12.png"  # Update with your test image path
img = load_img(image_path, target_size=(224, 224))  # Resize image to match input size
img_array = img_to_array(img) / 255.0  # Normalize the image
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

# Predict the class
predictions = model.predict(img_array)
predicted_class_idx = np.argmax(predictions[0])

# Map predicted index to class name
predicted_class_name = class_names[predicted_class_idx]
print("Predicted Class:", predicted_class_name)
