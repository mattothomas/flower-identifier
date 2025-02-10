import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array

# ---------------------------------------------------------------------
# Set parameters and directory paths
# ---------------------------------------------------------------------
base_dir = './flowers'  # Change this to point to the folder containing your flower images (organized in subfolders per class)
img_size = 224          # Resize images to 224x224 pixels
batch_size = 64         # Batch size for training

# ---------------------------------------------------------------------
# Create data generators for training and validation
# ---------------------------------------------------------------------
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2  # Use 20% of images for validation
)

valid_datagen = ImageDataGenerator(
    rescale=1.0/255,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='training'
)

validation_generator = valid_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    subset='validation'
)

# ---------------------------------------------------------------------
# Build the Convolutional Neural Network model
# ---------------------------------------------------------------------
model = Sequential()

# First convolutional block
model.add(Conv2D(64, (5, 5), padding='same', activation='relu', input_shape=(img_size, img_size, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Second convolutional block
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Third convolutional block
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fourth convolutional block
model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Fully connected layers for classification
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(5, activation='softmax'))  # Change 5 to the number of flower classes you have

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Print out model architecture
model.summary()

# ---------------------------------------------------------------------
# Train the model
# ---------------------------------------------------------------------
epochs = 10  # Adjust the number of epochs as needed
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=epochs
)

# Save the trained model for later use
model.save('flower_model.h5')

# ---------------------------------------------------------------------
# Load a test image and perform prediction
# ---------------------------------------------------------------------
test_img_path = 'img.jpg'  # Replace with the path to your test image
test_image = load_img(test_img_path, target_size=(img_size, img_size))
plt.imshow(test_image)
plt.title("Test Image")
plt.axis('off')
plt.show()

# Prepare the image for prediction
test_image = img_to_array(test_image)
test_image = np.expand_dims(test_image, axis=0)  # Add a batch dimension

# Predict the class of the test image
prediction = model.predict(test_image)
print("Prediction probabilities:", prediction)