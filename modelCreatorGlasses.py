import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Directory paths
train_dir = './glassData/train'
validation_dir = './glassData/validation'

# Image dimensions
Your_Image_Height, Your_Image_Width = 150, 150

# Data generators
train_datagen = ImageDataGenerator(rescale=1./255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(Your_Image_Height, Your_Image_Width),
    batch_size=20,
    class_mode='categorical')  # Use 'categorical' for binary classification

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(Your_Image_Height, Your_Image_Width),
    batch_size=20,
    class_mode='categorical')  # Use 'categorical' for binary classification

# Create a base MobileNetV2 model with pre-trained weights
base_model = MobileNetV2(input_shape=(Your_Image_Height, Your_Image_Width, 3), include_top=False, weights='imagenet')

# Add custom layers on top of the base model for binary classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
x = Dense(512, activation='relu')(x)
predictions = Dense(2, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model (optional)
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=100,  # Adjust based on the size of your dataset
    epochs=30,  # Adjust as needed
    validation_data=validation_generator,
    validation_steps=50)  # Adjust as needed

# Save the model
model.save('glasses_detector_mobilenetv2.model')
