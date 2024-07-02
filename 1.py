from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define paths to your training and validation data directories
train_data_dir = "./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
val_data_dir = "./GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"

# Set data augmentation parameters (adjust as needed)
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Define number of classes in your dataset
num_classes = 43  # Replace with the actual number of classes

# Load training and validation data generators with random resizing
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),  # Target size for resizing
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

# Load pre-trained ResNet50 model (without final layers)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))  # None for variable input size

# Freeze the pre-trained layers (optional)
base_model.trainable = False

# Add custom classification layers
x = base_model.output
x = Flatten(data_format='channels_last')(x)
x = Dense(1024, activation='relu')(x)  # Adjust number of units as needed
x = Dropout(0.5)(x)

# Final layer with softmax for multi-class classification (replace with sigmoid for multi-label)
predictions = Dense(num_classes, activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model (adjust epochs and other parameters as needed)
model.fit(train_generator, epochs=10, validation_data=val_generator)

# Save the trained model (optional)
model.save('my_image_classifier.h5')
