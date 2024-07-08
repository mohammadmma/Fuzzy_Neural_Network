from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from os import listdir
from os.path import isfile, join
import cv2  # Assuming you're using OpenCV for image loading

"""in this file i try to define load_images_and_labels function but it DOESN'T WORK TOO"""


def load_images_and_labels(data_dir):
    """
    Loads images and labels from a directory structure with subfolders for classes.

    Args:
        data_dir: Path to the directory containing subfolders for each class.

    Returns:
        X: A list of tuples containing (image_data, image_path).
        y: A list of corresponding labels (integers representing class indices).
    """
    X = []
    y = []

    # Loop through subdirectories (assuming they represent classes)
    for class_dir in listdir(data_dir):
        if isfile(join(data_dir, class_dir)):
            continue  # Skip any files within the data_dir (not subdirectories)

        # Get the class label (assuming folder name represents the class)
        label = class_dir

        # Loop through images in the class subdirectory
        for filename in listdir(join(data_dir, class_dir)):
            if not isfile(join(data_dir, class_dir, filename)):
                continue  # Skip any non-files within the class subdirectory

            # Load the image using OpenCV
            img = cv2.imread(join(data_dir, class_dir, filename))

            # Check if image loading failed (img is None)
            if img is None:
                print(f"Error loading image: {join(data_dir, class_dir, filename)}")
                continue  # Skip to the next image

            # Convert the image to a NumPy array and extract the path
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB if needed
            img = img.astype('float32') / 255.0  # Normalize to range [0, 1]
            # print(f"this is img = {img}")
            image_path = join(data_dir, class_dir, filename)
            # print(f"this is image path = {image_path}")
            # print(f"the label of this photo is {label}")

            # Store image data and path together (can be a tuple or dictionary)
            X.append((img, image_path))
            y.append(label)  # Append the class label
            # print(f"this is X= {X}")

    return X, y


# Load images and labels from your dataset directory
X_all, y_all = load_images_and_labels("./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images")

# Unpack data (assuming X and y contain tuples with image data and paths)
# X_with_paths, y = zip(*X_all)  # Unpack tuples into separate lists

# Splitting data with directory paths
X_train_with_paths, X_val_with_paths, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42
)

# print(f"x train with path = {X_train_with_paths}")
# print(f"x val with path = {X_val_with_paths}")
# print(f"y train = {y_train}")
# print(f"y val = {y_val}")


# Define paths to your training and validation data directories
train_data_dir = "./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
val_data_dir = "./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"

# X_train, X_val, y_train, y_val = train_test_split(X_all, y_all, test_size=0.2, random_state=42)
# print("Xtrain", X_train, X_val, end='\n')

# Set data augmentation parameters (adjust as needed)
train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

# Define number of classes in your dataset
num_classes = 43  # Replace with the actual number of classes

# Extract directory paths from training and validation data
# print(X_train_with_paths)
# X_train, _ = zip(*X_train_with_paths)  # Unpack only the first element (image data)
# X_val, _ = zip(*X_val_with_paths)

# train_data = []
# train_data_paths = []
# for img, path in X_train_with_paths:
#     train_data.append(img)
#     train_data_paths.append(path)
#
# val_data = []
# val_data_paths = []
# for img, path in X_val_with_paths:
#     val_data.append(img)
#     val_data_paths.append(path)
X_train, X_train_paths = zip(*X_train_with_paths)
print(f"X train = {X_train[0]}")
X_val, X_val_paths = zip(*X_val_with_paths)
# Load training and validation data generators with random resizing
train_generator = train_datagen.flow_from_directory(
    directory=X_train[0],
    target_size=(224, 224),  # Target size for resizing
    batch_size=32,
    class_mode='categorical',
    classes=list(set(y_train))
)

val_generator = val_datagen.flow_from_directory(
    directory=X_val[0],
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    classes=list(set(y_val))
)

# Load pre-trained ResNet50 model (without final layers)
base_model = ResNet50(weights='imagenet', include_top=False,
                      input_shape=(224, 224, 3))  # None for variable input size

# Print the actual output shape for verification (optional)
print(base_model.output.shape)

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
