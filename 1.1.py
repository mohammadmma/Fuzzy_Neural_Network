from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt

"""this file is a enhanced version of 'DCNN.py' combined with fuzzy system and fuzzy rules, contain different dataset 
for trainin and test, BUT VALIDATION DATASET IS NOT VALID!!!!"""


# Step 1: Train the DCNN
train_data_dir = "./GTSRB_Final_Training_Images/GTSRB/Final_Training/Images"
val_data_dir = "./GTSRB_Final_Test_Images/GTSRB/Final_Test/Images"

train_datagen = ImageDataGenerator(rescale=1. / 255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1. / 255)

num_classes = 43  # Replace with the actual number of classes

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical'
)

base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False

x = base_model.output
x = Flatten()(x)
x = Dense(1024, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(train_generator, epochs=10, validation_data=val_generator)
model.save('my_image_classifier.h5')


# Step 2: Define Fuzzy Logic Rules
def create_fuzzy_system(num_classes):
    confidence = ctrl.Antecedent(np.arange(0, 1.01, 0.01), 'confidence')
    ambiguity = ctrl.Consequent(np.arange(0, 1.01, 0.01), 'ambiguity')

    confidence['low'] = fuzz.trimf(confidence.universe, [0, 0, 0.5])
    confidence['medium'] = fuzz.trimf(confidence.universe, [0, 0.5, 1])
    confidence['high'] = fuzz.trimf(confidence.universe, [0.5, 1, 1])

    ambiguity['low'] = fuzz.trimf(ambiguity.universe, [0, 0, 0.5])
    ambiguity['medium'] = fuzz.trimf(ambiguity.universe, [0, 0.5, 1])
    ambiguity['high'] = fuzz.trimf(ambiguity.universe, [0.5, 1, 1])

    rule1 = ctrl.Rule(confidence['low'], ambiguity['high'])
    rule2 = ctrl.Rule(confidence['medium'], ambiguity['medium'])
    rule3 = ctrl.Rule(confidence['high'], ambiguity['low'])

    ambiguity_ctrl = ctrl.ControlSystem([rule1, rule2, rule3])
    ambiguity_sim = ctrl.ControlSystemSimulation(ambiguity_ctrl)

    return ambiguity_sim


fuzzy_system = create_fuzzy_system(num_classes)


# Step 3: Combine DCNN Output with Fuzzy Logic
def predict_with_fuzzy_logic(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    model = load_model('my_image_classifier.h5')
    predictions = model.predict(img_array)
    class_probabilities = predictions[0]
    predicted_class = np.argmax(class_probabilities)
    confidence = class_probabilities[predicted_class]

    fuzzy_system.input['confidence'] = confidence
    fuzzy_system.compute()
    ambiguity = fuzzy_system.output['ambiguity']

    if ambiguity > 0.5:
        top_3_indices = np.argsort(class_probabilities)[-3:][::-1]
        print(
            f"Ambiguous prediction. Top 3 classes: {top_3_indices} with probabilities: {class_probabilities[top_3_indices]}")
    else:
        print(f"Predicted class: {predicted_class} with confidence: {confidence}")

    plt.imshow(img)
    plt.axis('off')
    plt.show()


# Test the combined system
predict_with_fuzzy_logic('./GTSRB_Final_Test_Images/GTSRB/Final_Test/Images/00000.jpg')
