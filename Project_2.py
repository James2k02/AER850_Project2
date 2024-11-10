from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

'''Data Processing'''

# We need to define the image size which is suppose to be 500 x 500
img_height = 500
img_width = 500
batch_size = 32 # determines how many images are processed in one batch when training

# Setting the directory paths for train and validation folder
# Notes: validation and test are different
# Validation: is used during the training process to evaluate the model's performance
#             after each training iteration (epoch)
# Test: is used to evaluate the final model's performance after training and tuning

train_directory = 'Data/train'
valid_directory = 'Data/valid'

# Creating an ImageDataGenerator for Training Data

# ImageDataGenerator is used to geenrate batches of tensor image data with real-time
# data augmentation. It helps enhance the diversity of the training dataset without
# collecting more data.

train_datagen = ImageDataGenerator(
    rescale = 1./255, # normalizes the pixel values, scaling them from [0, 255] to [0, 1]
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    rotation_range = 90,
    brightness_range = [0.1, 1.0]
    )

# many of the arguments for ImageDataGenerator are used to make the training data
# more diverse and robust by manipulating the image slightly so it learns for 
# different situations

# For example, zoom_range zooms in by a certain percentage and it simulates the effect
# of an image being taken at different distances which helps the model learns to 
# recognize features at different scales

# Creating an ImageDataGenerator for Validation Data (only apply rescaling)

# the validation data is only normalized with no augmentation so we can ensure 
# that the validation results reflect the model's performance without any distortions

valid_datagen = ImageDataGenerator(rescale = 1./255)

# Now, we can grab the images from the directories and create image data to use

train_generator = train_datagen.flow_from_directory(
    train_directory,
    target_size = (img_height, img_width), # Resizes images to 500 x 500
    batch_size = batch_size,
    color_mode = 'rgb', # Converts the greyscale images to RGB
    class_mode = 'categorical' # Used for multi-class classification (3 labels)
    )

valid_generator = valid_datagen.flow_from_directory(
    valid_directory,
    target_size = (img_height, img_width),
    batch_size = batch_size,
    color_mode = 'rgb',
    class_mode = 'categorical'
    )

# Checking if the labels to see if it split correctly

print("Class indices: ", train_generator.class_indices)

'''Neural Network Architecture Design'''

# Creating a sequential model where you can add each layer individually 
# This first model will be a simple one to allow for comparison with the 
# second model; will have no dropout layer since its already simple

model1 = Sequential()

# adding layers to the model

# The convlution layer is responsible for extracting features from the input data 
# by applying various operations. This layer helps the model identify different features
# such as simple shapes within the image. Each of the filters (or kernels) detects a specific
# type of feature like an edge or a horizontal line within the input data. For this project,
# the convolution layers will use the ReLU activation function.

# The max pooling layer reduces the spatial dimensions of the input feature maps while
# retaining the important parts. Basically just makes the model more computationally efficient 
# and robust to small translations or distortions

# can play around with the input shape at the beginning --> maybe smaller and no RGB
model1.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (500, 500, 3)))
model1.add(MaxPooling2D(pool_size = (2, 2)))

# increase in filters in order to capture and learn more complex features as we go
model1.add(Conv2D(64, (3, 3), activation = 'relu'))
model1.add(MaxPooling2D(pool_size = (2, 2)))

model1.add(Conv2D(128, (3, 3), activation = 'relu'))
model1.add(MaxPooling2D(pool_size = (2, 2)))

# flatten layer converts the 2D feature maps into a 1D vector to be passed into
# the fully connected layers
model1.add(Flatten())

# dense layer adds a fully connected layer with 128 neurons, this where it will
# begin to connect every input from previous neurons to the ones in this layer
# and begin to learn patterns and relationships
model1.add(Dense(128, activation='relu'))

# dropout layer will randomly set a certain % of input units to 0 during training
# in order to prevent overfitting 
model1.add(Dropout(0.2))

# final output layer that has 3 neurons (one for each class)
# softmax activation function used to convert output into a probability distribution
# across the classes
model1.add(Dense(3, activation='softmax'))

# compiling the model
model1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# summary of the model which shows the number of layers, output shape of each layer, 
# and total parameters in the model
model1.summary()


# Creating a second model (more complex)

model2 = Sequential()

# adding layers
model2.add(Conv2D(32, (3, 3), activation = 'relu', input_shape = (500, 500, 3)))
model2.add(MaxPooling2D(pool_size = (2, 2)))

model2.add(Conv2D(64, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))

model2.add(Conv2D(128, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))

model2.add(Conv2D(256, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))

model2.add(Flatten())

model2.add(Dense(512, activation = 'relu'))
model2.add(Dropout(0.3))

model2.add(Dense(256, activation = 'relu'))
model2.add(Dropout(0.5))

model2.add(Dense(3, activation = 'softmax'))

# model compilation
model2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# model summary
model2.summary()















