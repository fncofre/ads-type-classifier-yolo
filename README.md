# ML Model - Classification of images as text or display

# Table of contents
- [Data](#Data)
- [Section 2](#id-section2)

## Problem
On the ads-capture project, we have a lot of different case scenarios where the ads have different types. Some examples are html ads, images or text (and others like videos, ...).

For this reason, the classification of the ad_type, it's not an easy task, sometimes `XPaths` rules are not enough to detect the resources inside an ad, maybe because they are in another nested iframe, also can be that the rules that actually exists doesn't match the criteria for that ad, or just the `XPaths` are not enough or updated to cover all cases.

This is why, that this model is necessary, a tool that can classify in a more `human way` (using visualization) the ad_type, using the screenshot of the ad, to be analyzed and then predict the type of ad in this case just a boolean type model. (display or text)

## Solution
Implement a model that using the screenshot of the ad as input, can make the output of the predictions, if it's display or text.

## Implementation

First, as steps to follow the order used to do the model was:
- Think in which model will be used.
- See how the dataset required to be tabulated.
- Train/validate the models.
- Compare results.
- Export the model.
- Pre-process the inputs.
- Make predictions w/ the exported models.

### Data

This is the most fundamental part of the model, needs to have a nice amount of data for each category of classification (in this case display and text), and make sure to cover many edge cases w/ your data.
To be able, to work easily on the models the structure of the dataset is as follow:
```
├───model text ads
    └───dataset
        ├───train
        │   ├───display
        │   └───text
        └───val
            ├───display
            └───text
```

<details>
<summary> Click to see the code used to make my dataset w/ train and val folders automatically defining the percentage of the total data to be used for validation</summary>

```python
import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

# Define paths
base_dir = './ignition/outputs/'
dataset_dir = './dataset/'
train_dir = os.path.join(dataset_dir, 'train')
val_dir = os.path.join(dataset_dir, 'val')

# Create directories
os.makedirs(os.path.join(train_dir, 'text'), exist_ok=True)
os.makedirs(os.path.join(train_dir, 'display'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'text'), exist_ok=True)
os.makedirs(os.path.join(val_dir, 'display'), exist_ok=True)

# Define class folders
class_folders = {
    'text': os.path.join(base_dir, 'output_texts_google_ads'),
    'display': os.path.join(base_dir, 'output_displays_google_ads')
}

# Move files to train and val directories
for class_name, class_folder in class_folders.items():
    files = [f for f in os.listdir(class_folder) if os.path.isfile(os.path.join(class_folder, f))]
    files = np.array(files)

    # Split files into training and validation sets
    train_files, val_files = train_test_split(files, test_size=0.15, random_state=42)

    # Move training files
    for file in train_files:
        shutil.move(os.path.join(class_folder, file), os.path.join(train_dir, class_name, file))

    # Move validation files
    for file in val_files:
        shutil.move(os.path.join(class_folder, file), os.path.join(val_dir, class_name, file))

print("Data preparation complete.")
```
</details>

With the previously done, we should have our folders w/ the data ready to be used to train validate the model.

### Tabulation of the data

In my personal case, w/ the data that I use, I had to clean the dataset; this because I have many miss cases, like some displays on the text folders. (This caused by my source used to get the screenshots of the ads, that was not 100% correct)

For this reason, I have to manually clean the data, and define a criteria to say which ads are `display` and which ones are `text`. In summarize, the criteria is that the screenshot has just text and a very small logo w/ one/few colors then is text; on the other hand all the others are display.

- Text example

- Text example w/ small Logo

- Display example w/ small image

- Display example

### Train a model from 0

The first idea of model, was a deep learning one, using Convolutional Neural Network (CNN).

The information for training the model was sourced from [TensorFlow's CNN official docs](https://www.tensorflow.org/tutorials/images/cnn).

<details>
<summary> Click to see the code used to make the first CNN model</summary>

```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report


combined_dir = './ignition/outputs/'

img_width, img_height = 150, 150

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)



train_generator = datagen.flow_from_directory(
    directory=combined_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    classes=['output_texts_google_ads', 'output_displays_google_ads'],
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=combined_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    classes=['output_texts_google_ads', 'output_displays_google_ads'],
    subset='validation'
)

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    validation_data=validation_generator,
    epochs=20
)

validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred = (y_pred > 0.5).astype(int)

print(classification_report(validation_generator.classes, y_pred, target_names=['text', 'html']))

model.save('ad_classification_model.h5')
```
</details>

The results of the first model, were pretty bad reaching a `0.49` of accuracy looking at the classification report generated.

<details>
<summary> Click to see the code used to make the second CNN model</summary>

```python
import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

combined_dir = './ignition/outputs/'

img_width, img_height = 150, 150

datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    validation_split=0.2
)

train_generator = datagen.flow_from_directory(
    directory=combined_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    classes=['output_texts_google_ads', 'output_displays_google_ads'],
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    directory=combined_dir,
    target_size=(img_width, img_height),
    batch_size=32,
    class_mode='binary',
    classes=['output_texts_google_ads', 'output_displays_google_ads'],
    subset='validation'
)

# Define a simplified CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    MaxPooling2D(pool_size=(2, 2)),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),

    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(1, activation='sigmoid')
])

model.compile(optimizer=Adam(learning_rate=0.0005), loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=validation_generator.samples // validation_generator.batch_size,
    validation_data=validation_generator,
    epochs=30,
    verbose=1
)

validation_generator.reset()
y_pred = model.predict(validation_generator, verbose=1)
y_pred = (y_pred > 0.5).astype(int)

print(classification_report(validation_generator.classes, y_pred, target_names=['text', 'html']))

model.save('ad_classification_model_reduced.h5')
```
</details>

The second reach `0.52` of accuracy, being very bad as the previous one.

So after these results, I discard the idea of train a model from zero, first by the results, and second that the main reason for this results it's that my quantity of data may not be enough to train a model from 0, to classify what I want. For this reason, then I jump to the following idea...

### Re-train the YOLOv8 Model

YOLOv8 is a model w/ a lot of work for object detection, so visual recognition for this case. In short words, this model is used as a pre-trained model that will have better results that the previous ones, and we can train to detect what we want w/ our dataset.

So first, we need to get the library to work w/ YOLOv8
```
pip install ultralytics
```
Then download the `pt` (PyTorch) of the YOLO model to use. I used the yolov8n-cls.pt getted from the official website. [Download official yolov8n-cls.pt](https://github.com/ultralytics/assets/releases/download/v8.2.0/yolov8n-cls.pt)

So we are ready to go, now we need to train the YOLO model as the following script. (very simple)

```python
from ultralytics import YOLO

model = YOLO("yolov8n-cls.pt")

path = "./ignition/dataset/"
results = model.train(data=path, epochs=10, imgsz=64)
```

Here the main part is the train parameters, where the data means the dataset path, the epochs meaning the number of trains time that the model will do on all the data, adjusting the parameters on each epoch; and finally the imgsz meaning the size that the images will have while training the model. (64x64 is a common resolution, very fast to work with, and keep a visually useful sample.) This parameters van be modified to look for better results.

Running the previous script, will generate a `runs` folder on our `root`, that will have all our different trains, having a lot of useful assets to see how was the training, and the results from each epoch; and also the `best.pt` that it's the model that we want to use to do the predictions. (we have a lot more of data, but it's the more straightforward one)

- Train batch_0 image:

- Val batch_0 image:

- Results graphs:

### Make predictions

Now w/ the `best.pt` w/ can make predictions, to see how the model is working, to do this I made the following script:

```python
from ultralytics import YOLO

model = YOLO("./runs/classify/train2/weights/best.pt")

img_path = "./image_sample_ad/1.jpg"
results = model(img_path)

categories = results[0].names
category_predict = categories[results[0].probs.top1]
print(f"The screenshot is the type: {category_predict}")
```

This will return the predicted category, however as you will see, the execution it's not really fast and depends from the ultralytics library that's not very lightweight.

### Export the model

To have a better performance, and more lightweight libraries where the model will be use, we need to export the model.
To do this, I use `onnx` format; the script is very simple:

```python
from ultralytics import YOLO

model = YOLO("./runs/classify/train2/weights/best.pt")
model.export(format="onnx")
```

This will generate `onnx` file called `best.onnx`.

### Optimizing the exported model
The exported model now is ready to be used, however the inputs (images) to be predicted, needs to have the same format the the used for the training, meaning a pre-processing that before was made by the model itself.
However, now we are trying to use the onnx model so we need a new strategy.

#### Pre-process the image using torchvision transforms
The first one, is from torchvision use transform, to adjust the input image, to the desired parameters. The script looks as follows:

```python
import onnxruntime
from PIL import Image
import torchvision.transforms as transforms


def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor.numpy()

def predict(image):
    input_data = preprocess_image(image)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    return outputs[0]



onnx_model_path = "./model text ads/models/best.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)
image = Image.open("./image_sample_ad/1.jpg")
predictions = predict(image)
print(predictions)
```
This works, however we have similar problems where `torchvision` is very heavy, and also takes like 0.5-1 second to make the prediction. So we jump to the next solution.

#### Pre-process the image to predict using PIL

This is the one w/ better results for this project, and the best it's that we are just using `PIL` to make the pre-process of the image.
```python
import onnxruntime
import numpy as np
from PIL import Image
import io

def preprocess_image(image, target_size=(64, 64)):
    image = image.resize(target_size, Image.BILINEAR)
    image_array = np.array(image).astype(np.float32)
    image_array /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    image_array = np.expand_dims(image_array, axis=0)
    return image_array


def predict(image):
    input_data = preprocess_image(image)
    inputs = {session.get_inputs()[0].name: input_data}
    outputs = session.run(None, inputs)
    return outputs[0]


def convert_to_jpeg(image_path):
    with Image.open(image_path) as image:
        if image.mode == 'RGBA':
            image = image.convert('RGB')
        image.save(image_path.replace(".png", ".jpg"), format='JPEG')


onnx_model_path = "./model text ads/models/best.onnx"
session = onnxruntime.InferenceSession(onnx_model_path)
image_path = "./model text ads/1.jpg"
if "png" in image_path:
    convert_to_jpeg(image_path)
    image_path = image_path.replace(".png", ".jpg")
image = Image.open(image_path)
predictions = predict(image)
print(predictions)
```

## Analyze the results

So after all the mentioned on the documentation, we got nice results reaching an accuracy on validation data of something around 96-98%, and the time to execute each prediction it's just negligible; and the best part, that on the project that we want to include our model, we will just need to have `onnxruntime` (vewry light), `numpy` used on most of the projects, and `PIL`.

So we can conclude that the project was a success, and the model was implemented on the main project, having a nice performance.

## Ideas to have better results / accuracy of classification

As mentioned, the model works fine, however this doesn't means that is perfect, we still have a lot of issues, but that can be improved replicating the steps, and be more strict on some parts.

So ideas to improve the model are the followings:
- The data needs to be larger.
    - train - display : 2550
    - train - text : 1800
    - val - display : 450
    - val - text : 315
- The edge cases, have to be more.
    - there are many specific cases that are really detailed, and it's hard to the model to find a pattern while there are just a few of those cases.
- Have a better criteria on the data tabulation.
    - This is a personal mistake I guess, because I was very selective on which cases are text or not, so when I allowed `small logos` on the text ones, may be a mistake because it's hard to a human to match the criteria, so also will be hard for the model.
- Text ads are not depending of the color.
    - Most of the cases for texts ads, are just white background w/ letter black or blue, this make a pattern on the model; so text ads w/ other background like black, tends to be misclassified as display.
    - This can be improve, by inverting the colors on the text ads, just run a script, to iterate over all the python files, and randomly picks an ad, and invert the color to random ones. This will expand the patterns of the models, to detect the text that we want.
