# Setup

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import os
import multiprocessing
from tqdm import tqdm
import tensorflow as tf
import kerastuner as kt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# directory = '/kaggle/input/histopathologic-cancer-detection/'
directory = '/histopathologic-cancer-detection/'

trainSet = pd.read_csv(directory + 'train_labels.csv')
trainSet.head()

# Size, Dimesion, & Structure

trainSet.info()

# Images

id_label_dict = dict(zip(trainSet['id'], trainSet['label']))
train_imgs = os.listdir(directory + "train")
chosen_imgs = np.random.choice(train_imgs, 20, replace=False)
fig, axes = plt.subplots(2, 10, figsize=(30, 6))
axes = axes.flatten()

for idx, img in enumerate(chosen_imgs):
    openImg = Image.open(directory + "train/" + img)
    axes[idx].imshow(openImg)
    label = id_label_dict.get(img.split('.')[0])
    axes[idx].set_title(f'Label: {label}')
    axes[idx].set_xticks([])
    axes[idx].set_yticks([])

plt.tight_layout()

# Data cleaning procedures

## Convert images to appropriate file type

# prefetch
label_dict = dict(zip(trainSet['id'], trainSet['label']))

def convert_image(image_file, label=0, with_label=True, subset='train'):
    image_name = image_file.split('.')[0]
    output_dir = f'png/{subset}/{label}' if with_label else f'png/{subset}'
    output_path = f'{output_dir}/{image_name}.png'

    if not os.path.exists(output_path):
        with Image.open(directory + f'{subset}/{image_file}') as tiff_img:
            png = tiff_img.convert("RGB")
            png.save(output_path)

def process_images(image_files, labels=[], with_label=True, subset='train'):
    os.makedirs(f'png/{subset}', exist_ok=True)
    if with_label:
        for label in set(labels):
            os.makedirs(f'png/{subset}/{label}', exist_ok=True)

    tasks = [(filename, label_dict[filename.split('.')[0]] if with_label else 0, with_label, subset) for filename in image_files]
    num_processes = min(len(tasks), multiprocessing.cpu_count())
    with multiprocessing.Pool(processes=num_processes) as pool:
        list(tqdm(pool.imap(convert_image, tasks), total=len(tasks)))

process_images((trainSet['id'] + '.tif').tolist(), labels=trainSet['label'].tolist())
process_images(os.listdir(directory + "test"), with_label=False, subset='test')

train_dataset = tf.keras.utils.image_dataset_from_directory('/kaggle/working/png/train', 
    label_mode='binary',
    image_size=(96,96), 
    seed=42,
    validation_split=0.2,
    subset='training',
    batch_size=128
)

val_dataset = tf.keras.utils.image_dataset_from_directory('/kaggle/working/png/train', 
    label_mode='binary',
    image_size=(96,96), 
    seed=42,
    validation_split=0.2,
    subset='validation',
    batch_size=128
)

test_dataset = tf.keras.utils.image_dataset_from_directory('/kaggle/working/png/test',
    label_mode=None,
    image_size=(96,96),
    shuffle=False,
    batch_size=1
)

AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=AUTOTUNE)

train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
val_dataset = val_dataset.cache().prefetch(buffer_size=AUTOTUNE)

data_augmentation = tf.keras.Sequential([
  tf.keras.layers.RandomFlip("horizontal"),
  tf.keras.layers.RandomRotation(0.2),
])

train_dataset = train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y))

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_dataset = train_dataset.map(lambda x, y: (normalization_layer(x), y))
val_dataset = val_dataset.map(lambda x, y: (normalization_layer(x), y))
test_dataset = test_dataset.map(lambda x: normalization_layer(x))

train_dataset = train_dataset.shuffle(buffer_size=1000).cache().prefetch(buffer_size=AUTOTUNE)

# sample images w/ labels

plt.figure(figsize=(12, 8))

for i in range(20):
    plt.subplot(4, 5, i + 1)
    img_id = np.random.choice(trainSet['id'])
    img_label = trainSet[trainSet['id'] == img_id]['label'].values[0]
    img_path = os.path.join(directory, f"train/{img_id}.tif")
    img = Image.open(img_path)
    plt.imshow(img)
    plt.title(f"Label: {img_label}")
    plt.axis('off')
plt.tight_layout()


# label distribution

plt.figure(figsize=(6, 4))
sns.countplot(x='label', data=trainSet)
plt.title('Label Distribution')
plt.show()

# pixel distribution

def plot_image_intensity_distribution(img):
    plt.hist(img.ravel(), bins=256, range=[0,256])
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

img_id = np.random.choice(trainSet['id'])
img_path = os.path.join(directory, f"train/{img_id}.tif")
img = Image.open(img_path)
img_np = np.array(img)

plt.figure(figsize=(6, 4))
plot_image_intensity_distribution(img_np)
plt.title('Pixel Intensity Distribution')
plt.show()

# Model

model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(96, 96, 3)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Dropout(0.25),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer='adam',
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=['accuracy'])


history = model.fit(
    train_dataset, 
    epochs=10, 
    validation_data=val_dataset, 
    verbose=1,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=3, monitor='val_loss')]
)

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.title('Train vs Validation Accuracy Per Epoch')
plt.legend(loc='lower right')

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label = 'val_loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Train vs Validation Loss Per Epoch')

def model_builder(hp):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Rescaling(1./255, input_shape=(96, 96, 3)))

    # first Dense layer
    hp_units = hp.Int('units', min_value=32, max_value=512, step=32)
    model.add(tf.keras.layers.Dense(units=hp_units, activation='relu'))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid'))
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])

    return model

tuner = kt.Hyperband(model_builder,
    objective='val_accuracy',
    max_epochs=10,
    factor=3,
    directory='my_dir',
    project_name='intro_to_kt'
)

tuner.search(train_dataset, epochs=50, validation_data=val_dataset)
best_hps=tuner.get_best_hyperparameters(num_trials=1)[0]

# Create submission entry

test_imgs = os.listdir("/kaggle/working/png/test")
model1_pred_df = pd.DataFrame(columns=['id', 'label'])
test_imgs=sorted(test_imgs)
predictions = model.predict(test_dataset)
model1_pred_df['id'] = [filename.split('.')[0] for filename in test_imgs]
model1_pred_df['label'] = np.round(predictions.flatten()).astype('int')
model1_pred_df
model1_pred_df.to_csv('predictions.csv', index=False)

# training/validation accuracy: Shows how the accuracy on the training and validation sets evolved during training

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
epochs = range(1, len(acc) + 1)

plt.plot(epochs, acc, 'b', label='Training acc')
plt.plot(epochs, val_acc, 'r', label='Validation acc')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# training/validation loss: Displays the loss on the training and validation sets throughout the training epochs

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.plot(epochs, loss, 'b', label='Training loss')
plt.plot(epochs, val_loss, 'r', label='Validation loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# confusion matrix: Helps in understanding the true positives, true negatives, false positives, and false negatives

val_predictions = model.predict(val_dataset)
val_predictions = [1 if x > 0.5 else 0 for x in val_predictions]

# True labels
val_labels = []
for images, labels in val_dataset.unbatch():
    val_labels.extend(labels.numpy())

cm = confusion_matrix(val_labels, val_predictions)
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.show()