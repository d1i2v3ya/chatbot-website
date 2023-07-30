import numpy as np 
import pandas as pd
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="darkgrid")

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.utils import class_weight
df = pd.read_csv(r'C:/Users/M9bin/OneDrive/Documents/brainstroke/brain_stroke.csv')
print(df.head(10))
print(df.shape)
print(df.info())
print(df.isnull().sum())
print(df['stroke'].value_counts())
dataset = pd.get_dummies(df)
print(dataset.head())
X = dataset.copy()
y = X.pop('stroke')
ax = sns.countplot(x=y)
plt.show()
from imblearn.over_sampling import RandomOverSampler
ros = RandomOverSampler(random_state=0)
X, y = ros.fit_resample(X, y) # type: ignore
ax = sns.countplot(x=y)
plt.show()
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
obj_norm = MinMaxScaler().fit(X) # type: ignore
X_normalized = obj_norm.transform(X) 
print(X_normalized[0])
train_features, test_features, train_labels, test_labels = train_test_split(X_normalized, y, test_size = 0.3, random_state = 0)
print(train_features.shape, train_labels.shape)
print(test_features.shape, test_labels.shape)
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, losses
from keras.callbacks import EarlyStopping, LearningRateScheduler
from keras.optimizers import Adam

print(tf.__version__)
device_name = tf.test.gpu_device_name()
if "GPU" not in device_name:
    print("GPU device not found")
print('Found GPU at: {}'.format(device_name))
print("[INFO] Class weighting...")
class_weights = class_weight.compute_class_weight('balanced', 
                                                  classes=np.unique(np.ravel(train_labels, order='C')),
                                                  y=np.ravel(train_labels, order='C'))

class_weight_dict = dict(enumerate(class_weights))
print(class_weight_dict)
def plot_cm(labels, predictions, p=0.5):
    cm = confusion_matrix(labels, predictions > p)
    plt.figure(figsize=(8,8))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title('Confusion matrix')
    plt.ylabel('Actual label')
    plt.xlabel('Predicted label')
    print(f'Actual is 0 and Prediction is 0: ', cm[0][0])
    print(f'Actual is 0 and Prediction is 1: ', cm[0][1])
    print(f'Actual is 1 and Prediction is 0: ', cm[1][0])
    print(f'Actual is 1 and Prediction is 1: ', cm[1][1])
    print(f'Total Correct Prediction: {np.sum(cm[0][0] + cm[1][1])} / {np.sum(cm)}')
    plt.show()
model = Sequential([
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(2, activation='sigmoid')
  ])
EPOCHS = 150
INIT_LR = 1e-1

lr_schedule = LearningRateScheduler(
    lambda epoch: 1e-8 * 10**(epoch/20))
optim_LR = tf.keras.optimizers.legacy.Adam(learning_rate=INIT_LR, decay=INIT_LR / EPOCHS,amsgrad=False)

model.compile(optimizer=optim_LR, 
              loss=losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(
    train_features,
    train_labels,
    epochs=EPOCHS,
    verbose=0, # type: ignore
    validation_split = 0.2,
    callbacks=[lr_schedule],
    class_weight=class_weight_dict)
lrs = 1e-8 * 10**(np.arange(EPOCHS)/20)
import plotly.express as px

fig = px.line(x=lrs, y=history.history["loss"], title='Learning rate vs loss', log_x=True)
fig.show()

fig, axs = plt.subplots(2, 1, figsize=(15,15))
axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train','Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

model = Sequential([
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(128, activation='relu'),
      layers.Dropout(0.3),
      layers.Dense(2, activation='sigmoid')
  ])
es = EarlyStopping(monitor='val_accuracy', verbose=1, patience=20, mode='max', restore_best_weights=True)

LR = 1e-2
EPOCHS = 250
optim = tf.keras.optimizers.legacy.Adam(learning_rate=LR, decay=LR / EPOCHS,amsgrad=False)

model.compile(optimizer=optim, 
              loss=losses.binary_crossentropy, metrics=['accuracy'])
history = model.fit(
    train_features,
    train_labels,
    epochs=EPOCHS,
    verbose=0, # type: ignore
    validation_split = 0.2,
    callbacks=[es],
    class_weight=class_weight_dict)

fig, axs = plt.subplots(2, 1, figsize=(15,15))

axs[0].plot(history.history['loss'])
axs[0].plot(history.history['val_loss'])
axs[0].title.set_text('Training Loss vs Validation Loss')
axs[0].set_xlabel('Epochs')
axs[0].set_ylabel('Loss')
axs[0].legend(['Train','Val'])
axs[1].plot(history.history['accuracy'])
axs[1].plot(history.history['val_accuracy'])
axs[1].title.set_text('Training Accuracy vs Validation Accuracy')
axs[1].set_xlabel('Epochs')
axs[1].set_ylabel('Accuracy')
axs[1].legend(['Train', 'Val'])
plt.show()

print("[INFO] Calculating model accuracy")
train_scores = model.evaluate(train_features, train_labels)
test_scores = model.evaluate(test_features, test_labels)
print(f"Test Accuracy (on train dataset): {train_scores[1]*100}")
print(f"Test Accuracy (on test dataset): {test_scores[1]*100}")
prediction = np.round(model.predict(test_features))
print(classification_report(test_labels, prediction))
test_predictions_baseline = model.predict(test_features)
plot_cm(test_labels, test_predictions_baseline)

import pickle 
pickle.dump(model, open('C:/Users/M9bin/OneDrive/Documents/brainstroke/models/model.pkl','wb'))
model=pickle.load(open('C:/Users/M9bin/OneDrive/Documents/brainstroke/models/model.pkl','rb'))

'''
import tensorflow as tf

# ... (define and train your model)

# Save the model using Keras's save() method
model.save('C:/Users/M9bin/OneDrive/Documents/brainstroke/weights.keras')

# Load the model using Keras's load_model() method
model = tf.keras.models.load_model('C:/Users/M9bin/OneDrive/Documents/brainstroke/weights.keras')
'''




