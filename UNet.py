#%% Import Libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from cv2 import waitKey,imread,resize
#from skimage.transform import resize
from skimage.io import imshow
from skimage import img_as_bool
from skimage.transform import resize as skresize
from tqdm import tqdm 
from tensorflow.keras.layers import Input,Conv2D,Dropout,MaxPooling2D,concatenate,Conv2DTranspose
from runLengthEncodeDecode import rle_decode,rle_encode,masks_as_image
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard

%matplotlib inline

#%% Build Training Data
# Work paths
TRAIN_PATH = 'train_v2/'
TEST_PATH = 'test_v2/'
MASK_PATH = 'train_ship_segmentations_v2.csv'

# Define input diemsnions
[W, H, C] = [768, 768, 3]

# Retrieve data
train_ids = next(os.walk(TRAIN_PATH))[2]
test_ids = next(os.walk(TEST_PATH))[2]

X_test = np.zeros((len(test_ids), W, H, C), dtype=np.uint8)

X_train = np.zeros((len(train_ids), W, H, C), dtype=np.uint8)
Y_train = np.zeros((len(train_ids), W, H, 1), dtype=np.bool)

# Load masks data
mask_df = pd.read_csv(MASK_PATH)

# Load training data
for n, img_id in tqdm(enumerate(train_ids), total=len(train_ids)):
    # Load images
    img = imread(TRAIN_PATH + img_id)
    [mask_h, mask_w, _] = np.shape(img)
    #img = resize(img, (H, W))
    X_train[n] = img
    
    # Load masks
    mask = mask_df.EncodedPixels[mask_df.ImageId==img_id].to_list()
    if str(mask[0]) == 'nan':
        mask = np.zeros((W,H,1), dtype=bool)
    else:
        mask = rle_decode(str(mask[0]), (mask_h, mask_w))
        #mask = skresize(mask, (H,W))
        mask = np.expand_dims(mask, -1)
    Y_train[n] = mask

# Load testing data
for n, img_id in tqdm(enumerate(test_ids), total=len(test_ids)):
    # Load images
    img = imread(TEST_PATH + img_id)
    [mask_h, mask_w, _] = np.shape(img)
    #img = resize(img, (H, W))
    X_test[n] = img


    
#%% Plot
image_x = 29
[fig, (ax1, ax2)] = plt.subplots(1,2, figsize = (10,5))
ax1.imshow(X_train[image_x])
ax2.imshow(np.squeeze(Y_train[image_x]))

#%%
# Normalize Inputs
print('Normalizing training data...\n')
mean = np.mean(X_train, axis=0)
std = np.std(X_train, axis=0)
X_train = (X_train - mean)/std
print('Done normalizing.\n')

#%% Define Model

# Build the model
inputs = Input((H, W, C))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)


c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same')(c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid')(c9)

model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.PrecisionAtRecall(0.9)])

model.summary()

#%% Save checkpoints
CHECKPOINT_PATH = 'Unet_model_weights.h5'
callbacks = [
    ModelCheckpoint(CHECKPOINT_PATH, verbose=1, save_best_only=True),
    EarlyStopping(patience=2, monitor='val_loss'),
    TensorBoard(log_dir='logs')]

#%% Build Model
results = model.fit(X_train, Y_train, validation_split=0.1, batch_size = 16, epochs=15, callbacks=callbacks)

#%% Save Model
model.save(CHECKPOINT_PATH)

#%% Predict
Y_test_pred = model.predict(X_test)

#%% Plot Results
fig, m_axs = plt.subplots(10, 2, figsize=(10, 40))
for n, (ax1, ax2) in enumerate(m_axs):
    ax1.imshow(X_test[n])
    ax2.imshow(np.squeeze(Y_test_pred[n]))
    