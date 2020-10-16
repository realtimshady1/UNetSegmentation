#%% Import Libraries
import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm 
from tensorflow.keras.layers import Input,Conv2D,MaxPooling2D,concatenate,Conv2DTranspose
from runLengthEncodeDecode import rle_decode,rle_encode,masks_as_image
from tensorflow.keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#%matplotlib inline

#%% Work Constants
# Work paths
TRAIN_PATH = '../data/train_v2/'
TEST_PATH = '../data/test_v2/'
DF_NAME = 'train_ship_segmentations_v2.csv'
CHECKPOINT_PATH = 'Unet_model_weights.h5'

# Training Variables
[W, H, C] = [768, 768, 3]
TRAIN_BATCH = 32
TEST_BATCH = 1
EPOCHS = 1

# Calulate iterations needed per epoch given a batch size
STEP_SIZE_TRAIN = len([f for f in os.listdir(TRAIN_PATH)]) // TRAIN_BATCH
STEP_SIZE_TEST = len([f for f in os.listdir(TEST_PATH + 'ships/')]) // TEST_BATCH

#%% Mask RLE conversion generator function
def decodeGenerator(generator, shape):
    '''
    generator function that calls for the train_generator
    modify the y output by decoding the elements
    return x and y
    '''
    while True:
        [x_out, y] = next(generator)
    
        # Initialize y_out
        y_out = np.zeros(shape, dtype=np.bool)
        
        # Cycle through masks in y
        for i, mask in enumerate(y):
            # Check whether it is a NaN
            if str(mask) != 'nan':
                #Decode the mask
                y_out[i] = rle_decode(str(mask), shape[1:3])
    
        yield x_out, y_out


#%% Load Training Data
df = pd.read_csv(DF_NAME)

print('Loading Data....\n')
datagen = ImageDataGenerator(rescale=1./255., validation_split=0.2)
train_generator = datagen.flow_from_dataframe(dataframe=df,
                                              directory=TRAIN_PATH,
                                              x_col='ImageId',
                                              y_col='EncodedPixels',
                                              target_size=(W, H),
                                              batch_size=TRAIN_BATCH,
                                              class_mode='raw')

#%%
test_generator = datagen.flow_from_directory(directory=TEST_PATH,
                                             target_size=(W, H),
                                             class_mode=None,
                                             batch_size=TEST_BATCH)

print('Done.')

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
model.compile(optimizer='adam', 
              loss='binary_crossentropy', 
              metrics=['Accuracy'])

model.summary()

#%% Save checkpoints
callbacks = [ModelCheckpoint(CHECKPOINT_PATH,verbose=1,save_best_only=True),
             EarlyStopping(patience=2,monitor='val_loss'),
             TensorBoard(log_dir='logs')]

#%% Train model
results = model.fit(decodeGenerator(train_generator,[TRAIN_BATCH, W, H]),
                   steps_per_epoch=STEP_SIZE_TRAIN,
                   epochs=EPOCHS)
model.save(CHECKPOINT_PATH)
model.load_weights(CHECKPOINT_PATH)

#%% Predict
test_generator.reset()
Y_test_pred = model.predict(test_generator, steps=STEP_SIZE_TEST)

#%% Plot Results
test_generator.reset()
fig, m_axs = plt.subplots(10, 2, figsize=(10, 40))
for n, (ax1, ax2) in enumerate(m_axs):
    ax1.imshow(np.squeeze(next(test_generator)))
    ax2.imshow(np.squeeze(Y_test_pred[n]))
    
