
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, concatenate, Conv2DTranspose, BatchNormalization, Dropout, Lambda,ZeroPadding2D

def wnet_model(n_class = 4, height = 266 ,width = 266,channels = 1 ):
    inputs = Input((height,width,channels))
    x1 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(inputs)
    x1 = Dropout(0.1)(x1)
    x1 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(x1)
    p1 = MaxPooling2D(pool_size=(2,2))(x1)
    
    p1 = MaxPooling2D(pool_size=(2,2),strides=(1,1))(p1)
   

    x2 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(p1)
    x2 = Dropout(0.1)(x2)
    x2 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(x2)
    p2 = MaxPooling2D((2,2))(x2)
    

    x3 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(p2)
    x3 = Dropout(0.1)(x3)
    x3 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(x3)
    p3 = MaxPooling2D((2,2))(x3)
    
    p3 = MaxPooling2D(pool_size=(2,2),strides=(1,1))(p3)
    

    x4 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(p3)
    x4 = Dropout(0.1)(x4)
    x4 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(x4)
    p4 = MaxPooling2D((2,2))(x4)
    

    x5 = Conv2D(1024,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(p4)
    x5 = Dropout(0.1)(x5)
    x5 = Conv2D(1024,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(x5)

    #Encoder Reconstruction
    y5 = Conv2DTranspose(128,(2,2),strides = (2,2),padding='same')(x5)
    
    y5 = concatenate([y5,x4])
    q5 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(y5)
    q5 = Dropout(0.1)(q5)
    q5 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(q5)

    y4 = Conv2DTranspose(64,(3,3),strides = (2,2),padding='same')(q5)
    
    y4 = ZeroPadding2D(padding=(1,1))(y4)
    
    y4 = concatenate([y4,x3])
    q4 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(y4)
    q4 = Dropout(0.1)(q4)
    q4 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(q4)

    y3 = Conv2DTranspose(32,(3,3),strides = (2,2),padding='same')(q4)
    
    y3 = concatenate([y3,x2])
    q3 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(y3)
    q3 = Dropout(0.1)(q3)
    q3 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(q3)

    y2 = Conv2DTranspose(16,(3,3),strides = (2,2),padding='same')(q3)
    
    y2 = ZeroPadding2D(padding=(1,1))(y2)
    
    y2 = concatenate([y2,x1])
    q2 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(y2)
    q2 = Dropout(0.1)(q2)
    q2 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same', name="encoder")(q2)
      
     
    # Decoder
    dec_x1 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(q2)
    dec_x1 = Dropout(0.1)(dec_x1)
    dec_x1 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_x1)
    dec_p1 = MaxPooling2D(pool_size=(2,2))(dec_x1)

    dec_p1 = MaxPooling2D(pool_size=(2,2),strides=(1,1))(dec_p1)


    dec_x2 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_p1)
    dec_x2 = Dropout(0.1)(dec_x2)
    dec_x2 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_x2)
    dec_p2 = MaxPooling2D((2,2))(dec_x2)


    dec_x3 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_p2)
    dec_x3 = Dropout(0.1)(dec_x3)
    dec_x3 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_x3)
    dec_p3 = MaxPooling2D((2,2))(dec_x3)
    

    dec_p3 = MaxPooling2D(pool_size=(2,2),strides=(1,1))(dec_p3)


    dec_x4 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_p3)
    dec_x4 = Dropout(0.1)(dec_x4)
    dec_x4 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_x4)
    dec_p4 = MaxPooling2D((2,2))(dec_x4)


    dec_x5 = Conv2D(1024,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_p4)
    dec_x5 = Dropout(0.1)(dec_x5)
    dec_x5 = Conv2D(1024,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_x5)

    # decoder recontruct
    dec_y5 = Conv2DTranspose(128,(2,2),strides = (2,2),padding='same')(dec_x5)

    dec_y5 = concatenate([dec_y5,dec_x4])
    dec_q5 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_y5)
    dec_q5 = Dropout(0.1)(dec_q5)
    dec_q5 = Conv2D(512,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_q5)

    dec_y4 = Conv2DTranspose(64,(3,3),strides = (2,2),padding='same')(dec_q5)

    dec_y4 = ZeroPadding2D(padding=(1,1))(dec_y4)

    dec_y4 = concatenate([dec_y4,dec_x3])
    dec_q4 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_y4)
    dec_q4 = Dropout(0.1)(dec_q4)
    dec_q4 = Conv2D(256,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_q4)

    dec_y3 = Conv2DTranspose(32,(3,3),strides = (2,2),padding='same')(dec_q4)

    dec_y3 = concatenate([dec_y3,dec_x2])
    dec_q3 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_y3)
    dec_q3 = Dropout(0.1)(dec_q3)
    dec_q3 = Conv2D(128,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_q3)

    dec_y2 = Conv2DTranspose(16,(3,3),strides = (2,2),padding='same')(dec_q3)

    dec_y2 = ZeroPadding2D(padding=(1,1))(dec_y2)

    dec_y2 = concatenate([dec_y2,dec_x1])
    dec_q2 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_y2)
    dec_q2 = Dropout(0.1)(dec_q2)
    dec_q2 = Conv2D(64,(3,3),kernel_initializer='he_normal',activation='relu',padding='same')(dec_q2)
    
    outputs = Conv2D(n_class,(1,1),activation='softmax',name="decoder")(dec_q2)
    model = Model(inputs = [inputs],outputs = [outputs])
    return model


model,enc = wnet_model(4,266,266,1)
model.compile(optimizer='adam', 
loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),metrics=['accuracy'])
model.summary()
