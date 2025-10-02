***************************************************mcp***************************************************

n = int(input("enter number of inputs: "))
inputs=[0]*n
weights=[1]*n
for x in range(n):
    t = int(input("enter bit :"))
    inputs[x]=t

# sum of products
sum = 0
for x in range(n):
    sum += inputs[x]*weights[x]

# giving threshold value
if sum>=n:
    print("output: 1")
else:
    print("output: 0")


***********************************************************************************************************




















******************************************************perceptron************************************************

import numpy as np

class Perceptron2D:
    def __init__(self, input_size, learning_rate, epochs):
        self.weights = np.zeros(input_size) # weights mei first index bias hai
        self.bias = 0
        self.learning_rate = learning_rate
        self.epochs = epochs

    def predict(self, inputs):
        summation = np.dot(inputs, self.weights) + self.bias
        return 1 if summation > 0 else 0   #THIS IS SIMPLE THRESHOLD  0 1 ACTIVATION FUNCTION

    def train(self, training_data, labels):
        for epoch in range(self.epochs):
            for inputs, label in zip(training_data, labels):
                prediction = self.predict(inputs)
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error


# training data
training_data = np.array([[2, 4], [6, 8], [1, 3], [5, 7], [10, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]])
labels = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Create and train the perceptron
input_size = 2  # Number of features
learning_rate = 0.01
epochs = 100
perceptron_2d = Perceptron2D(input_size, learning_rate, epochs)
perceptron_2d.train(training_data, labels)


# Test the perceptron
test_inputs = np.array([[2, 4], [1, 3], [6, 7], [10, 11]])
for test_input in test_inputs:
    prediction = perceptron_2d.predict(test_input)
    print(f"Inputs: {test_input}, Predicted Label: {prediction}")

***********************************************************************************************************************************
















*************************************************************cnn*****************************************************************

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical


# Load and preprocess the CIFAR-10 dataset
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
train_images, test_images = train_images / 255.0, test_images / 255.0  #normalize
train_labels, test_labels = to_categorical(train_labels), to_categorical(test_labels) #encode

# Building the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))

# Compile the model
model.compile(optimizer='adam',
                loss='categorical_crossentropy',
                metrics=['accuracy'])

# Train the model
model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))

# Evaluate the model
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")


*****************************************************************************************************************************

















************************************************************lstm***************************************************************

from tensorflow.keras.datasets import imdb
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding , Dense , LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer


# Get the word index
word_index = imdb.get_word_index()

# Calculate vocabulary size
vocab_size = len(word_index) + 1  # Add 1 for padding token

print("Vocabulary Size:", vocab_size)

mx_len = 200
max_words = 10000
embending_dims = 128
(X_train , y_train), (X_test, y_test) = imdb.load_data(num_words = max_words)
print(X_train.shape)


X_train = pad_sequences(X_train, maxlen=mx_len)
X_test = pad_sequences(X_test, maxlen=mx_len)


model = Sequential()
model.add(Embedding(input_dim = max_words , output_dim = embending_dims , input_length = mx_len))
model.add(LSTM(128))#dropout=0.2, recurrent_dropout=0.2)
model.add(Dense(1 , activation = 'sigmoid'))


model.summary()
model.compile(optimizer = 'adam' , loss = 'binary_crossentropy' , metrics = ['accuracy'])
model.fit(X_train, y_train, epochs=5, batch_size=32, validation_split=0.2)



input_sentence = "This movie was fantastic! The acting was superb"

tokenizer = Tokenizer()
tokenizer.fit_on_texts([input_sentence])

input_sequence = tokenizer.texts_to_sequences([input_sentence])
input_sequence_padded = pad_sequences(input_sequence, maxlen=mx_len)
# Make predictions
predictions = model.predict(input_sequence_padded)

# Assuming binary classification (positive/negative sentiment)
if predictions[0][0] >= 0.5:
    print(f"The sentence '{input_sentence}' is positive.")
else:
    print(f"The sentence '{input_sentence}' is negative.")



**************************************************************************************************************************
















*************************************************************auto encoders****************************************************

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from keras import layers, losses
from keras.datasets import fashion_mnist
from keras.models import Model
import seaborn as sns


(x_train, _), (x_test, _) = fashion_mnist.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
print (x_train.shape)
print (x_test.shape)

class Autoencoder(Model):
    def __init__(self, latent_dim, shape):
    super(Autoencoder, self).__init__()
    self.latent_dim = latent_dim
    self.shape = shape
    self.encoder = tf.keras.Sequential([
        layers.Flatten(),
        layers.Dense(latent_dim, activation='relu'),
    ])
    self.decoder = tf.keras.Sequential([
        layers.Dense(tf.math.reduce_prod(shape), activation='sigmoid'),
        layers.Reshape(shape)
    ])

    def call(self, x):
    encoded = self.encoder(x)
    decoded = self.decoder(encoded)
    return decoded


shape = x_test.shape[1:]
latent_dim = 64
autoencoder = Autoencoder(latent_dim, shape)

autoencoder.compile(optimizer='adam', loss=losses.MeanSquaredError())

autoencoder.fit(x_train, x_train,
                epochs=10,
                shuffle=True,
                validation_data=(x_test, x_test))

encoded_imgs = autoencoder.encoder(x_test).numpy()
decoded_imgs = autoencoder.decoder(encoded_imgs).numpy()


n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i])
    plt.title("original")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i])
    plt.title("reconstructed")
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()


mean_square_error = np.mean(np.power(x_test-decoded_imgs,2))

mse_dist = []
for i in range(len(x_test)):
    mse_dist.append(str(round(np.mean(np.power(x_test[i]-decoded_imgs[i],2)),3)))

mse_dist.sort()
sns.countplot(mse_dist)


*************************************************************************************************************************************













***************************************************************GD'S*******************************************************************


************common part for everything**************************


import numpy as np
import matplotlib.pyplot as plt

X = [0.3,0.5,0.7,0.4]
Y = [0.2,0.6,0.9,0.3]


def sigmoid(w,x,b):
    return (1.0/(1.0+np.exp(-(w*x + b))))

def error(w,b,X,Y):
    errors = 0.0
    for x,y in zip(X,Y):
    y_pred = sigmoid(w,x,b)
    errors += 0.5*(y_pred - y)**2

    return errors

def grad_w(w,b,X,Y):
    y_pred = sigmoid(w,X,b)
    return ((y_pred - Y)*y_pred*(1-y_pred)*X)


def grad_b(w,b,X,Y):
    y_pred = sigmoid(w,X,b)
    return ((y_pred - Y)*y_pred*(1-y_pred))


def plts(errors:list):
    index = [i for i in range(len(errors))]
    plt.plot(index , errors)
    plt.xlabel("epoch")
    plt.ylabel("Error")
    plt.show()

********************************************************************



**************************vanila gd***************************************

# Gradient decent vanialla

def gradient_decent():
    w, b, lr, epoch = 2,-2,1,100
    errors = []
    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")

    for i in range(epoch):
    errors.append(error(w,b,X,Y))
    dw,db = 0,0
    for x,y in zip(X,Y):
        # print(x,y)
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)

    w = w - lr*dw
    b = b - lr*db

    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")
    return errors


list_error = gradient_decent()

plts(list_error)

*********************************************************************

************************Stochastic gd*********************************

# stocastic gradient decent
import random

def stocastic_gradient_decent():
    w, b, lr, epoch = 2,-2,1,100
    errors = []
    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")

    for i in range(epoch):
    errors.append(error(w,b,X,Y))
    dw,db = 0,0
    # print(random.randint(0,3))
    ran = random.randint(0,3)
    # for x,y in zip(X,Y):
        # print(x,y)
    dw+=grad_w(w,b,X[ran],Y[ran])
    db+=grad_b(w,b,X[ran],Y[ran])

    w = w - lr*dw
    b = b - lr*db

    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")
    return errors


list_error = stocastic_gradient_decent()

plts(list_error)

************************************************************************


*********************************mini-batch***************************

# mini batch gradient decent
import random

def batch_gradient_decent():
    w, b, lr, epoch = 2,-2,1,100
    errors = []

    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    batch_size = 2

    for i in range(epoch):
    errors.append(error(w,b,X,Y))
    dw,db = 0,0
    X_b= X[0:batch_size]
    Y_b= Y[0:batch_size]
    for x,y in zip(X_b,Y_b):
        # print(x,y)
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)

    w = w - lr*dw
    b = b - lr*db

    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")
    return errors


list_error = batch_gradient_decent()

plts(list_error)

*************************************************************************



************************momentum gd***********************************

def momentum_gradient_decent():
    w,b,lr,epochs = 2,-2,0.1,100
    errors = []
    print(X,Y)
    prev_v_w, prev_v_b , gamma = 0,0,0.9
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    for i in range(epochs):
    dw,db = 0,0
    errors.append(error(w,b,X,Y))
    for x,y in zip(X,Y):
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)
    v_w = gamma * prev_v_w + lr*dw
    v_b= gamma * prev_v_b + lr*db

    w = w - v_w
    b = b - v_b
    prev_v_w = v_w
    prev_v_b = v_b

    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")

    return errors


list_error = momentum_gradient_decent()

plts(list_error)

*****************************************************************


***************************nestrov gd******************************

def nestrov_gradient_decent():
    w,b,lr,epochs = 2,-2,0.1,100
    errors = []
    print(X,Y)
    prev_v_w, prev_v_b , gamma = 0,0,0.9
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    for i in range(epochs):
    dw,db = 0,0
    errors.append(error(w,b,X,Y))
    for x,y in zip(X,Y):
        dw+=grad_w(w - gamma * prev_v_w,b - gamma * prev_v_b,x,y)
        db+=grad_b(w - gamma * prev_v_w,b - gamma * prev_v_b,x,y)
    v_w = gamma * prev_v_w + lr*dw
    v_b= gamma * prev_v_b + lr*db

    w = w - v_w
    b = b - v_b
    prev_v_w = v_w
    prev_v_b = v_b

    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")

    return errors


list_error = nestrov_gradient_decent()

plts(list_error)


*******************************************************************


**********************************ada grad gd******************************

def adagrad():
    w,b,lr,epochs= 2,-2,0.1,100
    v_w,v_b, eps =0,0 ,1e-8
    errors = []
    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    for i in range(epochs):
    dw,db = 0,0
    errors.append(error(w,b,X,Y))
    for x,y in zip(X,Y):
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)

    v_w = v_w + dw**2
    v_b = v_b + db**2

    w = w - (lr/(np.sqrt(v_w) + eps))*dw
    b = b - (lr/(np.sqrt(v_b) + eps))*db


    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")

    return errors

list_error = adagrad()

plts(list_error)

**************************************************************************


***********************RMS***********************************************

def RMS():
    w,b,lr,epochs= 2,-2,0.1,100
    v_w,v_b, eps  , beta=0,0 ,1e-8 , 0.9
    errors = []
    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    for i in range(epochs):
    dw,db = 0,0
    errors.append(error(w,b,X,Y))
    for x,y in zip(X,Y):
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)

    v_w = beta*v_w + (1-beta)*dw**2
    v_b = beta*v_b + (1-beta)*db**2

    w = w - (lr/(np.sqrt(v_w) + eps))*dw
    b = b - (lr/(np.sqrt(v_b) + eps))*db


    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")

    return errors

list_error = RMS()

plts(list_error)

**************************************************************************


**************************************************adam gd************************

def adam():
    w,b,lr,epochs= 2,-2,0.1,100
    v_w,v_b, eps  , beta=0,0 ,1e-8 , 0.9
    m_w,m_b = 0,0
    beta2=0.999
    errors = []
    print(X,Y)
    print(f"Error before updating weights: {error(w,b,X,Y)}")
    for i in range(epochs):
    dw,db = 0,0
    errors.append(error(w,b,X,Y))
    for x,y in zip(X,Y):
        dw+=grad_w(w,b,x,y)
        db+=grad_b(w,b,x,y)

    m_w = beta*v_w + (1-beta)*dw
    m_b = beta*v_b + (1-beta)*db


    v_w = beta2*v_w + (1-beta2)*dw**2
    v_b = beta2*v_b + (1-beta2)*db**2

    m_w_hat = m_w/(1-beta**(i+1))
    m_b_hat = m_b/(1-beta**(i+1))

    v_w_hat = v_w/(1-beta2**(i+1))
    v_b_hat = v_b/(1-beta2**(i+1))

    w = w - (lr/(np.sqrt(v_w_hat) + eps))*m_w_hat
    b = b - (lr/(np.sqrt(v_b_hat) + eps))*m_b_hat



    print(f"Error after updating weights: {error(w,b,X,Y)} weight : {w} bias : {b} ")

    return errors

list_error = adam()

plts(list_error)


********************************************************************************

**************************************************************************************************************************************