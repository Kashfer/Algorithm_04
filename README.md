# Algorithm_04 - 20170458 최주호

#첫번째


![1](https://user-images.githubusercontent.com/94188547/173534544-b666a59b-f8c0-4619-a529-65384358fe6b.PNG)
![2](https://user-images.githubusercontent.com/94188547/173534592-5b3a00ff-1847-4c44-91bf-49c200b66626.PNG)
![3](https://user-images.githubusercontent.com/94188547/173534603-7a28a949-e7b9-4778-ac8f-118c20d5b22b.PNG)
![4](https://user-images.githubusercontent.com/94188547/173534618-74e7dc66-0107-4983-9e86-76973a130994.PNG)
![5](https://user-images.githubusercontent.com/94188547/173534631-1e0c8d8e-fd80-4801-8b18-524dfe46332a.PNG)
![6](https://user-images.githubusercontent.com/94188547/173534642-a6b0abcc-8d89-47a1-88e5-fa4b824dde63.PNG)

#두번째


![7](https://user-images.githubusercontent.com/94188547/173534665-8a67955c-41c8-4eb1-91ed-8a4215d9be03.PNG)
![8](https://user-images.githubusercontent.com/94188547/173534671-c2071642-0797-40ba-bb5d-c5cf2585e08e.PNG)
![9](https://user-images.githubusercontent.com/94188547/173534678-87d654ac-27a2-4465-93e1-a421b1ae3360.PNG)
![10](https://user-images.githubusercontent.com/94188547/173534685-02d40b1a-f7ec-4305-8f1b-63cefb83a648.PNG)
![11](https://user-images.githubusercontent.com/94188547/173534708-7e856c86-e869-40a0-968c-29f899727962.PNG)
![12](https://user-images.githubusercontent.com/94188547/173534717-a5dc0322-cbc0-40b2-bb61-b24d5e263912.PNG)

#세번째


![13](https://user-images.githubusercontent.com/94188547/173534769-f911bafd-8ca6-40c0-a9cf-7c3cca7301e5.PNG)
![14](https://user-images.githubusercontent.com/94188547/173534777-9e66eb49-82f2-4f66-87ea-90bf889028ae.PNG)
![15](https://user-images.githubusercontent.com/94188547/173534790-fc575fd9-a578-48b4-8ddd-970b5b6845dc.PNG)
![16](https://user-images.githubusercontent.com/94188547/173534793-3b47821a-e87e-4138-a829-6d442bf14411.PNG)
![17](https://user-images.githubusercontent.com/94188547/173534795-5215a256-aa9f-456b-b260-fec51719c976.PNG)
![18](https://user-images.githubusercontent.com/94188547/173534801-65ca6ffd-4fd9-434c-a153-4e634c96f7df.PNG)


#코드

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
import numpy as np 
import matplotlib.pyplot as plt 

mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print('Shape of Train images :',train_images.shape)
print('Shape of Train labels : ', train_labels.shape)
print('\nShape of Test images : ', test_images.shape)
print("Shape of Test labels : ",test_labels.shape)

print('Train labels : ',train_labels)
print(train_images[1])
print('First 10 Train images in MNIST dataset\n')
for i in range(10):
    plt.subplot(1, 10, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(train_images[i])
plt.show()
print('\nTrain labels match with Train label sequentialy\n',train_labels[:10])

train_images = tf.reshape(train_images, [-1, 28, 28, 1])
test_images = tf.reshape(test_images, [-1, 28, 28, 1])

def select_model(model_number):
    if model_number == 1:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2 
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 3

    if model_number == 2:
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape=(28,28,1)),     # layer 1 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3 
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 4
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 5
                    
    if model_number == 3: 
        model = keras.models.Sequential([
                    keras.layers.Conv2D(32, (3,3), activation = 'relu', input_shape = (28, 28,1)),  # layer 1
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 2
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 3
                    keras.layers.Conv2D(64, (3,3), activation = 'relu'),                            # layer 4
                    keras.layers.MaxPool2D((2,2)),                                                  # layer 5
                    keras.layers.Conv2D(128, (3,3), activation = 'relu'),                           # layer 6
                    keras.layers.Flatten(),
                    keras.layers.Dense(10, activation = 'softmax')])                                # layer 7
    
    return model 

model = select_model()
model.summary()

model.compile(
    optimizer = 'adam',
    loss = 'sparse_categorical_crossentropy',
    metrics = ['accuracy']
)

model.fit(train_images, train_labels,  epochs = 5)

test_loss, accuracy = model.evaluate(test_images, test_labels, verbose = 2)
print('\nTest loss : ', test_loss)

test_images = tf.cast(test_images, tf.float32)
pred = model.predict(test_images)
Number = [0,1,2,3,4,5,6,7,8,9]

print('Prediction : ', pred.shape)
print('Test labels : ', test_labels.shape)

def plot_image(i, predictions_array, true_label, img):
  predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])

  plt.imshow(img, cmap=plt.cm.binary)

  predicted_label = np.argmax(predictions_array)
  if predicted_label == true_label:
    color = 'blue'
  else:
    color = 'red'

  plt.xlabel("{} {:2.0f}% ({})".format(Number[predicted_label],
                                100*np.max(predictions_array),
                                Number[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
  predictions_array, true_label = predictions_array[i], true_label[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)
  plt.xticks(Number)

  thisplot[predicted_label].set_color('red')
  thisplot[true_label].set_color('blue')
  
  (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
  
  i = 1
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, pred, test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, pred,  test_labels)
plt.show()
print('Test accuracy :', accuracy)

num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, pred, test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, pred, test_labels)
plt.show()

def error_mnist(prediction_array, true_label):
    error_index = []
    
    for i in range(true_label.shape[0]):
        if np.argmax(prediction_array[i]) != true_label[i]:
            error_index.append(i)
    return error_index

# change num_cols, num_rows if you want to see more result.  
def plot_error(index, prediction_array, true_label):
    num_cols = 5
    num_rows = 5
    plt.figure(figsize=(2*2*num_cols, 2*num_rows))

    assert len(index) < num_cols * num_rows
    for i in range(len(index)):
        plt.subplot(num_rows, 2*num_cols, 2*i+1)
        idx = index[i]
        plt.imshow(test_images[idx])
        plt.subplot(num_rows, 2*num_cols, 2*i+2)
        plt.bar(range(10), prediction_array[idx])
        plt.xticks(Number)
        
index = error_mnist(pred, test_labels)
index_slice = index[:10]
print(index[:10])

plot_error(index_slice, pred, test_labels)
