from keras.datasets import imdb 
import numpy as np 
from keras import models
from keras import layers
from keras import optimizers
import matplotlib.pyplot as plt
#载入数据 
(train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000) 
#print(train_data[0] ,"------------\n",train_labels[:10])
#print(max([max(seq) for seq in train_data]))
#准备数据:one-hot编码
#将整数序列编码为二进制矩阵 
def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))      
    for i, sequence in enumerate(sequences): 
        #enumerate() 函数用于将一个可遍历的数据对象组合为一个索引序列，同时列出数据和数据下标        
        results[i, sequence] = 1.      
    return results 
#数据向量化
x_train = vectorize_sequences(train_data)   
x_test = vectorize_sequences(test_data)

#标签向量化
y_train = np.asarray(train_labels).astype('float32') 
#array仍会copy出一个副本，占用新的内存，但asarray不会
y_test = np.asarray(test_labels).astype('float32')

#构建网络 
model = models.Sequential()
model.add(layers.Dense(16, activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))


#验证
#留出验证集 
x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_test[10000:]

#编译模型
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
#history = model.fit(partial_x_train,partial_y_train,epochs=4,batch_size=512,validation_data=(x_val,y_val))


model.fit(x_train, y_train, epochs=4, batch_size=512) 
results = model.evaluate(x_test, y_test)


'''
 history_dict = history.history 
 history_dict.keys() 
 dict_keys(['val_acc', 'acc', 'val_loss', 'loss']) 
 字典中包含4 个条目，对应训练过程和验证过程中监控的
'''
'''
#　绘制训练损失和验证损失 
history_dict = history.history
loss_values = history_dict['loss'] 
val_loss_values = history_dict['val_loss'] 
epochs = range(1, len(loss_values) + 1) 
plt.plot(epochs, loss_values, 'bo', label='Training loss') 
plt.plot(epochs, val_loss_values, 'b', label='Validation loss') 
plt.title('Training and validation loss') 
plt.xlabel('Epochs') 
plt.ylabel('Loss') 
plt.legend() 
plt.show()

plt.clf()    
acc = history_dict['acc']  
val_acc = history_dict['val_acc'] 
plt.plot(epochs, acc, 'bo', label='Training acc') 
plt.plot(epochs, val_acc, 'b', label='Validation acc') 
plt.title('Training and validation accuracy') 
plt.xlabel('Epochs') 
plt.ylabel('Accuracy') 
plt.legend() 
plt.show()



print('test:\n',model.predict(x_test) )
'''