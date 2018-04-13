#  coding: utf8
import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout, Flatten
"""
Đoạn test này nhằm kiểm tra model có thể nhớ dữ liệu đã train qua nhiều vòng train với dữ liệu khác nhau hay không. Tạo 4 bộ dự liệu, 
cho chạy 4 vòng train, kiểm tra kết quả
nếu đoán đúng thì model có thể train nhiều tập dữ liệu khác nhau 
"""
# Tao du liệu
x1 = np.ndarray(shape=(8,2),dtype=int,buffer=np.array([[1,1],[2,2],[3,3],[4,4],[1,1],[2,2],[3,3],[4,4]]))
y1 = np.ndarray(shape=(8,1),dtype=int,buffer=np.array([1,1,1,1,1,1,1,1]))

x2 = np.ndarray(shape=(8,2),dtype=int,buffer=np.array([[-1,1],[-2,2],[-3,3],[-4,4],[-1,1],[-2,2],[-3,3],[-4,4]]))
y2 = np.ndarray(shape=(8,1),dtype=int,buffer=np.array([2,2,2,2,2,2,2,2]))

x3 = np.ndarray(shape=(4,2),dtype=int,buffer=np.array([[-1,-1],[-2,-2],[-3,-3],[-4,-4]]))
y3 = np.ndarray(shape=(4,1),dtype=int,buffer=np.array([3,3,3,3]))

x4 = np.ndarray(shape=(8,2),dtype=int,buffer=np.array([[1,-1],[2,-2],[3,-3],[4,-4],[1,-1],[2,-2],[3,-3],[4,-4]]))
y4 = np.ndarray(shape=(8,1),dtype=int,buffer=np.array([4,4,4,4,4,4,4,4]))

x_test = np.ndarray(shape=(24,2),dtype=int,buffer=np.array([[2,5],[-6,7],[-8,-9],[6,-4],
                                                           [11,11],[-15,15],[-16,-13],[16,-14],
                                                           [21,21],[-25,25],[-26,-23],[26,-24],
                                                           [31,31],[-35,35],[-36,-33],[36,-34],
                                                            [42,45],[-46,47],[-48,-49],[46,-44],
                                                            [52,55],[-56,57],[-58,-59],[56,-54],
                                                           ]))
# convert class vectors to binary class matrices
num_classes = 5
y1 = keras.utils.to_categorical(y1, num_classes)
y2 = keras.utils.to_categorical(y2, num_classes)
y3 = keras.utils.to_categorical(y3, num_classes)
y4 = keras.utils.to_categorical(y4, num_classes)
"""
#tao model learning
model = Sequential()

model = Sequential()
model.add(Dense(32, input_dim=2))
model.add(Dropout(0.5))
#model.add(Flatten())
model.add(Dense(num_classes, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              )

model.fit(x1,y1, batch_size=1, epochs=50, verbose=1)
model.fit(x2,y2, batch_size=1, epochs=50, verbose=1)
model.fit(x3,y3, batch_size=1, epochs=50, verbose=1)
model.fit(x4,y4, batch_size=1, epochs=50, verbose=1)
print (model.predict_classes(x_test))
"""
train_data = [[x1,y1],[x2,y2],[x3,y3],[x4,y4]]
model_list = []
for x,y in train_data:
	model = Sequential()
	model.add(Dense(32, input_dim=2))
	model.add(Dropout(0.5))
	#model.add(Flatten())
	model.add(Dense(num_classes, activation='softmax'))
	model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              )
	model.fit(x,y, batch_size=1, epochs=50, verbose=1)
	model_list.append(model)
