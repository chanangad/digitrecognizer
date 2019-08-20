#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import pandas as pd
from keras import backend as K


# In[ ]:


train_data = pd.read_csv('../input/train.csv').values
test_data = pd.read_csv('../input/test.csv').values


# In[ ]:


train_x = train_data[:,1:].reshape(train_data.shape[0],1,28,28)
test_x = test_data.reshape(test_data.shape[0],1,28,28)
X_train = train_x / 255.0
X_test = test_x /255.0
y_train = train_data[:,0]


# In[ ]:


X_test[0]


# In[ ]:


from sklearn import preprocessing
lb = preprocessing.LabelBinarizer()
y_train = lb.fit_transform(y_train)


# In[ ]:


X_train.shape


# In[ ]:


classifier = Sequential()

# Step 1 - Convolution
classifier.add(Conv2D(16, (3, 3), input_shape = (1,28,28), activation = 'relu',data_format='channels_first'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=2))

# Adding a second convolutional layer
classifier.add(Conv2D(16, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2),strides=2))

# Step 3 - Flattening
classifier.add(Flatten())

# Step 4 - Full connection
classifier.add(Dense(units = 64, activation = 'relu'))
classifier.add(Dense(units = 10, activation = 'sigmoid'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])


# In[ ]:


42000/140


# In[ ]:


classifier.fit(X_train, y_train,
          epochs=20,
          batch_size= 140)


# In[ ]:


a= classifier.predict(X_test)
a


# In[ ]:


y = [i.argmax() for i in a]


# In[ ]:


ans = pd.DataFrame(columns=['ImageId','Label'])
ans


# In[ ]:


len(y)


# In[ ]:


ans['ImageId'] = pd.Series(range(1,28001))


# In[ ]:


ans['Label'] = pd.Series(y)


# In[ ]:


ans


# In[ ]:


ans.to_csv('submission.csv',index=False)


# In[ ]:


<a href="submission.csv"> Download File </a>


# <a href="submission.csv"> Download File </a>
# 
# 

# In[ ]:




