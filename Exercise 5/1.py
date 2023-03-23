import datetime
log_folder = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

import tensorflow as tf

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

from tensorflow.keras.callbacks import TensorBoard

callbacks = [TensorBoard(log_dir=log_folder,
                         histogram_freq=1,
                         write_graph=True,
                         write_images=True,
                         update_freq='epoch',
                         profile_batch=2,
                         embeddings_freq=1)]

try:
   model =  tf.keras.models.load_model('tf/mnist')
   print('Loading Model')
except:
   print('Creating Model')
   model = tf.keras.models.Sequential([
      tf.keras.layers.Flatten(input_shape=(28, 28)),
      tf.keras.layers.Dense(512, activation='relu'),
      tf.keras.layers.Dropout(0.2),
      tf.keras.layers.Dense(10, activation='softmax')])

   model.compile(optimizer='sgd',
      loss='sparse_categorical_crossentropy',
      metrics=['accuracy'])
   model.fit(X_train, y_train,
            epochs=10,
            validation_split=0.2,
            callbacks=callbacks)
   model.save('tf/mnist')
#%load_ext tensorboard
log_folder = 'logs'
#%tensorboard --logdir={log_folder}