from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Reshape
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers.merge import concatenate
from tensorflow.keras import utils
from keras import regularizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, Conv1D, MaxPooling1D, Flatten, Dropout, BatchNormalization, Reshape, GlobalAveragePooling1D
from keras.callbacks import LearningRateScheduler
from keras import regularizers
from sklearn.metrics import accuracy_score

def apply_CNN(X, y, channel):
    X_scaled = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    y_train = utils.to_categorical(y_train, 2)
    y_test = utils.to_categorical(y_test, 2)
    # reshape for fitting a model in a way: (number_of_samples, number_of_time_points, number_of_channels)
    if channel == 'single':
        length = X_scaled.shape[1]
        chan_num = 1
    elif channel == 'multi':
        length = X_scaled.shape[1]
        chan_num =
    X_train = X_train.reshape(-1,length,chan_num)
    X_test = X_test.reshape(-1,length,chan_num)
    # create a model
    model = Sequential()
    # create first convolutional layer with 32 filters of sliding window width 5
    model.add(Conv1D(32, 5, activation='relu', input_shape=(X_train.shape[1:])))
    # additional convolutional layer allows model to learn more complex features
    model.add(Conv1D(32, 5, activation='relu'))
    # use max pooling to reduce size of feature matrix
    model.add(MaxPooling1D())
    # the same for second set of convolutional layers
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(Conv1D(64, 5, activation='relu'))
    model.add(GlobalAveragePooling1D())
    # last layer
    model.add(Dense(2, activation='softmax'))
    # model should be compiled before fitting
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    print(model.summary())
    # TRAIN NETWORKS
    epochs = 20
    history = model.fit(X_train,y_train, epochs = epochs, validation_split = 0.1, verbose=1)
    print("CNN: Epochs={0:d}, Train accuracy={1:.5f}, Validation accuracy={2:.5f}".format(epochs,max(history.history['acc']),max(history.history['val_acc']) ))
    y_test_pred = np.argmax(y_test, axis = 1)
    print('Accuracy: %s' %accuracy_score(y_test, y_test_pred))
    return 0

