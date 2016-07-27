from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
import gym

atari = gym.make('SpaceInvaders-v0')
atari.reset()


model = Sequential()
model.add(Dense(200, input_shape=(124800,), activation="relu"))
model.add(Dense(200, activation="relu"))
model.add(Dense(6))


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])


