import pandas as pd
import nltk
import keras
from keras import layers
from keras import callbacks
from sklearn.model_selection import train_test_split


#TODO: Text cleaning
# For this dummy script, the target label will just be 'target'
tweets=[['text1', 'label1'], ['text2', 'label2']] # Import our labeled tweets here
df = pd.DataFrame(tweets)

for tweet in df[:, 0]: # Text column of the pd df
    tweet = nltk.word_tokenize(tweet) # Tokenize each labeled tweet
    # (in the sample script they just clean rather than tokenizing but I think it would be easier to tokenize)
    tweet = ' '.join(tweet) # Convert the tokenized tweet back into a string (we can debate whether to do this)

targets = pd.get_dummies(df['targets'].values) # Convert the labels into integers, if they aren't already integers

#TODO: Split the data into train and test data
X = [] # Extract the tweet text from the pd df (this is circular, we can amend)
Y=[] # Extract the target values from the pd df
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.10, random_state = 42)


#TODO: Decide on MAX_NB_WORDS (the vocabulary size - most frequent words) and EMBEDDING_DIM (# layers)
MAX_NB_WORDS = 50000
EMBEDDING_DIM = 100


#TODO: This is where the model begins:
model = keras.Sequential()
model.add(layers.Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
model.add(layers.SpatialDropout1D(0.2))
model.add(layers.LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(layers.Dense(13, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 5
batch_size = 64

history = model.fit(X_train, Y_train, epochs=epochs, batch_size=batch_size,
                    validation_split=0.1,callbacks=[callbacks.EarlyStopping(monitor='val_loss', patience=3, min_delta=0.0001)])

# Evaluate accuracy of the model
accr = model.evaluate(X_test,Y_test)
print('Test set\n  Loss: {:0.3f}\n  Accuracy: {:0.3f}'.format(accr[0],accr[1]))