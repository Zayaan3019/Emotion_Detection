from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, LSTM, Dense, Dropout, concatenate
from tensorflow.keras.models import Model

def build_emotion_model(input_shape=(48, 48, 1), lstm_units=64, num_classes=7):
    """
    Multi-branch CNN-BiLSTM model for emotion detection.
    """
    # CNN Branch
    cnn_input = Input(shape=input_shape)
    cnn_base = ResNet50(weights=None, include_top=False, input_tensor=cnn_input)
    cnn_output = Flatten()(cnn_base.output)

    # BiLSTM Branch
    lstm_input = Input(shape=(48, 48))  # Placeholder for temporal data
    lstm = LSTM(lstm_units, return_sequences=True)(lstm_input)
    lstm_output = LSTM(lstm_units)(lstm)

    # Concatenate and classify
    combined = concatenate([cnn_output, lstm_output])
    dense = Dense(128, activation='relu')(combined)
    dropout = Dropout(0.5)(dense)
    output = Dense(num_classes, activation='softmax')(dropout)

    return Model(inputs=[cnn_input, lstm_input], outputs=output)
