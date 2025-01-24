from tensorflow.keras.optimizers import Adam
from models.emotion_model import build_emotion_model
from preprocess import preprocess_image, augment_data
from sklearn.model_selection import train_test_split

def train_model(data_path, epochs=20, batch_size=32):
    # Load and preprocess data
    X, y = load_data(data_path)  # Implement `load_data` in utils.py
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Build model
    model = build_emotion_model(input_shape=(48, 48, 1), num_classes=7)

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Train model with data augmentation
    train_generator = augment_data(X_train, y_train)
    history = model.fit(
        train_generator,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size
    )

    # Save model
    model.save('models/emotion_model.h5')
    return history
