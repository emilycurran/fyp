from tensorflow.keras import layers, models

def build_model(input_shape=(256, 1)):
    model = models.Sequential([
        # First Conv Block with increased filters
        layers.Conv1D(128, 5, activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv1D(128, 5, strides=2, activation='relu', padding='same'),  # Replacing MaxPooling

        # Second Conv Block
        layers.Conv1D(256, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(256, 5, strides=2, activation='relu', padding='same'),

        # Third Conv Block
        layers.Conv1D(512, 5, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv1D(512, 5, strides=2, activation='relu', padding='same'),

        # Global Pooling Instead of Flatten
        layers.GlobalAveragePooling1D(),
        
        # Fully Connected Layers (Reduced Size)
        layers.Dense(128, activation='relu'),  # Reduced from 256 to 128
        layers.Dropout(0.5),
        layers.BatchNormalization(),

        layers.Dense(8, activation='softmax')  # Adjust output size for your task
    ])
    return model
