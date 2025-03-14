from tensorflow.keras import layers, models

def build_model(input_shape=(128, 128, 1)):
    model = models.Sequential([
        # First Conv Block with increased filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), strides=2, activation='relu', padding='same'),  # Replacing MaxPooling

        # Second Conv Block
        layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3, 3), strides=2, activation='relu', padding='same'),

        # Third Conv Block
        layers.Conv2D(512, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(512, (3, 3), strides=2, activation='relu', padding='same'),

        # Global Pooling Instead of Flatten
        layers.GlobalAveragePooling2D(),
        
        # Fully Connected Layers (Reduced Size)
        layers.Dense(128, activation='relu'),  # Reduced from 256 to 128
        layers.Dropout(0.5),
        layers.BatchNormalization(),

        layers.Dense(8, activation='softmax')  # Adjust output size for your task
    ])
    return model
