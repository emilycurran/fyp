import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np

import paper_model 
import paper_model_1d 
import utils

# get the directory where the script is located
parent_dir = os.path.dirname(os.path.abspath(__file__))
print("running")

def run_experiment(num_epochs, batch_size, train_split=0.8):
    
    #get the data 
    train_imgs, val_imgs, train_series, val_series, train_labels, val_labels, _ = utils.split_data() 
     
    # define the learning rate schedule
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate = 0.0001,  
        decay_steps= 1000,  
        decay_rate=0.95,
        staircase=False  
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule)

    # ___________________________________2d_____________________________________________

    print("training 2D cnn")

    model_2d = paper_model.build_model(input_shape = (128, 128, 1))
    model_2d.compile(optimizer=optimizer, loss ="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=True)

    history_2d = model_2d.fit(train_imgs, train_labels, validation_data=(val_imgs, val_labels), epochs=num_epochs)
    for layer in model_2d.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # Freeze BatchNorm so it stops updating running statistics

    model_2d.save(os.path.join(parent_dir, "2D_CNN.h5"), include_optimizer = False)

    # save training results
    plt.figure(figsize=(8, 6))
    plt.plot(history_2d.history["loss"], label= "Training Loss")
    plt.plot(history_2d.history["val_loss"], label="Validation Loss")
    plt.title("2D Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(f"2D_CNN_Loss.png"))
    plt.close() 

    plt.figure(figsize =(8, 6))
    plt.plot(history_2d.history["accuracy"], label="Training Accuracy")
    plt.plot(history_2d.history["val_accuracy"], label="Validation Accuracy")
    plt.title("2D Accuracy vs Epochs")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(parent_dir, "2D_CNN_Accuracy.png"))
    plt.close()    

    #_________________________________________1d_______________________________________________
    print("training 1d cnn") 

    model_1d = paper_model_1d.build_model(input_shape =(256, 1)) 
    model_1d.compile(optimizer= optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"], jit_compile=True)


    history_1d = model_1d.fit(train_series, train_labels, validation_data=(val_series, val_labels), epochs=num_epochs)
    for layer in model_1d.layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False  # Freeze BatchNorm so it stops updating running statistics


    model_1d.save(os.path.join(parent_dir, "1D_CNN.h5"), include_optimizer = False)

    # save trainin results
    plt.figure(figsize=(8, 6))
    plt.plot(history_1d.history["loss"], label= "Training Loss") 
    plt.plot(history_1d.history["val_loss"], label= "Validation Loss")
    plt.title("Time-Series Model Loss vs Epoch")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(parent_dir, f"1D_CNN_Loss.png"))
    plt.close()

    plt.figure(figsize= (8, 6))
    plt.plot(history_1d.history["accuracy"], label="Training Accuracy") 
    plt.plot(history_2d.history["val_accuracy"], label= "Validation Accuracy")
    plt.title("1D CNN Accuracy vs Epochs")
    plt.legend()  
    plt.grid(True) 
    plt.savefig(os.path.join(parent_dir, "1D_CNN_Accuracy.png"))
    plt.close() 

    return

def main():
    run_experiment(num_epochs = 10
        , batch_size = 16)

if __name__ == "__main__":
    main()