import tensorflow as tf  
import numpy as np 
import matplotlib.pyplot as plt  
import seaborn as sns 
from sklearn.metrics import confusion_matrix,  classification_report 
import os
 
import utils 

parent_dir = os.path.dirname(os.path.abspath(__file__) )  

def evaluate_model(model, x_test, y_test):
    y_pred_probs = model.predict(x_test)  # get probability scores
    y_pred = np.argmax(y_pred_probs,  axis=1)  # convert to class labels
    return y_pred 

def plot_confusion_matrix_log(y_true, y_pred, model_name): 
    # apply log scalingfor colours, but keep original counts for display
    cm = confusion_matrix(y_true, y_pred)

    cm_log = np.log1p( cm)#log transform for colour scaling

    plt.figure(figsize=(8,  6) )
    ax = sns.heatmap(cm_log, annot=cm, fmt="d" , cmap="inferno", cbar=True, 
                     xticklabels=sorted(set(y_true)), yticklabels = sorted(set(y_true)))

    plt.xlabel("Predicted Label")   
    plt.ylabel("True Label")  
    plt.title(f"Confusion Matrix: {model_name} (Log Color Scale, Actual Counts)") 
    plt.savefig(os.path.join( parent_dir, f"{model_name}.png"))
    print("fig saved")

def main():
    model_1d = os.path.join(parent_dir, "1D_CNN")  
    model_1d  = tf.keras.models.load_model(os.path.join(parent_dir , model_1d))

    model_2d = os.path.join(parent_dir, "2D_CNN")  
    model_2d =  tf.keras.models.load_model(os.path.join( parent_dir, model_2d))
    print("successfully load_model for testing")

    _, test_imgs, _, test_series, _ , test_labels, label_map = utils.split_data()
    y_pred_2d = evaluate_model(model_2d,test_imgs, test_labels)
    plot_confusion_matrix_log(test_labels, y_pred_2d, "2D confusion matrix")

    y_pred_test_1d = evaluate_model(model_1d,test_series, test_labels)
    plot_confusion_matrix_log(test_labels , y_pred_test_1d, "1D confusion matrix")

if __name__ ==  "__main__":
    main()
