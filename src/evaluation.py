from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import os

def evaluate_model(model, X_test, y_test, labels, save_path="../assets/visuals/confusion_matrix_svm.png"):
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    print(f"📈 Accuracy: {acc*100:.2f}%")
    print(classification_report(y_test, y_pred, target_names=labels))
    
    cm = confusion_matrix(y_test, y_pred)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=labels,
                yticklabels=labels)
    plt.title("Confusion Matrix — SVM")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()
