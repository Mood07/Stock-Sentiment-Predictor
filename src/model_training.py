from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC

def train_svm_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )
    model = LinearSVC()
    model.fit(X_train, y_train)
    
    print("🎯 SVM model training completed!")
    return model, X_train, X_test, y_train, y_test
