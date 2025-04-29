import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
import joblib

# Set style for plots
plt.style.use('default')

# Function to parse filename
def parse_filename(filename):
    movement_code = filename.split('_')[-1]
    joint = movement_code[0]  # A, K, M
    orientation = movement_code[1]  # T, R, M
    direction = movement_code[2]  # P, N, M
    range_type = movement_code[3]  # F, H
    leg = movement_code[4]  # L, R
    
    return {
        'joint': joint,
        'orientation': orientation,
        'direction': direction,
        'range_type': range_type,
        'leg': leg
    }

# Function to load transform data
def load_transform_data(base_path):
    features = []
    labels = []
    
    for patient_folder in os.listdir(base_path):
        if not patient_folder.startswith('t_f_'):
            continue
            
        patient_path = os.path.join(base_path, patient_folder)
        
        for movement_folder in os.listdir(patient_path):
            if not movement_folder.startswith('t_f_'):
                continue
                
            movement_path = os.path.join(patient_path, movement_folder)
            movement_info = parse_filename(movement_folder)
            
            # Load Fourier transform data
            ft_path = os.path.join(movement_path, 'fourier_transform')
            if os.path.exists(ft_path):
                ft_file = None
                for file in os.listdir(ft_path):
                    if file.endswith('.xlsx') and 'fourier_transform_results' in file:
                        ft_file = os.path.join(ft_path, file)
                        break
                
                if ft_file:
                    try:
                        ft_data = pd.read_excel(ft_file)
                        # Convert to numpy array and flatten
                        ft_features = ft_data.values.flatten()
                        
                        # Load corresponding wavelet transform data
                        wt_path = os.path.join(movement_path, 'wavelet_transform')
                        if os.path.exists(wt_path):
                            wt_file = None
                            for file in os.listdir(wt_path):
                                if file.endswith('.xlsx') and 'wavelet_transform_results' in file:
                                    wt_file = os.path.join(wt_path, file)
                                    break
                            
                            if wt_file:
                                try:
                                    wt_data = pd.read_excel(wt_file)
                                    # Convert to numpy array and flatten
                                    wt_features = wt_data.values.flatten()
                                    
                                    # Combine features
                                    combined_features = np.concatenate([ft_features, wt_features])
                                    features.append(combined_features)
                                    labels.append(movement_info['joint'])
                                    print(f"Successfully loaded data for {movement_folder}")
                                except Exception as e:
                                    print(f"Error loading wavelet transform data for {movement_folder}: {str(e)}")
                    except Exception as e:
                        print(f"Error loading Fourier transform data for {movement_folder}: {str(e)}")
    
    if not features:
        raise ValueError("No data was successfully loaded. Please check the file paths and formats.")
    
    return np.array(features), np.array(labels)

# Function to plot and save learning curves
def plot_learning_curve(estimator, X, y, title, save_path):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=5, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10)
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_mean, label='Training score', color='blue', linewidth=2)
    plt.plot(train_sizes, test_mean, label='Cross-validation score', color='red', linewidth=2)
    
    plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color='blue')
    plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color='red')
    
    plt.title(title, fontsize=14, pad=20)
    plt.xlabel('Training Examples', fontsize=12)
    plt.ylabel('Score', fontsize=12)
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Function to plot and save confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes, save_path):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix', fontsize=14, pad=20)
    plt.colorbar()
    
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, fontsize=12)
    plt.yticks(tick_marks, classes, fontsize=12)
    
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                    horizontalalignment="center",
                    verticalalignment="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12)
    
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

# Main execution
print("Loading data...")
base_path = 'ft_wt_transform'
try:
    X, y = load_transform_data(base_path)
    print(f"Successfully loaded {len(X)} samples")
except Exception as e:
    print(f"Error loading data: {str(e)}")
    exit(1)

print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Applying PCA for dimensionality reduction...")
pca = PCA(n_components=0.95)  # Keep 95% of variance
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

print("Training hybrid model...")
# Base models
rf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42)
gb = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
svm = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
mlp = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=1000, random_state=42)

# Create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('gb', gb),
        ('svm', svm),
        ('mlp', mlp)
    ],
    voting='soft'  # Use soft voting for probability estimates
)

# Create pipeline
model = Pipeline([
    ('scaler', scaler),
    ('pca', pca),
    ('voting', voting_clf)
])

# Plot and save learning curves
print("Generating learning curves...")
plot_learning_curve(model, X_train, y_train, 
                   'Learning Curves (Hybrid Model)', 
                   'learning_curves.png')

# Train the model
print("Training the model...")
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Plot and save confusion matrix
print("Generating confusion matrix...")
plot_confusion_matrix(y_test, y_pred, ['A', 'K', 'M'], 'confusion_matrix.png')

# Save the model
print("Saving model...")
joblib.dump(model, 'joint_movement_classifier.joblib')

print("\nAll visualizations and models have been saved successfully!")
print("Files created:")
print("- learning_curves.png")
print("- confusion_matrix.png")
print("- joint_movement_classifier.joblib")

# Print model description
print("\nModel Description:")
print("This hybrid model combines multiple machine learning algorithms:")
print("1. Random Forest: 200 trees with max depth of 20")
print("2. Gradient Boosting: 200 estimators with learning rate 0.1")
print("3. Support Vector Machine: RBF kernel with C=10")
print("4. Neural Network: MLP with hidden layers (100, 50)")
print("\nPreprocessing steps:")
print("1. Standard Scaling: Normalize features to zero mean and unit variance")
print("2. PCA: Dimensionality reduction keeping 95% of variance")
print("\nEnsemble Method:")
print("Soft Voting: Combines probability estimates from all base models")