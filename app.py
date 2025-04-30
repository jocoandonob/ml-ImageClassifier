import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import tempfile
from PIL import Image
import io
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import seaborn as sns

# Set page configuration
st.set_page_config(
    page_title="Image Classification with Scikit-Learn",
    page_icon="🖼️",
    layout="wide"
)

# Initialize session state variables if they don't exist
if 'model' not in st.session_state:
    st.session_state.model = None
if 'pca' not in st.session_state:
    st.session_state.pca = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'class_names' not in st.session_state:
    st.session_state.class_names = None
if 'test_images' not in st.session_state:
    st.session_state.test_images = None
if 'test_labels' not in st.session_state:
    st.session_state.test_labels = None
if 'trained' not in st.session_state:
    st.session_state.trained = False

# App title and description
st.title("Image Classification with Scikit-Learn")
st.markdown("""
This application allows you to train and test machine learning models for image classification. 
You can upload your own images or use sample images for classification.
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Dataset Preparation", "Model Training", "Model Evaluation", "Prediction"])

# Helper function to load and preprocess images
def load_and_preprocess_images(image_files, target_size=(32, 32), flatten=True):
    """
    Load and preprocess images from a list of file paths or uploaded files.
    
    Args:
        image_files: List of image files (file paths or uploaded files)
        target_size: Size to resize images to
        flatten: Whether to flatten the image to a 1D array
        
    Returns:
        List of preprocessed images
    """
    images = []
    for img_file in image_files:
        try:
            # Open and convert to RGB
            img = Image.open(img_file).convert('RGB')
            # Resize
            img = img.resize(target_size)
            # Convert to numpy array and normalize
            img_array = np.array(img) / 255.0
            
            if flatten:
                # Flatten the image to a 1D array
                img_array = img_array.reshape(-1)
            
            images.append(img_array)
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    return np.array(images)

# Helper function to plot sample images
def plot_sample_images(images, labels, class_names, num_rows=2, num_cols=5):
    """
    Plot a grid of sample images with their class labels.
    
    Args:
        images: Array of images
        labels: Array of class indices
        class_names: List of class names
        num_rows: Number of rows in the grid
        num_cols: Number of columns in the grid
        
    Returns:
        Matplotlib figure
    """
    num_images = min(num_rows * num_cols, len(images))
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(12, 5))
    
    for i, ax in enumerate(axes.flat):
        if i < num_images:
            # Reshape the flattened image back to 2D for display if needed
            if len(images[i].shape) == 1:
                img_reshape = images[i].reshape((32, 32, 3))
                ax.imshow(img_reshape)
            else:
                ax.imshow(images[i])
                
            ax.set_title(class_names[labels[i]])
            ax.axis('off')
        else:
            ax.axis('off')
    
    plt.tight_layout()
    return fig

# Helper function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    """
    Plot a confusion matrix.
    
    Args:
        y_true: Array of true class indices
        y_pred: Array of predicted class indices
        class_names: List of class names
        
    Returns:
        Matplotlib figure
    """
    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Plot confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    return fig

# Dataset Preparation Page
if page == "Dataset Preparation":
    st.header("Dataset Preparation")
    
    # Create two tabs for different dataset options
    dataset_tab = st.tabs(["Upload Images"])
    
    with dataset_tab[0]:  # Upload Images Tab
        st.info("Upload multiple images for different classes.")
        
        # Input for class names
        class_input = st.text_input("Enter class names (comma separated)", "cat,dog")
        class_names = [c.strip() for c in class_input.split(',')]
        
        # Create columns for each class
        cols = st.columns(len(class_names))
        uploaded_images = {}
        
        for i, col in enumerate(cols):
            with col:
                st.write(f"Upload {class_names[i]} images:")
                uploaded_images[class_names[i]] = st.file_uploader(
                    f"Choose {class_names[i]} images",
                    type=["jpg", "jpeg", "png"],
                    accept_multiple_files=True,
                    key=f"upload_{class_names[i]}"
                )
        
        # Process uploaded images
        if st.button("Process Images"):
            with st.spinner("Processing images..."):
                # Check if images were uploaded
                has_images = any(len(imgs) > 0 for imgs in uploaded_images.values())
                
                if not has_images:
                    st.warning("Please upload at least one image for each class.")
                else:
                    # Prepare data for training
                    X = []  # Images
                    y = []  # Labels
                    
                    for class_idx, class_name in enumerate(class_names):
                        if len(uploaded_images[class_name]) > 0:
                            # Process images for this class
                            class_images = load_and_preprocess_images(uploaded_images[class_name])
                            X.extend(class_images)
                            y.extend([class_idx] * len(class_images))
                    
                    X = np.array(X)
                    y = np.array(y)
                    
                    # Split into train and test sets
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=0.2, random_state=42
                    )
                    
                    # Store in session state
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.class_names = class_names
                    
                    # Get a subset of test images for display
                    test_display_indices = np.random.choice(
                        len(X_test), 
                        size=min(10, len(X_test)), 
                        replace=False
                    )
                    
                    st.session_state.test_display_images = X_test[test_display_indices]
                    st.session_state.test_display_labels = y_test[test_display_indices]
                    
                    st.success(f"Dataset prepared with {len(X_train)} training images and {len(X_test)} test images.")
                    
                    # Display sample images
                    if len(st.session_state.test_display_images) > 0:
                        st.subheader("Sample Test Images")
                        fig = plot_sample_images(
                            [img.reshape(32, 32, 3) for img in st.session_state.test_display_images],
                            st.session_state.test_display_labels,
                            class_names
                        )
                        st.pyplot(fig)

# Model Training Page
elif page == "Model Training":
    st.header("Model Training")
    
    if 'X_train' not in st.session_state:
        st.warning("Please prepare the dataset first before training the model.")
    else:
        st.success(f"Dataset is ready with {len(st.session_state.class_names)} classes: {', '.join(st.session_state.class_names)}")
        
        # Model configuration
        st.subheader("Model Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            model_type = st.selectbox(
                "Select Model Type", 
                ["Support Vector Machine", "Random Forest"]
            )
            
            # Feature reduction with PCA
            use_pca = st.checkbox("Use PCA for dimensionality reduction", value=True)
            if use_pca:
                n_components = st.slider(
                    "Number of PCA components", 
                    min_value=10, 
                    max_value=min(100, st.session_state.X_train.shape[1]),
                    value=50
                )
            
        with col2:
            if model_type == "Support Vector Machine":
                kernel = st.selectbox("Kernel", ["linear", "rbf", "poly", "sigmoid"], index=1)
                C = st.select_slider(
                    "Regularization parameter (C)",
                    options=[0.1, 1.0, 10.0, 100.0],
                    value=1.0
                )
            else:  # Random Forest
                n_estimators = st.slider("Number of trees", 10, 200, 100)
                max_depth = st.slider("Maximum depth", 3, 20, 10)
        
        # Training button
        if st.button("Train Model"):
            with st.spinner("Training model..."):
                try:
                    # Scale the data
                    scaler = StandardScaler()
                    X_train_scaled = scaler.fit_transform(st.session_state.X_train)
                    st.session_state.scaler = scaler
                    
                    # Apply PCA if selected
                    if use_pca:
                        pca = PCA(n_components=n_components)
                        X_train_pca = pca.fit_transform(X_train_scaled)
                        st.session_state.pca = pca
                    else:
                        X_train_pca = X_train_scaled
                        st.session_state.pca = None
                    
                    # Create and train the model
                    if model_type == "Support Vector Machine":
                        model = SVC(kernel=kernel, C=C, probability=True)
                        model_info = f"SVM with {kernel} kernel, C={C}"
                    else:  # Random Forest
                        model = RandomForestClassifier(
                            n_estimators=n_estimators, 
                            max_depth=max_depth, 
                            random_state=42
                        )
                        model_info = f"Random Forest with {n_estimators} trees, max_depth={max_depth}"
                    
                    # Train the model
                    model.fit(X_train_pca, st.session_state.y_train)
                    
                    # Store model in session state
                    st.session_state.model = model
                    st.session_state.model_info = model_info
                    st.session_state.trained = True
                    
                    st.success("Model trained successfully!")
                    st.info(f"Model: {model_info}")
                    
                except Exception as e:
                    st.error(f"Error training model: {str(e)}")

# Model Evaluation Page
elif page == "Model Evaluation":
    st.header("Model Evaluation")
    
    if not st.session_state.trained or st.session_state.model is None:
        st.warning("Please train a model first before evaluation.")
    else:
        st.success(f"Model is ready for evaluation: {st.session_state.model_info}")
        
        if st.button("Evaluate Model"):
            with st.spinner("Evaluating model on test dataset..."):
                try:
                    # Preprocess the test data
                    X_test_scaled = st.session_state.scaler.transform(st.session_state.X_test)
                    
                    # Apply PCA if it was used in training
                    if st.session_state.pca is not None:
                        X_test_pca = st.session_state.pca.transform(X_test_scaled)
                    else:
                        X_test_pca = X_test_scaled
                    
                    # Get predictions
                    y_pred = st.session_state.model.predict(X_test_pca)
                    
                    # Calculate accuracy
                    accuracy = accuracy_score(st.session_state.y_test, y_pred)
                    
                    # Display results
                    st.subheader("Evaluation Results")
                    st.metric("Test Accuracy", f"{accuracy:.2%}")
                    
                    # Display classification report
                    st.subheader("Classification Report")
                    class_report = classification_report(
                        st.session_state.y_test, 
                        y_pred, 
                        target_names=st.session_state.class_names
                    )
                    st.text(class_report)
                    
                    # Display confusion matrix
                    st.subheader("Confusion Matrix")
                    fig = plot_confusion_matrix(st.session_state.y_test, y_pred, st.session_state.class_names)
                    st.pyplot(fig)
                    
                    # Display some predictions on test samples
                    st.subheader("Sample Predictions")
                    
                    if hasattr(st.session_state, 'test_display_images') and len(st.session_state.test_display_images) > 0:
                        # Process display test images
                        display_images_scaled = st.session_state.scaler.transform(st.session_state.test_display_images)
                        
                        if st.session_state.pca is not None:
                            display_images_pca = st.session_state.pca.transform(display_images_scaled)
                        else:
                            display_images_pca = display_images_scaled
                        
                        # Get predictions
                        display_preds = st.session_state.model.predict(display_images_pca)
                        
                        # Show images with predictions
                        cols = st.columns(5)
                        for i, col in enumerate(cols):
                            if i < len(display_preds):
                                col.image(
                                    st.session_state.test_display_images[i].reshape(32, 32, 3), 
                                    caption=f"True: {st.session_state.class_names[st.session_state.test_display_labels[i]]}\nPred: {st.session_state.class_names[display_preds[i]]}", 
                                    use_column_width=True
                                )
                        
                except Exception as e:
                    st.error(f"Error evaluating model: {str(e)}")

# Prediction Page
elif page == "Prediction":
    st.header("Make Predictions on New Images")
    
    if not st.session_state.trained or st.session_state.model is None:
        st.warning("Please train a model first before making predictions.")
    else:
        st.success(f"Model is ready for predictions: {st.session_state.model_info}")
        
        st.write("Upload an image to classify:")
        
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            # Display the uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess the image
            img = image.resize((32, 32))
            img_array = np.array(img) / 255.0
            
            # Handle grayscale images
            if len(img_array.shape) == 2:
                img_array = np.stack((img_array,) * 3, axis=-1)
            # Handle RGBA images
            elif img_array.shape[2] == 4:
                img_array = img_array[:, :, :3]
                
            # Flatten the image
            img_array = img_array.reshape(1, -1)
            
            if st.button("Predict"):
                with st.spinner("Making prediction..."):
                    try:
                        # Preprocess the image
                        img_scaled = st.session_state.scaler.transform(img_array)
                        
                        # Apply PCA if it was used in training
                        if st.session_state.pca is not None:
                            img_pca = st.session_state.pca.transform(img_scaled)
                        else:
                            img_pca = img_scaled
                        
                        # Make prediction
                        pred_class = st.session_state.model.predict(img_pca)[0]
                        
                        # Get probabilities if the model supports it
                        if hasattr(st.session_state.model, 'predict_proba'):
                            proba = st.session_state.model.predict_proba(img_pca)[0]
                            confidence = proba[pred_class] * 100
                            
                            st.subheader("Prediction Result")
                            st.write(f"Predicted class: **{st.session_state.class_names[pred_class]}**")
                            st.write(f"Confidence: **{confidence:.2f}%**")
                            
                            # Display bar chart of class probabilities
                            st.subheader("Class Probabilities")
                            fig, ax = plt.subplots(figsize=(10, 5))
                            ax.bar(st.session_state.class_names, proba * 100)
                            ax.set_ylabel('Probability (%)')
                            ax.set_title('Class Probabilities')
                            plt.xticks(rotation=45, ha='right')
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.subheader("Prediction Result")
                            st.write(f"Predicted class: **{st.session_state.class_names[pred_class]}**")
                            
                    except Exception as e:
                        st.error(f"Error making prediction: {str(e)}")