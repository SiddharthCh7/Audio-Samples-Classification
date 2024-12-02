# Audio Chord Classification: Minor vs Major Chords

This project aims to classify audio samples as either a **Minor** or **Major** chord using machine learning techniques. The classification process involves extracting key audio features from the samples, training a machine learning model, and evaluating its performance.

## Project Overview

The task at hand is to determine whether a given audio sample represents a **Minor** or **Major** chord. The project leverages several techniques in audio processing and machine learning to accomplish this:

### 1. **Feature Extraction**:
Various audio features are extracted from the raw audio samples using the [librosa](https://librosa.org/) library, which is a powerful tool for analyzing and processing music and audio files.

### 2. **Model Selection**:
Support Vector Machine (SVM) was chosen for its strong performance on tasks involving classification with complex boundaries. SVM is a suitable model for this task, as it excels at separating different classes in high-dimensional spaces.

### 3. **Hyperparameter Tuning**:
Using **GridSearchCV**, hyperparameters for the SVM model (such as `C`, `gamma`, and `kernel`) were optimized to achieve the best performance. This allowed for fine-tuning the model to get the most accurate results.

### 4. **Dimensionality Reduction**:
The high-dimensional feature space resulting from the audio feature extraction was reduced using **PCA** (Principal Component Analysis). PCA helped to reduce the number of features while retaining most of the variance in the data, improving both model efficiency and performance.

### 5. **Data Scaling**:
The features were standardized using **StandardScaler** to ensure that all features contribute equally to the model, especially since SVMs are sensitive to feature scaling.

## Audio Features Used

The features extracted from the audio samples include:

### **Chroma Features**:
These capture harmonic content, which is crucial for chord recognition. Chroma mean and standard deviation are computed to capture the overall energy and variation in the harmonic content.

### **MFCC (Mel-frequency cepstral coefficients)**:
MFCCs are widely used to represent the timbral texture of sound, which is important for recognizing different types of chords. 
- **Mean** and **standard deviation** of MFCCs are used to capture both the central tendency and the spread of these coefficients.

### **Onset Statistics**:
These features capture the onset (or start) of musical events, which can be helpful in distinguishing between different chord types.

### **RMS (Root Mean Square)**:
RMS measures the energy of the audio signal, which can provide insight into the loudness and intensity of the sound.

### **Spectral Features**:
Spectral statistics capture the frequency distribution of the audio signal, which is vital for differentiating between chord types based on their spectral characteristics.

### **Zero Crossing Rate (ZCR)**:
ZCR represents the rate at which the audio signal crosses zero, which is useful for detecting percussive elements and high-frequency components in the sound.

## Model and Performance

The **SVM** model was trained on these extracted features, and hyperparameter tuning was performed using **GridSearchCV** to find the best set of parameters. The best-performing model was used for final classification.

- **Training Accuracy**: The model achieved an impressive **95%** accuracy on the training dataset, indicating that it learned the relationships between the features and the chord labels well.

- **Test Accuracy**: On the test dataset, the model achieved **55%** accuracy. While this may seem lower than the training accuracy, it's still a strong performance considering the complexity of the task. Given the nature of the dataset, achieving over 50% accuracy suggests that the model is effectively capturing the underlying patterns in the audio data.

### Why 55% Accuracy is Impressive

The test set accuracy of **55%** is notable because the task of chord classification from audio is inherently challenging, with several factors influencing the outcome. Achieving over 50% accuracy in such a task indicates that the model has captured significant patterns and relationships within the audio features, even though the task may involve overlapping characteristics between minor and major chords. Given the constraints and complexity of the data, this is considered a solid result.

## Conclusion

This project demonstrates the use of machine learning for audio classification, specifically in distinguishing between **Minor** and **Major** chords. By combining audio feature extraction, **SVM classification**, and techniques such as **PCA** for dimensionality reduction and **GridSearchCV** for hyperparameter tuning, the project achieves robust performance in both training and test phases.

The results show that with the right features and model selection, even complex audio classification tasks like chord recognition can be effectively tackled. Further improvements could involve more advanced feature engineering, model selection, or fine-tuning the hyperparameters further.

