# Final-Project-Sentiment-Analysis-Application
Utilizing machine learning, this program analyzes Instagram comments to identify cyberbullying instances by classifying sentiments as positive or negative.

# Objective
The objective of this project is to develop a machine learning model for sentiment analysis of Instagram comments, specifically focusing on identifying cyberbullying comments. The model will classify comments as either positive or negative sentiment based on their content.

# Model Description
The model utilizes the Naive Bayes algorithm for sentiment classification. The text data is vectorized using the CountVectorizer, which converts text into numerical features. The Multinomial Naive Bayes classifier is trained on the vectorized data to predict the sentiment of the comments.

# Evaluation Metrics
The effectiveness of the model is measured using accuracy score and classification report. Accuracy score provides an overall measure of the model's correctness, while the classification report gives insights into precision, recall, and F1-score for each sentiment class.

# Dataset Overview
The dataset used in this project is sourced from Kaggle, titled "Sentiment Analysis". It consists of Instagram comments labeled with sentiments (positive or negative) and includes columns such as 'Sentiment' and 'Instagram Comment Text'. The dataset has undergone minimal preprocessing, mainly selecting relevant columns and removing irrelevant information.

# Methodology
- Data Loading: Load the dataset using pandas library.
- Preprocessing: Select relevant columns and perform minimal preprocessing.
- Model Building: Split the dataset into training and testing sets. Convert text data into numerical features using CountVectorizer. Train the Naive Bayes classifier on the training data.
- Model Evaluation: Evaluate the model's performance using accuracy score and classification report on the testing data.
- Model Saving: Save the trained model and vectorizer for future use.
- User Interface: Create a simple user interface using Gradio for testing the model's predictions interactively.

# Challenges and Learnings
- Data Quality: Ensuring the quality and relevance of the dataset for sentiment analysis.
- Feature Engineering: Deciding on the appropriate features and preprocessing techniques.
- Model Selection: Choosing the suitable algorithm and fine-tuning parameters.
- Deployment: Integrating the model into a user-friendly interface for easy access.

# Future Improvements
- Advanced Models: Experimenting with deep learning architectures for improved performance.
- Hyperparameter Tuning: Fine-tuning model parameters for enhanced accuracy.
- Real-time Monitoring: Implementing feedback mechanisms for continuous model improvement.

# Demo
A demo of the model in action can be accessed here.

# Result
<img width="718" alt="image" src="https://github.com/Rifqiakmals12/Final-Project-Sentiment-Analysis-Application/assets/72428679/e52125d3-1cfd-4520-96e9-b9e829e7036b">
<img width="718" alt="image" src="https://github.com/Rifqiakmals12/Final-Project-Sentiment-Analysis-Application/assets/72428679/fc43a5fd-31f6-400e-9fca-5bc88fe8d476">

