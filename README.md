# NLP Part-of-Speech Tagging for Low Resource Indian Languages

![Project Logo/Visualization Here]

## Project Overview

This repository contains a natural language processing (NLP) project that focuses on part-of-speech (POS) tagging for the Bhojpuri language, with successful extensions to Maithili and Magahi languages. The project leverages machine learning techniques and linguistic features to accomplish accurate POS tagging. This README will provide a comprehensive understanding of the project, its objectives, methodologies, and results, while highlighting your skills and knowledge in the field of NLP.

## Table of Contents

1. Introduction
2. Project Methodology
3. Technology Stack
4. Data
5. Results
6. How to use
7. Project Files' Structure
8. Key Highlights
9. Contributing

## 1. Introduction
In the realm of natural language processing, part-of-speech tagging is a fundamental task, and this project focuses on the Bhojpuri language while demonstrating its adaptability to Maithili and Magahi languages. The project aims to accurately label each word in a given sentence with its corresponding part of speech, an essential step in various NLP applications like language understanding, sentiment analysis, and machine translation.


## 2. Project Methodology
The project's methodology encompasses several key steps:

I. **Data preprocessing and feature engineering:** We meticulously preprocessed linguistic features, converting them into a sparse matrix, and explore character n-gram features for enhanced model performance.
   
II. **Model development:** A Random Forest Classifier is used to train the model on the combined features.
   
III. **Testing and evaluation:** Rigorous testing is conducted on the model using a dedicated testing dataset. This phase generates an accuracy score and a detailed classification report, providing insights into the model's performance.


## 3. Technology Stack
- **Python**: The primary programming language for NLP and machine learning.
- **scikit-learn:** Leveraged for its machine learning capabilities and tools.
- **pandas:** Used for data manipulation and preprocessing.
- **NumPy:** Essential for numerical operations.
- **TfidfVectorizer:** Employed for feature extraction.
- **Random Forest Classifier:** The model's backbone for accurate POS tagging.
- **GitHub:** Our platform for version control and collaboration, making the project accessible and open to contributions.


## 4. Data
The project utilizes a dataset from [data source here], which contains linguistic data for the Bhojpuri language. This dataset is divided into training and testing sets to facilitate model development and evaluation.


## 5. Results
The project achieves an accuracy of approximately 78.18% on the Bhojpuri dataset, demonstrating its effectiveness in part-of-speech tagging. It is important to note that achieving an accuracy rate of approximately 78% in part-of-speech tagging for the Bhojpuri language is a significant accomplishment, particularly in the context of low-resource and less-researched languages. Bhojpuri, along with its closely related languages like Maithili and Magahi, often lacks the extensive linguistic resources and research attention enjoyed by more widely spoken languages. This 78% accuracy underscores the model's adaptability and its potential to advance research and technology in regions where language resources have traditionally been limited.


## 6. CHANGE: How to Use
To reproduce the results and utilize this NLP model, follow these steps:
Clone the repository: git clone [repository URL]
Navigate to the src/ directory.
Open the Jupyter Notebook and run the code provided, or execute it in your preferred Python environment.
Make sure to have the required libraries installed.
[Include any additional instructions or tips for users]


## 7. Project Files' Structure
The project's structure is organized as follows:

- `data/`       :         Contains datasets used for training and testing the model.
- `src/`         :        Source code and Google Collab Notebook
- `README.md`     :       This README file

  
## 8. Key Highlights
- **Multilingual Adaptability:** One of the standout features of this project is its adaptability to different languages. While the primary focus is on Bhojpuri, the model has been successfully extended to Maithili and Magahi languages, showcasing its versatility and robustness.

- **Impressive Accuracy:** We have achieved an accuracy rate of approximately 78.18% on the Bhojpuri dataset, highlighting the model's effectiveness in part-of-speech tagging. The accuracy is a testament to the quality of the model and the meticulousness of the development process.

- **Feature Engineering:** Our approach leverages a combination of linguistic features and character n-gram features to improve model performance. By employing techniques such as TF-IDF and Random Forest Classification, we have constructed a reliable and efficient model.

- **Extensive Testing:** The project has undergone rigorous testing and validation on multiple languages, including Maithili and Magahi. This demonstrates the model's ability to adapt and excel in different linguistic contexts.


## 9. Contributing
Contributions and suggestions are welcome. If you find issues or have ideas for improvements, please open an issue or submit a pull request.

