# Contract Document Comparison Tool

This project is a Flask web application designed to compare business contract documents based on the sentiment analysis of clauses and sub-clauses. The application identifies deviations, performs Named Entity Recognition (NER), and calculates cosine similarity between the clauses of two documents. The comparison results and deviations are highlighted in a generated PDF report.

## Features

- **Upload and Compare PDF Documents**: Upload two PDF files for comparison - a template and an actual contract.
- **Text Extraction**: Extract text from the uploaded PDF files using OCR.
- **Clause Identification**: Identify and classify clauses and sub-clauses within the documents.
- **Clause Type Prediction**: Predict the type of each clause using a pre-trained logistic regression model and TF-IDF vectorizer.
- **Deviation Detection**: Detect deviations between the template and actual contract clauses.
- **Named Entity Recognition (NER)**: Perform NER on the identified clauses to extract entities.
- **Cosine Similarity Calculation**: Calculate the cosine similarity between the corresponding clauses of the template and actual contract.
- **PDF Generation**: Generate a PDF report highlighting the deviations and including the comparison results.

## Dependencies

The project uses the following libraries:

- Flask
- joblib
- pdf2image
- pytesseract
- re
- concurrent.futures
- nltk
- difflib
- scikit-learn
- transformers
- io
- reportlab

## Installation

1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/contract-comparison-tool.git
   cd contract-comparison-tool
   ```

2. Install the required dependencies:
   ```sh
   pip install -r requirements.txt
   ```

3. Download and install Tesseract OCR from [here](https://github.com/tesseract-ocr/tesseract).

4. Ensure you have the `nltk` resources:
   ```sh
   python -m nltk.downloader punkt
   python -m nltk.downloader stopwords
   ```

## Usage

1. Place your trained logistic regression model and TF-IDF vectorizer in the `model` directory:
   ```
   model/
   ├── logistic_regression_model.pkl
   └── tfidf_vectorizer.pkl
   ```

2. Run the Flask application:
   ```sh
   python app.py
   ```

3. Open a web browser and go to `http://127.0.0.1:5000/`.

4. Upload the template and the contract document you want to compare.

5. View the comparison results and download the generated PDF report.

## Project Structure

```
contract-comparison-tool/
├── app.py
├── templates/
│   └── index.html
├── static/
│   └── styles.css
├── uploads/
├── model/
│   ├── logistic_regression_model.pkl
│   └── tfidf_vectorizer.pkl
├── requirements.txt
└── README.md
```

## Contributing

Contributions are Siddhi , Sujal ,Swakshan , Srushti and Komal.

