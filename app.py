import os
import time
import joblib
from flask import Flask, request, render_template, redirect, flash, send_from_directory
from werkzeug.utils import secure_filename
from pdf2image import convert_from_path
import pytesseract
import re
from concurrent.futures import ThreadPoolExecutor
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from difflib import ndiff
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# Define allowed file extensions and upload folder
ALLOWED_EXTENSIONS = {'pdf'}
UPLOAD_FOLDER = 'uploads'

# Load the trained logistic regression model and TF-IDF vectorizer
model_path = os.path.join("model", "logistic_regression_model.pkl")
tfidf_vectorizer_path = os.path.join("model", "tfidf_vectorizer.pkl")
classifier = joblib.load(model_path)
tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
stop_words = set(stopwords.words('english'))

# Initialize Flask app
app = Flask(__name__)
app.secret_key = 'supersecretkey'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize NER pipeline
ner_pipeline = pipeline("ner", model="dbmdz/bert-large-cased-finetuned-conll03-english", revision="f2482bf", aggregation_strategy="simple")

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    start_time = time.time()
    pages = convert_from_path(pdf_path, 300)
    with ThreadPoolExecutor() as executor:
        texts = list(executor.map(pytesseract.image_to_string, pages))
    text = ''.join(texts)
    end_time = time.time()
    print(f"Text extraction from PDF took {end_time - start_time:.2f} seconds")
    return text

def identify_clauses(text):
    """Identify clauses and sub-clauses in the text."""
    clauses = {}
    clause_pattern = re.compile(r'(?:^|\n)(?P<clause>\d+)\.\s*(?P<content>.*?)(?=\n\d+\.|\Z)', re.DOTALL)
    sub_clause_pattern = re.compile(r'(?:^|\n)(?P<sub_clause>\d+\.\d+)\s*(?P<content>.*?)(?=\n\d+\.\d+|\n\d+\.|\Z)', re.DOTALL)
    
    for match in clause_pattern.finditer(text):
        clause_name = f"clause {match.group('clause')}"
        clause_content = match.group('content').strip()
        clauses[clause_name] = {'content': clause_content, 'sub_clauses': {}}
        
        for sub_match in sub_clause_pattern.finditer(clause_content):
            sub_clause_name = f"sub-clause {sub_match.group('sub_clause')}"
            sub_clause_content = sub_match.group('content').strip()
            clauses[clause_name]['sub_clauses'][sub_clause_name] = sub_clause_content
    
    return clauses

def preprocess_text(text):
    """Preprocess the text for classification."""
    tokens = word_tokenize(text)
    tokens = [word.lower() for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)

def predict_clause_type(text):
    """Predict the clause type for a given text."""
    processed_text = preprocess_text(text)
    text_tfidf = tfidf_vectorizer.transform([processed_text])
    prediction = classifier.predict(text_tfidf)
    return prediction[0]

def compare_texts(template_text, actual_text):
    """Compare two texts and provide a summary of deviations."""
    processed_template = preprocess_text(template_text)
    processed_actual = preprocess_text(actual_text)
    diff = ndiff(processed_template.split(), processed_actual.split())
    
    deviations = {
        'template': template_text,
        'actual': actual_text,
        'diff': []
    }
    
    for line in diff:
        if line.startswith('- '):
            deviations['diff'].append(f"Template has '{line[2:]}' which is missing in Actual.")
        elif line.startswith('+ '):
            deviations['diff'].append(f"Actual has '{line[2:]}' which is missing in Template.")
    
    return deviations

def print_deviation_summary(deviations, clause_type):
    """Print deviations in the desired format."""
    summary = []
    summary.append(f"1. {clause_type} Clause:")
    summary.append(f"   - Template: {deviations['template']}")
    summary.append(f"   - Actual: {deviations['actual']}")
    summary.append("   - Deviation:")
    for deviation in deviations['diff']:
        summary.append(f"     - {deviation}")
    return '\n'.join(summary)

def calculate_cosine_similarity(text1, text2):
    """Calculate cosine similarity between two texts."""
    text1_tfidf = tfidf_vectorizer.transform([preprocess_text(text1)])
    text2_tfidf = tfidf_vectorizer.transform([preprocess_text(text2)])
    return cosine_similarity(text1_tfidf, text2_tfidf)[0][0]

def generate_highlighted_pdf(deviations_list):
    """Generate a single PDF containing all highlighted deviations."""
    pdf_filename = "highlighted_deviations.pdf"
    buffer = BytesIO()

    # Create a PDF document
    doc = SimpleDocTemplate(buffer, pagesize=letter)
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle(name='Highlighted', textColor=colors.red))

    content = []

    for index, deviations in enumerate(deviations_list, start=1):
        content.append(Paragraph(f"{index}. {deviations['clause_type']} Clause:", styles['Heading1']))
        content.append(Spacer(1, 12))

        content.append(Paragraph("Template Text:", styles['Heading2']))
        content.append(Paragraph(deviations['template'], styles['Normal']))
        content.append(Spacer(1, 12))

        content.append(Paragraph("Actual Text:", styles['Heading2']))
        content.append(Paragraph(deviations['actual'], styles['Normal']))
        content.append(Spacer(1, 12))

        content.append(Paragraph("Deviation:", styles['Heading2']))
        for deviation in deviations['diff']:
            content.append(Paragraph(deviation, styles['Highlighted']))
        content.append(Spacer(1, 12))

    doc.build(content)
    pdf_data = buffer.getvalue()
    buffer.close()

    # Save PDF file
    pdf_path = os.path.join(app.config['UPLOAD_FOLDER'], pdf_filename)
    with open(pdf_path, 'wb') as f:
        f.write(pdf_data)

    return pdf_path


@app.route('/', methods=['GET', 'POST'])
def upload_files():
    """Handle file upload, clause extraction, and classification."""
    if request.method == 'POST':
        if 'file_to_compare' not in request.files or 'template_file' not in request.files:
            flash('Please upload both files')
            return redirect(request.url)
        
        file_to_compare = request.files['file_to_compare']
        template_file = request.files['template_file']
        
        if file_to_compare.filename == '' or template_file.filename == '':
            flash('Please select both files')
            return redirect(request.url)
        
        if file_to_compare and allowed_file(file_to_compare.filename) and template_file and allowed_file(template_file.filename):
            filename_to_compare = secure_filename(file_to_compare.filename)
            filename_template = secure_filename(template_file.filename)
            
            file_path_to_compare = os.path.join(app.config['UPLOAD_FOLDER'], filename_to_compare)
            file_path_template = os.path.join(app.config['UPLOAD_FOLDER'], filename_template)
            
            file_to_compare.save(file_path_to_compare)
            template_file.save(file_path_template)

            # Extract text from uploaded documents
            text_to_compare = extract_text_from_pdf(file_path_to_compare)
            text_template = extract_text_from_pdf(file_path_template)

            # Identify clauses and sub-clauses
            clauses_to_compare = identify_clauses(text_to_compare)
            clauses_template = identify_clauses(text_template)
            
            # Predict clause types, compare texts, perform NER, and calculate cosine similarity
            results = []
            deviations_list = []
            for clause_name, clause_data in clauses_template.items():
                predicted_type = predict_clause_type(clause_data['content'])
                deviations = compare_texts(clause_data['content'], clauses_to_compare.get(clause_name, {}).get('content', ''))
                deviation_summary = print_deviation_summary(deviations, predicted_type)

                # Perform NER
                ner_results = ner_pipeline(clause_data['content'])
                ner_entities = ', '.join([f"{entity['word']} ({entity.get('entity-group', 'Unknown')})" for entity in ner_results])

                # Calculate cosine similarity
                cosine_sim = calculate_cosine_similarity(clause_data['content'], clauses_to_compare.get(clause_name, {}).get('content', ''))

                results.append({
                    'clause_name': clause_name,
                    'predicted_type': predicted_type,
                    'deviation_summary': deviation_summary,
                    'ner_entities': ner_entities,
                    'cosine_similarity': cosine_sim
                })

                deviations_list.append({
                    'clause_type': predicted_type,
                    'template': clause_data['content'],
                    'actual': clauses_to_compare.get(clause_name, {}).get('content', ''),
                    'diff': deviations['diff']
                })

            # Generate and save a single PDF with all highlighted deviations
            pdf_path = generate_highlighted_pdf(deviations_list)

            return render_template('index.html', text_to_compare=text_to_compare, text_template=text_template,
                                   clauses_to_compare=clauses_to_compare, clauses_template=clauses_template,
                                   results=results, pdf_path=pdf_path)  # pass the single pdf path to the template

    return render_template('index.html')

@app.route('/download/<filename>', methods=['GET'])
def download_pdf(filename):
    """Download the generated PDF file."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)



