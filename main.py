from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from img2table.document import PDF
from img2table.ocr import TesseractOCR
from openai import OpenAI
import os
import PyPDF2
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
import torch.nn.functional as F
import json
import pandas as pd

load_dotenv()

app = Flask(__name__)
# replace with your upload folder path
app.config['UPLOAD_FOLDER'] = './uploads'

api_key = os.getenv('SECRET_KEY')
debug_mode = os.getenv('DEBUG')
client = OpenAI(api_key=api_key)

tokenizer = AutoTokenizer.from_pretrained(
    'sentence-transformers/all-MiniLM-L6-v2')
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')


def mean_pooling(model_output, attention_mask):
    # First element of model_output contains all token embeddings
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(
        -1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def output(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # This is an example; use the latest suitable model
        messages=[
            {"role": "system", "content": "You are a language detection assistant."},
            {"role": "user", "content": f"This is a text where there are question number and corresponding answers. the numbers that stand alone are the qstn numbers and the long descriptive ones are the answers. map them correctly accoding to the order.So you need to now output a json file which has the question number as the key and the value as the corresponding answer. There should be no other words in this: '{text}'"}
        ]
    )
    return completion.choices[0].message.content


def extract_question_answers(pdf_path):
    question_answers = []

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text = page.extract_text()
            lines = text.split('\n')

            for line in lines:

                # Assuming the question number and answer are separated by a blank space
                parts = line.split(' ')

                if len(parts) == 2:
                    question_number = parts[0]
                    answer = parts[1]
                    question_answers.append((question_number, answer))

    return text


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/submit', methods=['POST'])
def submit():
    if 'pdf' not in request.files:
        return 'No file part', 400
    file = request.files['pdf']
    if file.filename == '':
        return 'No selected file', 400
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Instantiation of the pdf
        pdf = PDF(src="./uploads/" + filename)

        # Instantiation of the OCR, Tesseract, which requires prior installation
        ocr = TesseractOCR(lang="eng")

        # Table identification and extraction
        pdf_tables = pdf.extract_tables(ocr=ocr)

        # We can also create an excel file with the tables
        pdf.to_xlsx('questions.xlsx', ocr=ocr)

        return render_template('submit.html', filename=filename)
    else:
        return 'Invalid file', 400


@app.route('/submit_key', methods=['POST'])
def submit_key():
    file = request.files['file']
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        # Instantiation of the pdf
        pdf = PDF(src="./uploads/" + filename)

        # Instantiation of the OCR, Tesseract, which requires prior installation
        ocr = TesseractOCR(lang="eng")

        # Table identification and extraction
        pdf_tables = pdf.extract_tables(ocr=ocr)

        # We can also create an excel file with the tables
        pdf.to_xlsx('answer_key.xlsx', ocr=ocr)

        return render_template('submit_key.html', filename=filename)
    else:
        return 'Invalid file', 400


@app.route('/upload_answer_sheet', methods=['POST'])
def upload_answer_sheet():
    file = request.files['answer_sheet']
    if file and file.filename.endswith('.pdf'):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

        question_answers = extract_question_answers("uploads/" + filename)
        print(question_answers)
        text = output(question_answers)
        text = json.loads(text)
        print(type(text))
        print(text)
        score = {}
        total = 0
        # Specify the path to your Excel file
        file_path = './answer_key.xlsx'
        column_names = ['qstn_no', 'scheme']
        # Read the Excel file
        df = pd.read_excel(file_path, header=None, names=column_names)

        # Print the data
        answer_keys = df.iloc[:, 1]
        qstn_nos = df.iloc[:, 0]
        print(type(answer_keys))
        print(type(qstn_nos))
        for key, value in text.items():
            print(f"Question: {key}, Answer: {value}")
            for i in range(len(answer_keys)):
                if int(qstn_nos[i]) == int(key):
                    keys_sentences = answer_keys[i].split('-')
                    sentences = []
                    sentences.append(value)
                    for item in keys_sentences:
                        sentences.append(item)
                    sentences = [s for s in sentences if s]
                    print(sentences)
                    encoded_input = tokenizer(
                        sentences, padding=True, truncation=True, return_tensors='pt')
                    with torch.no_grad():
                        model_output = model(**encoded_input)
                    # Perform pooling
                    sentence_embeddings = mean_pooling(
                        model_output, encoded_input['attention_mask'])

                    # Normalize embeddings
                    sentence_embeddings = F.normalize(
                        sentence_embeddings, p=2, dim=1)

                    # Compute cosine similarities
                    # Assuming the first sentence is the source sentence
                    source_embedding = sentence_embeddings[0]
                    cosine_similarities = torch.matmul(
                        sentence_embeddings, source_embedding.unsqueeze(-1)).squeeze(-1)

                    print("Cosine Similarities:")
                    for i, sentence in enumerate(sentences):
                        print(
                            f"Similarity with '{sentence}': {cosine_similarities[i].item()}")
                    # Convert cosine similarities to scores, applying a threshold
                    threshold = 0.5
                    scores = torch.where(
                        cosine_similarities > threshold, 5 * (cosine_similarities + 1), torch.tensor(0.0))

                    # Calculate average score excluding the unrelated sentence
                    # Excludes the first source sentence itself and the unrelated sentence
                    relevant_scores = scores[1:-1]
                    average_score = torch.mean(relevant_scores)

                    print(
                        f"Overall score out of 10: {average_score.item():.2f}")
                    score[key] = average_score.item()
                    total += average_score.item()

                else:
                    pass

        return render_template('upload_success.html', filename=filename, total=total, score=score)
    else:
        return 'Invalid file', 400


if __name__ == '__main__':
    app.run(debug=True)
