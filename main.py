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


load_dotenv()

app = Flask(__name__)
# replace with your upload folder path
app.config['UPLOAD_FOLDER'] = './uploads'

api_key = os.getenv('SECRET_KEY')
debug_mode = os.getenv('DEBUG')
client = OpenAI(api_key=api_key)


def output(text):
    completion = client.chat.completions.create(
        model="gpt-3.5-turbo",  # This is an example; use the latest suitable model
        messages=[
            {"role": "system", "content": "You are a language detection assistant."},
            {"role": "user", "content": f"This is a text where there are question number and corresponding answers. So you need to now output a json file which has the question number as the key and the value as the corresponding answer. There should be no other words in this: '{text}'"}
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

        for key, value in text.items():
            print(f"Question: {key}, Answer: {value}")
        return render_template('upload_success.html', filename=filename, text=text)
    else:
        return 'Invalid file', 400


if __name__ == '__main__':
    app.run(debug=True)
