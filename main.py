from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
import os
from img2table.document import PDF
from img2table.ocr import TesseractOCR


app = Flask(__name__)
# replace with your upload folder path
app.config['UPLOAD_FOLDER'] = './uploads'


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


if __name__ == '__main__':
    app.run(debug=True)
