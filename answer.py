import PyPDF2


def extract_question_answers(pdf_path):
    question_answers = []

    with open(pdf_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)

        for page in pdf_reader.pages:
            text = page.extract_text()
            lines = text.split('\n')

            print(text)
            print(lines)

            for line in lines:
                print(line)
                # Assuming the question number and answer are separated by a blank space
                parts = line.split(' ')

                if len(parts) == 2:
                    question_number = parts[0]
                    answer = parts[1]
                    question_answers.append((question_number, answer))
            print("iterating")

    return question_answers


# Usage example
pdf_path = './data/a1.pdf'
question_answers = extract_question_answers(pdf_path)

# Print the extracted question number and answer pairs
for question_number, answer in question_answers:
    print(f"Question Number: {question_number}, Answer: {answer}")
