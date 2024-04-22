import os
from openai import OpenAI
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

    return text


# Usage example
pdf_path = './data/a1.pdf'
question_answers = extract_question_answers(pdf_path)

# Print the extracted question number and answer pairs
for question_number, answer in question_answers:
    print(f"Question Number: {question_number}, Answer: {answer}")


api_key = "sk-"
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


text = ""

print(output(text))


api_key = "sk-"
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


text = ""

print(output(text))
