import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import csv

nltk.download('punkt', force=True)
nltk.download('stopwords')

# pre processes the data
def process_data(input_file,output_name):
    csv.field_size_limit(10**7) 
    with open(input_file, 'r') as file:
        reader = csv.reader(file)
        rows = list(reader)

    # go through the text column
    for row in rows:
        row[2] = convert_to_lowercase(row[2])
        row[2] = remove_special_characters_numbers(row[2])
        # tokenzie the words to remove stopwords
        row[2] = tokenize_text(row[2])
        row[2] = remove_stopwords(row[2])
        # convert back to string
        row[2] = ' '.join(row[2])

    # write processed data to new file
    with open(output_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# convert everything to lowercase
def convert_to_lowercase(text):
    lowercased_text = text.lower()
    return lowercased_text

# remove all special characters
def remove_special_characters_numbers(text):
    clean_text = re.sub(r'[^a-zA-Z\s]', '', text)
    return clean_text

# makes words into tokens
def tokenize_text(text):
    tokens = word_tokenize(text)
    return tokens

# removes stop words from tokens
def remove_stopwords(tokens):
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    return filtered_tokens

if __name__ == "__main__":
    # training formatting
    process_data('train.csv','processed_train.csv')

