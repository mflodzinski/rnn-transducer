import pandas as pd

# path = 'files/train_sentences_val.csv'
# df = pd.read_csv(path)


def check_row_count():
    counter = {}

    for idx, row in df.iterrows():
        text = row["text"]
        if text not in counter:
            counter[text] = 1
        else:
            counter[text] += 1

    sorted_dict_by_values = {k: v for k, v in sorted(counter.items(), key=lambda item: item[1])}

    for k, v in sorted_dict_by_values.items():
        print(k, v, '\n')


def check_num_speakers():
    speakers = []
    for idx, row in df.iterrows():
        speaker = row['audio_path'].split('/')[3]
        if speaker not in speakers:
            speakers.append(speaker)
    print(speakers)
    print(len(speakers))

def check_num_audios():
    audios = []
    for idx, row in df.iterrows():
        audio = row['audio_path'].split('/')[4]
        if audio not in audios:
            audios.append(audio)
    print(audios)
    print(len(audios))

def check_num_texts():
    texts = []
    for idx, row in df.iterrows():
        text = row['text']
        if text not in texts:
            print(text)
            texts.append(text)
    print(len(texts))


import pandas as pd

# df =  pd.read_csv('files/test.csv')
# distinct_rows = df.drop_duplicates(subset=['text'])
# distinct_rows.to_csv('files/test_newest.csv', index=False)

# check_row_count()
# import csv
# Define the phrase to remove
# phrases_to_remove = [
#     'she_had_your_dark_suit_in_greasy_wash_water_all_year',
#     'dont_ask_me_to_carry_an_oily_rag_like_that'
# ]

# # Read the CSV file and filter rows
# with open(input_file, 'r') as csv_file:
#     csv_reader = csv.reader(csv_file)
#     header = next(csv_reader)  # Read header
#     filtered_rows = [row for row in csv_reader if not any(phrase in row[1] for phrase in phrases_to_remove)]

# # Write filtered rows to a new CSV file
# with open(output_file, 'w', newline='') as csv_output_file:
#     csv_writer = csv.writer(csv_output_file)
#     csv_writer.writerow(header)  # Write header
#     csv_writer.writerows(filtered_rows)

# import pandas as pd
# path = 'files/test_newest.csv'
# new_file = 'files/test_final.csv'
# df = pd.read_csv(new_file)

# column_names = ["audio_path", "text", "start", "end"]

# data = []

# for idx, row in df.iterrows():
#     audio_path = row['audio_path']
#     file, ext = audio_path.split('.')
#     word_file = file + '.WRD'

#     with open(word_file, 'r') as f:
#         for line in f:
#             if line == "": continue
#             elements = line.split()  # Split the line by whitespace

#             start, end, word = elements
#             word = word.replace("'", "")
#             new_row = [audio_path, word, start, end]
#             if (int(end) - int(start)) >= 320:
#                 data.append(new_row)

# new_df = pd.DataFrame(data, columns=column_names)
# new_df.to_csv(new_file, index=False)

# Sort the DataFrame by the length of words in the 'Text' column

# path = 'files/train_final.csv'
# sec_path = 'dupa.txt'

# a = []
# df = pd.read_csv(path)

# for i, row in df.iterrows():
#     a.append(row['text'])

# with open(sec_path, 'r+') as f:
#     lines = f.readlines()

#     # Move the file pointer to the beginning for writing
#     f.seek(0)

#     # Add "asd" at the end of each line and write it back to the file
#     for i, line in enumerate(lines):
#         modified_line = line.strip() + ' -> ' + a[i] + '\n' # Modify each line
#         f.write(modified_line)  # Write modified line back to the fileimport pandas as pd


# import pandas as pd

# # Load the CSV file into a DataFrame
# df = pd.read_csv('files/test_newest.csv')

# # Remove the last character from each row in the 'text' column
# df['text'] = "<" + df['text'] + ">"

# # Save the modified DataFrame back to the CSV file
# df.to_csv('files/test_newest.csv', index=False)

import pandas as pd

# Read the CSV file into a DataFrame
file_path = 'files/train_sentences.csv'
data = pd.read_csv(file_path)

# Remove duplicates based on the 'text' column
a = data['text'].str.len().max()
# Write the updated DataFrame without duplicates back to a new CSV file
# output_file_path = 'files/train_sentences.csv'
# data.to_csv(file_path, index=False)
print(a)