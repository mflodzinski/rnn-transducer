import pandas as pd

path = "files/test_newest.csv"
df = pd.read_csv(path)


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

check_row_count()
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


