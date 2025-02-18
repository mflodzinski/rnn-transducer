import sys
from PyQt5.QtWidgets import *
import pandas as pd
import yaml
from utils import AttrDict
from model import Transducer
from tokenizer import CharTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from utils import computer_cer
from pydub import AudioSegment
from pydub.playback import play

class KeywordRecognizer(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
    def initUI(self):
        self.setStyleSheet("""
            QWidget {
                background-color: #CECECE;
                font-family: 'Helvetica Neue';
            }
            QLabel {
                color: #333333;
                font-size: 16px;
                margin-bottom: 5px;
            }
            QTextEdit {
                border: 1px solid #cccccc;
                padding: 5px;
                border-radius: 4px;
                background-color: white;
                color: #333333;
                font-size: 25px;
            }
            QPushButton {
                background-color: #138d75;
                color: white;
                font-size: 14px;
                padding: 10px;
                border-radius: 4px;
                border: none;
            }
            QPushButton:hover {
                background-color: #0e6655;
            }
        """) 

        layout = QVBoxLayout()
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10) 

        label1 = QLabel("Wybierz Nagranie")
        layout.addWidget(label1)

        self.audioList = QListWidget()
        self.addAudioFiles()
        layout.addWidget(self.audioList)

        label2 = QLabel("Wprowadź Słowa Do Wykrycia")
        layout.addWidget(label2)

        self.keywordInput = QTextEdit()
        self.keywordInput.textChanged.connect(self.onTextChanged)
        layout.addWidget(self.keywordInput)

        self.recognizeButton = QPushButton("Wykryj Słowa")
        self.recognizeButton.clicked.connect(self.recognizeKeywords_or_reset)
        self.flag = True
        layout.addWidget(self.recognizeButton)

        self.playButton = QPushButton("Odtwórz Nagranie")
        self.playButton.clicked.connect(self.play_wav_file)
        layout.addWidget(self.playButton)

        self.setLayout(layout)
        self.setWindowTitle('Wykrywanie Słów Kluczowych')
        self.setGeometry(450, 450, 900, 600)
        
        self.load_model()


    def addAudioFiles(self):
        audio_paths = 'files/core_test_set.csv'
        paths_df = pd.read_csv(audio_paths)
        self.audioList.addItems(paths_df['audio_path'])

    def load_model(self):
        config_path = 'config/config.yaml'
        model_path = 'timit/rnnt/2enc1dec_model.chkpt'
        configfile = open(config_path)
        config = AttrDict(yaml.load(configfile, Loader=yaml.FullLoader))
        checkpoint = torch.load(model_path)
        model = Transducer(config.model)

        tokenizer = CharTokenizer()
        tokenizer = tokenizer.load_tokenizer('files/tokenizer.json')

        model.encoder.load_state_dict(checkpoint['encoder'])
        model.decoder.load_state_dict(checkpoint['decoder'])
        model.joint.load_state_dict(checkpoint['joint'])

        self.model = model
        self.model.eval()
        self.config = config
        self.tokenizer = tokenizer

    def onTextChanged(self):
        special_chars = ['', '\n']
        text = self.keywordInput.toPlainText()
        if text=='': return
        char = text[-1]
        char = char.lower()
        cursor = self.keywordInput.textCursor()
        if char.isalpha() or char in special_chars:
            pass
        else:
            self.keywordInput.setText(text[:-1])
            cursor.movePosition(cursor.End)
            self.keywordInput.setTextCursor(cursor)

    def play_wav_file(self):
        selectedAudio = self.audioList.currentItem().text()
        audio = AudioSegment.from_wav(selectedAudio + '.wav')
        play(audio)

    def load_mfcc(self, path):
        first_slash_index = path.find("/")
        path = (
            path[: first_slash_index + 1]
            + "MFCC/"
            + path[first_slash_index + 1 :]
        )
        path, ext = path.split(".")
        path = f"{path}.npy"
        aud = torch.tensor(np.load(path))
        aud = aud.permute(0, 2, 1)
        return aud, aud.shape[1]
    

    def greedy_decode(self, inputs, inputs_length):
        blank = self.model.config.blank
        zero_token = torch.LongTensor([[blank]])
        f, _ = self.model.encoder(inputs, None)

        if inputs.is_cuda:
            zero_token = zero_token.cuda()

        def decode(inputs, lengths):
            token_list = []
            u = 0
            t = 0
            gu, hidden = self.model.decoder(zero_token)
            umax = self.model.config.max_length

            while t < lengths and u < umax:
                h = self.model.joint(inputs[t].view(-1), gu.view(-1))
                out = F.log_softmax(h, dim=0)
                _, pred = torch.max(out, dim=0)
                pred = int(pred.item())

                if pred != blank:
                    token_list.append(pred)
                    token = torch.LongTensor([[pred]])
                    if zero_token.is_cuda:
                        token = token.cuda()
                    gu, hidden = self.model.decoder(token, hidden=hidden)
                    u += 1
                else:
                    t += 1

            return token_list

        decoded_seq = decode(f.squeeze(0), inputs_length)
        return decoded_seq


    def recognize(self, path):
        mfcc, size = self.load_mfcc(path)
        transcription = self.greedy_decode(mfcc, size)
        special_tokens = [0,28,29]
        print(transcription)
        return [t for t in transcription if t not in special_tokens]

    def separate_list_by_number(self, arr, separator):
        result = []
        sublist = []
        
        for num in arr:
            if num == separator:
                if sublist:
                    result.append(sublist)
                    sublist = []
            else:
                sublist.append(num)
        
        if sublist:
            result.append(sublist)
        
        return result
    
    def get_similarity(self, preds, labels):
        dist, total = computer_cer([preds], [labels])
        percent = (1 - dist/total) * 100
        return max(percent, 0)

    def get_preds_and_labels(self):
        separator = self.tokenizer._token_to_id[self.config.data.separator]
        selectedAudio = self.audioList.currentItem().text()

        keywords_plain_text = self.keywordInput.toPlainText().strip()
        keywords_list = keywords_plain_text.split('\n')
        keywords_ids = [self.tokenizer.tokens2ids(keyword_list) for keyword_list in keywords_list]

        transcription = self.recognize(selectedAudio)
        print(''.join(self.tokenizer.ids2tokens([transcription])[0]))

        transcriptions_ids = self.separate_list_by_number(transcription, separator)
        return transcriptions_ids, keywords_ids
    
    def recognizeKeywords_or_reset(self):
        if self.flag:
            transcriptions_ids, keywords_ids = self.get_preds_and_labels()
            max_percent_list = [0] * len(keywords_ids)
            if keywords_ids==[[]]: return 

            for idx, keyword_ids in enumerate(keywords_ids):
                for transcription_ids in transcriptions_ids:
                    percent = self.get_similarity(transcription_ids, keyword_ids)
                    max_percent_list[idx] = max(max_percent_list[idx], percent)
            
            keywords_plain_text = self.keywordInput.toPlainText()
            keywords_list = keywords_plain_text.split('\n')
            final_text = ''

            for kw, per in zip(keywords_list, max_percent_list):
                final_text += f'{kw} -> {round(per,2)}%\n'
            self.keywordInput.setText(final_text)
            self.flag = not self.flag
            self.recognizeButton.setText('Restart')
        else:
            self.keywordInput.clear()
            self.flag = not self.flag
            self.recognizeButton.setText('Wykryj słowa')

def main():
    app = QApplication(sys.argv)
    ex = KeywordRecognizer()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()
