#!/usr/bin/env python
"""Download and preprocess datasets. Supported datasets are:
  * English female: LJSpeech (https://keithito.com/LJ-Speech-Dataset/)
  * Mongolian male: MBSpeech (Mongolian Bible)
"""
__author__ = 'Erdene-Ochir Tuguldur'

import os
import sys
import csv
import time
import argparse
import fnmatch
import librosa
import pandas as pd
import soundfile as sf

from hparams import HParams as hp
from zipfile import ZipFile
from audio import preprocess
from utils import download_file
from datasets.mb_speech import MBSpeech
from datasets.ts_speech import TSSpeech, text_normalize
from datasets.lj_speech import LJSpeech

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--dataset", required=True, choices=['ljspeech', 'mbspeech', 'tsspeech'], help='dataset name')
args = parser.parse_args()

if args.dataset == 'ljspeech':
    dataset_file_name = 'LJSpeech-1.1.tar.bz2'
    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    dataset_path = os.path.join(datasets_path, 'LJSpeech-1.1')

    if os.path.isdir(dataset_path) and False:
        print("LJSpeech dataset folder already exists")
        sys.exit(0)
    else:
        dataset_file_path = os.path.join(datasets_path, dataset_file_name)
        if not os.path.isfile(dataset_file_path):
            url = "http://data.keithito.com/data/speech/%s" % dataset_file_name
            download_file(url, dataset_file_path)
        else:
            print("'%s' already exists" % dataset_file_name)

        print("extracting '%s'..." % dataset_file_name)
        os.system('cd %s; tar xvjf %s' % (datasets_path, dataset_file_name))

        # pre process
        print("pre processing...")
        lj_speech = LJSpeech([])
        preprocess(dataset_path, lj_speech)

elif args.dataset == 'mbspeech':
    dataset_name = 'MBSpeech-1.0'
    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    dataset_path = os.path.join(datasets_path, dataset_name)

    if os.path.isdir(dataset_path) and False:
        print("MBSpeech dataset folder already exists")
        sys.exit(0)
    else:
        bible_books = ['01_Genesis', '02_Exodus', '03_Leviticus']
        for bible_book_name in bible_books:
            bible_book_file_name = '%s.zip' % bible_book_name
            bible_book_file_path = os.path.join(datasets_path, bible_book_file_name)
            # if not os.path.isfile(bible_book_file_path):
            #     url = "https://s3.us-east-2.amazonaws.com/bible.davarpartners.com/Mongolian/" + bible_book_file_name
            #     download_file(url, bible_book_file_path)
            # else:
            #     print("'%s' already exists" % bible_book_file_name)

            print("extracting '%s'..." % bible_book_file_name)
            zipfile = ZipFile(bible_book_file_path)
            zipfile.extractall(datasets_path)

    dataset_csv_file_path = os.path.join(datasets_path, '%s-csv.zip' % dataset_name)
    dataset_csv_extracted_path = os.path.join(datasets_path, '%s-csv' % dataset_name)
    # if not os.path.isfile(dataset_csv_file_path):
    #     url = "https://www.dropbox.com/s/dafueq0w278lbz6/%s-csv.zip?dl=1" % dataset_name
    #     download_file(url, dataset_csv_file_path)
    # else:
    #     print("'%s' already exists" % dataset_csv_file_path)

    print("extracting '%s'..." % dataset_csv_file_path)
    zipfile = ZipFile(dataset_csv_file_path)
    zipfile.extractall(datasets_path)

    sample_rate = 44100  # original sample rate
    total_duration_s = 0

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    wavs_path = os.path.join(dataset_path, 'wavs')
    if not os.path.isdir(wavs_path):
        os.mkdir(wavs_path)

    metadata_csv = open(os.path.join(dataset_path, 'metadata.csv'), 'w')
    metadata_csv_writer = csv.writer(metadata_csv, delimiter='|')


    def _normalize(s):
        """remove leading '-'"""
        s = s.strip()
        if s[0] == '—' or s[0] == '-':
            s = s[1:].strip()
        return s


    def _get_mp3_file(book_name, chapter):
        book_download_path = os.path.join(datasets_path, book_name)
        wildcard = "*%02d - DPI.mp3" % chapter
        for file_name in os.listdir(book_download_path):
            if int(file_name.split("-")[0]) == int(chapter):
                return os.path.join(book_download_path, file_name)
        return None


    def _convert_mp3_to_wav(book_name, book_nr):
        global total_duration_s
        chapter = 1
        while True:
            try:
                i = 0
                chapter_csv_file_name = os.path.join(dataset_csv_extracted_path, "%s_%02d.csv" % (book_name, chapter))
                df = pd.read_csv(chapter_csv_file_name, sep="|")
                print("processing %s..." % chapter_csv_file_name)
                mp3_file = _get_mp3_file(book_name, chapter)
                print("processing %s..." % mp3_file)
                assert mp3_file is not None
                samples, sr = librosa.load(mp3_file, sr=sample_rate, mono=True)
                assert sr == sample_rate

                for index, row in df.iterrows():
                    start, end, sentence = row['start'], row['end'], row['sentence']
                    assert end > start
                    duration = end - start
                    duration_s = int(duration / sample_rate)
                    if duration_s > 10:
                        continue  # only audios shorter than 10s

                    total_duration_s += duration_s
                    i += 1
                    sentence = _normalize(sentence)
                    fn = "MB%d%02d-%04d" % (book_nr, chapter, i)
                    metadata_csv_writer.writerow([fn, sentence, sentence])  # same format as LJSpeech
                    wav = samples[start:end]
                    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=hp.sr)  # use same sample rate as LJSpeech
                    sf.write(os.path.join(wavs_path, fn + ".wav"), wav, hp.sr)
                chapter += 1
            except FileNotFoundError:
                break


    # _convert_mp3_to_wav('01_Genesis', 1)
    # _convert_mp3_to_wav('02_Exodus', 2)
    _convert_mp3_to_wav('03_Leviticus', 3)
    metadata_csv.close()
    print("total audio duration: %ss" % (time.strftime('%H:%M:%S', time.gmtime(total_duration_s))))

    # pre process
    print("pre processing...")
    mb_speech = MBSpeech([])
    preprocess(dataset_path, mb_speech)

elif args.dataset == 'tsspeech':
    from df.enhance import enhance, init_df, load_audio, save_audio
    df_model, df_state, _ = init_df()

    dataset_name = 'TSSpeech'
    datasets_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'datasets')
    dataset_path = os.path.join(datasets_path, dataset_name)

    data_names = [
         'ulger7', 'ulger8', 'ulger1', 'ulger2', 'ulger3', 'ulger4', 'ulger5', 'ulger6',
         'lit1','lit2', 'lit3','bart1', 'bart2',
         'geser1','geser2','geser3'
        ]


    
    total_duration_s = 0

    if not os.path.isdir(dataset_path):
        os.mkdir(dataset_path)
    wavs_path = os.path.join(dataset_path, 'wavs')
    if not os.path.isdir(wavs_path):
        os.mkdir(wavs_path)

    rows = []
    
    metadata_csv = open(os.path.join(dataset_path, f'metadata.csv'), 'w', encoding='utf-8')

    metadata_csv_writer = csv.writer(metadata_csv, delimiter='|')
    for data_name in data_names:
        with open(os.path.join(dataset_path, f'output/{data_name}.csv'), 'r', encoding='utf-8') as csv_file:
            csv_file_reader = csv.reader(csv_file, delimiter='|')
            for row in csv_file_reader:
                file_name = row[0]
                row[0] = data_name + "_" + row[0].split("/")[-1][:-4]
                row[1] = text_normalize(row[1])
                if len(row[1]) < 170 and len(row[1].split(" ")) > 2:
                    row[2] = row[1]
                    data_path = os.path.join(datasets_path, f"/output/{data_name}")
                    if file_name.endswith(".wav"):
                        # copy file
                        src = os.path.join(dataset_path, file_name)
                        dst = os.path.join(wavs_path, row[0]+".wav")

                        audio, _ = load_audio(src, sr=df_state.sr())
                        # Denoise the audio
                        enhanced = enhance(df_model, df_state, audio)
                        # Save for listening
                        save_audio(dst, enhanced, df_state.sr())

                        # get wav duration
                        duration = librosa.get_duration(path=dst)
                        if duration < 17 and duration > 0.9:
                            rows.append(row[:-1])
                            total_duration_s += duration
                        else:
                            # delete destination file
                            print("deleted...", dst)
                            os.remove(dst)

    # print total_duration_s as HH:MM:SS
    total_duration_s = int(total_duration_s)
    total_duration_s = time.strftime('%H:%M:%S', time.gmtime(total_duration_s))


    print(f"Total training wav duration: {total_duration_s}")
        
    metadata_csv_writer.writerows(rows)
    metadata_csv.close()

    def _normalize(s):
        """remove leading '-'"""
        s = s.strip()
        if s[0] == '—' or s[0] == '-':
            s = s[1:].strip()
        return s




    
   

    # pre process
    print("pre processing...")
    ts_speech = TSSpeech([])
    
    preprocess(dataset_path, ts_speech)