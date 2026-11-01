# CAP5610-Review-Classification-Project
Deep Learning Approaches to Product Review Classification using MLP, CNN, LSTM, Transformers

Uses https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023

Uses https://huggingface.co/stanfordnlp/glove/resolve/main/glove.840B.300d.zip


## Steps to use embedded dataset
Go to gloveEmbedTest.py found in the preprocessing folder and run it
Running that will generate the review and item metadata as well as GloVe embed the reviews

After that's finished running, in whatever file you want to use the embedding in, add 
'import sys'
'sys.path.append(str(Path(__file__).resolve().parent.parent)'

then you can do
'from preprocessing.readGloveTest import loadData'
'from preprocessing.gloveEmbedTest import CATEGORIES'
'from preprocessing.gloveEmbedTest import ITEMS_OUTPUT'

Example usage
for category, config in CATEGORIES.items():
  for asin, matrices in loadData.stream_category("arrays.h5", category):
    do stuff with data
