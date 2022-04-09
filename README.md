# Deep Learning Project 
## CV task

## NLP task
In the folder of the nlp_task, you will find models for semantic analysis on a tweet dataset regarding the Coronavirus. 
The semantic analysis is performed using a pretrained version of the BERT model and an untrained version of it, as well
as a baseline LSTM model.

### LSTM model
TEXTTEXTTEXT

### BERT model
The BERT model can be found in the ```BERT_pretrained.py``` file. The model is designed using the Google colab notebook 
found [here](https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO). This model is pretrained using 
the ```bert-base-uncased``` and finetuned according the semantic analysis task with 5 different semantic labels. Before
training and testing, the tweets are cleaned up using a cleanup function as defined by Edgar Jonathan for another BERT
model on the same dataset, found [here](https://www.kaggle.com/code/edgardjonathan/bert-deep-learning).

PRETRAINED UNTRAINED