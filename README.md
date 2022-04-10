# Deep Learning Project
***

## CV task
***
TBD

## NLP task
***
In the folder of the nlp_task, you will find models for semantic analysis on a tweet dataset regarding the Coronavirus. 
The semantic analysis is performed using a pretrained version of the BERT model and an untrained version of it, as well
as a baseline LSTM model.

### LSTM model
TBD

### BERT model
The BERT model can be found in the ```BERT_model.py``` file. The model is designed using the Google colab notebook 
found [here](https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO). This model is pretrained using 
the ```bert-base-uncased``` model, and finetuned according the semantic analysis task with 5 different semantic labels. 
Before training and testing, the tweets are cleaned up using a cleanup function as defined by Edgar Jonathan for another
BERT model on the same dataset, found [here](https://www.kaggle.com/code/edgardjonathan/bert-deep-learning).

The BERT model in the file can be run in both the pretrained and the untrained manner by commenting or uncommenting the
following lines:
```python
if __name__ == "__main__":
    # uncomment these lines to run an untrained BERT model
    print("\n\n----------------- untrained ------------------")
    main(True)
    
    # uncomment these lines to run an pretrained BERT model using bert-base-uncased
    print("\n\n----------------- pretrained -----------------")
    main(False)
```