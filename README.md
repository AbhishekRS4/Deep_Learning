# Deep Learning Project
***

## CV task - Cassava Leaf disease image classification
***
In the folder [src/cv_task](src/cv_task), the models for the Cassava leaf disease image classification.
### CNN models
* The models can be found in [src/cv_task/models.py](src/cv_task/models.py)
* Use the following to list all possible commandline parameters for training script
```
python src/cv_task/train.py --help
```
* For training with default parameters, run
```
python src/cv_task/train.py
```
* Use the following to list all possible commandline parameters for testing script
```
python src/cv_task/test.py --help
```
* For testing with default parameters, run
```
python src/cv_task/test.py
```
* To generate various plots, the following jupyter notebook [src/cv_task/plot_graphs.ipynb](src/cv_task/plot_graphs.ipynb) is useful

## NLP task - semantic analysis on tweet dataset regarding Coronavirus
***
* In the folder of the [src/nlp_task](src/nlp_task), you will find models for semantic analysis on a tweet dataset regarding the Coronavirus.
* The semantic analysis is performed using a pretrained version of the BERT model and an untrained version of it, as well
as a baseline LSTM model.

### LSTM model
* The baseline model can be found in the ```base_lstm_model.py``` file. The program can easily be run by running ```python base_lstm_model.py```. It will print some data statistics, a model summary, a classification report and it will generate a confusion matrix in the directory that this file is run from.

### BERT model
* The BERT model can be found in the ```BERT_model.py``` file. The model is designed using the Google colab notebook
found [here](https://colab.research.google.com/drive/1ywsvwO6thOVOrfagjjfuxEf6xVRxbUNO). This model is pretrained using
the ```bert-base-uncased``` model, and finetuned according the semantic analysis task with 5 different semantic labels.
Before training and testing, the tweets are cleaned up using a cleanup function as defined by Edgar Jonathan for another
BERT model on the same dataset, found [here](https://www.kaggle.com/code/edgardjonathan/bert-deep-learning).
* The BERT model in the file can be run in both the pretrained and the untrained manner by commenting or uncommenting the
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
