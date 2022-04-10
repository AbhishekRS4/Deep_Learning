import re
import sys
import torch
import emoji
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
import matplotlib.pyplot as plt

from tqdm import trange

from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from tensorflow.keras.preprocessing.sequence import pad_sequences

from pytorch_pretrained_bert import BertTokenizer, BertConfig
from pytorch_pretrained_bert import BertAdam, BertForSequenceClassification

log = open("print_log.log", "a")
sys.stdout = log


# cleaner as used in similar BERT application (source: https://www.kaggle.com/code/edgardjonathan/bert-deep-learning)
def cleaner(tweet):
    # remove links
    tweet = "".join(re.sub("(\w+:\/\/\S+)", " ", tweet))

    # remove hashtags
    tweet = "".join(re.sub("(#[A-Za-z0-9_]+)", " ", tweet))

    # remove user mention
    tweet = "".join(re.sub("(@[A-Za-z0-9_]+)", " ", tweet))

    # remove none alphanumeric and aposthrope
    tweet = "".join(re.sub("([^0-9A-Za-z \t'])", " ", tweet))

    # remove extra whitespace
    tweet = " ".join(tweet.split())

    # remove emoji unicode
    tweet = "".join(c for c in tweet if c not in emoji.UNICODE_EMOJI)  # Remove Emojis

    # remove leading and trailing space
    tweet = tweet.strip()

    return tweet


def randomize_model(model):
    for module_ in model.named_modules():
        if isinstance(module_[1],(torch.nn.Linear, torch.nn.Embedding)):
            module_[1].weight.data.normal_(mean=0.0, std=model.config.initializer_range)
        elif isinstance(module_[1], torch.nn.LayerNorm):
            module_[1].bias.data.zero_()
            module_[1].weight.data.fill_(1.0)
        if isinstance(module_[1], torch.nn.Linear) and module_[1].bias is not None:
            module_[1].bias.data.zero_()
    return model


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def main(untrained):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    torch.cuda.get_device_name(0)

    df = pd.read_csv("Corona_NLP_train.csv", header=0, encoding='latin-1')

    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: cleaner(x))
    tweets = df.OriginalTweet.values

    # add BERT tokens: CLS for classification and SEP for separator
    tweets = ["[CLS] " + tweet + " [SEP]" for tweet in df.OriginalTweet.values]

    # {'Extremely Negative': 5481, 'Extremely Positive': 6624, 'Negative': 9917, 'Neutral': 7713, 'Positive': 11422}
    le = preprocessing.LabelEncoder()
    le.fit(df.Sentiment)
    labels = le.transform(df.Sentiment)
    unique, counts = np.unique(labels, return_counts=True)
    print(dict(zip(unique, counts)))

    # Tokenize with BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = [tokenizer.tokenize(twt) for twt in tweets]
    print("Tokenize the first tweet:")
    print(tokenized_texts[0])

    # advised length
    MAX_LEN = 128

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]

    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")

    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i>0) for i in seq]
        attention_masks.append(seq_mask)

    # Use train_test_split to split our data into train and validation sets for training

    train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, labels,
                                                                                        random_state=2018, test_size=0.1)
    train_masks, validation_masks, _, _ = train_test_split(attention_masks, input_ids,
                                                           random_state=2018, test_size=0.1)
    # Convert all of our data into torch tensors, the required datatype for our model
    train_inputs = torch.tensor(train_inputs)
    validation_inputs = torch.tensor(validation_inputs)
    train_labels = torch.tensor(train_labels)
    validation_labels = torch.tensor(validation_labels)
    train_masks = torch.tensor(train_masks)
    validation_masks = torch.tensor(validation_masks)

    # Select a batch size for training. For fine-tuning BERT on a specific task, the authors recommend a batch size of
    # 16 or 32
    batch_size = 32

    # Create an iterator of our data with torch DataLoader. This helps save on memory during training because, unlike a
    # for loop, with an iterator the entire dataset does not need to be loaded into memory
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
    validation_sampler = SequentialSampler(validation_data)
    validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=batch_size)

    # Uncomment the following line for weight randomization
    # model = randomize_model(model)

    if untrained:
        config = BertConfig(vocab_size_or_config_json_file=30522)
        model = BertForSequenceClassification(config, num_labels=5)
    else:
        # Load BertForSequenceClassification, the pretrained BERT model with a single linear classification layer on top
        model = BertForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=5)

    model.cuda()

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'gamma', 'beta']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay_rate': 0.0}
    ]

    # This variable contains all of the hyperparemeter information our training loop needs
    optimizer = BertAdam(optimizer_grouped_parameters, lr=2e-5, warmup=.1)

    t = []

    # Store our loss and accuracy for plotting
    train_loss_set = []

    # Number of training epochs (authors recommend between 2 and 4)
    epochs = 10

    # trange is a tqdm wrapper around the normal python range
    for _ in trange(epochs, desc="Epoch"):

        # Training

        # Set our model to training mode (as opposed to evaluation mode)
        model.train()

        # Tracking variables
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0

        # Train the data for one epoch
        for step, batch in enumerate(train_dataloader):
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            # Clear out the gradients (by default they accumulate)
            optimizer.zero_grad()
            # Forward pass
            b_input_ids = torch.tensor(b_input_ids).to(device).long()
            b_labels = torch.tensor(b_labels).to(device).long()
            loss = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask, labels=b_labels)
            train_loss_set.append(loss.item())
            # Backward pass
            loss.backward()
            # Update parameters and take a step using the computed gradient
            optimizer.step()

            # Update tracking variables
            tr_loss += loss.item()
            nb_tr_examples += b_input_ids.size(0)
            nb_tr_steps += 1

        print("Train loss: {}".format(tr_loss / nb_tr_steps))

        # Validation

        # Put model in evaluation mode to evaluate loss on the validation set
        model.eval()

        # Tracking variables
        eval_loss, eval_accuracy = 0, 0
        nb_eval_steps, nb_eval_examples = 0, 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)
            # Unpack the inputs from our dataloader
            b_input_ids, b_input_mask, b_labels = batch
            b_input_ids = torch.tensor(b_input_ids).to(device).long()
            b_labels = torch.tensor(b_labels).to(device).long()
            # Telling the model not to compute or store gradients, saving memory and speeding up validation
            with torch.no_grad():
                # Forward pass, calculate logit predictions
                logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            tmp_eval_accuracy = flat_accuracy(logits, label_ids)

            eval_accuracy += tmp_eval_accuracy
            nb_eval_steps += 1

        print("Validation Accuracy: {}".format(eval_accuracy / nb_eval_steps))

    plt.figure(figsize=(15,8))
    plt.title("Training loss")
    plt.xlabel("Batch")
    plt.ylabel("Loss")
    plt.plot(train_loss_set)
    if untrained:
        plt.savefig('loss_BERT_untrained.png', bbox_inches='tight')
    else:
        plt.savefig('loss_BERT_pretrained.png', bbox_inches='tight')

    #################################### TESTING ########################################
    df = pd.read_csv("Corona_NLP_test.csv", header=0, encoding='latin-1')
    df['OriginalTweet'] = df['OriginalTweet'].apply(lambda x: cleaner(x))
    tweets = df.OriginalTweet.values

    # add BERT tokens: CLS for classification and SEP for separator
    tweets = ["[CLS] " + tweet + " [SEP]" for tweet in df.OriginalTweet.values]

    # labels = df['Sentiment'].map({'Extremely Negative': 0, 'Negative': 1, 'Neutral': 2, 'Positive': 3, 'Extremely Positive': 4})
    le = preprocessing.LabelEncoder()
    le.fit(df.Sentiment)
    labels = le.transform(df.Sentiment)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenized_texts = [tokenizer.tokenize(twt) for twt in tweets]

    # Use the BERT tokenizer to convert the tokens to their index numbers in the BERT vocabulary
    input_ids = [tokenizer.convert_tokens_to_ids(x) for x in tokenized_texts]
    # Pad our input tokens
    input_ids = pad_sequences(input_ids, maxlen=MAX_LEN, dtype="long", truncating="post", padding="post")
    # Create attention masks
    attention_masks = []

    # Create a mask of 1s for each token followed by 0s for padding
    for seq in input_ids:
        seq_mask = [float(i > 0) for i in seq]
        attention_masks.append(seq_mask)

    prediction_inputs = torch.tensor(input_ids)
    prediction_masks = torch.tensor(attention_masks)
    prediction_labels = torch.tensor(labels)

    prediction_data = TensorDataset(prediction_inputs, prediction_masks, prediction_labels)
    prediction_sampler = SequentialSampler(prediction_data)
    prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=batch_size)

    # Prediction on test set

    # Put model in evaluation mode
    model.eval()

    # Tracking variables
    predictions, true_labels = [], []

    # Predict
    for batch in prediction_dataloader:
        # Add batch to GPU
        batch = tuple(t.to(device) for t in batch)
        # Unpack the inputs from our dataloader
        b_input_ids, b_input_mask, b_labels = batch
        b_input_ids = torch.tensor(b_input_ids).to(device).long()
        b_labels = torch.tensor(b_labels).to(device).long()
        # Telling the model not to compute or store gradients, saving memory and speeding up prediction
        with torch.no_grad():
            # Forward pass, calculate logit predictions
            logits = model(b_input_ids, token_type_ids=None, attention_mask=b_input_mask)

        # Move logits and labels to CPU
        logits = logits.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    # # Import and evaluate each test batch using Matthew's correlation coefficient
    # from sklearn.metrics import matthews_corrcoef
    # matthews_set = []
    #
    # for i in range(len(true_labels)):
    #   matthews = matthews_corrcoef(true_labels[i],
    #                  np.argmax(predictions[i], axis=1).flatten())
    #   matthews_set.append(matthews)
    #
    # print(matthews_set)

    # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
    flat_predictions = [item for sublist in predictions for item in sublist]
    flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
    flat_true_labels = [item for sublist in true_labels for item in sublist]

    # print("matthews corrcoef:", matthews_corrcoef(flat_true_labels, flat_predictions))

    accuracy = accuracy_score(y_true=flat_true_labels, y_pred=flat_predictions)
    print("\n\nAccuracy: %.2lf " % (accuracy*100))

    print("\n\nClassification Report: ")
    print(classification_report(flat_true_labels, flat_predictions))

    cf_matrix = confusion_matrix(flat_true_labels, flat_predictions)
    cf_matrix = pd.DataFrame(cf_matrix)
    plt.figure(figsize=(10,7))
    sns.set(font_scale=1.6)
    sns.heatmap(cf_matrix, annot=True, fmt="g")
    if untrained:
        plt.savefig('confusion_matrix_BERT_untrained.png', bbox_inches='tight')
    else:
        plt.savefig('confusion_matrix_BERT_pretrained.png', bbox_inches='tight')


if __name__ == "__main__":
    # uncomment these lines to run an untrained BERT model
    # print("\n\n----------------- untrained ------------------")
    # main(True)

    # uncomment these lines to run an pretrained BERT model using bert-base-uncased
    print("\n\n----------------- pretrained -----------------")
    main(False)
