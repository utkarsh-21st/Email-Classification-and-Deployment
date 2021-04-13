## Email Clasification
### Classification using DistilBERT and XLNet

![Home Page](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/app/data/images/home_img.png "Home Page")

The email-dataset (.msg files) that I originally used in this project are confidential and thus can't be open-sourced, so in the [jupyter-notebook](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/text_classification.ipynb "jupyter-notebook"), I've used a text-complaint dataset, which is stored in a CSV file, to train the model.
Here is the link for that - [Click](https://drive.google.com/file/d/10LSWKtWAOOSv1l-SIvzr6sPI-niXcxbZ/view "Click").

If you do have .msg files, you may train the model accordingly as per the instructions provided in the notebook.
Go through the [Jupyter Notebook](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/text_classification.ipynb "Jupyter Notebook") and train the model.

Having trained a model, let's deploy it using Flask:
The below deployment provides a UI to browse and read .msg files, classifies them, and place each file to its predicted folder 
Note: It currently implements DistilBERT in its backend however you can make relevant alterations to the tasks.py to make it run for XLNet.

### Deployment using Flask
#### How to run?
- Install dependencies: See [requirements.txt](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/requirements.txt "requirements.txt")
- Clone this repository
```shell
git clone https://github.com/utkarsh-21st/Email-Classification-and-Deployment.git
```
- After going through the jupyter-nb, you would have 2 files saved - model and model_data. Put the saved model file into the `app/model` directory and the `model_data`  folder to the `app` directory.

- Now fire up a terminal, cd to your project directory, and run the below command:

```shell
python main.py
```
Head over to the URL (localhost) which would be displayed in your Terminal window.
(similar to http://127.0.0.1:5000/)

![Home Page](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/app/data/images/home_img.png "Home Page")

------------

## Overview Of Different Approaches for Text Classification [January 2021]
**Bag of Words representation**, which was much used before the advent of
Deep Learning, approaches the problem in the following way:
Having a dataset of sequences, it first creates a vocabulary (unique-words)
out of it. Then a numerical feature-vector is generated for every
text-sequence. The feature-vector is an array of size of vocabulary, where
each element represents the occurrences (in that sequence) of the
corresponding word in the vocabulary.
Then all the vectors and corresponding class labels are fed to a
classification algorithm such as SVM or an ensemble-based like Gradient
Boosting or a Neural Network.
The features could be improved by re-weighting them with what’s called
**Tf–idf (Term Frequency Inverse Document Matrix)**.
It turns out that word-embeddings learned through Neural Networks
perform much better than the above mentioned.
Next is the idea of word-embeddings which is also a numerical vector
representation for a word. Two famous ones are **word2vec** and **GloVe**.
These are learned embeddings that take into consideration occurrence and
co-occurrence information as well. They also capture semantic-similarity

between words and are known to perform better than ‘simple’ Bag of Words
representation methods.
**RNN (Recurrent Neural Network)** is currently one of the best approaches to
tackle sequence-based problems. The major advantage is that they
maintain a state for keeping track of context while reading the sequence in
a sequential manner.
When using RNN’s for this task, we can use pre-trained embeddings,
word2vec, GloVe, by representing each word by these embeddings.
Particularly, **LSTM (Long Short Term Memory)**, which is a kind of RNN can
be used for this task, even better, **BiDirectional LSTM**, which reads the
sequence from either side can be used.
Next, there is **BERT (Bidirectional Encoder Representations from
Transformers)** which is currently one of the best models out there. It uses a
transformer, an attention mechanism that learns contextual relations
between words in a sequence. It is a Deep Neural Network consist of
hundreds of millions of parameters so training it from scratch requires
enormous computational resources as well as a huge dataset. So we are
going to make use of a pre-trained model and fine-tune it for our specific
task.

Finally, there is **XLNet**, which is the current state-of-the-art model on
various NLP tasks. XLNet is an extension of the Transformer-XL model. We
are going to make use of a pre-trained model and fine-tune it for our
specific task.
