## Email Clasification
### Classification using DistilBERT and XLNet

The email-dataset (.msg files) that I originally used in this project are confidential and thus can't be open-sourced, so in the [jupyter-notebook](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/text_classification.ipynb "jupyter-notebook"), I've used a text-complaint dataset, which is stored in a CSV file, to train the model.
Here is the link for that - [Click](https://drive.google.com/file/d/10LSWKtWAOOSv1l-SIvzr6sPI-niXcxbZ/view "Click")
If you do have .msg files, you may train the model accordingly as per the instructions provided in notebook.
Go through the [Jupyter Notebook](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/text_classification.ipynb "Jupyter Notebook") and train the model.

Having trained a model, now lets deploy it using Flask:
The below deployment provides a UI to browse and read .msg files, classifies them and place each file to its predicted folder 
Note: It currently implements DistilBERT in its backend however you can make relevant alterations to the tasks.py to make it run for XLNet.

### Deployment using Flask

The project directory looks like so:
├── app
│   ├── data
│   │   ├── images
│   │   └── model_data
│   ├── __init__.py
│   ├── model
│   ├── tasks.py
│   ├── templates
│   │   ├── about.html
│   │   ├── index.html
│   │   └── template.html
│   └── views.py
└── run.py

#### How to run?
- Install dependencies: See [requirements.txt](https://github.com/utkarsh-21st/attendence-face-recognition/blob/master/requirements.txt "requirements.txt")
- Clone this repository
```shell
git clone https://github.com/utkarsh-21st/Email-Classification-and-Deployment.git
```
- After going through the jupyter-nb, you would have 2 files saved - model and model_data. Put the saved model file into the `app/model` directory and the `model_data`  folder to `app` directory.

- Now fire up a terminal, cd to your project directory and run the below command:

```shell
python main.py
```
Go to the URL (localhost) which would be displayed in your Terminal window.
(similar to http://127.0.0.1:5000/)
and there we have the Home Page - 
![Home Page](https://github.com/utkarsh-21st/Email-Classification-and-Deployment/blob/master/app/data/images/home_img.png "Home Page")

