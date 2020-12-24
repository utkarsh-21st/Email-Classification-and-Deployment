from app import app, app_dir, temp_dir, result_dir
from flask import render_template, jsonify, request
import os
from app.tasks import load_model, load_data, predict, load_labelencoder, load_tokenizer
import shutil


@app.route('/', methods=["GET", "POST"])
def index():
    upload_status = ''
    model_state_msg = 'Model not found!'
    classify_enable = 0

    if 'model' in globals():
        model_state_msg = 'Model Loaded successfully!'
    else:
        global model
        global tokenizer
        global le
        global max_length

        model = load_model()
        le, num_classes = load_labelencoder()
        tokenizer, max_length = load_tokenizer(num_classes)
        model_state_msg = 'Model loaded succesfully!'

    if 'model' not in os.listdir(app_dir):
        os.mkdir(app_dir / 'model')
        return render_template("index.html", model_state_msg=model_state_msg,
                               classify_enable=classify_enable)
    elif len(os.listdir(app_dir / 'model')) == 0:
        return render_template("index.html", model_state_msg=model_state_msg,
                               classify_enable=classify_enable)

    if request.method == 'POST':

        if request.files:
            if os.path.exists(temp_dir / 'messages'):
                shutil.rmtree(temp_dir / 'messages')
            os.mkdir(temp_dir / 'messages')
            files = request.files.getlist('files')
            upload_status = 'Files uploaded!'
            classify_enable = 1
            for file in files:
                file.save(temp_dir / 'messages' / file.filename)
            load_data()
            return render_template('index.html',
                                   model_state_msg=model_state_msg,
                                   upload_status=upload_status,
                                   classify_enable=classify_enable)

    return render_template('index.html',
                           model_state_msg=model_state_msg,
                           upload_status=upload_status,
                           classify_enable=classify_enable)


@app.route('/about')
def about():
    return render_template('about.html')


@app.route('/classify')
def classify():
    predict(max_length=max_length, le=le, model=model, tokenizer=tokenizer)
    # redirect('/')
    return jsonify({'result': f'Done! Go to {result_dir} for classified messages'})
