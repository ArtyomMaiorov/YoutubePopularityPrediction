from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import requests

import joblib

app = Flask(__name__)

model = joblib.load('random_forest_model.pkl')


def extract_features(video_info):
    features = {}
    features['title'] = video_info['title']
    features['channel_title'] = video_info['channelTitle']
    features['category_id'] = video_info['categoryId']
    features['publish_time'] = video_info['publishedAt']
    features['tags'] = ', '.join(video_info['tags']) if 'tags' in video_info else ''
    features['description'] = video_info['description']
    return features


def predict(video_id, api_key='API_KEY'):
    url = f'https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={api_key}'
    response = requests.get(url)
    video_info = response.json()['items'][0]['snippet']

    features = extract_features(video_info)
    features_df = pd.DataFrame([features])

    prediction = model.predict(features_df)
    return prediction[0]

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    thumbnail_url = None
    if request.method == 'POST':
        video_id = request.form['video_id'].split('=')[-1]
        prediction = predict(video_id)
        thumbnail_url = f'https://img.youtube.com/vi/{video_id}/default.jpg'
    return render_template('index.html', prediction=prediction, thumbnail_url=thumbnail_url)

@app.route('/reset', methods=['GET'])
def reset():
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
