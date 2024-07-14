import os
import pandas
import catboost
import logging
import pickle
from catboost import CatBoostClassifier
from utils.plots import plot_candles, plot_lines

from dotenv.main import load_dotenv

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


def get_figure(data, save_path, score):

    fig = plot_candles(data, title=f"Model Prediction {score:.2f}", show=False)
    fig = plot_lines(data, y=['predict'], fig=fig, c=['#d5c4a1'], secondary_y=True, heat='predict', show=False, heat_size=30)
    
    fig.update_layout(legend=dict(orientation="h", bgcolor='rgba(0,0,0,0)')) 
    fig.update_layout(
        font_family="Bahnschrift",
        font_color="#f9f5d7",
        title_font_family="Bahnschrift",
        title_font_color="#fabd2f",
        legend_title_font_color="#f9f5d7",
        title=dict(font=dict(size=50))
    )

    logger.info("Saving image")
    fig.write_image(save_path, format='png', width=1000, height=800, scale=2)
    logger.info("Image saved successfuly -> {images/result.png}")

    


def make_predict(df):
    
    load_dotenv()

    PREPROCESSING_PATH = os.environ['PREPROCESSING_PATH']
    MODEL_PATH = os.environ['MODEL_PATH']
    IMG_SAVE_PATH = os.environ['IMG_SAVE_PATH']

    logger.info("Loading preprocessing")
    with open(PREPROCESSING_PATH, 'rb') as f:
        pipeline = pickle.load(f)

    logger.info("Loading model")
    model = CatBoostClassifier()
    model.load_model(MODEL_PATH)

    logger.info("Preprocessing")
    data = pipeline.transform(df)

    data['predict'] = model.predict_proba(data)[:, 1]
    

    sample = data.tail(12)
    last_score = sample['predict'].tail(1).values[0]
    get_figure(sample, IMG_SAVE_PATH, last_score)

    return last_score
