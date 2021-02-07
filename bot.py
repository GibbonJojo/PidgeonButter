import tweepy
import urllib
import numpy as np
import cv2
import logging
import time
from tensorflow.keras.models import load_model
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()


class Bot:
    def __init__(self, config: str, model: str):
        self.config_path = config
        with open(config) as f:
            self.config = json.load(f)
        self.api = self.setup_api()
        self.model = load_model(model)
        self.since_id = self.config["SINCE_ID"]

    def setup_api(self):
        logger.info("Setting up API")
        auth = tweepy.OAuthHandler(self.config["CONSUMER_KEY"], self.config["CONSUMER_SECRET"])
        auth.set_access_token(self.config["ACCESS_TOKEN"], self.config["ACCESS_TOKEN_SECRET"])
        api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)
        try:
            api.verify_credentials()
        except Exception as e:
            logger.error("Error creating API", exc_info=True)
            raise e
        logger.info("API created")
        return api

    def convert_images(self, url: str):
        logger.info(f"Converting image with url {url}")
        resp = urllib.request.urlopen(url)
        image = np.asarray(bytearray(resp.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (self.config["IMG_SIZE"], self.config["IMG_SIZE"]))
        image = image / 255.0
        return np.array(image).reshape(-1, self.config["IMG_SIZE"], self.config["IMG_SIZE"], 3)

    def predict(self, images):
        return [int(np.round(self.model.predict(image)[0][0])) for image in images]

    def process_media(self, tweet, target_id=None, target_author=None):
        target_id = target_id if target_id else tweet.id
        target_author = target_author if target_author else tweet.author.screen_name
        logger.info(f"Answering to {tweet.user.name}")
        images = [self.convert_images(media["media_url"])
                  for media in tweet.extended_entities["media"]
                  if media["type"] == "photo"]
        logger.info(f"Predicting")

        predictions = self.predict(images)
        logger.info(f"Predictions: {predictions}")

        if len(predictions) == 1:
            self.api.update_status(
                status=f"@{target_author} {self.config['CATEGORIES'][predictions[0]].capitalize()}",
                in_reply_to_status_id=target_id)
        else:
            answer = f"@{target_author}\n"
            for i, prediction in enumerate(predictions):
                answer += f"{i+1}: {self.config['CATEGORIES'][prediction].capitalize()}\n"
            self.api.update_status(status=answer, in_reply_to_status_id=target_id)

    def update_since_id(self):
        self.config["SINCE_ID"] = self.since_id
        with open(self.config_path, "w") as config:
            json.dump(self.config, config)

    def check_mentions(self):
        logger.info("Retrieving mentions")
        new_since_id = self.since_id
        for tweet in tweepy.Cursor(self.api.mentions_timeline, since_id=self.since_id).items():
            self.since_id = max(tweet.id, new_since_id)

            if hasattr(tweet, "extended_entities"):
                if "media" in tweet.extended_entities:
                    self.process_media(tweet)
            elif tweet.in_reply_to_status_id:
                parent_tweet = self.api.get_status(tweet.in_reply_to_status_id)
                if hasattr(parent_tweet, "extended_entities"):
                    if "media" in parent_tweet.extended_entities:
                        self.process_media(parent_tweet, tweet.id, tweet.author.screen_name)

        self.update_since_id()

    def run(self):
        logger.info("Starting Bot")
        while True:
            try:
                self.check_mentions()
            except Exception as e:
                logger.warning(e)
            logger.info("Waiting...")
            time.sleep(15)
