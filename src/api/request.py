from __future__ import annotations

import json
import logging
import time

import requests
from requests.auth import HTTPDigestAuth

from config.state_init import StateManager


class RequestData:
    def __init__(self, state: StateManager):
        self.api_config = state.api_config
        self.creds = state.api_config.api_creds
        self.params = state.api_config.api_params

    def make_request(self, endpoint, params=None):
        url = f"{self.params['BASE_URL']}/{endpoint}"
        logging.debug(f"FULL URL: {url}?{params}")

        try:
            logging.debug("Attempting request...")
            response = requests.get(
                url, params, auth=HTTPDigestAuth(self.creds["USERNAME"], self.creds["PASSWORD"])
            )
            if response.status_code == 200:
                logging.debug(f"SUCCESSFUL Request: {response.status_code}")
                return response.json()
            if response.status_code == 429:
                logging.error(
                    f"API limit reached ERROR: {response.status_code}, Sleeping for {self.api_config.sleep_interval} seconds..."
                )
                time.sleep(int(self.api_config.sleep_interval))
                return None
            else:
                logging.error(f"ERROR: {response.status_code}, {response.text}")
                return None
        except requests.RequestException as e:
            logging.error(f"Request failed: {e}")
            return None

    def main(self, endpoint, params):
        try:
            response = self.make_request(endpoint, json.dumps(params) if params else None)
            # self.file_handler.save_json(response, Path(f'{self.save_path}/{filename}.json'))
        except Exception as e:
            logging.error(f"Error: {e}")
        return response
