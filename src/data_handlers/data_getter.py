from beaker.cache import CacheManager, cache_regions, cache_region
from beaker.util import parse_cache_config_options
import slate3k as slate # This is a fork of the standard slate3k on pip to work with 3+: https://github.com/TakesxiSximada/slate3k
import requests
import progressbar

import os 
import re
import logging
import pickle
import json
import base64
import atexit
from datetime import date, timedelta

legiscan_key = os.getenv("LEGISCAN_KEY")
propublica_key = os.getenv("PROPUBLICA_KEY")

# Set cache options
cache_opts = {
    'cache.type': 'file',
    'cache.data_dir': '/tmp/cache/data',
    'cache.lock_dir': '/tmp/cache/lock'
}

# Create cache regions
cache_regions.update({
    'day_term': {
        'expire': 86400
    },
    'hour_term': {
        'expire': 3600
    }
})

# Create cache for this instance
cache = CacheManager(**parse_cache_config_options(cache_opts))

class APIManager:
    def __init__(self):
        atexit.register(self.__on_exit)

        # Load the bill texts from the persistent cache if it's less than a week old
        self.bill_texts = self.__load_pickle("pickles/billtext_cache.p")
        self.bill_subjects = self.__load_pickle("pickles/billsubject_cache.p")

        # Since I have a limited number of API calls to LegiScan, keep track of the number of calls we're making
        self.legiscan_call_count = 0

    def get_cached_bill_texts(self):
        return self.bill_texts

    def get_cached_bill_subjects(self):
        return self.bill_subjects

    def get_legiscan_call_count(self):
        return self.legiscan_call_count

    @cache.region('day_term', 'bill_list')
    def get_bill_list(self, max_len):
        '''Get a list of size `max_len` or less of bills from US Congress.'''
        return [x for x in self.__legiscan_call("getMasterList", ("state", "US"))["masterlist"].values()][1:max_len + 1]

    @cache.region('day_term', 'bill_text')
    def get_bill_text(self, ls_id):
        '''Get the text of a bill from its LegiScan ID.'''

        # Check if we've already gotten this bill's text
        if ls_id in self.bill_texts:
            return self.bill_texts[ls_id]

        # Return an empty string if there are no texts associated with a bill (don't know why this happens)
        if len(self.get_bill_info(ls_id)["bill"]["texts"]) == 0:
            return ""

        # Haven't been able to find a bill with more than one text, so this line might need to be changed
        doc_id = self.get_bill_info(ls_id)["bill"]["texts"][0]["doc_id"]

        res = self.__legiscan_call("getBillText", ("id", doc_id))

        if(res["status"] == "ERROR"):
            print("Unable to get text for bill {}".format(ls_id))

        content = ""

        # slate expects a file object, so we must create a temporary PDF.
        with open("tmp.pdf", "wb") as tmp:
            tmp.write(base64.b64decode(res["text"]["doc"]))
            
        # slate doesn't like wb+, so we have to re-open the file :(
        with open("tmp.pdf", "rb") as tmp:
            reader = slate.PDF(tmp)
            content = ''.join(reader).replace("\n", " ").replace("-", "")

        # Remove versioning data from content
        content = re.sub(r'VerDate ((?!VerDate).)*?\f\d*', "", content)

        # Clear unicode characters that we don't care about
        content = (content.encode('ascii', 'ignore')).decode('utf-8')
            
        if os.path.exists("tmp.pdf"):
            os.remove("tmp.pdf")

        self.bill_texts[res["text"]["bill_id"]] = content

        return content

    @cache.region('day_term', 'bill_training')
    def construct_bill_training_data(self, ls_id):
        '''Construct one entry for BERT training data based on this bill. Returns a tuple with a key in the first entry and dict of data in the second.'''
        content = self.get_bill_text(ls_id)
        info = self.get_bill_info(ls_id)
        subject = self.get_primary_subject(info)
        return (self.__create_bill_key(info), {"subject": subject, "content": content})

    @cache.region('day_term', 'bill_info')
    def get_bill_info(self, ls_id):
        '''Get the information for a single bill from LegiScan.'''
        return self.__legiscan_call("getBill", ("id", ls_id))

    @cache.region('day_term', 'bill_subject')
    def get_primary_subject(self, bill):
        '''Get the primary subject of the LegiScan bill object.'''

        bill_key = self.__create_bill_key(bill)

        if bill_key in self.bill_subjects:
            return self.bill_subjects[bill_key]

        # Reconstruct the session number from the provided session name
        session = ''.join([i for i in bill["bill"]["session"]["session_name"] if i.isdigit()])
        bill_id = bill["bill"]["bill_number"].lower().replace("b", "r")
        
        res = self.__propublica_call("{sess}/bills/{bill}.json".format(sess=session, bill=bill_id))

        if res["status"] != "OK":
            print("ProPublica call failed: {}".format(res))

        # Find best bill option if result is ambiguous
        # This seems unlikely to happen, but I don't want to happen unhandled while trying to retrieve training data
        best_result = 0
        if len(res["results"]) > 1:
            print("WARNING: Bill {} is ambiguous ({} results found). Guessing best result.".format(bill["bill"]["bill_number"], len(res["results"])))
            for i, option in enumerate(res["results"]):
                if option["title"] == bill["bill"]["title"]:
                    best_result = i

        self.bill_subjects[bill_key] = res["results"][best_result]["primary_subject"]

        return res["results"][best_result]["primary_subject"]

    def __on_exit(self):
        '''Store bill data in persistent cache on exit'''
        if not os.path.exists('pickles'):
            os.makedirs('pickles')
        self.__cache(self.get_cached_bill_texts(), "pickles/billtext_cache.p")
        self.__cache(self.get_cached_bill_subjects(), "pickles/billsubject_cache.p")

    def __cache(self, dict_to_cache, path):
        '''Timestamp and cache a dict to the given path.'''
        dict_to_cache["saved_on"] = date.today()
        pickle.dump(dict_to_cache, open(path, "wb"))

    def __load_pickle(self, path):
        '''Attempts to load a cached file, only if the file was stored less than a week ago. Otherwise, returns an empty dictionary.'''
        if os.path.exists(path):
            cache_load = pickle.load(open(path, "rb"))
            if cache_load["saved_on"] > date.today() + timedelta(days = -7):
                print("Loaded cache at {}.".format(path))
                return cache_load
        return {}

    def __create_bill_key(self, bill):
        '''Create a unique, cross-API compatible key for the provided LegiScan bill object.'''
        session = ''.join([i for i in bill["bill"]["session"]["session_name"] if i.isdigit()])
        bill_id = bill["bill"]["bill_number"].lower().replace("b", "r")
        return "{}/{}".format(session, bill_id)

    def __legiscan_call(self, endpoint, *args):
        '''Generalized function for calling LegiScan API endpoints.'''
        if legiscan_key is None:
            return "LegiScan API key is not defined. Please set the LEGISCAN_KEY environment variable to a valid API key."

        self.legiscan_call_count += 1

        # Build URL
        url="https://api.legiscan.com/?key={key}&op={endpoint}".format(key=legiscan_key, endpoint=endpoint)
        for arg in args:
            url += "&{}={}".format(arg[0], arg[1])
        
        # print("LEGISCAN API CALL (#{}): {}".format(self.legiscan_call_count, url))
        # Send request
        r = requests.get(url)
        return json.loads(r.text)

    def __propublica_call(self, endpoint, *args):
        '''Call the given path within the Propublica API, starting after the API version (v1).'''
        if propublica_key is None:
            return "ProPublica API key is not defined. Please set the PROPUBLICA_KEY environment variable to a valid API key."

        url="https://api.propublica.org/congress/v1/{endpoint}".format(endpoint=endpoint)
        
        for arg in args:
            url += "&{}={}".format(arg[0], arg[1])
        headers = {"X-API-Key": propublica_key}
        # print("PROPUBLICA API CALL: {}".format(url))
        # Send request
        r = requests.get(url, headers=headers)

        return json.loads(r.text)

# Test code ----------------------------------------------------------

def _test():
    api = APIManager()

    def _test_bill_list():
        print(api.get_bill_list(10)[1]['bill_id'])

    def _test_bill_text():
        txt = api.get_bill_text(1502653)

    def _test_bill_subject():
        bills = api.get_bill_list(10)
        for bill in bills:
            print(api.get_primary_subject(api.get_bill_info(bill["bill_id"])))

    #_test_bill_list()
    #_test_bill_text()
    _test_bill_subject()

# Main code ----------------------------------------------------------

def build_dataset(count, path):
    '''Creates a pickle file of a given number of bill texts and categories.'''
    api = APIManager()

    data = {}

    print("Building dataset of size {}...".format(count))

    for bill in progressbar.progressbar(api.get_bill_list(count)):
        res = api.construct_bill_training_data(bill["bill_id"])
        data[res[0]] = res[1]

    print("Creating pickle file at {}...".format(path))

    json.dump(data, open("{}.json".format(path), "w"))
    pickle.dump(data, open(path, "wb"))

    print("Dataset saved.")

if __name__ == "__main__":
    # Disable PDFMiner warnings from slate (no better way to do this, unfortunately -- https://stackoverflow.com/a/33926649)
    logging.propagate = False
    logging.getLogger().setLevel(logging.ERROR)

    if not os.path.exists('raw_data'):
        os.makedirs('raw_data')

    build_dataset(2000, "raw_data/dataset.p")