# This code has been taken and modified from the sample code provided in Task 1's Problem Statement
# Reference - https://github.com/carstonhernke/scrape-earnings-transcripts/blob/master/scrape_earnings_calls.py

import requests
import time
import os
from bs4 import BeautifulSoup


DATA_FOLDER = 'Data'
DEBUG = False


class HTMLExtractor:

    def __init__(self):
        return

    ''' Extract a specific page hosted on 'url' and write the contents of it to the article_name.html file '''

    def grab_page(self, url, article_name):
        if DEBUG:
            print("Attempting to get page: " + url)
        page = requests.get(url)
        page_html = page.text
        soup = BeautifulSoup(page_html, 'html.parser')
        content = soup.find("div", {"class": "sa-art article-width"})

        article_name = article_name.replace('/', '-')
        article_name = article_name.strip('-')

        if content != None:
            filename = '{}'.format(article_name)
            file = open(os.path.join(
                DATA_FOLDER, filename.lower() + ".html"), 'w')
            file.write(str(content))
            file.close()
            if DEBUG:
                print("Successfully Saved")

    ''' Extract the list of arcticles on a given page and extract each of them sequentially '''

    def process_list_page(self, page):
        origin_page = "https://seekingalpha.com/earnings/earnings-call-transcripts" + \
            "/" + str(page)
        if DEBUG:
            print("Getting page {}".format(origin_page))
        page = requests.get(origin_page)
        page_html = page.text
        soup = BeautifulSoup(page_html, 'html.parser')
        article_list = soup.find_all(
            "li", {'class': 'list-group-item article'})
        if DEBUG:
            print("The page returned {} articles".format(len(article_list)))
        for article in range(0, len(article_list)):
            page_url = article_list[article].find_all("a")[0].attrs['href']
            url = "https://seekingalpha.com" + page_url
            self.grab_page(url, page_url)
            time.sleep(1)


if __name__ == "__main__":
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER)
    htmlExtractor = HTMLExtractor()
    for page in range(1, 2):
        htmlExtractor.process_list_page(page)
