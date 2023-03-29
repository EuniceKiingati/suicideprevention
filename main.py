import time
import pickle
import os
import numpy as np

from selenium.webdriver.common.keys import Keys
from selenium import webdriver
from usefulMethods import vect

driver = webdriver.Chrome()


def login():
  driver.get("http://www.twitter.com/login")
  time.sleep(5)
  if not("login" in driver.current_url.lower()):
    print(driver.title)
    print(dir(driver))
    print("login" in driver.title)
    print(driver.current_url)
    return
  print(os.getenv("TWITTER_USERNAME"))
  usernameInput = driver.find_element_by_css_selector("[name=username]")
  usernameInput.clear()
  usernameInput.send_keys(os.getenv("TWITTER_USERNAME"))
  usernameInput.send_keys(Keys.ENTER) 
  time.sleep(3)
  #next button
  # driver.find_element_by_class_name("css-901oao").click()
  # time.sleep(2)
  passwordInput = driver.find_element_by_name("password")
  passwordInput.send_keys(os.getenv("TWITTER_PASSWORD"))
  passwordInput.send_keys(Keys.ENTER)

login()
time.sleep(5)
visitedTweets = []

clf = ''
with open("classifier.pickle", 'rb') as f:
        clf = pickle.load(f)

while True:
  # login()
  # time.sleep(5)
  tweetsInFeed = driver.find_elements_by_css_selector("[data-testid='tweet']")

  for tweet in tweetsInFeed:
    tweetTextDisp=""
    try:
      tweetTextDisp = tweet.find_element_by_css_selector('[lang="en"][dir="auto"]')
    except:
      continue
    print(tweetTextDisp.text, "\n\n\n\n\n\n")

    label = {0: 'negative', 1: 'positive'}
    transformedText = vect.transform([tweetTextDisp.text])
    prediction = label[clf.predict(transformedText)[0]]
    probability = np.max(clf.predict_proba(transformedText))*100
    print('Prediction: %s\nProbability: %.2f%%' %
          (prediction, probability))

    if(prediction == label[1] and probability > 50 and not tweetTextDisp.text in visitedTweets ): # if tweet is predicted as suicidal reply to it
      visitedTweets.append(tweetTextDisp.text)
      commentBtn = tweet.find_elements_by_css_selector("svg")[1]
      commentBtn.click()
      commentTextArea = driver.find_element_by_css_selector("[data-testid='tweetTextarea_0']")
      commentTextArea.send_keys(" If you or someone you know is contemplating suicide, please do not hesitate to talk to someone Kenya Hotline: +254 722 178 177, tafadhali pata usaidizi kwa kupigia hii namba")
      commentSbmtBtn = driver.find_element_by_css_selector('[data-testid="tweetButton"]')
      commentSbmtBtn.click()

  time.sleep(20)
  driver.refresh()
 