import numpy as np
import pandas as pd
from datetime import datetime
from langdetect import detect

# Games info

# Read all games files
ps4_df = pd.read_csv('bayesian/data/ps4_games.csv')
switch_df = pd.read_csv('bayesian/data/switch_games.csv')
xbox_df = pd.read_csv('bayesian/data/xboxone_games.csv')


# The columns user_score, user_pos, user_mixed and user_neg 
# must have numbers 

def obj_to_numeric(df, cols):
    for col in cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        df.loc[df[col].isnull(), col] = 0


# PS4 cleaning              
obj_to_numeric(ps4_df, ['user_score', 'user_pos', 'user_mixed', 'user_neg'])

# Fill meta_overview and user_overview null values with their corresponding categories when there's no score.

ps4_df.loc[ps4_df['meta_overview'].isnull(), 'meta_overview'] = 'No score yet'
ps4_df.loc[ps4_df['user_overview'].isnull(), 'user_overview'] = 'No user score yet'

print(ps4_df.info())
ps4_df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

# No outliers but the columns are in different range. 
# They should be in the same range in order to be able to compare them.

ps4_df['n_user_score'] = ps4_df['user_score'] * 10

# release_date column contain strings, for instance "Jul 19, 2016". We must transform those strings into datetime.
ps4_df['release_date'] = ps4_df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%d-%m-%Y')))


print(ps4_df.info())
# E:/ds_practice/case_study/bayesian/cleaned_data
ps4_df.to_csv('bayesian/cleaned_data/ps4_games_cleaned.csv', index=False, encoding = 'utf-8')

#---------------------------------

# Xbox One

obj_to_numeric(xbox_df, ['user_score', 'user_pos', 'user_mixed', 'user_neg'])

xbox_df.loc[xbox_df['meta_overview'].isnull(), 'meta_overview'] = 'No score yet'
xbox_df.loc[xbox_df['user_overview'].isnull(), 'user_overview'] = 'No user score yet'

xbox_df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

xbox_df['n_user_score'] = xbox_df['user_score'] * 10
xbox_df['release_date'] = xbox_df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%d-%m-%Y')))

xbox_df.to_csv('bayesian/cleaned_data/xboxone_games_cleaned.csv', index=False, encoding = 'utf-8')

# switch 

obj_to_numeric(switch_df, ['user_score', 'user_pos', 'user_mixed', 'user_neg'])

switch_df.loc[switch_df['meta_overview'].isnull(), 'meta_overview'] = 'No score yet'
switch_df.loc[switch_df['user_overview'].isnull(), 'user_overview'] = 'No user score yet'

switch_df.describe().loc[['min', 'max'], ['meta_score', 'user_score']]

switch_df['n_user_score'] = switch_df['user_score'] * 10
switch_df['release_date'] = switch_df['release_date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%d-%m-%Y')))

switch_df.to_csv('bayesian/cleaned_data/switch_games_cleaned.csv', index=False, encoding = 'utf-8')


# Merge three csv

consoles = ['ps4', 'xboxone', 'switch']

tables = [pd.read_csv(f'bayesian/cleaned_data/{c}_games_cleaned.csv', lineterminator='\n') for c in consoles]

for t in tables: print(t.shape)

dataFrame = pd.concat(tables)
print(dataFrame.shape)

dataFrame.to_csv('bayesian/cleaned_data/cleanedgames.csv', index=False, encoding = 'utf-8')



# Reviews

'''
Meta reviews
Load critics reviews of each platform
'''

meta_reviews = pd.read_csv('bayesian/data/meta_reviews.csv', lineterminator='\n')

ps4_rdf =  meta_reviews.loc[meta_reviews['platform'] == 'PlayStation 4']
#print(ps4_rdf.info())
switch_rdf =  meta_reviews.loc[meta_reviews['platform'] == 'Switch']
#print(switch_rdf.info())
xbox_rdf =  meta_reviews.loc[meta_reviews['platform'] == 'Xbox One']
#print(xbox_rdf.info())

print(ps4_rdf.shape)
print(switch_rdf.shape)
print(xbox_rdf.shape)

# check null score 

print(ps4_rdf.loc[ps4_rdf['score'].isnull()])
print(switch_rdf.loc[switch_rdf['score'].isnull()])
print(switch_rdf.loc[switch_rdf['score'].isnull()])

meta_reviews['date'] = meta_reviews['date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%Y-%m-%d')))

meta_reviews.to_csv('bayesian/cleaned_data/cleanedmetareviews.csv', index=False, encoding = 'utf-8')

# user reviews 

user_reviews = pd.read_csv('bayesian/data/user_reviews.csv', lineterminator='\n')

ps4_udf =  user_reviews.loc[user_reviews['platform'] == 'PlayStation 4']
#print(ps4_rdf.info())
switch_udf =  user_reviews.loc[user_reviews['platform'] == 'Switch']
#print(switch_rdf.info())
xbox_udf =  user_reviews.loc[user_reviews['platform'] == 'Xbox One']
#print(xbox_rdf.info())

print(ps4_udf.shape)
print(switch_udf.shape)
print(xbox_udf.shape)

# check null score 

print(ps4_udf.loc[ps4_udf['score'].isnull()])
print(switch_udf.loc[switch_udf['score'].isnull()])
print(xbox_udf.loc[xbox_udf['score'].isnull()])

user_reviews['date'] = user_reviews['date'].apply(lambda x: pd.to_datetime(datetime.strptime(x, '%Y-%m-%d')))

user_reviews['score'] = user_reviews['score'] * 10
# Merge

review_df = pd.concat([meta_reviews, user_reviews]).reset_index(drop=True)

def print_examples(review_df, qty=1):
    for i in range(qty):
        print(review_df.iloc[i]['text'])
        print('\n')

print_examples(review_df.loc[review_df['score'] > 85], 5)

print_examples(review_df.loc[review_df['score'] < 40], 5)

'''

A lot of reviews contain the game name. This could be a problem by the time we train a sentiment analysis classifier. 
The algorithm could use that information to classify a review as positive or negative,
 something that won't be useful for new unseen data (of different games). 
 Let's remove them (and also transform all texts to lower case)
'''

review_df['text'] = review_df.apply(lambda x: x.text.lower().replace(f'{(x["title"]).lower()}', ''), 1)
print_examples(review_df.loc[review_df['score'] < 40], 5)

'''
Printing reviews I noticed that some of them are not english. For example:
'''

review_df.iloc[132654]['text']
review_df.iloc[132648]['text']

# the langdetect library is useful to find the language of a given text
def detect_lang(row):
    try:
        lang = detect(row.text)
    except:
        lang = "error"
    return lang

review_df['lang'] = review_df.apply(lambda x: detect_lang(x), 1)
print(review_df['lang'].value_counts()[:10])

# We are only interested in english reviews

review_df = review_df.loc[review_df['lang'] == 'en']

review_df.to_csv('bayesian/cleaned_data/cleanedreviews.csv', index=False, encoding = 'utf-8')





