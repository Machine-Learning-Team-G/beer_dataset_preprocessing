#!/bin/python3 

import pandas as pd
import csv

# read files
df1 = pd.read_csv('beer_reviews.csv', encoding="utf-8")
df2 = pd.read_csv('train.csv', encoding="utf-8")

# remove non-ascii characters which cause errors
df1.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
df2.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

# beer_name to beer_id mapping
df1_name = df1.copy(deep=True)[['beer_beerid', 'beer_name']]
df2_name = df2.copy(deep=True)[['beer/beerId', 'beer/name']]
df1_name.drop_duplicates(inplace=True)
df2_name.drop_duplicates(inplace=True)

df1_id_to_name = {}
df2_id_to_name = {}
for beer_id, beer_name in df1_name.itertuples(index=False):
	df1_id_to_name[beer_id] = beer_name
for beer_id, beer_name in df2_name.itertuples(index=False):
	df2_id_to_name[beer_id] = beer_name

# drop columns
df1.drop(columns=['brewery_name', 'beer_name', 'review_time', 'index'], inplace=True, axis=1)
df2.drop(columns=['user/birthdayRaw', 'beer/name', 'user/birthdayUnix', 'user/ageInSeconds', \
'review/timeStruct', 'review/text', 'user/gender', 'review/timeUnix', 'index'], inplace=True, axis=1)

# rename columns
df2.rename(columns={'beer/ABV':'beer_abv', 'beer/beerId':'beer_beerid', 'beer/brewerId':'brewery_id', \
'beer/style':'beer_style', 'review/appearance':'review_appearance', 'review/aroma':'review_aroma',\
'review/overall':'review_overall', 'review/palate':'review_palate', 'review/taste':'review_taste',\
'user/profileName':'review_profilename'}, inplace=True)

# reorder columns
df1 = df1[sorted(df1.columns)]
df2 = df2[sorted(df2.columns)]

# remove values not in both sets
df1_styles = sorted(df1['beer_style'].unique())
df2_styles = sorted(df2['beer_style'].unique())
df2_drop_rows = []
for style in df2_styles:
	if style not in df1_styles:
		df2_drop_rows = df2[df2['beer_style'] == style].index
		df2.drop(df2_drop_rows, inplace=True)

df1_ids = sorted(df1['beer_beerid'].unique())
df2_ids = sorted(df2['beer_beerid'].unique())
for beer_id in df2_ids:
	if beer_id not in df1_ids:
		df2_drop_rows = df2[df2['beer_beerid'] == beer_id].index
		df2.drop(df2_drop_rows, inplace=True)

df1_brewers = sorted(df1['brewery_id'].unique())
df2_brewers = sorted(df2['brewery_id'].unique())
for brewer_id in df2_brewers:
	if brewer_id not in df1_brewers:
		df2_drop_rows = df2[df2['brewery_id'] == brewer_id].index
		df2.drop(df2_drop_rows, inplace=True)

# label encode names for easier use
df1['profilename_encoded'] = df1['review_profilename'].astype('category').cat.codes
df1.drop(columns=['review_profilename'], inplace=True)
df2['profilename_encoded'] = df2['review_profilename'].astype('category').cat.codes
df2.drop(columns=['review_profilename'], inplace=True)

# label encode beer style
df1['beer_style'] = df1['beer_style'].astype('category')
df1['beer_style_encoded'] = df1['beer_style'].cat.codes
id_to_style = dict(enumerate(df1['beer_style'].cat.categories))
style_to_id = {v: k for k, v in id_to_style.items()}
df1.drop(columns=['beer_style'], inplace=True)

df2['beer_style_encoded'] = df2['beer_style'].map(style_to_id)
df2.drop(columns=['beer_style'], inplace=True)


# normalize data
review_cols = [
    'review_appearance',
    'review_aroma',
    'review_palate',
    'review_taste',
    'review_overall'
]
df1[review_cols] = df1.groupby('profilename_encoded')[review_cols].transform(lambda x: (x - x.mean()) / x.std())
df2[review_cols] = df2.groupby('profilename_encoded')[review_cols].transform(lambda x: (x - x.mean()) / x.std())

# save to files
df1.to_csv('training_set.csv')
df2.to_csv('testing_set.csv')
with open('id_to_style.csv', mode='w', newline='') as file:
	writer = csv.DictWriter(file, id_to_style.keys())
	writer.writeheader()
	writer.writerow(id_to_style)
with open('trainingset_id_to_name.csv', mode='w', newline='') as file:
	writer = csv.DictWriter(file, df1_id_to_name.keys())
	writer.writeheader()
	writer.writerow(df1_id_to_name)
with open('testingset_id_to_name.csv', mode='w', newline='') as file:
	writer = csv.DictWriter(file, df2_id_to_name.keys())
	writer.writeheader()
	writer.writerow(df2_id_to_name)
