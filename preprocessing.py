#!/bin/python3 

import pandas as pd
import csv
# import numpy as np # 스케일링을 제거하여 더 이상 필요하지 않습니다.
import sys

# --- 1. read files ---
print("Loading data...")
try:
    df1 = pd.read_csv('./dataset/beer_reviews.csv', encoding="utf-8")
    df2 = pd.read_csv('./dataset/train.csv', encoding="utf-8")
except FileNotFoundError:
    print("ERROR: 'dataset' folder not found or files (beer_reviews.csv, train.csv) are missing.")
    print("Please ensure the files are in the correct directory.")
    sys.exit()

print("Data loaded.")

# --- 2. drop columns with NaN (MODIFIED per user request) ---
# MODIFICATION: 컬럼 중 하나라도 비어있다면 해당 행(row)을 모두 드랍합니다.
original_shape_df1 = df1.shape
original_shape_df2 = df2.shape

df1.dropna(inplace=True)
df2.dropna(inplace=True)

print(f"df1 shape after strict dropna: {original_shape_df1} -> {df1.shape}")
print(f"df2 shape after strict dropna: {original_shape_df2} -> {df2.shape}")

# --- 2b. Fill missing ABV (REMOVED) ---
# Step 2의 강력한 dropna로 인해 이 단계는 더 이상 필요하지 않습니다.

# --- 3. remove non-ascii characters ---
df1.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)
df2.replace({r'[^\x00-\x7F]+': ''}, regex=True, inplace=True)

# --- 4. beer_name to beer_id mapping (IMPROVED) ---
print("Creating name maps...")
df1_name = df1.copy(deep=True)[['beer_beerid', 'beer_name']]
df2_name = df2.copy(deep=True)[['beer/beerId', 'beer/name']]
df1_name.drop_duplicates(inplace=True)
df2_name.drop_duplicates(inplace=True)

df1_id_to_name = pd.Series(df1_name.beer_name.values, index=df1_name.beer_beerid).to_dict()
df2_id_to_name = pd.Series(df2_name['beer/name'].values, index=df2_name['beer/beerId']).to_dict()

# --- 5. drop columns ---
df1.drop(columns=['brewery_name', 'beer_name', 'review_time', 'index'], inplace=True, axis=1)
df2.drop(columns=['user/birthdayRaw', 'beer/name', 'user/birthdayUnix', 'user/ageInSeconds', \
'review/timeStruct', 'review/text', 'user/gender', 'review/timeUnix', 'index'], inplace=True, axis=1)

# --- 6. rename columns ---
df2.rename(columns={'beer/ABV':'beer_abv', 'beer/beerId':'beer_beerid', 'beer/brewerId':'brewery_id', \
'beer/style':'beer_style', 'review/appearance':'review_appearance', 'review/aroma':'review_aroma',\
'review/overall':'review_overall', 'review/palate':'review_palate', 'review/taste':'review_taste',\
'user/profileName':'review_profilename'}, inplace=True)

# --- 7. reorder columns ---
df1 = df1[sorted(df1.columns)]
df2 = df2[sorted(df2.columns)]

# --- 8. remove values not in both sets (IMPROVED) ---
# (이 로직은 맥주, 양조장, 스타일에 대해 df1을 기준으로 df2를 필터링합니다)
print("Pruning test set (items)...")
df1_styles = set(df1['beer_style'].unique())
df1_ids = set(df1['beer_beerid'].unique())
df1_brewers = set(df1['brewery_id'].unique())

df2 = df2[df2['beer_style'].isin(df1_styles)]
df2 = df2[df2['beer_beerid'].isin(df1_ids)]
df2 = df2[df2['brewery_id'].isin(df1_brewers)]
print(f"df2 shape after item pruning: {df2.shape}")

# --- 9. label encode names for easier use (MODIFIED: Keep only overlapping users) ---
print("Filtering for overlapping profile names (users)...")

# 1. df1과 df2에서 고유한 사용자 이름 집합을 가져옵니다.
df1_profiles = set(df1['review_profilename'].unique())
df2_profiles = set(df2['review_profilename'].unique())

# 2. 두 집합에 *모두* 존재하는 사용자(교집합)를 찾습니다.
common_profiles = df1_profiles.intersection(df2_profiles)
print(f"Found {len(common_profiles)} users present in both df1 and df2.")

if not common_profiles:
    print("Warning: No common users found between df1 and df2. This may result in empty datasets.")

# 3. 두 데이터프레임 모두 "공통 사용자"의 리뷰만 남도록 필터링합니다.
df1 = df1[df1['review_profilename'].isin(common_profiles)]
df2 = df2[df2['review_profilename'].isin(common_profiles)]

print(f"df1 shape after filtering for common users: {df1.shape}")
print(f"df2 shape after filtering for common users: {df2.shape}")

# 4. 이제 "공통 사용자" 목록을 기반으로 인코딩 맵을 생성합니다.
common_profile_list = sorted(list(common_profiles))
profile_to_id = {name: i for i, name in enumerate(common_profile_list)}

df1['profilename_encoded'] = df1['review_profilename'].map(profile_to_id)
df2['profilename_encoded'] = df2['review_profilename'].map(profile_to_id)

df1.drop(columns=['review_profilename'], inplace=True)
df2.drop(columns=['review_profilename'], inplace=True)

# --- 10. label encode beer style (Correct) ---
print("Encoding beer styles...")
df1['beer_style'] = df1['beer_style'].astype('category')
df1['beer_style_encoded'] = df1['beer_style'].cat.codes
id_to_style = dict(enumerate(df1['beer_style'].cat.categories))
style_to_id = {v: k for k, v in id_to_style.items()}

df1.drop(columns=['beer_style'], inplace=True)

# Apply the *same* mapping to df2
df2['beer_style_encoded'] = df2['beer_style'].map(style_to_id)
df2.drop(columns=['beer_style'], inplace=True)

# --- 11. normalize data (REMOVED per user request) ---
print("Skipping review score normalization.")

# --- 12. save to files --- (이제 11번 단계가 됩니다)
print("Saving processed files...")
df1.to_csv('training_set.csv', index=False)
df2.to_csv('testing_set.csv', index=False)

# 딕셔너리 저장 로직
with open('id_to_style.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(id_to_style.keys())
    writer.writerow(id_to_style.values())

with open('trainingset_id_to_name.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(df1_id_to_name.keys())
    writer.writerow(df1_id_to_name.values())

with open('testingset_id_to_name.csv', mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(df2_id_to_name.keys())
    writer.writerow(df2_id_to_name.values())

print("Preprocessing complete!")
print(f"Saved 'training_set.csv' with shape: {df1.shape}")
print(f"Saved 'testing_set.csv' with shape: {df2.shape}")