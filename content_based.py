import pandas as pd
import sys
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# --- Helper function for mapping files ---
def load_mapping_file(filepath, id_col_name, val_col_name):
    """
    Loads mapping files that are formatted as two rows (one for IDs, one for names).
    """
    try:
        df_raw = pd.read_csv(filepath, header=None)
        if df_raw.shape[0] < 2:
            print(f"Warning: File {filepath} has fewer than 2 rows.")
            return pd.DataFrame(columns=[id_col_name, val_col_name]).set_index(id_col_name)
        
        mapping_df = pd.DataFrame({
            id_col_name: df_raw.iloc[0],
            val_col_name: df_raw.iloc[1]
        })
        # Convert ID column to numeric, handling potential errors
        mapping_df[id_col_name] = pd.to_numeric(mapping_df[id_col_name], errors='coerce')
        mapping_df = mapping_df.dropna(subset=[id_col_name]) # Drop rows where ID conversion failed
        mapping_df = mapping_df.drop_duplicates(subset=[id_col_name]).set_index(id_col_name)
        
        # Ensure index is of integer type if possible
        if pd.api.types.is_float_dtype(mapping_df.index):
            try:
                mapping_df.index = mapping_df.index.astype(int)
            except ValueError:
                pass # Keep as float if conversion fails
                
        return mapping_df
    except pd.errors.EmptyDataError:
        print(f"Warning: File {filepath} is empty.")
        return pd.DataFrame(columns=[id_col_name, val_col_name]).set_index(id_col_name)
    except Exception as e:
        print(f"Error loading mapping file {filepath}: {e}")
        return pd.DataFrame(columns=[id_col_name, val_col_name]).set_index(id_col_name)

# --- Load mapping files ---
print("Loading mapping files...")
full_name_map = None
style_map = None
try:
    test_name_map = load_mapping_file("testingset_id_to_name.csv", "beer_beerid", "beer_name")
    train_name_map = load_mapping_file("trainingset_id_to_name.csv", "beer_beerid", "beer_name")
    full_name_map = pd.concat([test_name_map, train_name_map])
    full_name_map = full_name_map[~full_name_map.index.duplicated(keep='first')]
    print(f"Loaded {len(full_name_map)} unique beer names.")
except Exception as e:
    print(f"Error loading name maps: {e}")

try:
    style_map = load_mapping_file("id_to_style.csv", "beer_style_encoded", "style_name")
    print(f"Loaded {len(style_map)} unique beer styles.")
except Exception as e:
    print(f"Error loading style map: {e}")

TRAIN_FILE_NAME = "training_set.csv" #"training_set.csv" 
TEST_FILE_NAME = "testing_set.csv" #"testing_set.csv"


print(f"\n--- Attempting to load '{TRAIN_FILE_NAME}' ---")
try:
    # Load the training set
    train_df = pd.read_csv(TRAIN_FILE_NAME)
    print(f"Successfully loaded '{TRAIN_FILE_NAME}'.")
    
    # --- Step 1: Building Item Profiles from Training Set ---
    print("\n--- Step 1: Building Item Profiles from Training Set ---")
    
    required_cols = ['beer_beerid', 'beer_abv', 'beer_style_encoded']
    if not all(col in train_df.columns for col in required_cols):
         print(f"Error: '{TRAIN_FILE_NAME}' is missing required columns. Found: {train_df.columns}")
         raise ValueError("Missing required columns in training set")

    train_beer_profiles = train_df[required_cols].drop_duplicates(subset=['beer_beerid'])
    print(f"Found {len(train_beer_profiles)} unique beers in the training set.")

    # --- Step 2: Preprocessing Training Profiles ---
    print("\n--- Step 2: Preprocessing Training Profiles ---")
    train_beer_profiles = train_beer_profiles.dropna(subset=['beer_style_encoded'])
    train_beer_profiles['beer_style_encoded'] = train_beer_profiles['beer_style_encoded'].astype(int)
    
    mean_abv_train = train_beer_profiles['beer_abv'].mean()
    train_beer_profiles['beer_abv'] = train_beer_profiles['beer_abv'].fillna(mean_abv_train)
    
    train_beer_profiles = train_beer_profiles.set_index('beer_beerid')
    print(f"Processed {len(train_beer_profiles)} unique beer profiles from training set.")

    # --- Step 3: Vectorizing Training Profiles ---
    print("\n--- Step 3: Vectorizing Training Profiles ---")
    train_profiles_processed = pd.get_dummies(train_beer_profiles, columns=['beer_style_encoded'])
    
    scaler = MinMaxScaler()
    train_profiles_processed['beer_abv'] = scaler.fit_transform(train_profiles_processed[['beer_abv']])
    
    trained_style_columns = [col for col in train_profiles_processed.columns if 'beer_style_encoded_' in col]
    print(f"Model was trained on {len(trained_style_columns)} style features.")

    # --- Step 4: Calculating Similarity Matrix ---
    print("\n--- Step 4: Calculating Similarity Matrix (The 'Trained Model') ---")
    item_similarity_matrix_trained = cosine_similarity(train_profiles_processed)
    
    item_sim_df_trained = pd.DataFrame(
        item_similarity_matrix_trained,
        index=train_profiles_processed.index,
        columns=train_profiles_processed.index
    )
    print("Trained similarity matrix shape:", item_sim_df_trained.shape)
    
    # --- Step 5: Validation using 'testing_set.csv' ---
    print(f"\n--- Step 5: Validating Model with '{TEST_FILE_NAME}' ---")
    
    try:
        test_df = pd.read_csv(TEST_FILE_NAME)
        print(f"Loaded '{TEST_FILE_NAME}' for validation.")
        
        # Using z-score > 1.0 as the threshold
        LIKE_THRESHOLD = 1.0
        print(f"Using 'review_overall > {LIKE_THRESHOLD}' (z-score) as 'liked' threshold.")
        
        if 'review_overall' not in test_df.columns or 'profilename_encoded' not in test_df.columns:
             print(f"Error: '{TEST_FILE_NAME}' is missing 'review_overall' or 'profilename_encoded' for validation.")
             raise ValueError("Missing required columns in testing set")

        test_df = test_df.dropna(subset=['review_overall', 'profilename_encoded'])
        
        liked_beers_df = test_df[test_df['review_overall'] > LIKE_THRESHOLD]
        user_item_likes = liked_beers_df.groupby('profilename_encoded')['beer_beerid'].apply(list)
        user_item_likes = user_item_likes[user_item_likes.apply(len) >= 2]
        
        print(f"Found {len(user_item_likes)} users in the test set with 2 or more 'liked' (overall > {LIKE_THRESHOLD}) beers.")
        
        if len(user_item_likes) == 0:
            print(f"Cannot perform validation: No users found with 2+ liked beers > {LIKE_THRESHOLD}.")
        else:
            total_hits = 0
            total_recommendations = 0
            total_relevant_items = 0
            total_users_tested = 0
            N_recs = 10 # Number of recommendations to generate
            
            train_model_beers = set(item_sim_df_trained.index)
            
            for user_id, liked_list in user_item_likes.items():
                seed_beers = [beer for beer in liked_list if beer in train_model_beers]
                
                if not seed_beers:
                    continue 
                
                seed_beer_id = seed_beers[0]
                ground_truth_beers = set(liked_list) - {seed_beer_id}
                
                if not ground_truth_beers:
                    continue 
                    
                if seed_beer_id not in item_sim_df_trained.index:
                    continue 
                    
                sim_scores = item_sim_df_trained[seed_beer_id]
                top_scores = sim_scores.sort_values(ascending=False)
                top_N_indices = top_scores.index[1:N_recs+1]
                
                hits = 0
                for rec_beer_id in top_N_indices:
                    if rec_beer_id in ground_truth_beers:
                        hits += 1
                
                total_hits += hits
                total_recommendations += N_recs 
                total_relevant_items += len(ground_truth_beers)
                total_users_tested += 1

            if total_users_tested > 0:
                print(f"\n--- Validation Results (Precision@{N_recs} and Recall@{N_recs}) ---")
                
                # Calculate Precision
                precision = (total_hits / total_recommendations) * 100 if total_recommendations > 0 else 0
                
                # Calculate Recall
                recall = (total_hits / total_relevant_items) * 100 if total_relevant_items > 0 else 0
                
                print(f"Total users tested: {total_users_tested}")
                print(f"Total 'hits' (recommended beer was in user's other liked list): {total_hits}")
                print(f"Total recommendations made: {total_recommendations}")
                print(f"Total relevant items (ground truth): {total_relevant_items}")
                print(f"Average Precision@{N_recs}: {precision:.4f}%")
                print(f"Average Recall@{N_recs}: {recall:.4f}%")
            else:
                print("\n--- Validation Not Possible ---")
                print(f"Validation ran, but no users from the test set had 'liked' beers (>{LIKE_THRESHOLD}) that were also present in the '{TRAIN_FILE_NAME}'.")
                print("This means there is no overlap between the highly-rated beers in the test set and the beers in the training set.")
                print("Cannot calculate precision or recall.")

    except Exception as e:
        print(f"An error occurred during the validation phase with '{TEST_FILE_NAME}': {e}")
        import traceback
        traceback.print_exc()

except Exception as e:
    print(f"\n--- ERROR ---")
    print(f"An unexpected error occurred while loading '{TRAIN_FILE_NAME}': {e}")
    import traceback
    traceback.print_exc()