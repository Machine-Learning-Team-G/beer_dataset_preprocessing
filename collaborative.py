import pandas as pd
#from sklearn.preprocessing import StandardScaler, MinMaxScaler
#from sklearn.decomposition import PCA
from surprise import Dataset, Reader, SVD, NMF
from surprise.model_selection import cross_validate, train_test_split, GridSearchCV
from surprise import accuracy
from collections import defaultdict
import sys

# -----------------------------------------------------------------
# --- PART 0: SETUP AND DATA LOADING ---
# -----------------------------------------------------------------

# Helper function to load the unusually formatted mapping files
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
        mapping_df[id_col_name] = pd.to_numeric(mapping_df[id_col_name], errors='coerce')
        mapping_df = mapping_df.dropna(subset=[id_col_name])
        mapping_df = mapping_df.drop_duplicates(subset=[id_col_name]).set_index(id_col_name)
        
        if pd.api.types.is_float_dtype(mapping_df.index):
            try:
                mapping_df.index = mapping_df.index.astype(int)
            except ValueError:
                pass
                
        return mapping_df
    except pd.errors.EmptyDataError:
        print(f"Warning: File {filepath} is empty.")
        return pd.DataFrame(columns=[id_col_name, val_col_name]).set_index(id_col_name)
    except Exception as e:
        print(f"Error loading mapping file {filepath}: {e}")
        return pd.DataFrame(columns=[id_col_name, val_col_name]).set_index(id_col_name)


print("Loading mapping files...")
full_name_map = None
try:
    test_name_map = load_mapping_file("testingset_id_to_name.csv", "beer_beerid", "beer_name")
    train_name_map = load_mapping_file("trainingset_id_to_name.csv", "beer_beerid", "beer_name")
    full_name_map = pd.concat([test_name_map, train_name_map])
    full_name_map = full_name_map[~full_name_map.index.duplicated(keep='first')]
    print(f"Loaded {len(full_name_map)} unique beer names.")
except Exception as e:
    print(f"Error loading name maps: {e}")

# Define File Names
TRAIN_FILE_NAME = "training_set.csv"
TEST_FILE_NAME = "testing_set.csv"
LIKE_THRESHOLD = 4.0 # Unscaled ratings (1-5). "Liked" = 4.0 or higher.

# Hyperparameters
n_factors = 100
n_epochs = 50


def tuning(dataset): # 이걸 토대로 최적화를 진행하려 했으나 시간이 매우 오래 걸리는 문제가 발생해서 수작업으로 진행했습니다.
    param_grid = {
    "n_factors": [50, 100, 150],
    "n_epochs": [20, 30, 50],
    "lr_all": [0.002, 0.005, 0.01],
    "reg_all": [0.02, 0.05, 0.1],
    "biased": [True] 
    }
    gs = GridSearchCV(SVD, param_grid, measures=["rmse"], cv=3, joblib_verbose=0)
    gs.fit(dataset)

    print(gs.best_score["rmse"], gs.best_params["rmse"])


try:
    print(f"\n--- Loading '{TRAIN_FILE_NAME}' for model training ---")
    train_df = pd.read_csv(TRAIN_FILE_NAME)
    print(f"Loaded {len(train_df)} reviews from '{TRAIN_FILE_NAME}'.")
    
    print(f"\n--- Loading '{TEST_FILE_NAME}' for model validation ---")
    test_df = pd.read_csv(TEST_FILE_NAME)
    print(f"Loaded {len(test_df)} reviews from '{TEST_FILE_NAME}'.")
    
    # -----------------------------------------------------------------
    # --- PART 1: PREPARE DATASET AND CROSS-VALIDATE (RMSE) ---
    # -----------------------------------------------------------------

    
    # The Reader object is needed to parse the unscaled ratings
    # The previous preprocessing step removed scaling, so we assume 1-5 scale.
    reader = Reader(rating_scale=(1, 5))
    
    # Load the training data into Surprise's format
    print("\nLoading training data into Surprise dataset...")
    train_data_surprise = Dataset.load_from_df(train_df[['profilename_encoded', 'beer_beerid', 'review_overall']], reader)
    
    # tuning(train_data_surprise)
    
    # Run 3-fold cross-validation (CV) on the *training set*
    # This gives us a baseline RMSE (Root Mean Squared Error)
    # RMSE tells us (on average) how far our rating *predictions* are from the *actual* rating
    print("\nRunning 3-fold cross-validation on the training set...")
    cv_results = cross_validate(SVD(), train_data_surprise, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    
    # -----------------------------------------------------------------
    # --- PART 2: BUILD FINAL MODEL ---
    # -----------------------------------------------------------------
    
    print("\nTraining final SVD model on the full training set...")
    # Build the full training set from the data
    trainset = train_data_surprise.build_full_trainset()
    
    # Instantiate and train the SVD model
    model = SVD(n_factors, n_epochs, random_state=42, lr_all=0.01)
    model.fit(trainset)
    print("Model training complete.")
    
    # -----------------------------------------------------------------
    # --- PART 3: VALIDATE ON THE *TEST SET* (Precision/Recall) ---
    # -----------------------------------------------------------------
    
    print(f"\n--- Validating model on '{TEST_FILE_NAME}' ---")
    print(f"Using 'review_overall >= {LIKE_THRESHOLD}' as the 'like' threshold.")
    
    # Load the test data into Surprise's format
    test_data_surprise = Dataset.load_from_df(test_df[['profilename_encoded', 'beer_beerid', 'review_overall']], reader)
    
    # Convert to a 'testset' format (a list of raw rating tuples)
    # This format is required for the accuracy.precision_recall_at_k function
    testset = test_data_surprise.build_full_trainset().build_testset()
    
    # Get predictions for *all* ratings in the test set
    predictions = model.test(testset)
    
    # Calculate Precision and Recall at k=10
    def precision_recall_at_k(predictions, k=10, threshold=LIKE_THRESHOLD):
        """Return precision and recall at k for each user."""
        
        # First, map the predictions to each user.
        user_est_true = defaultdict(list)
        for uid, _, true_r, est, _ in predictions:
            user_est_true[uid].append((est, true_r))

        precisions = dict()
        recalls = dict()
        for uid, user_ratings in user_est_true.items():

            # Sort user ratings by estimated value
            user_ratings.sort(key=lambda x: x[0], reverse=True)

            # Number of relevant items
            n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)

            # Number of recommended items in top k
            n_rec_k = sum((est >= threshold) for (est, _) in user_ratings[:k])

            # Number of relevant and recommended items in top k
            n_rel_and_rec_k = sum(((true_r >= threshold) and (est >= threshold))
                                  for (est, true_r) in user_ratings[:k])

            # Precision@k: Proportion of recommended items that are relevant
            precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0

            # Recall@k: Proportion of relevant items that are recommended
            recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

        # Average across all users
        avg_precision = sum(p for p in precisions.values()) / len(precisions)
        avg_recall = sum(r for r in recalls.values()) / len(recalls)
        
        return avg_precision, avg_recall

    avg_precision, avg_recall = precision_recall_at_k(predictions, k=10, threshold=LIKE_THRESHOLD)
    
    print("\n--- Validation Results (Precision@10 and Recall@10) ---")
    print(f"Average Precision@10: {avg_precision*100:.4f}%")
    print(f"Average Recall@10: {avg_recall*100:.4f}%")
    
    # Calculate F1 Score
    if (avg_precision + avg_recall) > 0:
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall)
        print(f"Average F1-Score@10: {f1_score*100:.4f}%")
    else:
        print("F1-Score@10: 0.0000%")
        
    # -----------------------------------------------------------------
    # --- PART 4: DEMO RECOMMENDATION FUNCTION ---
    # -----------------------------------------------------------------
    
    def get_collaborative_recommendations(user_id, model, train_df, full_name_map, n=10):
        """
        Generates Top-N recommendations for a user based on the SVD model.
        """
        print("\n" + "="*50)
        print(f"RECOMMENDATIONS FOR USER: {user_id}")
        print("="*50)
        
        # Get the set of all beer IDs in the training set (our "universe")
        all_beer_ids = set(train_df['beer_beerid'].unique())
        
        # Get the set of beer IDs this user has *already rated*
        try:
            rated_beer_ids = set(train_df[train_df['profilename_encoded'] == user_id]['beer_beerid'])
            print(f"This user has rated {len(rated_beer_ids)} beers.")
        except:
            print("Could not find this user in the training set.")
            return

        # Get the set of beers to predict (all beers - rated beers)
        beers_to_predict = all_beer_ids - rated_beer_ids
        print(f"Calculating predictions for {len(beers_to_predict)} unrated beers...")
        
        # Predict the rating for each unrated beer
        preds = [model.predict(user_id, beer_id) for beer_id in beers_to_predict]
        
        # Sort predictions by estimated rating (highest first)
        top_preds = sorted(preds, key=lambda x: x.est, reverse=True)[:n]
        
        # Get the beer IDs and predicted ratings
        top_n_recommendations = []
        for pred in top_preds:
            beer_id = pred.iid
            predicted_rating = pred.est
            
            # Get beer name
            try:
                beer_name = full_name_map.loc[beer_id, 'beer_name']
            except KeyError:
                beer_name = f"Unknown Name (ID: {beer_id})"
                
            top_n_recommendations.append((beer_name, predicted_rating))

        print("\nTop 10 Collaborative Filtering Recommendations:")
        for beer_name, rating in top_n_recommendations:
            print(f"  - {beer_name} (Predicted Rating: {rating:.2f})")
            
        return top_n_recommendations

    # -----------------------------------------------------------------
    # --- PART 5: RUN DEMO ---
    # -----------------------------------------------------------------
    
    # Find the most active user in the training set for the demo
    demo_user_id = train_df['profilename_encoded'].value_counts().index[0]
    
    # Show what this user *actually* liked (for comparison)
    print("\n--- Running demo for most active user ---")
    print(f"Demo User ID (profilename_encoded): {demo_user_id}")
    
    user_actual_likes = train_df[
        (train_df['profilename_encoded'] == demo_user_id) & 
        (train_df['review_overall'] >= LIKE_THRESHOLD)
    ].sort_values(by='review_overall', ascending=False)
    
    print(f"\nTop 5 *actual* likes for this user (for comparison):")
    for beer_id in user_actual_likes['beer_beerid'].head(5).values:
        try:
            print(f"  - {full_name_map.loc[beer_id, 'beer_name']}")
        except:
            print(f"  - Unknown Name (ID: {beer_id})")

    # Get and print the new recommendations
    get_collaborative_recommendations(demo_user_id, model, train_df, full_name_map, n=10)


except FileNotFoundError as e:
    print(f"Error: Could not find file {e.filename}. Please ensure files are present.")
except pd.errors.EmptyDataError:
    print(f"Error: One of the files '{TRAIN_FILE_NAME}' or '{TEST_FILE_NAME}' is empty.")
except Exception as e:
    print(f"An error occurred during the analysis: {e}")
    import traceback
    traceback.print_exc()