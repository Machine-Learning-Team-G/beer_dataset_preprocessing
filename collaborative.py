import os
from surprise import Reader, Dataset, AlgoBase
from surprise import KNNBasic, SVD
from surprise import accuracy
import pandas as pd
import numpy as np
import collections
import time
import seaborn as sns
import matplotlib.pyplot as plt

# CBF를 위한 scikit-learn 임포트
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import hstack

# --- 0. 전역 설정 및 출력 폴더 생성 ---
train_file = "training_set.csv"
test_file = "testing_set.csv"
beer_feature_file = "beers.csv" # CBF를 위한 맥주 특징 파일 (가정)

REVIEWS = ['review_overall', 'review_appearance', 'review_aroma', 'review_palate', 'review_taste']
READER = Reader(rating_scale=(0, 5)) 
MIN_COUNT_THRESHOLD = 100
K = 10
THRESHOLD = 4.5
N_USERS_ANALYSIS = 5 

BASE_OUTPUT_DIR = "analysis_results"
REPORT_DIR = os.path.join(BASE_OUTPUT_DIR, "reports")
VIZ_DIR = os.path.join(BASE_OUTPUT_DIR, "visualizations")
SVD_VIZ_DIR = os.path.join(VIZ_DIR, "svd_latent_features")
HEATMAP_DIR = os.path.join(VIZ_DIR, "comparison_heatmaps")

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SVD_VIZ_DIR, exist_ok=True)
os.makedirs(HEATMAP_DIR, exist_ok=True)
print(f"결과물은 '{BASE_OUTPUT_DIR}' 폴더 내에 저장됩니다.")


# --- 1. 데이터 로드 및 필터링 (최초 1회 실행) ---
print("--- 1. 데이터 로드 및 필터링 (최초 1회) ---")
train_df_global = pd.read_csv(train_file)
test_df_global = pd.read_csv(test_file)

# 1-1. CBF를 위한 맥주 특징 로드 및 전처리
try:
    beers_df = pd.read_csv(beer_feature_file)
    # CBF에서 사용할 'beer_style'과 'beer_abv'의 결측치 처리
    beers_df['beer_style'] = beers_df['beer_style'].fillna("Unknown")
    # 도수(abv) 결측치는 0.0으로 채움
    beers_df['beer_abv'] = beers_df['beer_abv'].fillna(0.0)
    print(f"'{beer_feature_file}' 로드 성공. {len(beers_df)}개의 맥주 특징 로드됨.")
    
    # CBF 모델에 필요한 필수 열 확인
    if 'beer_beerid' not in beers_df.columns or 'beer_style' not in beers_df.columns or 'beer_abv' not in beers_df.columns:
        print(f"경고: '{beer_feature_file}'에 'beer_beerid', 'beer_style', 'beer_abv' 열 중 하나가 없습니다.")
        print("Content-Based Filtering은 비활성화됩니다.")
        RUN_CBF = False
    else:
        RUN_CBF = True
        
except FileNotFoundError:
    print(f"경고: Content-Based Filtering에 필요한 '{beer_feature_file}' 파일을 찾을 수 없습니다.")
    print("CBF 모델을 제외하고 분석을 계속합니다.")
    RUN_CBF = False
    beers_df = None # 변수 초기화

# 1-2. 평점 데이터 필터링
train_beer_counts = train_df_global['beer_beerid'].value_counts()
beers_over_min_train = set(train_beer_counts[train_beer_counts > MIN_COUNT_THRESHOLD].index)
test_beer_counts = test_df_global['beer_beerid'].value_counts()
beers_over_min_test = set(test_beer_counts[test_beer_counts > MIN_COUNT_THRESHOLD].index)
common_beers_over_min = beers_over_min_test.intersection(beers_over_min_train)

common_beer_ids = list(common_beers_over_min)
num_common_items = len(common_beer_ids)
print(f"필터링 후 공통 항목(맥주) 개수: {num_common_items}")

train_df = train_df_global[train_df_global['beer_beerid'].isin(common_beers_over_min)].copy()
test_df = test_df_global[test_df_global['beer_beerid'].isin(common_beers_over_min)].copy()


# --- 2. Precision@K 및 Recall@K 계산 함수 정의 ---
def precision_recall_at_k(predictions, k=10, threshold=4.0):
    user_est_true = collections.defaultdict(list)
    for uid, iid, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()
    
    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        n_rec_k = user_ratings[:k]
        
        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rel_and_rec_k = sum((true_r >= threshold) for (_, true_r) in n_rec_k)

        precisions[uid] = n_rel_and_rec_k / k if k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    mean_precision = sum(precisions.values()) / len(precisions) if precisions else 0
    mean_recall = sum(recalls.values()) / len(recalls) if recalls else 0
    
    return mean_precision, mean_recall


# --- 3. 모델 학습 및 평가 파이프라인 ---
def evaluate_model(algo, name, train_data, test_data, k=10, threshold=4.0):
    start_time = time.time()
    
    trainset = train_data.build_full_trainset()
    testset_raw_3_tuples = [(uid, iid, r_ui) for (uid, iid, r_ui, *_) in test_data.raw_ratings]

    algo.fit(trainset)
    predictions = algo.test(testset_raw_3_tuples)
    
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    mean_precision, mean_recall = precision_recall_at_k(predictions, k=k, threshold=threshold)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    result_dict = {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        f"P@{k}": mean_precision,
        f"R@{k}": mean_recall,
        "Time (s)": elapsed_time
    }
    
    return result_dict, algo, trainset

# --- 3-1. Content-Based Filtering 커스텀 클래스 정의 ---
class ContentKNN(AlgoBase):
    def __init__(self, beer_features_df, k=40):
        AlgoBase.__init__(self)
        self.k = k
        # 'beer_beerid'를 인덱스로 사용하여 특징 DF를 미리 준비
        self.beer_features_df = beer_features_df.set_index('beer_beerid')
        self.sim_matrix = None
        self.global_mean = None
        self.item_profiles = None

    def fit(self, trainset):
        AlgoBase.fit(self, trainset)
        
        print("Fitting ContentKNN model...")
        self.global_mean = trainset.global_mean
        
        # 1. Build Item-Feature Matrix (Item Profiles)
        n_items = trainset.n_items
        # trainset의 내부 ID 순서대로 원본(raw) ID 리스트 생성
        all_raw_iids = [trainset.to_raw_iid(iid) for iid in range(n_items)]
        
        # 원본 ID 순서에 맞게 특징 DF 재정렬
        features_df = self.beer_features_df.reindex(all_raw_iids)
        
        # (중요) reindex 후 발생할 수 있는 결측치 재처리
        features_df['beer_style'] = features_df['beer_style'].fillna("Unknown")
        features_df['beer_abv'] = features_df['beer_abv'].fillna(0.0)
        
        print("  1/3: Preprocessing features (TF-IDF for style, Scaling for abv)...")
        # a. TF-IDF on 'beer_style'
        tfidf = TfidfVectorizer(stop_words='english')
        style_matrix = tfidf.fit_transform(features_df['beer_style'])
        
        # b. Scale 'beer_abv'
        scaler = MinMaxScaler()
        abv_matrix = scaler.fit_transform(features_df[['beer_abv']])
        
        # c. Combine item_profiles (n_items x n_features)
        item_profiles_sparse = hstack([style_matrix, abv_matrix])
        
        print("  2/3: Calculating content-based similarity matrix...")
        # 2. Compute Item-Item Content Similarity Matrix
        self.sim_matrix = cosine_similarity(item_profiles_sparse)
        
        print("  3/3: ContentKNN fit complete.")
        return self

    def estimate(self, u, i):
        # 'u' and 'i' are INNER IDs
        
        # 사용자가 훈련셋에 없거나, 아이템이 훈련셋에 없으면 (Cold Start)
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.global_mean
        
        # Get all items rated by user u
        try:
            user_ratings = self.trainset.ur[u] # list of (item_inner_id, rating)
        except KeyError:
            # (드물지만) 사용자가 testset에만 있는 경우
            return self.global_mean
        
        # Get similarities between target item 'i' and items user 'u' rated
        neighbors = []
        for (j, r) in user_ratings:
            # i, j는 모두 inner ID이므로 sim_matrix에서 직접 조회 가능
            sim = self.sim_matrix[i, j]
            neighbors.append((sim, r))
        
        # Sort by similarity and get top K
        neighbors.sort(key=lambda x: x[0], reverse=True)
        top_k_neighbors = neighbors[:self.k]
        
        # Calculate weighted average
        sum_sim = sum(abs(sim) for (sim, r) in top_k_neighbors)
        sum_weighted_ratings = sum(sim * r for (sim, r) in top_k_neighbors)
        
        if sum_sim == 0:
            return self.global_mean # No similar items rated
        
        prediction = sum_weighted_ratings / sum_sim
        
        # Clamp prediction to rating scale
        prediction = min(self.trainset.rating_scale[1], prediction)
        prediction = max(self.trainset.rating_scale[0], prediction)
        
        return prediction


# --- 최종 결과를 저장할 마스터 리스트 ---
all_model_results = []
all_user_analyses = []

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.max_colwidth', None)

# ==============================================================================
# ===== 메인 루프 시작: 5개의 리뷰 타겟에 대해 전체 분석 실행 =====
# ==============================================================================

for target_review in REVIEWS:
    
    print("\n" + "="*80)
    print(f"===== ANALYZING TARGET REVIEW: {target_review} =====")
    print("="*80)

    # --- 4. 모델별 데이터 준비 ---
    train_data = Dataset.load_from_df(train_df[['profilename_encoded', 'beer_beerid', target_review]], READER)
    test_data = Dataset.load_from_df(test_df[['profilename_encoded', 'beer_beerid', target_review]], READER)

    # --- 5. 모델 설정 및 비교 실행 ---
    # (동적으로 모델 딕셔너리 생성)
    MODELS_TO_RUN = {
        "Item-based CF (KNN)": KNNBasic(sim_options={"name": "cosine", "user_based": False}),
        "User-based CF (KNN)": KNNBasic(sim_options={"name": "cosine", "user_based": True}),
        "Model-based CF (SVD)": SVD(n_factors=100, random_state=42)
    }
    
    # CBF 모델은 RUN_CBF 플래그가 True일 때만 추가
    if RUN_CBF:
        MODELS_TO_RUN["Content-based CF (KNN)"] = ContentKNN(beer_features_df=beers_df, k=40)

    fitted_models = {} 
    analysis_trainsets = {} 
    results = []
    
    print(f"\n--- 5. [{target_review}] {len(MODELS_TO_RUN)}개 추천 모델 비교 평가 ---")

    for name, algo in MODELS_TO_RUN.items():
        print(f"-> [{target_review}] {name} 모델 평가 중...")
        result, fitted_algo, fitted_trainset = evaluate_model(algo, name, train_data, test_data, k=K, threshold=THRESHOLD)
        
        results.append(result)
        fitted_models[name] = fitted_algo
        analysis_trainsets[name] = fitted_trainset 
        
        print(f"   완료. RMSE: {result['RMSE']:.4f}, P@{K}: {result[f'P@{K}']:.4f}")

    results_df = pd.DataFrame(results).set_index("Model")
    results_df['target_review'] = target_review
    all_model_results.append(results_df)

    print(f"\n--- [{target_review}] 모델별 성능 비교 ---")
    print(results_df.to_markdown(floatfmt=".4f"))

    # (중요) SVD와 CBF는 시각화/분석에서 SVD 모델을 사용
    current_svd_model = fitted_models["Model-based CF (SVD)"]
    current_svd_trainset = analysis_trainsets["Model-based CF (SVD)"]

    # --- 6. SVD 잠재 특징 시각화 (n_factors=2로 재학습) ---
    print(f"\n--- 6. [{target_review}] SVD 잠재 특징 시각화 ---")
    
    N_FACTORS_VIZ = 2
    algo_svd_viz = SVD(n_factors=N_FACTORS_VIZ, random_state=42, verbose=False)
    trainset_viz = train_data.build_full_trainset() 
    algo_svd_viz.fit(trainset_viz) 

    try:
        user_raw_ids = [trainset_viz.to_raw_uid(uid) for uid in trainset_viz.all_users()]
        user_latent_df = pd.DataFrame(algo_svd_viz.pu, index=user_raw_ids)
        user_latent_df.columns = [f'Factor_{i+1}' for i in range(N_FACTORS_VIZ)]
        user_latent_df = user_latent_df.reset_index().rename(columns={'index': 'UserID'})

        item_raw_ids = [trainset_viz.to_raw_iid(iid) for iid in trainset_viz.all_items()]
        item_latent_df = pd.DataFrame(algo_svd_viz.qi, index=item_raw_ids)
        item_latent_df.columns = [f'Factor_{i+1}' for i in range(N_FACTORS_VIZ)]
        item_latent_df = item_latent_df.reset_index().rename(columns={'index': 'ItemID'})

        fig, ax = plt.subplots(figsize=(12, 12))
        ITEM_SAMPLE_SIZE = min(100, len(item_latent_df)) 
        sampled_items = item_latent_df.sample(n=ITEM_SAMPLE_SIZE, random_state=42)
        sns.scatterplot(
            x='Factor_1', y='Factor_2', data=sampled_items, ax=ax, 
            label=f'Items (N={ITEM_SAMPLE_SIZE})', alpha=0.6, s=70, color='blue'
        )
        USER_SAMPLE_SIZE = min(100, len(user_latent_df))
        sampled_users = user_latent_df.sample(n=USER_SAMPLE_SIZE, random_state=42)
        sns.scatterplot(
            x='Factor_1', y='Factor_2', data=sampled_users, ax=ax, 
            label=f'Users (N={USER_SAMPLE_SIZE})', alpha=0.6, s=50, marker='x', color='red'
        )
        
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(0, color='gray', linestyle='--', linewidth=0.8)
        
        ax.set_title(f'SVD Latent Factor Visualization ({target_review})', fontsize=16)
        ax.set_xlabel('Latent Factor 1', fontsize=14)
        ax.set_ylabel('Latent Factor 2', fontsize=14)
        ax.legend(title='Entity Type', fontsize=12)
        ax.grid(True, linestyle=':', alpha=0.7)

        plt.tight_layout()
        svd_save_path = os.path.join(SVD_VIZ_DIR, f"svd_latent_features_{target_review}.png")
        plt.savefig(svd_save_path)
        plt.close(fig) 
        print(f"SVD 잠재 특징 그래프가 '{svd_save_path}' 파일로 저장되었습니다.")

    except Exception as e:
        print(f"SVD 시각화 중 오류 발생: {e}")
        if 'plt' in locals() and plt.gcf().get_axes():
            plt.close('all') 

    # --- 7. 추천 전후 평점 매트릭스 비교 히트맵 ---
    print(f"\n--- 7. [{target_review}] 추천 전/후 비교 히트맵 시각화 ---")

    try:
        N_USERS_ITEMS = num_common_items 
        sampled_users_heatmap = np.random.choice(
            train_df['profilename_encoded'].unique(), 
            size=N_USERS_ITEMS, 
            replace=False
        ).tolist()
        sampled_items = common_beer_ids 

        before_matrix_df = train_df[train_df['profilename_encoded'].isin(sampled_users_heatmap)].pivot_table(
            index='profilename_encoded', 
            columns='beer_beerid', 
            values=target_review 
        )
        before_matrix_df = before_matrix_df.reindex(index=sampled_users_heatmap, columns=sampled_items) 

        after_matrix_data = []
        for user_id in sampled_users_heatmap:
            for item_id in sampled_items:
                pred = current_svd_model.predict(user_id, item_id)
                after_matrix_data.append({'profilename_encoded': user_id, 'beer_beerid': item_id, 'predicted_rating': pred.est})

        after_matrix_df = pd.DataFrame(after_matrix_data).pivot_table(
            index='profilename_encoded', 
            columns='beer_beerid', 
            values='predicted_rating'
        )
        after_matrix_df = after_matrix_df.reindex(index=sampled_users_heatmap, columns=sampled_items) 

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        sns.heatmap(
            before_matrix_df, ax=ax1, cmap='viridis', annot=True, fmt=".1f", linewidths=.5,
            cbar_kws={'label': f'Original Rating ({target_review})'}, vmin=0, vmax=5 
        )
        ax1.set_title(f'A: Before Recommendation (Originals)', fontsize=16)
        ax1.set_xlabel('Beer IDs (Filtered)')
        ax1.set_ylabel(f'User IDs ({N_USERS_ITEMS} Sampled)')

        sns.heatmap(
            after_matrix_df, ax=ax2, cmap='viridis', annot=True, fmt=".2f", linewidths=.5,
            cbar_kws={'label': 'Predicted Rating (SVD)'}, vmin=0, vmax=5 
        )
        ax2.set_title(f'B: After Recommendation (SVD Predictions)', fontsize=16)
        ax2.set_xlabel('Beer IDs (Filtered)')
        ax2.set_ylabel('') 
        ax2.set_yticks([]) 

        fig.suptitle(f'Rating Matrix Completion: Before vs After (Target: {target_review})', fontsize=20, y=1.03)
        plt.tight_layout()
        heatmap_save_path = os.path.join(HEATMAP_DIR, f"cf_before_after_comparison_heatmap_{target_review}.png")
        plt.savefig(heatmap_save_path)
        plt.close(fig) 
        print(f"추천 전/후 비교 히트맵이 '{heatmap_save_path}' 파일로 저장되었습니다.")
    
    except Exception as e:
        print(f"비교 히트맵 시각화 중 오류 발생: {e}")
        if 'plt' in locals() and plt.gcf().get_axes():
            plt.close('all')

    # --- 8. 개별 사용자 추천 결과 분석 (5명 요약 테이블) ---
    print(f"\n--- 8. [{target_review}] 개별 사용자 추천 결과 요약 테이블 ---")

    analysis_results = []
    
    try:
        available_users = test_df['profilename_encoded'].unique()
        if len(available_users) == 0:
            raise IndexError("테스트 셋에 사용자가 없습니다.")
        
        N_USERS_ANALYSIS_CURRENT = min(N_USERS_ANALYSIS, len(available_users))
        sampled_analysis_users = np.random.choice(
            available_users, 
            size=N_USERS_ANALYSIS_CURRENT, 
            replace=False
        ).tolist()

        for i, target_user_raw_id in enumerate(sampled_analysis_users):
            user_analysis = {"User ID": target_user_raw_id, "Target Review": target_review}
            
            # A. 평가 분석 (Test Set)
            user_test_data = test_df[test_df['profilename_encoded'] == target_user_raw_id]
            
            if user_test_data.empty:
                user_analysis["Actual Likes (Test)"] = "N/A (No Test Data)"
                user_analysis["Top-K Predictions (Test)"] = "N/A"
                user_analysis["P@K (Test)"] = 0.0
            else:
                actual_likes_list = user_test_data[user_test_data[target_review] >= THRESHOLD]['beer_beerid'].tolist()
                actual_likes_str = ", ".join(map(str, actual_likes_list)) if actual_likes_list else "None"
                user_analysis["Actual Likes (Test)"] = actual_likes_str
                
                user_test_predictions = []
                for _, row in user_test_data.iterrows():
                    item_id = row['beer_beerid']
                    actual_rating = row[target_review] 
                    pred = current_svd_model.predict(target_user_raw_id, item_id)
                    user_test_predictions.append((pred.est, actual_rating, item_id))

                user_test_predictions.sort(key=lambda x: x[0], reverse=True)

                hits = 0
                top_k_test_preds_list = []
                for j, (est, true_r, iid) in enumerate(user_test_predictions[:K]):
                    is_hit = (true_r >= THRESHOLD)
                    if is_hit: hits += 1
                    status = "HIT" if is_hit else "MISS"
                    top_k_test_preds_list.append(f"{j+1}. {iid} (Est: {est:.2f}, Act: {true_r:.1f} [{status}])")
                
                user_analysis["Top-K Predictions (Test)"] = "\n".join(top_k_test_preds_list) if top_k_test_preds_list else "None"
                user_analysis["P@K (Test)"] = hits / K if K > 0 else 0

            # B. 실제 추천 (Unseen Items)
            try:
                target_user_inner_id = current_svd_trainset.to_inner_uid(target_user_raw_id)
                seen_items_inner = set([iid for (iid, r) in current_svd_trainset.ur[target_user_inner_id]])
                all_items_inner = set(current_svd_trainset.all_items())
                items_to_recommend_inner = all_items_inner - seen_items_inner

                unseen_predictions = []
                for iid_inner in items_to_recommend_inner:
                    iid_raw = current_svd_trainset.to_raw_iid(iid_inner)
                    pred = current_svd_model.predict(target_user_raw_id, iid_raw)
                    unseen_predictions.append((pred.est, iid_raw))

                unseen_predictions.sort(key=lambda x: x[0], reverse=True)
                
                top_k_unseen_recs_list = []
                for j, (est, iid) in enumerate(unseen_predictions[:K]):
                    top_k_unseen_recs_list.append(f"{j+1}. {iid} (Est: {est:.2f})")

                user_analysis[f"Top-{K} Recommendations (Unseen)"] = "\n".join(top_k_unseen_recs_list) if top_k_unseen_recs_list else "None"

            except ValueError:
                user_analysis[f"Top-{K} Recommendations (Unseen)"] = "N/A (User not in Trainset)"

            analysis_results.append(user_analysis)
        
        all_user_analyses.extend(analysis_results)
        
        analysis_df = pd.DataFrame(analysis_results).set_index(["User ID", "Target Review"])
        print(f"\n--- [{target_review}] 5명 사용자 분석 요약 (Markdown Table) ---")
        print(analysis_df.to_markdown())

    except Exception as e:
        print(f"사용자 분석 중 오류 발생: {e}")

# ==============================================================================
# ===== 메인 루프 종료 =====
# ==============================================================================

print("\n" + "="*80)
print("===== 모든 리뷰 타겟 분석 완료 =====")
print("="*80)

# --- 10. 최종 리포트 생성 ---
try:
    if not all_model_results:
        print("경고: 분석된 모델 결과가 없습니다. 리포트를 생성할 수 없습니다.")
    else:
        # 10-1. 모델 성능 통합 리포트
        final_model_report = pd.concat(all_model_results)
        report_csv_path = os.path.join(REPORT_DIR, "final_model_evaluation_report_all_reviews.csv")
        report_md_path = os.path.join(REPORT_DIR, "final_model_evaluation_report_all_reviews.md")
        
        final_model_report.to_csv(report_csv_path)
        final_model_report.to_markdown(report_md_path)
        
        print("\n--- 10-1. 최종 모델 평가 통합 리포트 (모든 리뷰) ---")
        print(final_model_report.to_markdown(floatfmt=".4f"))
        print(f"\n-> '{report_csv_path}' 파일로 저장되었습니다.")

    if not all_user_analyses:
        print("경고: 사용자 분석 결과가 없습니다. 리포트를 생성할 수 없습니다.")
    else:
        # 10-2. 사용자 분석 통합 리포트
        final_user_report = pd.DataFrame(all_user_analyses)
        final_user_report = final_user_report.set_index(["Target Review", "User ID"])
        user_report_csv_path = os.path.join(REPORT_DIR, "final_user_analysis_report_all_reviews.csv")
        
        final_user_report.to_csv(user_report_csv_path)
        print("\n--- 10-2. 최종 사용자 분석 통합 리포트 (모든 리뷰) ---")
        print(final_user_report)
        print(f"\n-> '{user_report_csv_path}' 파일로 저장되었습니다.")

except Exception as e:
    print(f"최종 리포트 생성 중 오류 발생: {e}")


# --- 11. 베스트 케이스 요약 ---
print("\n" + "="*80)
print("===== 베스트 케이스 성능 요약 (터미널 출력) =====")
print("="*80)

if not all_model_results:
    print("분석된 모델 결과가 없어 베스트 케이스를 요약할 수 없습니다.")
else:
    try:
        precision_col = f"P@{K}"
        recall_col = f"R@{K}"

        report_for_summary = final_model_report.reset_index()

        # 1. Best RMSE (Lowest)
        best_rmse_idx = report_for_summary['RMSE'].idxmin()
        best_rmse_series = report_for_summary.loc[best_rmse_idx]
        print("\n[--- 베스트 RMSE (예측 정확도) ---]")
        print(f"  Model:         {best_rmse_series['Model']}")
        print(f"  Target Review: {best_rmse_series['target_review']}")
        print(f"  RMSE Value:    {best_rmse_series['RMSE']:.4f}")

        # 2. Best Precision@K (Highest)
        best_p_at_k_idx = report_for_summary[precision_col].idxmax()
        best_p_at_k_series = report_for_summary.loc[best_p_at_k_idx]
        print(f"\n[--- 베스트 Precision@{K} (Top-N 추천 품질) ---]")
        print(f"  Model:         {best_p_at_k_series['Model']}")
        print(f"  Target Review: {best_p_at_k_series['target_review']}")
        print(f"  P@{K} Value:    {best_p_at_k_series[precision_col]:.4f}")

        # 3. Best Recall@K (Highest)
        best_r_at_k_idx = report_for_summary[recall_col].idxmax()
        best_r_at_k_series = report_for_summary.loc[best_r_at_k_idx]
        print(f"\n[--- 베스트 Recall@{K} (선호 항목 커버리지) ---]")
        print(f"  Model:         {best_r_at_k_series['Model']}")
        print(f"  Target Review: {best_r_at_k_series['target_review']}")
        print(f"  R@{K} Value:    {best_r_at_k_series[recall_col]:.4f}")
        
        print("\n" + "="*80)

    except Exception as e:
        print(f"베스트 케이스 요약 중 오류 발생: {e}")


print("\n모든 분석 파이프라인이 완료되었습니다.")
print(f"생성된 파일들은 '{REPORT_DIR}' 및 '{VIZ_DIR}' 폴더에 저장되었습니다.")