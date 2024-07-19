import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from lightfm.evaluation import precision_at_k, auc_score

# Load data
# Assuming CSV files with columns: userId, productId, rating for interactions,
# and userId, userFeature for user features and productId, productFeature for item features
interactions = pd.read_csv('ecommerce_ratings.csv')
user_features = pd.read_csv('user_features.csv')
item_features = pd.read_csv('item_features.csv')

# Create a LightFM dataset
dataset = Dataset()

# Fit the dataset with user IDs and item IDs
dataset.fit(interactions['userId'], interactions['productId'])

# Build user and item features
dataset.fit_partial(
    users=user_features['userId'],
    items=item_features['productId'],
    user_features=user_features['userFeature'],
    item_features=item_features['productFeature']
)

# Build the interactions matrix
(interactions_matrix, weights) = dataset.build_interactions(
    [(x['userId'], x['productId'], x['rating']) for _, x in interactions.iterrows()]
)

# Build the user features matrix
user_features_matrix = dataset.build_user_features(
    [(x['userId'], [x['userFeature']]) for _, x in user_features.iterrows()]
)

# Build the item features matrix
item_features_matrix = dataset.build_item_features(
    [(x['productId'], [x['productFeature']]) for _, x in item_features.iterrows()]
)

# Instantiate the model
model = LightFM(loss='warp')

# Train the model
model.fit(interactions_matrix, 
          user_features=user_features_matrix,
          item_features=item_features_matrix,
          epochs=30, 
          num_threads=2)

# Evaluate the model
train_precision = precision_at_k(model, interactions_matrix, k=10, user_features=user_features_matrix, item_features=item_features_matrix).mean()
train_auc = auc_score(model, interactions_matrix, user_features=user_features_matrix, item_features=item_features_matrix).mean()

print(f'Train precision at k=10: {train_precision}')
print(f'Train AUC score: {train_auc}')

# Function to get top N product recommendations for a user
def get_top_n_recommendations(userId, n=10):
    # Get the user ID and item IDs
    user_id_map, user_feature_map, item_id_map, item_feature_map = dataset.mapping()
    user_index = user_id_map[userId]
    item_indices = list(item_id_map.values())
    
    # Predict scores for all items
    scores = model.predict(user_index, item_indices, user_features=user_features_matrix, item_features=item_features_matrix)
    
    # Get the top N items
    top_n_indices = scores.argsort()[-n:][::-1]
    top_n_item_ids = [list(item_id_map.keys())[list(item_id_map.values()).index(i)] for i in top_n_indices]
    
    return top_n_item_ids
user_id = 1
top_n_recommendations = get_top_n_recommendations(user_id, n=10)
print(f"Top 10 product recommendations for user {user_id}: {top_n_recommendations}")
