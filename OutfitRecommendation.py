import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from PIL import Image
from io import BytesIO
import os

# Reading the CSV data
data = pd.read_csv('https://storage.googleapis.com/clozify-data-ml/recommendation_data.csv')
image_directory = "https://storage.googleapis.com/clozify-data-ml/images/"

def recommend_items_by_sub_category(data, gender, season, emotion_category, top_m=3, top_n=5):
    # Step 1: Group data by subCategory
    sub_categories = data['subCategory'].unique()
    
    recommendations = {}

    for category in sub_categories:
        # Filter data for the current subCategory and input attributes
        category_data = data[
            (data['gender'] == gender) &
            (data['season'] == season) &
            (data['Emotion_Category'] == emotion_category) &
            (data['subCategory'] == category)
        ].reset_index(drop=True)
        
        if category_data.empty:
            recommendations[category] = {"recommendations": []}
            continue
        
        # Extract embeddings
        embeddings = category_data.loc[:, '0':'2047'].astype(float).values
        
        # Compute similarity
        similarity_matrix = cosine_similarity(embeddings)
        
        category_recommendations = []

        for idx in range(min(len(category_data), top_m)):
            similarities = similarity_matrix[idx]
            similar_indices = np.argsort(similarities)[::-1][1:top_n + 1]
            
            # Recommendation structure
            recommendation = {
                'recommendations_item': {
                    'image': f"{image_directory}{category_data.iloc[idx]['image']}",
                    'name': category_data.iloc[idx]['productDisplayName']
                },
                'more_recommended_items': [
                    {
                        'image': f"{image_directory}{category_data.iloc[i]['image']}",
                        'name': category_data.iloc[i]['productDisplayName']
                    } for i in similar_indices
                ]
            }

            category_recommendations.append(recommendation)
        
        recommendations[category] = {"recommendations": category_recommendations}
    
    return recommendations

# Example usage
recommendations_by_category = recommend_items_by_sub_category(
    data,
    gender='Women',
    season='Rainy',
    emotion_category='Contentment'
)

# Convert recommendations to JSON format
import json
recommendations_json = json.dumps(recommendations_by_category, indent=4)

# Output the recommendations JSON
print(recommendations_json)
