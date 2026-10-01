"""Module 2: Data Preprocessing & Cleaning"""
import re
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

class TextPreprocessor:
    def __init__(self):
        try:
            self.stop_words = set(stopwords.words('english'))
        except:
            print("Downloading NLTK data...")
            nltk.download('stopwords', quiet=True)
            nltk.download('punkt', quiet=True)
            self.stop_words = set(stopwords.words('english'))
    
    def clean_text(self, text):
        """Clean and normalize text"""
        if pd.isna(text):
            return ""
        
        text = str(text)
        
        # Remove HTML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remove emojis and special characters
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        return text
    
    def preprocess_dataframe(self, df):
        """Preprocess entire dataframe"""
        print("Preprocessing data...")
        
        # Handle missing values
        df['Text'] = df['Text'].fillna('')
        df['Summary'] = df['Summary'].fillna('')
        df['ProfileName'] = df['ProfileName'].fillna('Unknown')
        
        # Combine text and summary
        df['combined_text'] = df['Summary'] + ' ' + df['Text']
        
        # Clean text
        df['cleaned_text'] = df['combined_text'].apply(self.clean_text)
        
        # Remove duplicates
        df = df.drop_duplicates(subset=['cleaned_text'])
        
        # Remove empty reviews
        df = df[df['cleaned_text'].str.len() > 10]
        
        print(f"After preprocessing: {len(df)} reviews")
        
        return df.reset_index(drop=True)
