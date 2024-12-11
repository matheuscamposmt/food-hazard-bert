
def prepare_data(self, df, category):
    """
    Prepare data for a specific category
    """
    print(f"[DATA PREP] Total records: {len(df)}")
    
    # Combine title and text, ensure strings
    texts = (df['title'].fillna('') + ' ' + df['text'].fillna('')).astype(str)
    
    # Encode labels
    le = self.label_encoders[category]
    labels = le.fit_transform(df[category].fillna('Unknown'))
    
    print(f"[DATA PREP] Unique categories: {le.classes_}")
    print(f"[DATA PREP] Number of unique categories: {len(le.classes_)}")

    # Verify text format
    if not isinstance(texts, list):
        texts = texts.tolist()
    
    print(f"[DATA PREP] Sample texts (first 3):")
    for text in texts[:3]:
        print(f"  - {text[:100]}...")
    
    return texts, labels, le