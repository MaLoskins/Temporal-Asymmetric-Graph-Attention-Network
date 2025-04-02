"""
Synthetic Social Media Conversation Data Generator

This script generates synthetic social media conversation data with text content,
user interactions, and temporal information, suitable for preprocessing into
temporal graph format for TAGAN.
"""

import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
import os
from faker import Faker
import networkx as nx
import torch
from collections import defaultdict

# Initialize Faker for generating realistic text
fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)

class SocialMediaDataGenerator:
    """Generator for synthetic social media conversation data."""
    
    def __init__(
        self,
        num_users=50,
        num_threads=20,
        max_posts_per_thread=15,
        max_replies_per_post=5,
        time_span_days=5,
        controversial_ratio=0.3,
        output_dir="./data/raw"
    ):
        """
        Initialize the generator.
        
        Args:
            num_users: Number of users in the dataset
            num_threads: Number of conversation threads
            max_posts_per_thread: Maximum posts in each thread
            max_replies_per_post: Maximum replies to each post
            time_span_days: Time span of conversations in days
            controversial_ratio: Ratio of threads labeled as controversial
            output_dir: Directory to save generated data
        """
        self.num_users = num_users
        self.num_threads = num_threads
        self.max_posts_per_thread = max_posts_per_thread
        self.max_replies_per_post = max_replies_per_post
        self.time_span_days = time_span_days
        self.controversial_ratio = controversial_ratio
        self.output_dir = output_dir
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Topic categories
        self.topics = [
            "technology", "politics", "sports", "entertainment",
            "science", "health", "environment", "business"
        ]
        
        # Generate user profiles
        self.users = self._generate_users()
        
        # Sentiment words for post content generation
        self.positive_words = [
            "love", "great", "excellent", "amazing", "wonderful", "brilliant",
            "fantastic", "outstanding", "terrific", "superb", "happy", "joy"
        ]
        
        self.negative_words = [
            "hate", "terrible", "awful", "horrible", "disappointing", "poor",
            "bad", "mediocre", "frustrating", "annoying", "angry", "sad"
        ]
        
        self.neutral_words = [
            "okay", "fine", "average", "moderate", "reasonable", "fair",
            "acceptable", "decent", "standard", "normal", "common", "regular"
        ]
        
        # Words more likely to appear in controversial threads
        self.controversial_words = [
            "disagree", "argument", "debate", "wrong", "false", "incorrect",
            "misleading", "biased", "unfair", "controversial", "dispute", "conflict"
        ]
    
    def _generate_users(self):
        """Generate user profiles."""
        users = []
        for i in range(self.num_users):
            user = {
                "user_id": f"user_{i}",
                "name": fake.name(),
                "age": random.randint(18, 70),
                "interests": random.sample(self.topics, random.randint(1, 3)),
                "activity_level": random.choice(["low", "medium", "high"]),
                "join_date": fake.date_time_between(start_date="-2y", end_date="now").strftime("%Y-%m-%d")
            }
            users.append(user)
        return users
    
    def _generate_post_content(self, is_controversial=False, is_reply=False, parent_content=None):
        """
        Generate text content for a post or reply.
        
        Args:
            is_controversial: Whether the thread is controversial
            is_reply: Whether this is a reply
            parent_content: Content of the parent post if this is a reply
            
        Returns:
            Text content for the post
        """
        # Base content
        if is_reply and parent_content:
            # Sometimes reference the parent post
            if random.random() < 0.3:
                opening = random.choice([
                    "I agree that ", "I disagree that ", "You're right about ", 
                    "I don't think ", "Interesting point about ", "Regarding "
                ])
                # Extract a fragment from the parent content
                words = parent_content.split()
                if len(words) > 5:
                    fragment = " ".join(random.sample(words, min(5, len(words))))
                    content = opening + fragment + ". "
                else:
                    content = opening + parent_content + ". "
            else:
                content = ""
        else:
            # New thread starter
            topic = random.choice(self.topics)
            content = f"[{topic.upper()}] "
        
        # Add some sentences
        num_sentences = random.randint(1, 3)
        for _ in range(num_sentences):
            content += fake.sentence()
            
        # Add sentiment words
        if is_controversial:
            # Controversial threads have more extreme sentiment
            sentiment_words = random.sample(
                self.positive_words + self.negative_words + self.controversial_words,
                random.randint(1, 3)
            )
        else:
            # Non-controversial threads have more neutral sentiment
            sentiment_words = random.sample(
                self.positive_words + self.neutral_words,
                random.randint(0, 2)
            )
        
        # Insert sentiment words
        if sentiment_words:
            sentiment_phrase = " I feel " + " and ".join(sentiment_words) + " about this. "
            content += sentiment_phrase
            
        return content
    
    def generate_data(self):
        """
        Generate synthetic social media conversation data.
        
        Returns:
            DataFrame containing the generated data
        """
        # Initialize data structures
        posts_data = []
        thread_labels = {}
        post_id_counter = 0
        
        # Generate conversation threads
        for thread_id in range(self.num_threads):
            # Determine if this thread is controversial
            is_controversial = random.random() < self.controversial_ratio
            thread_labels[thread_id] = 1 if is_controversial else 0
            
            # Generate thread start time
            # Make sure threads start earlier to cover more time
            start_time = datetime.now() - timedelta(days=random.uniform(self.time_span_days*0.5, self.time_span_days))
            
            # Determine number of posts in this thread
            if is_controversial:
                # Controversial threads tend to have more posts - ensure at least 10 posts
                num_posts = random.randint(max(10, self.max_posts_per_thread//2), self.max_posts_per_thread)
            else:
                num_posts = random.randint(max(5, self.max_posts_per_thread//4), self.max_posts_per_thread // 2)
            
            # Create thread starter post
            starter_user = random.choice(self.users)
            starter_content = self._generate_post_content(is_controversial)
            
            starter_post = {
                "post_id": post_id_counter,
                "thread_id": thread_id,
                "user_id": starter_user["user_id"],
                "content": starter_content,
                "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                "parent_id": None,  # Thread starter has no parent
                "controversial": is_controversial
            }
            
            posts_data.append(starter_post)
            thread_posts = [starter_post]
            post_id_counter += 1
            
            # Generate replies
            for _ in range(1, num_posts):
                # Find a post to reply to (could be thread starter or another reply)
                parent_post = random.choice(thread_posts)
                parent_id = parent_post["post_id"]
                parent_content = parent_post["content"]
                
                # Choose user to make the reply (different from parent post user)
                remaining_users = [u for u in self.users if u["user_id"] != parent_post["user_id"]]
                reply_user = random.choice(remaining_users)
                
                # Generate reply time (after parent post)
                parent_time = datetime.strptime(parent_post["timestamp"], "%Y-%m-%d %H:%M:%S")
                
                # Create more distinct time windows by spreading replies over a longer period
                # For every 3rd post, add a larger time gap to ensure multiple snapshots
                if _ % 3 == 0:
                    # Create a bigger gap - between 4 to 8 hours
                    reply_time = parent_time + timedelta(hours=random.randint(4, 8))
                else:
                    # Normal gap - 5 minutes to 3 hours
                    reply_time = parent_time + timedelta(minutes=random.randint(5, 180))
                
                # Generate reply content
                reply_content = self._generate_post_content(
                    is_controversial, 
                    is_reply=True,
                    parent_content=parent_content
                )
                
                # Create reply post
                reply_post = {
                    "post_id": post_id_counter,
                    "thread_id": thread_id,
                    "user_id": reply_user["user_id"],
                    "content": reply_content,
                    "timestamp": reply_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "parent_id": parent_id,
                    "controversial": is_controversial
                }
                
                posts_data.append(reply_post)
                thread_posts.append(reply_post)
                post_id_counter += 1
        
        # Convert to DataFrame
        df = pd.DataFrame(posts_data)
        
        # Save raw data
        df.to_csv(os.path.join(self.output_dir, "social_media_data.csv"), index=False)
        
        # Save user profiles
        with open(os.path.join(self.output_dir, "user_profiles.json"), "w") as f:
            json.dump(self.users, f, indent=2)
        
        # Save thread labels
        thread_labels_df = pd.DataFrame({
            "thread_id": list(thread_labels.keys()),
            "controversial": list(thread_labels.values())
        })
        thread_labels_df.to_csv(os.path.join(self.output_dir, "thread_labels.csv"), index=False)
        
        return df

    def analyze_data(self, df):
        """
        Analyze generated data and print statistics.
        
        Args:
            df: DataFrame containing the generated data
        """
        print("\n===== SYNTHETIC SOCIAL MEDIA DATA STATISTICS =====")
        print(f"Total users: {self.num_users}")
        print(f"Total threads: {self.num_threads}")
        print(f"Total posts: {len(df)}")
        print(f"Thread starters: {len(df[df['parent_id'].isna()])}")
        print(f"Replies: {len(df[df['parent_id'].notna()])}")
        print(f"Controversial threads: {len(df[df['controversial'] == True])}")
        print(f"Non-controversial threads: {len(df[df['controversial'] == False])}")
        
        # Calculate average replies per thread
        thread_sizes = df.groupby("thread_id").size()
        print(f"Average posts per thread: {thread_sizes.mean():.2f}")
        
        # Calculate time span of conversations
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        time_span = df['timestamp'].max() - df['timestamp'].min()
        print(f"Time span of data: {time_span}")
        
        # User activity
        user_activity = df.groupby("user_id").size().sort_values(ascending=False)
        print(f"Most active user: {user_activity.index[0]} with {user_activity.iloc[0]} posts")
        print(f"Least active user: {user_activity.index[-1]} with {user_activity.iloc[-1]} posts")
        print("===== END OF STATISTICS =====\n")

def main():
    """Main function to generate synthetic social media data."""
    # Create output directory
    output_dir = "./data/raw"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize generator
    generator = SocialMediaDataGenerator(
        num_users=50,
        num_threads=100,
        max_posts_per_thread=20,
        max_replies_per_post=5,
        time_span_days=7,
        controversial_ratio=0.4,
        output_dir=output_dir
    )
    
    # Generate data
    print("Generating synthetic social media data...")
    df = generator.generate_data()
    
    # Print statistics
    generator.analyze_data(df)
    
    print(f"Data generated and saved to {output_dir}")
    
    return df

if __name__ == "__main__":
    main()