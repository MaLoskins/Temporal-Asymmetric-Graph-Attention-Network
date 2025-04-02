"""
Social Media Data Preprocessing for TAGAN

This script converts raw social media conversation data into temporal graph format
suitable for processing by TAGAN. It extracts text features from posts, creates
user interaction graphs, and organizes these into temporal snapshots.
"""

import os
import torch
import numpy as np
import pandas as pd
import networkx as nx
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize
import json
import pickle
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Union, Any, Set

# Import TAGAN preprocessing utilities
from src.tagan.data.preprocessing import preprocess_temporal_graph
from src.tagan.data.data_loader import TemporalGraphDataset, TemporalGraphDataLoader


class SocialMediaGraphProcessor:
    """
    Processor that converts social media conversations into temporal graphs.
    """
    
    def __init__(
        self,
        raw_data_dir="./data/raw",
        processed_data_dir="./data/processed",
        text_embedding_dim=16,
        snapshot_duration=2,  # hours - reduced to create more snapshots
        max_snapshots=20,     # increased to allow more snapshots
        min_nodes_per_snapshot=1  # reduced to allow smaller valid snapshots
    ):
        """
        Initialize the processor.
        
        Args:
            raw_data_dir: Directory containing raw data files
            processed_data_dir: Directory to save processed data
            text_embedding_dim: Dimension of text embeddings
            snapshot_duration: Duration of each snapshot in hours
            max_snapshots: Maximum number of snapshots to create
            min_nodes_per_snapshot: Minimum number of nodes per snapshot
        """
        self.raw_data_dir = raw_data_dir
        self.processed_data_dir = processed_data_dir
        self.text_embedding_dim = text_embedding_dim
        self.snapshot_duration = snapshot_duration * 3600  # Convert to seconds
        self.max_snapshots = max_snapshots
        self.min_nodes_per_snapshot = min_nodes_per_snapshot
        
        # Create output directory
        os.makedirs(processed_data_dir, exist_ok=True)
        
        # Initialize text vectorizer
        self.vectorizer = None
        self.svd = None
    
    def load_raw_data(self):
        """
        Load raw social media data.
        
        Returns:
            Tuple of (posts_df, users, thread_labels)
        """
        # Load posts data
        posts_path = os.path.join(self.raw_data_dir, "social_media_data.csv")
        posts_df = pd.read_csv(posts_path)
        
        # Convert timestamps to datetime objects
        posts_df['timestamp'] = pd.to_datetime(posts_df['timestamp'])
        
        # Sort by timestamp
        posts_df = posts_df.sort_values('timestamp')
        
        # Load user profiles
        users_path = os.path.join(self.raw_data_dir, "user_profiles.json")
        with open(users_path, 'r') as f:
            users = json.load(f)
        
        # Load thread labels
        labels_path = os.path.join(self.raw_data_dir, "thread_labels.csv")
        thread_labels = pd.read_csv(labels_path)
        
        return posts_df, users, thread_labels
    
    def create_text_embeddings(self, posts_df):
        """
        Create text embeddings for post content.
        
        Args:
            posts_df: DataFrame containing posts
            
        Returns:
            Dictionary mapping post_id to embedding vector
        """
        print("Creating text embeddings...")
        
        # Initialize TF-IDF vectorizer
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            min_df=2,
            max_df=0.8
        )
        
        # Fit and transform post content
        tfidf_matrix = self.vectorizer.fit_transform(posts_df['content'])
        print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
        
        # Reduce dimensionality with SVD to get fixed-size embeddings
        self.svd = TruncatedSVD(n_components=self.text_embedding_dim)
        text_embeddings = self.svd.fit_transform(tfidf_matrix)
        
        # Normalize embeddings
        text_embeddings = normalize(text_embeddings)
        print(f"Text embeddings shape: {text_embeddings.shape}")
        
        # Create dictionary mapping post_id to embedding
        post_embeddings = {}
        for i, post_id in enumerate(posts_df['post_id']):
            post_embeddings[post_id] = text_embeddings[i]
        
        return post_embeddings
    
    def encode_post_for_new_data(self, content):
        """
        Encode a new post content into an embedding vector.
        
        Args:
            content: Post content text
            
        Returns:
            Embedding vector for the post
        """
        if self.vectorizer is None or self.svd is None:
            raise ValueError("Vectorizer and SVD models are not initialized. Run create_text_embeddings first.")
        
        # Transform text to TF-IDF
        tfidf = self.vectorizer.transform([content])
        
        # Apply SVD
        embedding = self.svd.transform(tfidf)
        
        # Normalize
        embedding = normalize(embedding)
        
        return embedding[0]
    
    def convert_to_interaction_edges(self, posts_df):
        """
        Convert post-reply structure to user interaction edges.
        
        Args:
            posts_df: DataFrame containing posts
            
        Returns:
            DataFrame containing interaction edges
        """
        print("Converting to interaction edges...")
        
        # Create edges based on replies
        edges = []
        
        # Process each post that has a parent
        replies = posts_df[posts_df['parent_id'].notna()]
        
        for _, reply in replies.iterrows():
            # Find the parent post
            parent_post = posts_df[posts_df['post_id'] == reply['parent_id']].iloc[0]
            
            # Create an edge from parent author to reply author
            edge = {
                'source': parent_post['user_id'],
                'target': reply['user_id'],
                'timestamp': reply['timestamp'],
                'thread_id': reply['thread_id'],
                'controversial': reply['controversial'],
                'parent_post_id': parent_post['post_id'],
                'reply_post_id': reply['post_id']
            }
            edges.append(edge)
        
        # Convert to DataFrame
        edges_df = pd.DataFrame(edges)
        print(f"Created {len(edges_df)} interaction edges")
        
        return edges_df
    
    def create_node_attributes(self, posts_df, users, post_embeddings):
        """
        Create node attributes combining user profile and post content.
        
        Args:
            posts_df: DataFrame containing posts
            users: List of user dictionaries
            post_embeddings: Dictionary mapping post_id to embedding
            
        Returns:
            DataFrame containing node attributes
        """
        print("Creating node attributes...")
        
        # Create a mapping from user_id to user attributes
        user_dict = {user['user_id']: user for user in users}
        
        # Aggregate posts by user
        user_posts = defaultdict(list)
        user_timestamps = defaultdict(list)
        
        for _, post in posts_df.iterrows():
            user_id = post['user_id']
            post_id = post['post_id']
            user_posts[user_id].append(post_id)
            user_timestamps[user_id].append(post['timestamp'])
        
        # Create node attributes
        node_attributes = []
        
        for user_id, posts in user_posts.items():
            # Get user profile
            user = user_dict[user_id]
            
            # Convert categorical attributes to numeric
            activity_level_map = {'low': 0, 'medium': 1, 'high': 2}
            activity = activity_level_map.get(user['activity_level'], 1)
            
            # Average embeddings of all posts by this user
            user_embeddings = [post_embeddings[post_id] for post_id in posts]
            avg_embedding = np.mean(user_embeddings, axis=0)
            
            # Convert interests to one-hot encoding
            topic_map = {topic: i for i, topic in enumerate(
                ["technology", "politics", "sports", "entertainment", 
                 "science", "health", "environment", "business"]
            )}
            interests = np.zeros(len(topic_map))
            for interest in user['interests']:
                if interest in topic_map:
                    interests[topic_map[interest]] = 1
            
            # Combine all attributes
            attributes = {
                'user_id': user_id,
                'last_timestamp': max(user_timestamps[user_id]),
                'activity': activity,
                'age': user['age'],
                'post_count': len(posts),
                'embedding': avg_embedding,
                'interests': interests
            }
            
            node_attributes.append(attributes)
        
        # Convert to DataFrame
        nodes_df = pd.DataFrame(node_attributes)
        print(f"Created attributes for {len(nodes_df)} nodes")
        
        return nodes_df
    
    def prepare_temporal_graph_data(self, edges_df, nodes_df):
        """
        Prepare data for temporal graph processing.
        
        Args:
            edges_df: DataFrame containing edges
            nodes_df: DataFrame containing nodes
            
        Returns:
            DataFrame suitable for preprocess_temporal_graph function
        """
        print("Preparing temporal graph data...")
        
        # Create a DataFrame in the expected format for preprocess_temporal_graph
        graph_data = []
        
        for _, edge in edges_df.iterrows():
            # Get node attributes for source and target
            source_node = nodes_df[nodes_df['user_id'] == edge['source']].iloc[0]
            target_node = nodes_df[nodes_df['user_id'] == edge['target']].iloc[0]
            
            # Convert timestamp to Unix timestamp
            timestamp = edge['timestamp'].timestamp()
            
            # Combine node embeddings and interests as node attributes
            source_embedding = source_node['embedding']
            source_interests = source_node['interests']
            source_attrs = np.concatenate([
                [source_node['activity'], source_node['age'] / 100, source_node['post_count'] / 10],
                source_embedding,
                source_interests
            ])
            
            target_embedding = target_node['embedding']
            target_interests = target_node['interests']
            target_attrs = np.concatenate([
                [target_node['activity'], target_node['age'] / 100, target_node['post_count'] / 10],
                target_embedding,
                target_interests
            ])
            
            # Edge attributes
            edge_attrs = [
                float(edge['controversial']),  # Use controversial flag as edge feature
                1.0  # Dummy weight
            ]
            
            # Add to graph data
            interaction = {
                'timestamp': timestamp,
                'source': edge['source'],
                'target': edge['target'],
                'source_attrs': source_attrs.tolist(),
                'target_attrs': target_attrs.tolist(),
                'edge_attrs': edge_attrs,
                'thread_id': edge['thread_id']
            }
            
            graph_data.append(interaction)
        
        return pd.DataFrame(graph_data)
    
    def process_data(self):
        """
        Process raw data into temporal graph format for TAGAN.
        
        Returns:
            Tuple of (train_data, val_data, test_data, thread_labels)
        """
        # Load raw data
        posts_df, users, thread_labels = self.load_raw_data()
        
        # Create text embeddings
        post_embeddings = self.create_text_embeddings(posts_df)
        
        # Convert to interaction edges
        edges_df = self.convert_to_interaction_edges(posts_df)
        
        # Create node attributes
        nodes_df = self.create_node_attributes(posts_df, users, post_embeddings)
        
        # Prepare data for temporal graph processing
        graph_data = self.prepare_temporal_graph_data(edges_df, nodes_df)
        
        # Create temporal graph snapshots
        print("Creating temporal graph snapshots...")
        snapshots = preprocess_temporal_graph(
            data=graph_data,
            timestamp_col='timestamp',
            source_col='source',
            target_col='target',
            edge_attr_cols=['edge_attrs'],
            node_attr_cols=['source_attrs', 'target_attrs'],
            snapshot_duration=self.snapshot_duration,
            max_snapshots=self.max_snapshots,
            min_nodes_per_snapshot=self.min_nodes_per_snapshot,
            normalize_features=True
        )
        
        print(f"Created {len(snapshots)} temporal snapshots")
        
        # Group snapshots by thread_id to create sequences
        thread_snapshots = defaultdict(list)
        
        for snapshot in snapshots:
            # Extract thread_ids from this snapshot
            snapshot_edges = graph_data[
                (graph_data['timestamp'] >= snapshot['timestep']) &
                (graph_data['timestamp'] < snapshot['timestep'] + self.snapshot_duration)
            ]
            
            thread_ids = snapshot_edges['thread_id'].unique()
            
            # Associate snapshot with each thread_id that appears in it
            for thread_id in thread_ids:
                thread_snapshots[thread_id].append(snapshot)
        
        # Sort snapshots within each thread by timestep
        for thread_id in thread_snapshots:
            thread_snapshots[thread_id].sort(key=lambda x: x['timestep'])
        
        # Filter threads with too few snapshots (reduced requirement to get more data)
        min_snapshots_per_thread = 2
        valid_threads = {
            thread_id: snapshots
            for thread_id, snapshots in thread_snapshots.items()
            if len(snapshots) >= min_snapshots_per_thread
        }
        
        # If we have too few valid threads, relax the requirements
        if len(valid_threads) < 10:
            print("WARNING: Too few valid sequences with 2+ snapshots. Reducing minimum to 1...")
            min_snapshots_per_thread = 1
            valid_threads = {
                thread_id: snapshots
                for thread_id, snapshots in thread_snapshots.items()
                if len(snapshots) >= min_snapshots_per_thread
            }
        
        print(f"Found {len(valid_threads)} valid thread sequences with {min_snapshots_per_thread}+ snapshots")
        
        # Create sequences with labels
        sequences = []
        labels = []
        
        thread_label_dict = dict(zip(thread_labels['thread_id'], thread_labels['controversial']))
        
        for thread_id, thread_sequence in valid_threads.items():
            sequences.append(thread_sequence)
            # Get label from thread_labels
            label = thread_label_dict.get(thread_id, 0)
            labels.append(label)
        
        # Split into train, validation, and test sets
        num_sequences = len(sequences)
        train_size = int(0.7 * num_sequences)
        val_size = int(0.15 * num_sequences)
        
        # Create indices and shuffle
        indices = np.arange(num_sequences)
        np.random.seed(42)
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_sequences = [sequences[i] for i in train_indices]
        train_labels = [labels[i] for i in train_indices]
        
        val_sequences = [sequences[i] for i in val_indices]
        val_labels = [labels[i] for i in val_indices]
        
        test_sequences = [sequences[i] for i in test_indices]
        test_labels = [labels[i] for i in test_indices]
        
        # Save processed data
        processed_data = {
            'train_sequences': train_sequences,
            'train_labels': train_labels,
            'val_sequences': val_sequences,
            'val_labels': val_labels,
            'test_sequences': test_sequences,
            'test_labels': test_labels,
            'vectorizer': self.vectorizer,
            'svd': self.svd
        }
        
        with open(os.path.join(self.processed_data_dir, 'processed_social_media_data.pkl'), 'wb') as f:
            pickle.dump(processed_data, f)
        
        print(f"Data split - Train: {len(train_sequences)}, Val: {len(val_sequences)}, Test: {len(test_sequences)}")
        
        return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels

    def visualize_graph_snapshot(self, snapshot, title='Graph Snapshot', save_path=None):
        """
        Visualize a graph snapshot.
        
        Args:
            snapshot: Graph snapshot dictionary
            title: Plot title
            save_path: Path to save the visualization (optional)
        """
        # Extract data from snapshot
        x = snapshot['x']
        edge_index = snapshot['edge_index']
        
        # Create NetworkX graph
        G = nx.Graph()
        
        # Add nodes
        for i in range(x.size(0)):
            G.add_node(i)
        
        # Add edges
        for i in range(edge_index.size(1)):
            src = edge_index[0, i].item()
            dst = edge_index[1, i].item()
            G.add_edge(src, dst)
        
        # Create plot
        plt.figure(figsize=(10, 8))
        pos = nx.spring_layout(G)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', edge_color='gray', 
                node_size=500, font_size=10, font_weight='bold')
        
        plt.title(title)
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()


def main():
    """Main function to preprocess social media data for TAGAN."""
    # Set random seed
    np.random.seed(42)
    torch.manual_seed(42)
    
    # Create processor
    processor = SocialMediaGraphProcessor(
        raw_data_dir="./data/raw",
        processed_data_dir="./data/processed",
        text_embedding_dim=16,
        snapshot_duration=6,  # hours
        max_snapshots=10,
        min_nodes_per_snapshot=3
    )
    
    # Process data
    print("Processing social media data to temporal graph format...")
    train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels = processor.process_data()
    
    # Visualize a sample graph
    if train_sequences:
        print("Visualizing a sample graph snapshot...")
        sample_snapshot = train_sequences[0][0]  # First snapshot of first sequence
        save_path = os.path.join(processor.processed_data_dir, "sample_graph_snapshot.png")
        processor.visualize_graph_snapshot(
            sample_snapshot, 
            title="Sample Social Media Interaction Graph", 
            save_path=save_path
        )
        print(f"Sample visualization saved to {save_path}")
    
    print("Preprocessing complete!")
    return train_sequences, train_labels, val_sequences, val_labels, test_sequences, test_labels


if __name__ == "__main__":
    main()