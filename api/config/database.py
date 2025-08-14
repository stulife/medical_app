import os

class Config:
    # Neo4j Configuration
    NEO4J_URI = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    NEO4J_USERNAME = os.environ.get('NEO4J_USERNAME', 'neo4j')
    NEO4J_PASSWORD = os.environ.get('NEO4J_PASSWORD', 'password')
    
    # Model Configuration
    MODEL_NAME = os.environ.get('MODEL_NAME', 'custom_medical_model')
    
    # Feedback Configuration
    FEEDBACK_COLLECTION_ENABLED = os.environ.get('FEEDBACK_COLLECTION_ENABLED', 'True').lower() == 'true'