import logging
from ragdoll import settings

# Initialize logging
logging.basicConfig(level=logging.INFO)


# Create a Config instance via the shared AppConfig bootstrap
app = settings.get_app()
config_manager = app.config

# 1. Accessing the Ingestion Configuration
ingestion_config = config_manager.ingestion_config
logging.info(f"Ingestion Config: {ingestion_config}")
max_threads = ingestion_config.max_threads
batch_size = ingestion_config.batch_size
logging.info(f"Max Threads: {max_threads}")
logging.info(f"Batch Size: {batch_size}")

# 2. Accessing the Embeddings Configuration
logging.info("2. Accessing the Embeddings Configuration")
embeddings_config = config_manager.embeddings_config
logging.info(f"Embeddings Config: {embeddings_config}")
default_client_name = embeddings_config.default_client
client_configs = embeddings_config.clients

if default_client_name in client_configs:
    default_client_config = client_configs[default_client_name]
    logging.info(f"Default Client Config ({default_client_name}): {default_client_config}")
else:
    logging.warning(f"Unknown default client: {default_client_name}")

# 3. Accessing the Loaders Configuration
loaders_mapping = config_manager.get_loader_mapping()
logging.info(f"Loaders Config: {list(loaders_mapping.keys())}")

