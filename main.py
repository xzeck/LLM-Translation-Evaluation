import logging
import pandas as pd

from preprocess import PreProcessor
from embedding_model import EmbeddingModel
from qdrant_connection_handler import QDrantHandler
from RAG import RAG
from LLM import LLM
from benchmark import BenchMark
from TranslationStrategies import LLMTranslationStrategy, RAGTranslationStrategy

from config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

# Initialize configuration
logging.info("Loading configuration...")
config = Config()

EMBEDDING_MODEL_NAME = config.get_property("EMBEDDING_MODEL_NAME")
URL = config.get_property("URL")
MODEL_NAME = config.get_property("MODEL_NAME")
logging.info(f"Configuration loaded: EMBEDDING_MODEL_NAME={EMBEDDING_MODEL_NAME}, URL={URL}, MODEL_NAME={MODEL_NAME}")

# Preprocess the data
logging.info("Starting data preprocessing...")
preprocess = PreProcessor()

try:
    preprocess.read()
    logging.info("Data successfully read.")
except Exception as e:
    logging.error(f"Failed to read data: {e}")
    raise

try:
    df = preprocess.chunk_sentence(None)
    logging.info(f"Data successfully chunked. Number of rows: {len(df)}")
except Exception as e:
    logging.error(f"Failed to chunk data: {e}")
    raise

# Creating an embedding model
logging.info("Initializing embedding model...")
embedding_model = EmbeddingModel(model_name=EMBEDDING_MODEL_NAME)

# Create the qdrant handler
logging.info("Initializing QDrant handler...")
qdrant_handler = QDrantHandler(url=None, in_memory=True)

# Store the embeddings
logging.info("Storing embeddings in QDrant...")
try:
    qdrant_handler.store_embeddings_in_qdrant(df, embedding_model=embedding_model)
    logging.info("Embeddings successfully stored in QDrant.")
except Exception as e:
    logging.error(f"Failed to store embeddings in QDrant: {e}")
    raise

# Query the handler to see if we are getting data back
query = "Are you weak"
logging.info(f"Querying QDrant with: {query}")
try:
    result_translations = qdrant_handler.query_qdrant(query, embedding_model)
    logging.info(f"Query successful. Result: {result_translations}")
except Exception as e:
    logging.error(f"Failed to query QDrant: {e}")
    raise

# Models to benchmark
logging.info("Initializing LLM and RAG models...")
llm = LLM(MODEL_NAME, config.get_property("token"))
rag = RAG(MODEL_NAME, config.get_property("token"))

# Read test set
test_file = "split.csv"
logging.info(f"Reading test set from {test_file}...")
try:
    df_test = pd.read_csv(test_file, skiprows=range(1, 900), nrows=25)
    logging.info(f"Test set loaded successfully. Number of rows: {len(df_test)}")
except Exception as e:
    logging.error(f"Failed to load test set: {e}")
    raise

# Benchmarking
logging.info("Setting up benchmarking strategies...")
llm_translation_strategy = LLMTranslationStrategy(llm)
rag_translation_strategy = RAGTranslationStrategy(rag, qdrant_handler)

logging.info("Benchmarking LLM...")
benchmark_llm = BenchMark(llm_translation_strategy)

logging.info("Benchmarking RAG...")
benchmark_rag = BenchMark(rag_translation_strategy)

logging.info("Running LLM benchmark on the test dataset...")
try:
    benchmark_llm.benchmark_translation_system(df_test, embedding_model)
    logging.info("RAG benchmarking completed successfully.")
except Exception as e:
    logging.error(f"Failed during RAG benchmarking: {e}")
    raise


logging.info("Running RAG benchmark on the test dataset...")
try:
    benchmark_rag.benchmark_translation_system(df_test, embedding_model)
    logging.info("RAG benchmarking completed successfully.")
except Exception as e:
    logging.error(f"Failed during RAG benchmarking: {e}")
    raise
