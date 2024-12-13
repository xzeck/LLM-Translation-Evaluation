import logging
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from ITranslationStrategy import TranslationStrategy

class BenchMark:
    def __init__(self, translation_strategy: TranslationStrategy, log_level=logging.INFO, log_file=None):
        """
        Initialize BenchMark with optional logging configuration.
        
        Args:
            translation_strategy (TranslationStrategy): Translation strategy to be used
            log_level (int, optional): Logging level. Defaults to logging.INFO.
            log_file (str, optional): Path to log file. If None, logs to console. Defaults to None.
        """
        self.translation_strategy = translation_strategy
        
        # Configure logging
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_level)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if log_file is provided)
        if log_file:
            file_handler = logging.FileHandler(log_file)
            file_handler.setLevel(log_level)
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        self.logger.info(f"BenchMark initialized with translation strategy: {type(translation_strategy).__name__}")
    
    def benchmark_bleu(self, hypotheses, references):
        """
        Calculates BLEU score for a set of hypotheses and references.
        
        Args:
            hypotheses (list): Generated translations
            references (list): Reference translations
        
        Returns:
            float: BLEU score
        """
        self.logger.info(f"Calculating BLEU score for {len(hypotheses)} sentence pairs")
        
        try:
            references = [[ref.split()] for ref in references]  # BLEU expects nested lists
            hypotheses = [hyp.split() for hyp in hypotheses]
            
            smooth = SmoothingFunction()
            bleu = corpus_bleu(references, hypotheses, smoothing_function=smooth.method1)
            
            self.logger.info(f"BLEU score calculated: {bleu}")
            return bleu
        
        except Exception as e:
            self.logger.error(f"Error calculating BLEU score: {e}")
            raise
    
    def benchmark_meteor(self, hypotheses, references):
        """
        Calculates average METEOR score for a set of hypotheses and references.
        
        Args:
            hypotheses (list): Generated translations
            references (list): Reference translations
        
        Returns:
            float: Average METEOR score
        """
        self.logger.info(f"Calculating METEOR score for {len(hypotheses)} sentence pairs")
        
        try:
            meteor_scores = [meteor_score([ref], hyp) for ref, hyp in zip(references, hypotheses)]
            avg_meteor = sum(meteor_scores) / len(meteor_scores)
            
            self.logger.info(f"Average METEOR score: {avg_meteor}")
            return avg_meteor
        
        except Exception as e:
            self.logger.error(f"Error calculating METEOR score: {e}")
            raise
    
    def benchmark_translation_system(self, df, embedding_model):
        """
        Benchmarks the translation system on BLEU and METEOR using a test dataset.
        
        Args:
            df (DataFrame): DataFrame containing source and reference translations
            embedding_model: Embedding model used for translation
        """
        self.logger.info("Starting translation system benchmark")
        
        try:
            # Log input details
            self.logger.info(f"Total sentences to translate: {len(df)}")
            
            source_sentences = df['en']  # English source sentences
            reference_translations = df['fr']  # Corresponding French references
            
            # Generate translations with logging
            self.logger.info("Generating translations...")
            hypotheses = []
            for i, text in enumerate(source_sentences):
                translation = self.translation_strategy.translate(text, embedding_model)
                hypotheses.append(translation)
                
                # Optional: Log progress periodically to avoid excessive logging
                if (i + 1) % 100 == 0:
                    self.logger.info(f"Translated {i+1} sentences")
            
            # Compute BLEU
            bleu = self.benchmark_bleu(hypotheses, reference_translations)
            
            # Uncomment METEOR if needed
            # meteor = self.benchmark_meteor(hypotheses, reference_translations)
            
            self.logger.info("Translation system benchmark completed successfully")
            
            print(bleu)
            
            return bleu  # or return (bleu, meteor) if you uncomment METEOR
        
        except Exception as e:
            self.logger.error(f"Benchmark failed: {e}")
            raise