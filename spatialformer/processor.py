

# processor.py

from .graphsage import GraphSAGEModule

class Processor:
    @classmethod
    def run_graphsage(cls, 
                      transcript_file, 
                      postfix=None, 
                      threshold=10, 
                      config_path=None,
                      feature_num = 50,
                      hidden_dim = 64,
                      output_dim = 128,
                      total_step = 10000,
                      strategy =  "ddp",
                      lr = 0.001,
                      batch_size = 1024):
        # Initialize a class-level GraphSAGE instance
        cls.graphsage_instance = GraphSAGEModule()
        cls.graphsage_instance.get_subgraph(transcript_file, postfix, threshold)
        cls.graphsage_instance.train_model(
                                           feature_num = feature_num,
                                           hidden_dim = hidden_dim,
                                           output_dim = output_dim,
                                           total_step = total_step,
                                           strategy =  strategy,
                                           lr = lr,
                                           batch_size = batch_size)

    @classmethod
    def get_embedding(cls, checkpoint, token_path):
        # Ensure the GraphSAGE instance is initialized through `run_graphsage`
        if cls.graphsage_instance is None:
            raise ValueError("GraphSAGE instance is not initialized. Call `run_graphsage` first.")

        embeddings = cls.graphsage_instance.get_embedding(checkpoint, token_path)
        return embeddings
    


