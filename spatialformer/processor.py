

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
        model = cls.graphsage_instance.train_model(
                                           feature_num = feature_num,
                                           hidden_dim = hidden_dim,
                                           output_dim = output_dim,
                                           total_step = total_step,
                                           strategy =  strategy,
                                           lr = lr,
                                           batch_size = batch_size)
        return model
    @classmethod
    def get_embedding(cls, checkpoint, token_path):
        # Ensure the GraphSAGE instance is initialized through `run_graphsage`
        if cls.graphsage_instance is None:
            raise ValueError("GraphSAGE instance is not initialized. Call `run_graphsage` first.")

        embeddings = cls.graphsage_instance.get_embedding(checkpoint, token_path)
        return embeddings

    @classmethod
    def load_pretrained_model(cls, 
                              vocab_path,
                              hidden_dim,
                              output_dim,
                              batch_size,
                              checkpoint,
                              device):
        cls.graphsage_instance = GraphSAGEModule()
        model = cls.graphsage_instance.load_pretrained_model( vocab_path,
                                                    hidden_dim,
                                                    output_dim,
                                                    batch_size,
                                                    checkpoint,
                                                    device)
        return model
    @classmethod
    def build_graph(cls,
                    data_path,
                    vocab_path,
                    batch_size,
                    graph_path,
                    threshold,
                    split):
        cls.graphsage_instance = GraphSAGEModule()
        all_graph = cls.graphsage_instance.build_graph(
                    data_path,
                    vocab_path,
                    batch_size,
                    graph_path,
                    threshold,
                    split)
        return all_graph
    @classmethod
    def train_pattern_model(cls,
                    model,
                    lr,
                    output_dim,
                    strategy,
                    output_dir,
                    train_dataloader,
                    val_dataloader
                    ):
        cls.graphsagepattern_instance = GraphSAGEModule()
        cls.graphsage_instance.train_model(
                                model,
                                lr,
                                strategy,
                                output_dim,
                                output_dir,
                                train_dataloader,
                                val_dataloader
                             )
    def test_model(cls,
                   test_dataloader,
                   ckp_path):
        cls.graphsagepattern_instance = GraphSAGEModule()
        cls.graphsage_instance.test_model(
                                test_dataloader,
                                ckp_path
                             )
        

    


