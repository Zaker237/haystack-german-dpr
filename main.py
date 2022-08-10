import os
import logging

os.environ["TIKA_LOG_PATH"] = "/data/mboutchouang/models/logs"

from haystack.nodes import DensePassageRetriever
from haystack.document_stores import InMemoryDocumentStore

import hydra
from omegaconf import DictConfig

logging.basicConfig(format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

@hydra.main(config_path="config", config_name="main")
def main(config: DictConfig) -> None:

    retriever = DensePassageRetriever(
    document_store=InMemoryDocumentStore(),
        query_embedding_model=config.model.query_model,
        passage_embedding_model=config.model.passage_model,
        max_seq_len_query=64,
        max_seq_len_passage=256,
    )

    retriever.train(
        data_dir=config.data.doc_dir,
        train_filename=config.data.train_filename,
        dev_filename=config.data.dev_filename,
        #test_filename=dev_filename,
        n_epochs=1,
        batch_size=config.training.batch_size,
        grad_acc_steps=8,
        #save_dir=save_dir,
        evaluate_every=3000,
        embed_title=True,
        num_positives=1,
        num_hard_negatives=1,
        n_gpu=config.training.n_gpu
    )

    retriever.save(
        save_dir=config.save.save_dir,
        query_encoder_dir=config.save.query_encoder_dir,
        passage_encoder_dir=config.save.passage_encoder_dir
    )

if __name__ == "__main__":
    main()
