from synthesizer.synthesizer_dataset import SynthesizerDataset
from synthesizer.hparams import hparams
from synthesizer.preprocess import preprocess_dataset,preprocess_speaker,create_embeddings
from pathlib import Path

data_path = Path("datasets")
output_path  = Path("datasets/processed_comb")
encoder_path = Path("models/encoder.pt")

if __name__ == "__main__":
    preprocess_dataset(datasets_root= data_path, out_dir= output_path,n_processes= 1,
    skip_existing= True,subfolders="", no_alignments= True,hparams=hparams,datasets_name= "SG")

    create_embeddings(output_path,encoder_model_fpath=encoder_path,n_processes= 2)