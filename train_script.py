from synthesizer.synthesizer_dataset import SynthesizerDataset
from synthesizer.hparams import hparams
from synthesizer.preprocess import preprocess_dataset,preprocess_speaker,create_embeddings
from pathlib import Path
from synthesizer.train import train

processed_data_path = Path("datasets/output")
model_dir = Path("./models")
train_sub_dir = "seab"


train(run_id= train_sub_dir,syn_dir= processed_data_path,models_dir= model_dir,save_every= 10,backup_every= 0,hparams= hparams,force_restart= False)