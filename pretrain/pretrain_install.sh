
cd ../finetune
pip install -e .
pip install -q -U transformers datasets peft bitsandbytes
pip3 install peft
pip install tf-keras
pip3 install numpy==1.26.4
pip3 install transformers==4.51.3
pip3 uninstall byted-wandb 
pip3 install wandb
