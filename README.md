# LVLM-Interpret: An Interpretability Tool for Large Vision-Language Models
[[Project Page](https://intellabs.github.io/multimodal_cognitive_ai/lvlm_interpret/)] [[Paper](https://arxiv.org/abs/2404.03118)]

## Setup

- Update submodules

  `git submodule update --init --recursive`

- Install dependencies

  `pip install -r requirements.txt`


## Usage

Start the Gradio server:
```
python app.py --model_name_or_path Intel/llava-gemma-2b --load_8bit 
```
or
```
python app.py --model_name_or_path llava-hf/llava-1.5-7b-hf --load_8bit
```

Options:
```
usage: app.py [-h] [--model_name_or_path MODEL_NAME_OR_PATH] [--host HOST] [--port PORT] [--share] [--embed] [--load_4bit] [--load_8bit]

options:
  -h, --help            show this help message and exit
  --model_name_or_path MODEL_NAME_OR_PATH
                        Model name or path to load the model from
  --host HOST           Host to run the server on
  --port PORT           Port to run the server on
  --share               Whether to share the server on Gradio's public server
  --embed               Whether to run the server in an iframe
  --load_4bit           Whether to load the model in 4bit
  --load_8bit           Whether to load the model in 8bit

```
