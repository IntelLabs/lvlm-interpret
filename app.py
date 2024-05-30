import argparse 
import logging


from utils_gradio import build_demo

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_name_or_path", type=str, default="Intel/llava-gemma-2b",
                        help="Model name or path to load the model from")
    parser.add_argument("--host", type=str, default="0.0.0.0",
                        help="Host to run the server on")
    parser.add_argument("--port", type=int, default=7860,
                        help="Port to run the server on")
    parser.add_argument("--share", action="store_true",
                        help="Whether to share the server on Gradio's public server")
    parser.add_argument("--embed", action="store_true",
                        help="Whether to run the server in an iframe")
    parser.add_argument("--load_4bit", action="store_true",
                        help="Whether to load the model in 4bit")
    parser.add_argument("--load_8bit", action="store_true",
                        help="Whether to load the model in 8bit")
    args = parser.parse_args()

    assert not( args.load_4bit and args.load_8bit), "Cannot load both 4bit and 8bit models"

    demo = build_demo(args, embed_mode=False)
    demo.queue(max_size=5)
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=True
    )
