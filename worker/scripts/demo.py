import logging
from concurrent import futures

import grpc

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

from diffuse_worker import data_pb2_grpc
from diffuse_worker.config import WorkerConfig
from diffuse_worker.pipeline import Pipeline, PipelineStage
from diffuse_worker.server import InferenceWorkerServicer
from diffuse_worker.slicing import ModelSlice

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
MAX_MESSAGE_BYTES = 128 * 1024 * 1024
MAX_NEW_TOKENS = 120


def total_layers(model_id):
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    return getattr(cfg, "num_hidden_layers", None) or cfg.n_layer


def start_worker():
    options = [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=2), options=options
    )
    data_pb2_grpc.add_InferenceWorkerServicer_to_server(
        InferenceWorkerServicer(WorkerConfig()), server
    )
    port = server.add_insecure_port("127.0.0.1:0")
    server.start()
    return server, port


def main():
    total = total_layers(MODEL)
    mid = total // 2

    s1, p1 = start_worker()
    s2, p2 = start_worker()

    chan_options = [
        ("grpc.max_send_message_length", MAX_MESSAGE_BYTES),
        ("grpc.max_receive_message_length", MAX_MESSAGE_BYTES),
    ]
    c1 = grpc.insecure_channel(f"127.0.0.1:{p1}", options=chan_options)
    c2 = grpc.insecure_channel(f"127.0.0.1:{p2}", options=chan_options)

    stage1 = PipelineStage(c1, MODEL, 0, mid)
    stage2 = PipelineStage(c2, MODEL, mid, total)
    stage1.load()
    stage2.load()

    ref = ModelSlice()
    ref.load(MODEL, 0, total)
    tokenizer = ref.tokenizer
    pipe = Pipeline([stage1, stage2], tokenizer)

    print(f"\nDiffuse pipeline: 2 nodes, layers 0:{mid} and {mid}:{total}")
    print(f"Model: {MODEL}")
    print("Type a prompt (empty to quit).\n")

    while True:
        prompt = input("> ").strip()
        if not prompt:
            break
        messages = [{"role": "user", "content": prompt}]
        formatted = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        out = pipe.generate(formatted, max_new_tokens=MAX_NEW_TOKENS)
        answer = out[len(formatted):].strip()
        print(f"  {answer}\n")

    c1.close()
    c2.close()
    s1.stop(None)
    s2.stop(None)


if __name__ == "__main__":
    main()