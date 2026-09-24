import ollama
from PIL import Image

# 2B instruct: 0.5 s per answer when warm (11 s once, to load), and it sits in
# memory next to the chat model. The 4B took 12-14 s and evicted the chat model.
MODEL = "qwen3-vl:2b-instruct"
KEEP_ALIVE = "30m"
DEFAULT_QUESTION = "Describe what's in this image in 2 short sentences, focusing on the most notable objects."


def resize_for_vision(image_path, max_dim=384):
    img = Image.open(image_path)
    img.thumbnail((max_dim, max_dim))
    resized_path = image_path.rsplit(".", 1)[0] + "_small.jpg"
    img.save(resized_path)
    return resized_path


def look_at(image_path, question=DEFAULT_QUESTION, max_tokens=80):
    small_path = resize_for_vision(image_path)
    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "user", "content": question, "images": [small_path]}],
        think=False,
        keep_alive=KEEP_ALIVE,
        options={"num_predict": max_tokens, "temperature": 0.2},
    )
    return response["message"]["content"].strip()


def warmup(image_path):
    """Loads the model so the first real question is fast."""
    look_at(image_path, "Reply with one word: ok", max_tokens=3)


if __name__ == "__main__":
    print("Rocky sees:", look_at("latest.jpg"))
