import http.client
import json
import time

COUNT = 0


def connect():
    conn = http.client.HTTPSConnection("api.openai.com", timeout=5)
    headers = {
        'Authorization': 'Bearer sk-xxxxx',
        'Content-Type': 'application/json'
    }
    return conn, headers


def connect_openai():
    conn = http.client.HTTPSConnection("api.openai.com", timeout=5)
    headers = {
        'Authorization': 'Bearer sk-xxxxx',
        'Content-Type': 'application/json'
    }

    return conn, headers


def connect_gpt4():
    conn = http.client.HTTPSConnection("api.openai.com", timeout=5)
    headers = {
        'Authorization': 'Bearer sk-xxxxx',
        'Content-Type': 'application/json'
    }

    return conn, headers


def get_score_by_gpt(rels, heads, tails, sentences, flag=False, model_name="gpt-3.5-turbo-0613"):
    global COUNT
    if COUNT >= 10:
        COUNT = 0
        return [-1] * len(tails)
    if flag:
        conn, headers = connect_openai()
    else:
        conn, headers = connect()
    data = "Evaluate the following unrelated triples and the sentences generated from them which are not contextually connected to each other:\n "
    for rel, head, tail, sentence in zip(rels, heads, tails, sentences):
        data = data + f"[Triplet] Relationship: '{rel}', Head: '{head}', Tail: '{tail}'. [Sentence] {sentence} \n "

    if model_name != "gpt-3.5-turbo-0613":
        conn, headers = connect_gpt4()

    payload = json.dumps({
        "model": model_name,
        "messages": [
            {
                "role": "system",
                "content":
                   "You are an evaluator with comprehensive general knowledge. Your task is to assess the validity of the provided common sense triples and their corresponding sentences. Please respond only with 'Yes', 'No' or 'Unidentifiable'. 'Yes' means the triple and sentence are generally valid. 'No' means they are absolutely invalid. 'Unidentifiable' means they are usually invalid, and try not to use 'Unidentifiable'. Make your judgment based on the information given and your knowledge of the world."
            },
            {
                "role": "user",
                "content":
                   "Evaluate the following unrelated triples and the sentences generated from them which are not contextually connected to each other:\n "
                    "[Triplet] Relationship: 'AtLocation', Head: 'star', Tail: 'hollywood'. [Sentence] star is typically located in hollywood. \n "
                    "[Triplet] Relationship: 'Synonym', Head: 'depart', Tail: 'leave'. [Sentence] depart is synonymous with leave. \n "
                    "[Triplet] Relationship: 'RelatedTo', Head: 'dog' , Tail: 'cat'. [Sentence] dog has a connection with cat. \n "
                    "[Triplet] Relationship: 'EtymologicallyRelatedTo', Head: 'throughgoing', Tail: 'thoroughgo'. [Sentence] There’s a common etymological origin for throughgoing and thoroughgo. \n "
                    "[Triplet] Relationship: 'IsA', Head: 'signal_transduction_pathway', Tail: 'river'. [Sentence] signal_transduction_pathway is an example of river. \n "
            },
            {
                "role": "assistant",
                "content": "Yes Yes Yes Yes No"
            },
            {
                "role": "user",
                "content": data
            }
        ],
        "stream": False
    })
    try:
        conn.request("POST", "/v1/chat/completions", payload, headers)
        res = conn.getresponse()
        if res.getcode() == 200:
            COUNT = 0
            data = json.loads(res.read().decode("utf-8"))
            score = data["choices"][0]["message"]["content"]
            score = score.split()
            if len(score) != len(tails):
                return [-1] * len(tails)
            return score
        else:
            print("sleep: code", res.getcode())
            time.sleep(1)
            COUNT += 1
            return get_score_by_gpt(rels, heads, tails, sentences, flag=True)
    except Exception as e:
        print("exception: ", e)
        time.sleep(1)
        COUNT += 1
        return get_score_by_gpt(rels, heads, tails, sentences, flag=True)

