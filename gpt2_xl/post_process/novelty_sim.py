import http.client
import json
import time
from scipy.spatial.distance import cdist
from joblib import Parallel, delayed

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


def get_embedding_by_gpt(data, flag=False, model_name="text-embedding-ada-002"):
    global COUNT
    if COUNT >= 5:
        COUNT = 0
        return [-1]
    if flag:
        conn, headers = connect_openai()
    else:
        conn, headers = connect()

    payload = json.dumps({
        "model": model_name,
        "input": data
    })

    try:
        conn.request("POST", "/v1/embeddings", payload, headers)
        res = conn.getresponse()
        if res.getcode() == 200:
            COUNT = 0
            data = json.loads(res.read().decode("utf-8"))
            embedding_data_list = data['data']
            return [item['embedding'] for item in embedding_data_list]
        else:
            print("sleep: code", res.getcode())
            time.sleep(1)
            COUNT += 1
            return get_embedding_by_gpt(data, flag=True)
    except Exception as e:
        print("exception: ", e)
        time.sleep(1)
        COUNT += 1
        return get_embedding_by_gpt(data, flag=True)


def get_sim_by_embedding(answer, list_embedding_tails, threshold=0.85, n_jobs=-1):
    answer_embedding = get_embedding_by_gpt(answer)

    try:
        cosine_distance = cdist(answer_embedding, list_embedding_tails, 'cosine')[0]
        cosine_similarity = 1 - cosine_distance
        def is_similar(similarity):
            return similarity > threshold
        similar_indices = Parallel(n_jobs=n_jobs)(
            delayed(is_similar)(sim) for sim in cosine_similarity
        )

        if True in similar_indices:
            return True
        else:
            return False
    except Exception as e:
        print(e)
        return True


