import json
import os
import numpy as np
from text2vec import SentenceModel

# 文件路径
QA_FILE = r"D:\Desktop\问答系统\qa_pairs_with_id.json"
VECTOR_FILE = r"D:\Desktop\问答系统\qa_question_vectors.npy"

# 加载QA对从本地文件
def load_qa_pairs(file_path=QA_FILE):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print(f"找不到QA对文件: {file_path}")
        return []

# 加载 text2vec 中文语义匹配模型
def load_model():
    model = SentenceModel('shibing624/text2vec-base-chinese')
    return model

# 向量化问句并保存到文件
def vectorize_and_save_questions():
    qa_pairs = load_qa_pairs()
    if not qa_pairs:
        return

    model = load_model()

    # 获取所有问题并进行向量化
    questions = [pair['question'] for pair in qa_pairs]
    question_embeddings = model.encode(questions)

    # 保存向量化的问句
    np.save(VECTOR_FILE, question_embeddings)
    print(f"问句向量已保存到 {VECTOR_FILE}")

if __name__ == "__main__":
    vectorize_and_save_questions()
