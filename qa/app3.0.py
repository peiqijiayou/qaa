import streamlit as st
import json
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import openai
import re
import time

# 设置页面配置，确保这是第一个 Streamlit 命令
st.set_page_config(
    page_title="问答系统聊天机器人",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 在页面顶部设置一个欢迎标题，模拟 Apple 风格
st.title("🤖 欢迎使用问答系统")

# 侧边栏，增加新建对话按钮
st.sidebar.title("功能")
if st.sidebar.button("清空对话"):
    st.session_state['messages'] = [{"role": "assistant", "content": "您好，我是您的问答机器人，有什么可以帮您的吗？"}]
    st.session_state['is_responding'] = False
    st.rerun()

# 文件路径
VECTOR_FILE = r"D:\Desktop\问答系统\qa_question_vectors.npy"  # 存储的向量文件
HISTORY_FILE = r"D:\Desktop\问答系统\chat_history.json"
QA_FILE = r"D:\Desktop\问答系统\qa_pairs_with_id.json"

# 加载QA对从本地文件
def load_qa_pairs(file_path=QA_FILE):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        st.error(f"找不到QA对文件: {file_path}")
        return []

# 从文件中加载已向量化的问句
def load_vectorized_questions():
    if os.path.exists(VECTOR_FILE):
        question_embeddings = np.load(VECTOR_FILE)
        return question_embeddings
    else:
        st.error(f"找不到问句向量文件: {VECTOR_FILE}")
        return None

# 加载对话历史
def load_history():
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return []

# 保存对话历史到本地文件
def save_history(messages):
    with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=4)

# 加载 text2vec 中文语义匹配模型
@st.cache_resource
def load_model():
    model = SentenceModel('shibing624/text2vec-base-chinese')  # 使用 text2vec 模型
    return model

model = load_model()

# 生成句子嵌入
def get_sentence_embedding(sentence):
    embedding = model.encode(sentence)
    return embedding.reshape(1, -1)

# DeepSeek API配置
DEEPSEEK_API_KEY = "sk-3df88f6a2e72477aaa4c2108acd3a36c"  # 替换为实际的 DeepSeek API Key
openai.api_key = DEEPSEEK_API_KEY
openai.api_base = "https://api.deepseek.com"

# 验证答案是否正确的函数
def validate_answer(question, answer):
    prompt = (
        f"请判断以下回答是否正确地回答了问题。\n\n"
        f"问题: {question}\n\n"
        f"回答: {answer}\n\n"
        f"回答是否正确？请回答 '是' 或 '否'。"
    )

    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "请判断回答的正确性，并仅回复 '是' 或 '否'。"},
                {"role": "user", "content": prompt},
            ],
            stream=False
        )
        if response.choices:
            result = response.choices[0].message['content'].strip()
            return result.lower() == '是'
        else:
            st.error("验证回答时，DeepSeek API 未返回结果。")
            return False
    except Exception as e:
        st.error(f"验证回答时出错: {e}")
        return False

# 生成答案的函数
def generate_answer(question):
    try:
        response = openai.ChatCompletion.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是智能问答助手，请回答与上海财经大学浙江学院统计系有关的内容"},
                {"role": "user", "content": question},
            ],
            stream=False
        )
        if response.choices:
            return response.choices[0].message['content']
        else:
            return "抱歉，无法生成答案。"
    except Exception as e:
        return f"调用 DeepSeek API 时出错: {e}"

# 格式化LaTeX输出
def format_latex_output(answer):
    formatted_answer = re.sub(
        r"\\\[(.*?)\\]|\[(.*?)\]",
        lambda m: "$$ " + (m.group(1) or m.group(2)).replace('\n', ' ') + " $$",
        answer,
        flags=re.DOTALL
    )
    formatted_answer = re.sub(
        r"\\\((.*?)\\\)",
        lambda m: "$" + m.group(1).replace('\n', ' ') + "$",
        formatted_answer,
        flags=re.DOTALL
    )
    return formatted_answer

# 加载QA对和向量化数据
qa_pairs = load_qa_pairs()
question_embeddings = load_vectorized_questions()

if question_embeddings is None:
    st.error("请先生成问句向量化文件！")
else:
    # 初始化对话历史
    if 'messages' not in st.session_state:
        st.session_state['messages'] = load_history() or [
            {"role": "assistant", "content": "您好，我是您的问答机器人，有什么可以帮您的吗？"}]

    # 初始化会话状态以跟踪是否正在等待回答
    if 'is_responding' not in st.session_state:
        st.session_state['is_responding'] = False

    # 显示聊天历史
    for msg in st.session_state.messages:
        if msg['role'] == 'user':
            with st.chat_message("user"):
                st.write(msg['content'])
        else:
            with st.chat_message("assistant"):
                st.markdown(msg['content'])

    # 输入区域
    user_input = st.chat_input("请输入您的问题：")

    if user_input and not st.session_state['is_responding']:
        # 添加用户消息到历史记录
        st.session_state.messages.append({'role': 'user', 'content': user_input})
        # 存储用户输入到会话状态
        st.session_state['latest_user_input'] = user_input
        # 设置正在响应的标志
        st.session_state['is_responding'] = True
        # 保存历史对话记录
        save_history(st.session_state['messages'])
        # 触发重新运行
        st.rerun()

    # 生成并显示答案
    if st.session_state.get('is_responding') and 'latest_user_input' in st.session_state:
        user_question = st.session_state['latest_user_input']

        # 计算用户问题的嵌入
        user_embedding = get_sentence_embedding(user_question)
        similarities = [cosine_similarity(user_embedding, q_emb.reshape(1, -1))[0][0] for q_emb in question_embeddings]
        max_similarity = max(similarities)
        most_similar_idx = similarities.index(max_similarity)

        # 定义相似度阈值
        SIMILARITY_THRESHOLD = 0.7  # 或者根据你的需求选择合适的阈值

        # 检查相似度
        if max_similarity >= SIMILARITY_THRESHOLD:
            a1 = qa_pairs[most_similar_idx]['answer']
            final_answer = f"{a1}\n\n来源: {qa_pairs[most_similar_idx]['source']}"
        else:
            a2 = generate_answer(user_question)
            final_answer = f"{a2}\n\n来源: DeepSeek V2.5"

        # 格式化LaTeX输出
        formatted_answer = format_latex_output(final_answer)

        # 将助手的回复添加到历史记录
        st.session_state.messages.append({'role': 'assistant', 'content': ""})
        assistant_idx = len(st.session_state.messages) - 1

        # 创建占位符
        with st.chat_message("assistant"):
            message_placeholder = st.empty()

        # 逐字显示答案
        full_response = ""
        for char in formatted_answer:
            full_response += char
            message_placeholder.markdown(full_response + "▌")
            time.sleep(0.02)  # 调整流式显示速度

        # 移除末尾光标并渲染完整内容
        message_placeholder.markdown(full_response)

        # 更新历史记录中的助手回复
        st.session_state.messages[assistant_idx]['content'] = formatted_answer

        # 重置标志
        st.session_state['is_responding'] = False

        # 清除最新用户输入
        del st.session_state['latest_user_input']

        # 保存历史对话记录
        save_history(st.session_state['messages'])
