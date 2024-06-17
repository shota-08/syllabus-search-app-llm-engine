import streamlit as st
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()

from llm_engine import get_as_retriever_answer

st.set_page_config(page_title='Langchain x syllabus')
st.title('哲学科')

# 初期化
embedding = OpenAIEmbeddings(model= "text-embedding-3-small")

# chroma db呼び出し
persist_directory = "./docs/tetsugaku_chroma"
db = Chroma(collection_name="langchain_store", persist_directory=persist_directory, embedding_function=embedding)

# 定数定義
USER_NAME = "user"
ASSISTANT_NAME = "assistant"

# 質問文
query = st.chat_input('やりたいことを入力')

# チャットログセッションの初期化
if "tetsugaku_chat_log" not in st.session_state:
    st.session_state.tetsugaku_chat_log = []

if query:
    docs = get_as_retriever_answer(query, db)
    answer = docs["result"]
    source_documents = docs["source_documents"]

    # ログの表示
    for chat in st.session_state.tetsugaku_chat_log:
        with st.chat_message(chat["name"]):
            st.write(chat["msg"])

    # chat表示
    with st.chat_message(USER_NAME):
        st.write(query)

    with st.chat_message(ASSISTANT_NAME):
        st.write(answer)
        st.write("### おすすめの授業")
        st.write(f"- [{source_documents[0].metadata['title']}]({source_documents[0].metadata['url']})")
        st.write(f"- [{source_documents[1].metadata['title']}]({source_documents[1].metadata['url']})")
        st.write(f"- [{source_documents[2].metadata['title']}]({source_documents[2].metadata['url']})")
        st.write(f"- [{source_documents[3].metadata['title']}]({source_documents[3].metadata['url']})")

    # data表示
    with st.expander('docs', expanded=False):
        st.write(docs)

    # セッションにログを追加
    st.session_state.tetsugaku_chat_log.append({"name" : USER_NAME, "msg" : query})
    st.session_state.tetsugaku_chat_log.append({"name" : ASSISTANT_NAME, "msg" : answer})