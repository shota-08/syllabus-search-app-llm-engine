from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
load_dotenv()  # read local .env file

# LLMモデルの設定
llm_3 = ChatOpenAI(model="gpt-3.5-turbo-0125", temperature=0)
llm_4 = ChatOpenAI(model="gpt-4o", temperature=0)


def get_llm_summary(text):
    prompt_template = """
    # 命令
    あなたは、日本の大学の文学部国文学科の履修登録のアドバイザーです。
    履修登録を検討している大学生に対して、この講義で学べる題材は何か、200文字以内で簡潔に説明してください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・200文字以内で簡潔に答える
    ・講義で扱うジャンルや年代や題材のキーワードは、可能な限り説明に含めること
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・複数の要素がある場合、国文学における重要度の高い要素から優先して説明すること

    # 講義内容:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm_4
    answer = chain.invoke({"text": text})
    return answer.content

def get_llm_keyword(text):
    prompt_template = """
    # 命令
    あなたは、日本の大学の文学部国文学科の履修アシスタントです。
    以下の授業計画を参照し、文学部に関連するキーワードや作品名を抽出しなさい。
    もし内容がない場合、何も生成せずに「シラバス参照」と返しなさい。

    # 授業計画:
    {text}
    """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    chain = prompt | llm_4
    answer = chain.invoke({"text": text})
    return answer.content

def pass_text(query, title, text):
    query_with_text = f"""
    # 命令
    あなたは進路相談のアドバイザーです。
    学生の問い合わせに親切に回答してください。そして講座名と講座概要を参照しながら、おすすめの講座を簡潔に紹介してください。
    学生の問い合わせが、講座名と講座概要と関連性が無い場合、講座の紹介はしないでください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・おすすめした理由と一緒に、講座名と講座概要を簡潔に伝えること
    ・ジャンルや年代や題材などの講座概要にある具体的なキーワードを含めること
    ・講座概要に書いていないことは伝えてはいけない
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・会話が続くように、文章の最後に「他にどんなことを学びたいですか？」など付けること
    ・100文字以内で簡潔に答える

    # 講座名:
    {title}

    # 講座概要:
    {text}

    # 学生の問い合わせ:
    {query}
    """
    return query_with_text

def get_llm_answer(query, title, text):
    prompt = PromptTemplate(
        template = "{template}",
        input_variables = ["template"]
    )
    query_with_text = pass_text(query, title, text)
    chain = prompt | llm_3
    answer = chain.invoke({"template": query_with_text})
    return answer.content

def get_as_retriever_answer(query, db):
    prompt_template = """
    # 命令
    あなたは進路相談のアドバイザーです。
    学生の問い合わせに親切に回答してください。そして講座名と講座概要を参照しながら、おすすめの講座を簡潔に紹介してください。
    学生の問い合わせが、講座名と講座概要と関連性が無い場合、講座の紹介はしないでください。
    一度大きく深呼吸をしてステップバイステップで考えてください。

    # 制約条件:
    ・おすすめした理由と一緒に、講座名と講座概要を簡潔に伝えること
    ・ジャンルや年代や題材などの講座概要にある具体的なキーワードを含めること
    ・講座概要に書いていないことは伝えてはいけない
    ・単位取得の情報、欠席の情報、履修条件の情報は説明に含めてはいけない
    ・会話が続くように、文章の最後に「他にどんなことを学びたいですか？」など付けること
    ・100文字以内で簡潔に答える

    # 参考情報:
    {context}

    # 学生の問い合わせ:
    {question}

    # 回答文
    """
    prompt_qa = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain_type_kwargs = {"prompt": prompt_qa}
    retriever = db.as_retriever(search_kwargs={"k": 4})
    qa = RetrievalQA.from_chain_type(
        llm=llm_4,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs=chain_type_kwargs,
        return_source_documents=True,
    )
    result = qa(query)
    return result
