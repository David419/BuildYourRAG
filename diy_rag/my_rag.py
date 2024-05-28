import chromadb
from chromadb import Settings
from sentence_transformers import SentenceTransformer
from modelscope import AutoModelForCausalLM, AutoTokenizer

class MyRag:
    def __init__(self, db_path: str, db_name: str, embedding_path: str):
        """
        初始化函数，用于初始化载入chroma数据库、embedding模型和llm模型
        :param db_path: chroma数据库地址
        :param db_name: chroma数据库名称
        """
        # chroma数据库设置
        settings = Settings(allow_reset=True)  # 数据库设置，允许使用reset指令
        self.client = chromadb.PersistentClient(path=db_path, settings=settings)  # 根据指定路径和设置打开chroma数据库
        self.collection = self.client.get_or_create_collection(name=db_name, metadata={"hnsw:space": "cosine"})  # 载入collection，并指定距离计算公式为cosine

        # embedding模型设置 默认使用CPU进行计算
        self.embedding_model = SentenceTransformer(embedding_path)
        self.query_instruction = "为这个句子生成表示以用于检索相关文章："

        # llm模型设置
        self.llm_model = AutoModelForCausalLM.from_pretrained("qwen/Qwen1.5-4B-Chat")
        self.tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen1.5-4B-Chat")
        self.prompt = "请根据以下背景来回答问题：query\n背景：background"
    def db_query(self, single_query: str, return_n=3):
        """
        chroma数据库请求，输入为单条请求，输出为召回的所有信息，格式为字典
        :param single_query: 单条请求。例如: '上海有什么好吃的'
        :param return_n: 返回条数，默认返回3条相关信息
        :return:
        """
        results_num = return_n  # 返回相关结果条数

        q_embeddings = self.embedding_model.encode([self.query_instruction + single_query], normalize_embeddings=True)  # 转化请求数据为向量

        results = self.collection.query(
            query_embeddings=q_embeddings,
            n_results=results_num,
        )
        return results

    @staticmethod
    def extract_recall_result(return_result):
        """
        从数据库返回信息中抽取所需要的信息作为背景信息，并作为输入到llm中
        :param return_result: chroma数据库返回的结果。结果为一个dict
        :return: 返回背景信息字符串
        """
        texts = []  # 正文片段
        for element in return_result['metadatas'][0]:  # 因为只设置单条请求，因此只取第一个元素
            text = element['text'].replace(' ', '')  # 删除文本中的空格
            text = text.replace('\n', '')  # 删除文本中的换行符
            texts.append(text)
        return '\n'.join(texts)

    def get_response(self, single_query: str, db_recall_n=3, llm_output_n=512):
        # 从chroma中召回结果，并提取正文作为模型的背景输入
        db_recall = self.db_query(single_query, db_recall_n)
        background_info = self.extract_recall_result(db_recall)

        # 将请求和背景信息相结合，转化为模型可理解的格式
        input_content = self.prompt.replace('query', single_query)
        input_content = input_content.replace('background', background_info)
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": input_content}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt")

        # 模型计算并输出
        generated_ids = self.llm_model.generate(model_inputs.input_ids, max_new_tokens=llm_output_n)
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]  # 提取输出结果
        return response


if __name__ == "__main__":
    my_rag = MyRag(db_path='./news_db', db_name='news', embedding_path='embedding_model/bge-base-zh-v1_5')
    res = my_rag.db_query(single_query='APP的过渡期是多久')
    print(res)
    print(my_rag.extract_recall_result(res))
    print(my_rag.get_response(single_query='APP的过渡期是多久'))









