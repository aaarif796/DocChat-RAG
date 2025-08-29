from langchain_chroma import Chroma; from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

vs = Chroma(collection_name="docchat", embedding_function=embeddings, chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"), tenant=os.getenv("CHROMA_TENANT"), database=os.getenv("CHROMA_DATABASE"))