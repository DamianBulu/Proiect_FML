import os
import requests
from typing import List
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document


class LocalServerEmbeddings(Embeddings):
    """
        Clasă personalizată care conectează LangChain la endpoint-ul de embeddings din LM Studio.
    """
    def __init__(
        self,
        base_url:str="http://localhost:1234/v1",
            model:str="text-embedding-nomic-embed-text-v1.5", #Numele modelului de embedding din LM Studio
            timeout:int=500

    ):
        self.base_url=base_url.rstrip("/")
        self.model=model
        self.timeout=timeout

    def _embed(self,texts:List[str])->List[List[float]]:
        response=requests.post(
            f"{self.base_url}/embeddings",
            json={"model":self.model,"input":texts},
            timeout=self.timeout
        )
        response.raise_for_status()
        data=response.json()

        if "data" not in data:
            raise ValueError(f"Unexpected LM Studio embedding response: {data}")

        return [item["embedding"] for item in data["data"]]

    def embed_documents(self,texts:List[str])->List[List[float]]:
        return self._embed(texts)

    def embed_query(self,text:str)->List[float]:
        return self._embed([text])[0]


class MedicalKnowledgeRepo():
    def __init__(self,persist_directory: str="chroma_db"):
        self.persist_directory=persist_directory
        self.embedding_model=LocalServerEmbeddings()

        #conexiunea cu baza de date locală ChromaDB
        self.vectordb=Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_model
        )

    def seed_database(self,texts:List[str],metadatas:List[dict]=None):
        """
            Metodă utilitară pentru a popula baza de date cu documente medicale.
            Trebuie rulată o singură dată (sau când adaugam documente noi).
        """
        if not metadatas:
            metadatas=[{}]*len(texts)

        documents=[Document(page_content=t,metadata=m) for t,m in zip(texts,metadatas)]
        self.vectordb.add_documents(documents)
        print(f"S-au adăugat {len(documents)} fragmente în ChromaDB.")


    def get_knowledge(self, query:str,k:int=2)->str:
        """
            Caută în ChromaDB cele mai relevante 'k' fragmente de text pe baza întrebării.
        """
        try:
            #Efectuăm căutarea de similaritate folosind modelul de embeddings
            results=self.vectordb.similarity_search(query,k=k)

            if not results:
                return "No relevant medical informations found"

            #Formatăm rezultatele într-un string clar pe care Advisor-ul să îl poată citi
            knowledge_blocks=[]
            for i,res in enumerate(results):
                source=res.metadata.get("source","Unknown source")
                knowledge_blocks.append(f"Document {i+1} (Source: {source})\n{res.page_content}")

            return "\n\n".join(knowledge_blocks)
        except Exception as e:
            print(f"Eroare RAG: {str(e)}")
            return "Not available vectorial database"