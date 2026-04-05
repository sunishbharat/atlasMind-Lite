
from jql_embeddings import JQL_Embeddings
from pathlib import Path
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL



def main():
    print("Hello from amind-partial!")

    embedconfig = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")

    jql_embeddings = JQL_Embeddings(embedconfig)
    jql_embeddings.run(Path("data/jira-jql-annotated-queries.md"))
    #jql_embeddings.search_sample_jql_embeddings_db("What is the status of the tickets?")

if __name__ == "__main__":
    main()
