
from jql_embeddings import JQL_Embeddings
from jira_field_embeddings import Jira_Field_Embeddings
from pathlib import Path
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL


embedconfig = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")

def test_JQL_Embeddings():
    print("Testing JQL Embeddings!")

    jql_embeddings = JQL_Embeddings(embedconfig)
    documentProc, model = jql_embeddings.run(Path("data/jira_jql_annotated_queries.md"))
    return documentProc, model


def test_Jira_Field_Embeddings(documentProc, model):
    print("Testing Jira Field Embeddings!")

    jira_field_embeddings = Jira_Field_Embeddings(embedconfig, documentProc)
    jira_field_embeddings.run(Path("data/jira_fields.json"))


if __name__ == "__main__":
    documentProc, model = test_JQL_Embeddings()
    test_Jira_Field_Embeddings(documentProc, model)
