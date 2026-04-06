import asyncio
from jql_embeddings import JQL_Embeddings
from jira_field_embeddings import Jira_Field_Embeddings
from pathlib import Path
from dconfig import EmbeddingsConfig
from settings import EMBEDDING_MODEL
from atlasmind import AtlasMind


embedconfig = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")

def test_JQL_Embeddings():
    print("Testing JQL Embeddings!")

    jql_embeddings = JQL_Embeddings(embedconfig)
    documentProc, model = jql_embeddings.run(Path("data/jira_jql_annotated_queries.md"))
    return documentProc, model


def test_Jira_Field_Embeddings(documentProc, model):
    print("Testing Jira Field Embeddings!")
    jira_field_embeddings = Jira_Field_Embeddings(embedconfig, document_processor=documentProc)
    jira_field_embeddings.run(Path("data/jira_fields.json"))
    rows, _ = asyncio.run(jira_field_embeddings.search_jira_fields(
        "list all the issues that are assigned to me and status is blocked", model))
    print_rows(rows)
    rows, _ = asyncio.run(jira_field_embeddings.search_jira_fields(
        "Show me all easy tickets", model))
    print_rows(rows)

def print_rows(rows):
    print("-"*40)
    for row in rows:
        print(f"rows: {row}")


def test_atlasmind():
    embedconfig = EmbeddingsConfig(model_name="BAAI/bge-small-en-v1.5")
    atlasmind = AtlasMind(embedconfig)
    atlasmind.run()
    query = "List the 100 issues with major or minor priority for project ZOOKEEPER, create a bar graph grouped by month."
    print(asyncio.run(atlasmind.generate_jql(query)))


if __name__ == "__main__":
    test_atlasmind()
    #documentProc, model = test_JQL_Embeddings()
    #test_Jira_Field_Embeddings(documentProc, model)
