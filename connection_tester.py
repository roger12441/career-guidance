from neo4j import GraphDatabase

uri = "neo4j://localhost:7687"
username = "neo4j"
password = "caarg343"

driver = GraphDatabase.driver(uri, auth=(username, password))

with driver.session() as session:
    result = session.run("RETURN 'Connection successful!' AS message")
    for record in result:
        print(record["message"])
