#!/bin/bash

SCRIPT_DIR=$(dirname "$0")
PROJECT_ROOT_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)
CONFIG_FILE="$PROJECT_ROOT_DIR/.env"
# shellcheck disable=SC1090
source "$CONFIG_FILE"

docker run --name ragmongo --hostname localhost --userns=host --user root -e MONGODB_INITDB_ROOT_USERNAME=admin -e MONGODB_INITDB_ROOT_PASSWORD=admin -v ./database/raganything_db/mongo:/data/db -v ./database/mongo_keyfile:/data/configdb/keyfile -v ./database/mongot_mmlongbench:/data/mongot -p 27017:27017 -d mongodb/mongodb-atlas-local

docker run --name ragmongo_doc_bench --hostname localhost --userns=host --user root -e MONGODB_INITDB_ROOT_USERNAME=admin -e MONGODB_INITDB_ROOT_PASSWORD=admin -v ./database/raganything_db/mongo_docbench:/data/db -v ./database/mongo_keyfile:/data/configdb/keyfile -v ./database/mongot_docbench:/data/mongot -p 27018:27017 -d mongodb/mongodb-atlas-local

# docker cp ragmongo_doc_bench:/data/mongot ./database/

docker run -it --name ragneo4j --hostname localhost --userns=host --user root -p 7474:7474 -p 7687:7687 -v ./database/raganything_db/neo4j:/data -v ./database/raganything_db/neo4j_plugins:/plugins -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* neo4j

docker run -it --name ragneo4j_doc_bench --hostname localhost --userns=host --user root -p 7475:7474 -p 7688:7687 -v ./database/raganything_db/neo4j_docbench:/data -v ./database/raganything_db/neo4j_plugins:/plugins -e NEO4J_AUTH=none -e NEO4J_PLUGINS='["apoc"]' -e NEO4J_apoc_export_file_enabled=true -e NEO4J_apoc_import_file_enabled=true -e NEO4J_apoc_import_file_use__neo4j__config=true -e NEO4J_dbms_security_procedures_unrestricted=apoc.\\\* neo4j