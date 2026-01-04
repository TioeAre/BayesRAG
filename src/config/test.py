import os
from project_config import project_config

print(project_config.config_path)
print(project_config.DATA_ROOT)
print(os.getenv("TRACELOOP_HEADERS"))
print(os.getenv("PROJECT_ROOT"))
