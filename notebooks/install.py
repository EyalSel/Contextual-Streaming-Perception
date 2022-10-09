import json
from pathlib import Path

# Modify config json
config_path = Path.home()/".jupyter/jupyter_notebook_config.json"
if config_path.exists():
    with open(config_path, 'r') as f:
        jupyter_configs = json.load(f)
else:
    jupyter_configs = {}
config_options = {}
config_options["template_dirs"] = [str(Path(__file__).parent.absolute()/"templates")]
config_options["include_default"] = True
config_options["include_core_paths"] = True
jupyter_configs["JupyterLabTemplates"] = config_options
with open(config_path, 'w') as f:
    json.dump(jupyter_configs, f, sort_keys=True,
              indent=4, separators=(',', ': '))

