import json
path = '/home/alfrog/projects/VibeVoice_Optimization/notebooks/tesis_model_cuantization.ipynb'
with open(path) as f:
    nb = json.load(f)
for cell in nb['cells']:
    if cell.get('id') in ('generate-train-script', 'generate-merge-script'):
        cell['source'] = [s.replace(
            'cd {PROJECT_ROOT}\n',
            'cd {PROJECT_ROOT}\n\n# Inyectar VibeVoice_repo en PYTHONPATH\nexport PYTHONPATH="{PROJECT_ROOT}/VibeVoice_repo:$PYTHONPATH"\n'
        ) for s in cell['source']]
with open(path, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
print('Done')
