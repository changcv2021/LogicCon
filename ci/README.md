# GitHub Actions template

`tests.yml` defines dependency-light software tests on Python 3.11, 3.12 and 3.13.

To enable it, copy this file to `.github/workflows/tests.yml` and commit the change through an identity with workflow-write permission. No model downloads, GPU runner, or API secrets are needed. The template is distributed here so the code can be published using repository-write authorization alone.

The same checks can be run locally:

```bash
python -m pip install -e .
python -m unittest discover -s tests -v
logiccon-reason doctor
bash -n scripts/slurm_infer.sh
```
