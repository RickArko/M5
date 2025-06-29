micromamba create -f conda-env.yml

micromamba run -n m5 python -m ipykernel install --user --name m5 --display-name "m5"

micromamba run -n m5 python src/generate_data.py
micromamba run -n m5 python src/process.py