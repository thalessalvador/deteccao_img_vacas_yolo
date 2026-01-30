# Projeto: Detecção de Vacas com YOLO

Este projeto treina e avalia um modelo YOLO para detectar vacas (cattle torso) usando o dataset **cows2021-DatasetNinja**.  
Ele inclui scripts para **treinamento** e **predição** com visualização e métricas.

## Estrutura do dataset (necessário antes de rodar)

O Ultralytics espera `images/` e `labels/`. Como o dataset original usa `img/`, renomeie **uma vez** em cada split:

Baixe o dataset aqui:
```text
https://datasetninja.com/cows2021
```

```powershell
Rename-Item cows2021-DatasetNinja\detection_and_localisation-train\img images
Rename-Item cows2021-DatasetNinja\detection_and_localisation-val\img images
Rename-Item cows2021-DatasetNinja\detection_and_localisation-test\img images
```

As anotações originais continuam em `ann/`. O script `treinar.py` gera as labels YOLO em `labels/` e também guarda uma cópia em `ann_yolo/`.

## Ambiente virtual (.venv)

Crie e ative o ambiente:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
```

Instale as dependências:

```powershell
pip install ultralytics numpy pillow matplotlib
```

> Observação: o `ultralytics` instala o `torch`. Se houver problema, instale o PyTorch manualmente conforme a sua GPU.

## GPU (CUDA) vs CPU

Em algumas máquinas o PyTorch pode instalar **somente CPU**. Nesse caso o treino **não usa a GPU NVIDIA**.
Se o teste abaixo retornar `False no-cuda`, instale a versão CUDA do PyTorch:

```powershell
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"
```

Exemplo (CUDA 12.1):

```powershell
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

Depois, rode novamente o teste. Se aparecer `True` e o nome da sua GPU, o treino vai usar CUDA.

## Treinamento

Treina a partir de um modelo base (ex.: `yolov8n.pt`) e gera um modelo customizado:

```powershell
python .\treinar.py
```

Saída esperada: `yolov8n_dorso_vaca.pt` (configurável no `treinar.py`).

**Importante:** o `00061.jpg` e o `00061.jpg.json` são **excluídos explicitamente** do treino para evitar data leakage.

## Predição / Avaliação

Para rodar a predição e visualizar os boxes:

```powershell
python .\predicao_vacas.py
```

No arquivo `predicao_vacas.py`, ajuste:

```python
MODEL_PATH = Path("yolov8n_dorso_vaca.pt")
```

Se quiser comparar com o modelo base, troque para `yolov8n.pt`, por exemplo.
