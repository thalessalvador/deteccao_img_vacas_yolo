"""
Treinamento de um modelo YOLO customizado para deteccao de vacas (cattle torso).

O script:
1) Gera anotacoes YOLO (txt) dentro do proprio dataset.
2) EXCLUI explicitamente 00061.jpg e 00061.jpg.json (evita data leakage).
3) Treina um modelo base (ex: yolov8n.pt) e gera um novo .pt.
4) Salva o modelo final com nome definido pelo usuario.
5) Usa CUDA se disponivel, caso contrario usa CPU.
"""

import json
import os
import shutil
from pathlib import Path

from PIL import Image

try:
    import torch
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "ultralytics/torch are required. Install with: pip install ultralytics"
    ) from e

# -----------------------------
# Configuracoes principais
# -----------------------------
BASE_MODEL_PATH = Path("yolov8n.pt")  # Modelo base para iniciar o treino (pre-treinado).
OUTPUT_MODEL_PATH = Path("yolov8n_dorso_vaca.pt")  # Nome do modelo final treinado.

DATASET_ROOT = Path("cows2021-DatasetNinja")  # Pasta raiz do dataset original.
SPLITS = {
    "train": DATASET_ROOT / "detection_and_localisation-train",  # Treino
    "val": DATASET_ROOT / "detection_and_localisation-val",  # Validacao
    "test": DATASET_ROOT / "detection_and_localisation-test",  # Teste
}

# IMPORTANTE: arquivo a ser excluido do treino (data leakage)
EXCLUDE_IMAGE_NAME = "00061.jpg"  # Imagem usada no teste manual.
EXCLUDE_ANN_NAME = "00061.jpg.json"  # Anotacao correspondente a imagem excluida.

# Nomes de pastas/arquivos usados no dataset YOLO (sem duplicar imagens)
IMG_DIR_ORIGINAL = "images"  # Pasta original com imagens do dataset (apos renomear).
IMG_DIR_YOLO = "images"  # Alias com hardlinks para o YOLO (mesmo conteudo).
ANN_DIR_ORIGINAL = "ann"  # Pasta original com anotacoes JSON.
ANN_DIR_YOLO = "ann_yolo"  # Pasta com anotacoes YOLO geradas.
LABELS_DIR_YOLO = "labels"  # Pasta exigida pelo Ultralytics (labels YOLO).

# Arquivo data.yaml usado pelo Ultralytics
DATA_YAML_PATH = DATASET_ROOT / "data_yolo.yaml"

# Hiperparametros basicos (ajuste se necessario)
EPOCHS = 50  # Numero de epocas de treino.
IMGSZ = 640  # Tamanho de entrada da imagem (ex: 640 = 640x640).
BATCH = 16  # Tamanho do batch por iteracao.


def tem_cuda():
    """
    Verifica se ha GPU com suporte a CUDA disponivel.

    Parametros:
        None

    Retorna:
        bool: True se CUDA estiver disponivel, False caso contrario.
    """
    return bool(torch.cuda.is_available())


def garantir_diretorio(path: Path):
    """
    Cria o diretorio se ele nao existir.

    Parametros:
        path (Path): Caminho do diretorio.

    Retorna:
        None
    """
    path.mkdir(parents=True, exist_ok=True)


def ler_tamanho_imagem(img_path: Path, ann_path: Path):
    """
    Le o tamanho da imagem (largura/altura).

    Parametros:
        img_path (Path): Caminho da imagem.
        ann_path (Path): Caminho do JSON (pode conter size).

    Retorna:
        tuple[int, int]: (width, height).
    """
    try:
        data = json.loads(ann_path.read_text(encoding="utf-8"))
        size = data.get("size", {})
        w = int(size.get("width", 0))
        h = int(size.get("height", 0))
        if w > 0 and h > 0:
            return w, h
    except Exception:
        pass

    with Image.open(img_path) as im:
        return im.size


def json_para_labels_yolo(ann_path: Path, img_size):
    """
    Converte anotacoes JSON (retangulos) para linhas YOLO.

    Parametros:
        ann_path (Path): Caminho do JSON de anotacao.
        img_size (tuple[int, int]): (width, height).

    Retorna:
        list[str]: Linhas YOLO no formato: class x_center y_center w h.
    """
    width, height = img_size
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    lines = []
    for obj in data.get("objects", []):
        if obj.get("geometryType") != "rectangle":
            continue
        exterior = obj.get("points", {}).get("exterior", [])
        if len(exterior) != 2:
            continue
        (x1, y1), (x2, y2) = exterior
        x1, y1, x2, y2 = map(float, (x1, y1, x2, y2))
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # Converte para YOLO: x_center, y_center, w, h normalizados
        box_w = max(0.0, x2 - x1)
        box_h = max(0.0, y2 - y1)
        cx = x1 + box_w / 2.0
        cy = y1 + box_h / 2.0

        if width == 0 or height == 0:
            continue

        cx /= width
        cy /= height
        box_w /= width
        box_h /= height

        # Classe unica: 0 = cattle torso (ou "cow" para o nosso caso)
        lines.append(f"0 {cx:.6f} {cy:.6f} {box_w:.6f} {box_h:.6f}")
    return lines


def criar_link_ou_copiar(src: Path, dst: Path):
    """
    Tenta criar hard link; se falhar, copia o arquivo.

    Parametros:
        src (Path): Arquivo origem.
        dst (Path): Arquivo destino.

    Retorna:
        None
    """
    garantir_diretorio(dst.parent)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        shutil.copy2(src, dst)


def construir_dataset_yolo():
    """
    Construi o dataset no formato YOLO (images/labels) a partir do JSON,
    sem duplicar imagens (usa hardlinks na pasta images).

    Parametros:
        None

    Retorna:
        Path: Caminho do arquivo data.yaml gerado.
    """
    # Estrutura YOLO esperada (dentro de cada split original)
    for split in ("train", "val", "test"):
        base = SPLITS[split]
        garantir_diretorio(base / IMG_DIR_YOLO)
        garantir_diretorio(base / LABELS_DIR_YOLO)
        garantir_diretorio(base / ANN_DIR_YOLO)

    for split, base in SPLITS.items():
        img_dir = base / IMG_DIR_ORIGINAL
        ann_dir = base / ANN_DIR_ORIGINAL
        img_yolo_dir = base / IMG_DIR_YOLO
        ann_yolo_dir = base / ANN_DIR_YOLO
        labels_dir = base / LABELS_DIR_YOLO

        for img_path in img_dir.glob("*.jpg"):
            ann_path = ann_dir / f"{img_path.name}.json"

            # Excluir explicitamente o arquivo 00061 (data leakage)
            if img_path.name == EXCLUDE_IMAGE_NAME or ann_path.name == EXCLUDE_ANN_NAME:
                continue

            if not ann_path.exists():
                continue

            # Imagem (hardlink para evitar duplicacao)
            out_img = img_yolo_dir / img_path.name
            criar_link_ou_copiar(img_path, out_img)

            # Labels (gerados em ann_yolo e espelhados em labels)
            out_ann_yolo = ann_yolo_dir / f"{img_path.stem}.txt"
            out_label = labels_dir / f"{img_path.stem}.txt"
            img_size = ler_tamanho_imagem(img_path, ann_path)
            lines = json_para_labels_yolo(ann_path, img_size)
            out_ann_yolo.write_text("\n".join(lines), encoding="utf-8")
            criar_link_ou_copiar(out_ann_yolo, out_label)

    # data.yaml
    DATA_YAML_PATH.write_text(
        "\n".join(
            [
                f"path: {DATASET_ROOT.as_posix()}",
                f"train: {SPLITS['train'].name}/{IMG_DIR_YOLO}",
                f"val: {SPLITS['val'].name}/{IMG_DIR_YOLO}",
                f"test: {SPLITS['test'].name}/{IMG_DIR_YOLO}",
                "names:",
                "  0: cattle_torso",
            ]
        ),
        encoding="utf-8",
    )
    return DATA_YAML_PATH


def treinar_modelo(data_yaml: Path):
    """
    Treina o modelo YOLO e salva o resultado final.

    Parametros:
        data_yaml (Path): Caminho do arquivo data.yaml.

    Retorna:
        Path: Caminho do modelo final salvo.
    """
    device = 0 if tem_cuda() else "cpu"
    print("Modelo base:", BASE_MODEL_PATH.name)
    print("Dispositivo:", "CUDA" if device != "cpu" else "CPU")
    print("Arquivo de dados:", data_yaml.as_posix())

    model = YOLO(str(BASE_MODEL_PATH))
    results = model.train(
        data=str(data_yaml),
        imgsz=IMGSZ,
        epochs=EPOCHS,
        batch=BATCH,
        device=device,
    )

    # Salva o melhor modelo com o nome escolhido
    save_dir = Path(results.save_dir)
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise SystemExit(f"Best model not found: {best_pt}")

    shutil.copy2(best_pt, OUTPUT_MODEL_PATH)
    return OUTPUT_MODEL_PATH


def main():
    """
    Fluxo principal: prepara o dataset YOLO e treina o modelo.

    Parametros:
        None

    Retorna:
        None
    """
    data_yaml = construir_dataset_yolo()
    out_model = treinar_modelo(data_yaml)
    print("Modelo final salvo em:", out_model.as_posix())


if __name__ == "__main__":
    main()
