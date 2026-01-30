"""
Detecção de vacas com YOLO e comparação com rótulos (GT) do DatasetNinja.

O script:
1) Lê as caixas de ground-truth do JSON.
2) Executa YOLO na imagem e filtra a classe definida em CLASSE_ALVO.
3) Desenha boxes (GT e predições) em memória.
4) Exibe as imagens na tela.
5) Calcula IoU por predição e AP@0.5 (mAP@0.5 para 1 classe e 1 imagem).
"""

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

try:
    from ultralytics import YOLO
except Exception as e:
    raise SystemExit(
        "ultralytics is required. Install with: pip install ultralytics"
    ) from e

IMG_PATH = Path("cows2021-DatasetNinja/detection_and_localisation-train/images/00061.jpg")
ANN_PATH = Path("cows2021-DatasetNinja/detection_and_localisation-train/ann/00061.jpg.json")
#MODEL_PATH = Path("yolov8n.pt")  # modelo base (COCO), classe esperada: "cow"
#MODEL_PATH = Path("yolo11n.pt")  # modelo base (COCO), classe esperada: "cow"
MODEL_PATH = Path("yolov8n_dorso_vaca.pt")  # modelo local, classe esperada: "cattle_torso"
CLASSE_ALVO = "cattle_torso"  # Use "cow" para modelos base; "cattle_torso" para o modelo treinado.
CONF_THRES = 0.25  # Limiar de confiança das predições da YOLO (filtra detecções fracas).
IOU_THRES = 0.5  # Limiar de IoU para considerar uma predição como verdadeiro positivo.


def carregar_caixas_gt(path: Path):
    """
    Carrega caixas GT (x1,y1,x2,y2) do JSON do DatasetNinja.

    Parâmetros:
        path (Path): Caminho do arquivo JSON de anotação.

    Retorna:
        list[list[float]]: Lista de caixas no formato [x1, y1, x2, y2].
    """
    data = json.loads(path.read_text(encoding="utf-8"))
    boxes = []
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
        boxes.append([x1, y1, x2, y2])
    return boxes


def executar_yolo(image_path: Path, model_path: Path, conf: float):
    """
    Executa YOLO na imagem e retorna lista de predições com box, classe e confiança.

    Parâmetros:
        image_path (Path): Caminho da imagem de entrada.
        model_path (Path): Caminho do modelo YOLO (.pt).
        conf (float): Limiar de confiança para as predições.

    Retorna:
        list[dict]: Lista de predições com chaves: box, conf, cls_id e name.
    """
    model = YOLO(str(model_path))
    results = model(str(image_path), conf=conf, verbose=False)
    preds = []
    for r in results:
        if r.boxes is None:
            continue
        for box, cls_id, score in zip(r.boxes.xyxy, r.boxes.cls, r.boxes.conf):
            cls_id = int(cls_id)
            name = r.names.get(cls_id, str(cls_id))
            preds.append(
                {
                    "box": box.cpu().numpy().astype(float).tolist(),
                    "conf": float(score),
                    "cls_id": cls_id,
                    "name": name,
                }
            )
    return preds


def filtrar_classe(preds, classe_alvo):
    """
    Filtra apenas a classe desejada nas predições.

    Parametros:
        preds (list[dict]): Lista de predicoes geradas pela YOLO.
        classe_alvo (str): Nome da classe a manter (ex: "cow", "cattle_torso").

    Retorna:
        list[dict]: Lista de predicoes apenas da classe desejada.
    """
    alvo = classe_alvo.lower().strip()
    return [p for p in preds if p["name"].lower() == alvo]


def calcular_iou(a, b):
    """
    Calcula IoU entre duas caixas no formato [x1,y1,x2,y2].

    Parâmetros:
        a (list[float]): Caixa A [x1, y1, x2, y2].
        b (list[float]): Caixa B [x1, y1, x2, y2].

    Retorna:
        float: Valor do IoU entre 0 e 1.
    """
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    a_area = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    b_area = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = a_area + b_area - inter
    return inter / union if union > 0 else 0.0


def casar_predicoes(gt_boxes, preds, iou_thr=0.5):
    """
    Faz o matching GT x predição por IoU (greedy, por confiança).

    Parâmetros:
        gt_boxes (list[list[float]]): Caixas ground-truth.
        preds (list[dict]): Predições da YOLO.
        iou_thr (float): Limiar de IoU para considerar TP.

    Retorna:
        list[dict]: Lista com predição, IoU e flag TP/FP.
    """
    preds = sorted(preds, key=lambda p: p["conf"], reverse=True)
    gt_used = [False] * len(gt_boxes)
    matches = []
    for p in preds:
        best_iou = 0.0
        best_j = -1
        for j, gt in enumerate(gt_boxes):
            if gt_used[j]:
                continue
            v = calcular_iou(p["box"], gt)
            if v > best_iou:
                best_iou = v
                best_j = j
        is_tp = best_iou >= iou_thr and best_j >= 0
        if is_tp:
            gt_used[best_j] = True
        matches.append({"pred": p, "iou": best_iou, "tp": is_tp})
    return matches


def calcular_ap(matches, num_gt):
    """
    Calcula AP (VOC-style) a partir das correspondências.

    Parâmetros:
        matches (list[dict]): Lista de matches retornada por casar_predicoes.
        num_gt (int): Número total de caixas GT.

    Retorna:
        float: Valor de AP.
    """
    if num_gt == 0:
        return 0.0
    tps = np.array([1 if m["tp"] else 0 for m in matches])
    fps = 1 - tps
    tp_cum = np.cumsum(tps)
    fp_cum = np.cumsum(fps)
    recall = tp_cum / max(1, num_gt)
    precision = tp_cum / np.maximum(1, (tp_cum + fp_cum))

    # Interpolated AP (VOC-style)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return float(ap)


def desenhar_caixas(image, gt_boxes, preds, title=None):
    """
    Desenha GT (verde) e predições (vermelho) e retorna a imagem resultante.

    Parâmetros:
        image (PIL.Image.Image): Imagem base.
        gt_boxes (list[list[float]]): Caixas GT.
        preds (list[dict]): Predições da YOLO.
        title (str | None): Título opcional para desenhar na imagem.

    Retorna:
        PIL.Image.Image: Imagem com os boxes desenhados.
    """
    img = image.copy()
    draw = ImageDraw.Draw(img)
    for box in gt_boxes:
        draw.rectangle(box, outline="lime", width=3)
    for p in preds:
        box = p["box"]
        label = f"{p['name']} {p['conf']:.2f}"
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0] + 3, box[1] + 3), label, fill="red")
    if title:
        draw.text((10, 10), title, fill="yellow")
    return img


def exibir_imagens(images, titles):
    """
    Exibe uma linha de imagens com títulos usando matplotlib.

    Parâmetros:
        images (list[PIL.Image.Image]): Lista de imagens para exibição.
        titles (list[str]): Lista de títulos correspondentes.

    Retorna:
        None
    """
    cols = len(images)
    fig, axes = plt.subplots(1, cols, figsize=(6 * cols, 6))
    if cols == 1:
        axes = [axes]
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


def principal():
    """
    Fluxo principal: carrega dados, roda YOLO, exibe imagens e imprime métricas.

    Parâmetros:
        None

    Retorna:
        None
    """
    if not IMG_PATH.exists():
        raise SystemExit(f"Image not found: {IMG_PATH}")
    if not ANN_PATH.exists():
        raise SystemExit(f"Annotation not found: {ANN_PATH}")

    # 1) Carrega GT e roda YOLO
    gt_boxes = carregar_caixas_gt(ANN_PATH)
    preds = executar_yolo(IMG_PATH, MODEL_PATH, CONF_THRES)
    class_preds = filtrar_classe(preds, CLASSE_ALVO)

    img = Image.open(IMG_PATH).convert("RGB")

    # 2) Renderiza imagens em memória
    model_name = MODEL_PATH.name
    yolo_img = desenhar_caixas(img, [], class_preds, f"Predições da YOLO ({model_name})")
    overlay_img = desenhar_caixas(
        img, gt_boxes, class_preds, f"GT (green) + YOLO (red) ({model_name})"
    )

    # 3) Exibe na tela
    exibir_imagens(
        [yolo_img, overlay_img],
        [f"Predições da YOLO ({model_name})", f"Sobreposição GT + YOLO ({model_name})"],
    )
    # 4) Métricas
    # IoU (Intersection over Union): mede a sobreposição entre a caixa predita e a GT (Ground Truth - Anotação verdadeira),
    # calculando a área de interseção dividida pela área de união (0 a 1) 0 as caixas não se sobrepõem. 1 as caixas se sobrepõem completamente.
    # IOU_THRES (0.5) NÃO é confiança; ele define se uma predição vira TP no cálculo
    # da AP/mAP: se IoU >= 0.5, conta como acerto.
    # CONF_THRES (0.25) é o filtro de confiança das predições: abaixo disso a YOLO
    # descarta a detecção antes do cálculo das métricas.
    # mAP (mean Average Precision): média da AP entre classes; aqui usamos AP@0.5,
    # que integra a curva precisão-recall com IoU >= 0.5. Como é 1 classe/1 imagem,
    # mAP@0.5 = AP@0.5.
    matches = casar_predicoes(gt_boxes, class_preds, IOU_THRES)
    ap50 = calcular_ap(matches, len(gt_boxes))
    map50 = ap50  # single class, single image

    print("Modelo usado:", MODEL_PATH.name)
    print("Classe alvo:", CLASSE_ALVO)
    print("Caixas GT:", len(gt_boxes))
    print("Caixas preditas (classe alvo):", len(class_preds))
    print("Correspondências (ordenadas por confiança):")
    for i, m in enumerate(matches, 1):
        box = m["pred"]["box"]
        print(
            f"  {i}. conf={m['pred']['conf']:.3f} iou={m['iou']:.3f} tp={m['tp']} box={box}"
        )
    print(f"AP@0.5: {ap50:.4f}")
    print(f"mAP@0.5 (imagem/classe única): {map50:.4f}")

    # Se não houver vacas detectadas, mostre o que a YOLO encontrou
    if len(class_preds) == 0:
        print(f"Nenhuma predição da classe '{CLASSE_ALVO}' foi encontrada.")
        if len(preds) == 0:
            print("A YOLO não retornou nenhuma predição acima do CONF_THRES.")
        else:
            print("Outras classes detectadas (top 10 por confiança):")
            preds_sorted = sorted(preds, key=lambda p: p["conf"], reverse=True)
            for i, p in enumerate(preds_sorted[:10], 1):
                print(f"  {i}. {p['name']} conf={p['conf']:.3f} box={p['box']}")


if __name__ == "__main__":
    principal()
