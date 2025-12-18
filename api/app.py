from ultralytics import YOLO
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import math

model = YOLO("./yolo11_sixray_best.pt")

CLASS_COLORS = {
    "gun": (0, 0, 255), #rouge
    "knife": (204, 0, 204), #violet
    "wrench": (255, 51, 51), # bleu
    "pliers": (255, 255, 0), # turquoise
    "scissors": (0, 255, 255), #jaune
}

import time

def make_beep(sr=22050, duration=2, freq=880):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)

    phase = (time.time() % 1) * 2 * np.pi

    wave = 0.2 * np.sin(2 * np.pi * freq * t + phase)
    return (sr, wave.astype(np.float32))



HIGH_THREAT_CONF = 0.7

def detect_weapons(image: Image.Image, conf_threshold: float):
    if image is None:
        return None, None, None, None

    results = model(image, conf=conf_threshold)[0]
    img_bgr = results.orig_img.copy()
    detections_info = []

    boxes = results.boxes
    count = 0
    max_conf = 0.0

    if boxes is None or len(boxes) == 0:
        annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    else:
        count = len(boxes)
        # calc max conf
        max_conf = float(boxes.conf.max().item()) if hasattr(boxes, "conf") else max(float(b.conf[0]) for b in boxes)

        for idx, box in enumerate(boxes, start=1):
            cls_id = int(box.cls[0])
            cls_name = results.names[cls_id]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            color = CLASS_COLORS.get(cls_name.lower(), (255, 255, 255))

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, thickness=2)

            label = f"{idx}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x, text_y = x1, y1
            cv2.rectangle(img_bgr, (text_x, text_y - th), (text_x + tw, text_y), color, -1)
            cv2.putText(img_bgr, label, (text_x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

            detections_info.append({
                "id": idx,
                "classe": cls_name,
                "confiance": round(conf, 4)
            })

        annotated = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    annotated_pil = Image.fromarray(annotated)
    df = pd.DataFrame(detections_info, columns=["id", "classe", "confiance"])

    state = "ok" if count == 0 else ("danger" if max_conf >= HIGH_THREAT_CONF else "warn")

    if state == "danger":
        audio = make_beep()
    else:
        audio = None

    STATE_STYLE = {
        "ok":    ("🟢 AUCUNE MENACE", "#2e7d32", "#e8f5e9"),
        "warn":  ("🟠 MENACE MODÉRÉE", "#ef6c00", "#fff3e0"),
        "danger":("🔴 MENACE DÉTECTÉE", "#c62828", "#ffebee"),
    }

    title, color, bg = STATE_STYLE[state]

    status_html = f"""
    <div class="status-box" style="
        border-left: 6px solid {color};
        background: {bg};
        padding: 12px 16px;
        border-radius: 8px;
        margin-bottom: 10px;">
        <div style="font-size:18px;font-weight:700;color:{color};">
            {title}
        </div>
            <div style="margin-top:4px;color:#111;">
            Objets détectés :
            <b style="color:#000;">{count}</b>
            —
            Confiance max :
            <b style="color:#000;">{max_conf:.2f}</b>
        </div>
    </div>
    """



    return annotated_pil, df, status_html, audio


conf_slider = gr.Slider(
    minimum=0.1,
    maximum=1.0,
    value=0.5,
    step=0.05,
    label="Seuil de confiance minimum"
)

css = """
#beep_audio {
  display: none !important;
}
"""


demo = gr.Interface(
    fn=detect_weapons,
    inputs=[gr.Image(type="pil", label="Image rayon X à analyser"), conf_slider],
    outputs=[
        gr.Image(type="pil", label="Image annotée (bboxes + IDs)"),
        gr.Dataframe(headers=["id", "classe", "confiance"], label="Détails des objets détectés"),
        gr.HTML(label="Voyant"),
        gr.Audio(label="Bip (si menace)", autoplay=True, elem_id="beep_audio")
    ],
    title="Plateforme de Sûreté Aéroportuaire par Inspection Rayons X",
    description="Upload une image rayon X : le modèle dessine les bounding boxes avec un ID par objet et affiche la liste détaillée des détections.",
    flagging_mode="never",
    css=css
)

demo.launch(server_name="0.0.0.0", server_port=7865)
