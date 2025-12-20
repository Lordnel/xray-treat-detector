from ultralytics import YOLO
import gradio as gr
from PIL import Image
import numpy as np
import cv2
import pandas as pd
import time

model = YOLO("./models/yolo11_sixray_best.pt")

CLASS_COLORS = {
    "gun": (0, 0, 255), #rouge
    "knife": (204, 0, 204), #violet
    "wrench": (255, 51, 51), # bleu
    "pliers": (255, 255, 0), # turquoise
    "scissors": (0, 255, 255), #jaune
}

french_classes = {
    "gun": "pistolet",
    "knife": "couteau",
    "wrench": "clé",
    "pliers": "pince",
    "scissors": "ciseaux",
}


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
                "classe": french_classes.get(cls_name.lower(), cls_name),
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


css = """
#beep_audio {
  display: none !important;
}
"""

with gr.Blocks(css=css) as demo:

    gr.Markdown("# Plateforme de Sûreté Aéroportuaire Rayons X par Vision Artificielle")
    gr.Markdown("### Importez des images rayon X ou utilisez des exemples fournis : le modèle détecte les armes et objets dangereux")

    with gr.Row():
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Image rayon X à analyser"
            )

            conf_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.5,
                step=0.05,
                label="Seuil de confiance minimum"
            )

            with gr.Accordion("Images d’exemple", open=False):
                gr.Examples(
                    examples = [
                        ["samples_images/1.jpg", 0.75],
                        ["samples_images/2.jpg", 0.5],
                        ["samples_images/3.jpg", 0.6],
                        ["samples_images/4.jpg", 0.78],
                    ],
                    inputs=[image_input, conf_slider],
                )

            btn = gr.Button("Lancer la détection", variant="primary")

        with gr.Column(scale=1):
            img_out = gr.Image(
                label="Image annotée (bboxes + IDs)"
            )

            df_out = gr.Dataframe(
                headers=["id", "classe", "confiance"],
                label="Détails des objets détectés"
            )

            status_out = gr.HTML(label="Voyant")
            audio_out = gr.Audio(autoplay=True, elem_id="beep_audio")

    btn.click(
        fn=detect_weapons,
        inputs=[image_input, conf_slider],
        outputs=[img_out, df_out, status_out, audio_out]
    )

    gr.Markdown(
    """
    <div style="text-align:center; margin-top:30px; color:#777; font-size:12px;">
        Projet tutoré DevOps-MlOps 2025-2026 5A IIIA - HESTIM<br>
        © 2025 — Plateforme de Sûreté Aéroportuaire Rayons X par Vision Artificielle<br>
        Développée par <b>Prunel AKPLOGAN</b> & <b>Kenneth ADJETE</b><br>
        Encadrée par <b>Pr. KHIAT Azzedine</b>
    </div>
    """,
    elem_id="footer"
)


demo.launch(server_name="0.0.0.0", server_port=7865)
