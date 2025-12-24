# tests/test_app.py
import pytest
import numpy as np
from PIL import Image
import pandas as pd
from api.app import make_beep, detect_weapons, french_classes

def test_make_beep_format1():
    """Vérifie que la fonction make_beep retourne le bon format audio pour Gradio"""
    sr, wave = make_beep(duration=0.1) # durée courte pour le test
    assert sr == 22050
    assert isinstance(wave, np.ndarray)
    assert wave.dtype == np.float32

def test_french_classes_content():
    """Vérifie que les traductions essentielles sont présentes"""
    assert "gun" in french_classes
    assert french_classes["gun"] == "pistolet"
    assert "knife" in french_classes

def test_detect_weapons_none_input():
    """Vérifie que la fonction gère correctement une entrée vide (None)"""
    res_img, res_df, res_html, res_audio = detect_weapons(None, 0.5)
    assert res_img is None
    assert res_df is None
    assert res_html is None
    assert res_audio is None

def test_detect_weapons_with_blank_image():
    """Vérifie le comportement avec une image vide (sans objets)"""
    # Création d'une image noire de 100x100
    blank_img = Image.fromarray(np.zeros((100, 100, 3), dtype=np.uint8))
    
    annotated_img, df, status_html, audio = detect_weapons(blank_img, 0.9)
    
    # Doit retourner une image PIL
    assert isinstance(annotated_img, Image.Image)
    # Le DataFrame doit être vide (si aucune détection)
    assert isinstance(df, pd.DataFrame)
    # Le statut doit être "AUCUNE MENACE"
    assert "AUCUNE MENACE" in status_html
    # Pas d'audio si pas de danger
    assert audio is None