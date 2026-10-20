import json
from types import SimpleNamespace

import numpy as np

from core.result_parser import frame_counts, parse_detections, simplify_polygon
from fakes import fake_result

CFG = SimpleNamespace(SAVE_KEYPOINTS=True, SAVE_POLYGONS=True, POLYGON_EPSILON_PX=2.0, COORD_DECIMALS=0)


def test_deteccion_serializable():
    dets = parse_detections(fake_result("deteccion"), CFG)
    assert len(dets) == 2
    assert dets[0]["bbox_xyxy"] == [10, 21, 110, 221]
    assert "keypoints" not in dets[0]
    json.dumps(dets)


def test_pose_17_keypoints():
    dets = parse_detections(fake_result("pose"), CFG)
    assert len(dets[0]["keypoints"]) == 17 and len(dets[0]["keypoints"][0]) == 3


def test_segmentacion_area_y_poligono_simplificado():
    dets = parse_detections(fake_result("segmentacion"), CFG)
    assert dets[0]["mask_area_px"] == 2500
    assert len(dets[0]["polygon"]) == 4          # el punto colineal (50,1) se elimina


def test_sin_detecciones():
    assert parse_detections(fake_result("deteccion", n=0), CFG) == []
    assert frame_counts([])["n_detections"] == 0


def test_epsilon_cero_no_simplifica():
    pts = np.array([[0, 0], [1, 0], [2, 0], [2, 2], [0, 2]], dtype=np.float32)
    assert len(simplify_polygon(pts, 0)) == 5
