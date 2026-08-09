# Integration-test data

Fixed inputs for the real-network integration suite (`pytest -m integration`). Regenerate with `pipeline_v1/tests/prepare_integration_data.py`.

- `panels/<case_id>.png` — the pre-cropped panel for each DET/OOV/COL/SIZE case, produced by the real reading-order extraction (YOLO26n + `panel_ordering.reading_order`), so the fixture's panel IDs stay meaningful. The integration tests take these crops as input; they never run panel detection themselves.
- `pages/lay_001_page.png` — p006, the full-page illustration the LAY-001 layout test runs real YOLO on.

## Provenance of the last regeneration

```json
{
  "generated_at": "2026-08-09T20:39:52+02:00",
  "detector": "leoxs22/manga-panel-detector-yolo26n",
  "confidence": 0.25,
  "panel_inset": 0,
  "ordering": "panel_ordering.reading_order (right-to-left, top-to-bottom)",
  "fixture": "/home/phil/code/perso/manga_colorization/pipeline_v1/evaluation/v1_1_cases.json",
  "pages": {
    "CH134_004": {
      "source": "/home/phil/code/perso/manga_colorization/data/chapter_134/0134-004.png",
      "panels": 5,
      "boxes": [
        [
          618,
          0,
          1093,
          972
        ],
        [
          105,
          166,
          602,
          399
        ],
        [
          104,
          425,
          602,
          973
        ],
        [
          540,
          996,
          1199,
          1800
        ],
        [
          103,
          997,
          524,
          1648
        ]
      ]
    },
    "P003": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c001 (v01) - p003 [VIZ Media] [Digital] [1r0n].png",
      "panels": 6,
      "boxes": [
        [
          606,
          1,
          1382,
          1070
        ],
        [
          0,
          235,
          589,
          577
        ],
        [
          0,
          611,
          585,
          1070
        ],
        [
          740,
          1105,
          1382,
          1494
        ],
        [
          739,
          1524,
          1383,
          2088
        ],
        [
          0,
          1101,
          720,
          2244
        ]
      ]
    },
    "P004_005": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c001 (v01) - p004-p005 [VIZ Media] [Digital] [1r0n].png",
      "panels": 1,
      "boxes": [
        [
          23,
          0,
          2918,
          2250
        ]
      ]
    },
    "P007": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c001 (v01) - p007 [VIZ Media] [Digital] [1r0n].png",
      "panels": 7,
      "boxes": [
        [
          800,
          0,
          1389,
          696
        ],
        [
          151,
          0,
          783,
          696
        ],
        [
          939,
          1174,
          1387,
          1682
        ],
        [
          573,
          763,
          1387,
          1141
        ],
        [
          506,
          1176,
          972,
          1683
        ],
        [
          0,
          711,
          633,
          1710
        ],
        [
          1,
          1713,
          1389,
          2248
        ]
      ]
    },
    "P008": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c001 (v01) - p008 [VIZ Media] [Digital] [1r0n].png",
      "panels": 4,
      "boxes": [
        [
          109,
          0,
          1499,
          671
        ],
        [
          636,
          705,
          1500,
          1501
        ],
        [
          109,
          705,
          618,
          1500
        ],
        [
          116,
          1538,
          1342,
          2072
        ]
      ]
    },
    "P013": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c001 (v01) - p013 [VIZ Media] [Digital] [1r0n].png",
      "panels": 2,
      "boxes": [
        [
          12,
          1,
          1366,
          681
        ],
        [
          12,
          713,
          1367,
          2250
        ]
      ]
    },
    "P130": {
      "source": "/home/phil/code/perso/manga_colorization/data/page_per_volume/Frieren - Beyond Journey's End v01 (2021) (Digital) (1r0n) (f2)/Frieren - Beyond Journey's End - c005 (v01) - p130 [VIZ Media] [Digital] [1r0n].png",
      "panels": 6,
      "boxes": [
        [
          138,
          0,
          1364,
          656
        ],
        [
          738,
          690,
          1500,
          1199
        ],
        [
          738,
          1235,
          1499,
          1646
        ],
        [
          132,
          689,
          717,
          1648
        ],
        [
          869,
          1682,
          1500,
          2056
        ],
        [
          129,
          1682,
          847,
          2250
        ]
      ]
    }
  }
}
```
