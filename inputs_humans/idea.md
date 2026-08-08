Je veux la pipeline de colorisation de manga suivante. Pour une page de manga
1. il detect les cases avec https://huggingface.co/leoxs22/manga-panel-detector-yolo26n (il faut sauvegarder les position des cases pour la derniere etape/)
2. il extrait les cases dans un dossier 1_pannels et numerate les pannels dans l'ordre de lecture japonais
3. pour chaue pannel il demande a google/gemma-4-31b-it sur openrouter quels sont les personnages present (see /home/phil/code/perso/manga_colorization/character_detection_methods/character-detection-openrouter-vlm/run.py)
4. il color case par case avec la methode flux 9b + lora en passant au model le panel a coloriser et un atlas composed seulement des personnages detecte.
5. une fois la case colorier, elle est stitch a la page original a la bonne position