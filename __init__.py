from .save_image_custom import SaveImageCustom

NODE_CLASS_MAPPINGS = {
    "SaveUtility: SaveImageCustom": SaveImageCustom,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveUtility: SaveImageCustom": "Save Image (Dir + Name)",
}

