from modules.dumper import FeaturesDumper

MODULATOR_DUMPERS=0

def build_dumper(writer):
    global MODULATOR_DUMPERS
    MODULATOR_DUMPERS += 1

    if writer:
        return FeaturesDumper({
            "tag": f"modulation_map.{MODULATOR_DUMPERS}",
            "mode": "signed_split",
            "interval": 100,
            "expand": True
        }, writer)
    else:
        return None

