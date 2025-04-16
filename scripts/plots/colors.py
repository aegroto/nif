colors = {
    "NIF": "orangered",
    "Strumpler basic [2021]": "plum",
    "Strumpler meta-learned [2021]": "blue",
    "BPG": "cyan",
    "JPEG": "yellowgreen",
    "JPEG2000": "black",
    "Xie [2021]": "green",
    "Ballè (Factorized Prior) [2017]": "orange",
    "Ballè (Hyperprior) [2017]": "orchid",
    "COIN": "violet",
    "COIN++": "magenta",
    "WebP": "gray"
}

def codec_color(name):
    try:
        return colors[name]
    except KeyError:
        print(f"WARNING: Unassigned color for {name}")
        return "white"
