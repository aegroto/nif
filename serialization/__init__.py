import pickle
import sys
import struct
import torch

def save_buffer_to_file(buffer, path):
    with open(path, "wb") as file:
        file.write(buffer.getbuffer())

def serialize_state_dict(state_dict, path):
    with open(path, "wb") as file:
        pickle.dump(state_dict, file)

def deserialize_state_dict(path):
    with open(path, "rb") as file:
        return pickle.load(file)

# def serialize_state_dict(state_dict, buffer):
#     serialized_size = 0
#     for (name, param) in state_dict.items():
#         if type(param) is dict:
#             serialized_size = serialize_state_dict(param, buffer)
#         elif type(param) is float:
#             buffer.write(bytes(struct.pack("f", param)))
#         elif type(param) is int:
#             buffer.write(bytes(struct.pack("i", param)))
#         elif type(param) is bytes:
#             buffer.write(param)
#         else:
#             print(f"ERROR: Unable to serialize '{name}' ({type(param)})") 
#             sys.exit(1)

#         print(f"Written '{name}' ({type(param)}), buffer size: {len(buffer.getbuffer())}")

