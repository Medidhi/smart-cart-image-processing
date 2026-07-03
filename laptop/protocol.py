"""
Tiny length-prefixed frame protocol shared by server (Pi) and client (laptop).

Wire format per message:
    [4 bytes: json_len big-endian][json bytes][4 bytes: jpeg_len][jpeg bytes]

json payload = {"fps": float, "camera_id": int, "camera_name": str,
    "annotated": bool, "counts": {label: n}, "detections": [
    {"label","cls_id","score","box":[x1,y1,x2,y2],"center":[cx,cy]}, ...]}
camera_id / camera_name / annotated / cls_id are OPTIONAL additive fields
(older senders omit them; receivers must tolerate their absence).
"""
import json
import struct

_HDR = struct.Struct(">I")


def send_frame(sock, meta: dict, jpeg_bytes: bytes):
    payload = json.dumps(meta).encode("utf-8")
    sock.sendall(_HDR.pack(len(payload)) + payload
                 + _HDR.pack(len(jpeg_bytes)) + jpeg_bytes)


def _recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_frame(sock):
    """Blocking read of one message. Returns (meta_dict, jpeg_bytes)."""
    (json_len,) = _HDR.unpack(_recv_exact(sock, 4))
    meta = json.loads(_recv_exact(sock, json_len).decode("utf-8"))
    (jpeg_len,) = _HDR.unpack(_recv_exact(sock, 4))
    jpeg = _recv_exact(sock, jpeg_len)
    return meta, jpeg
