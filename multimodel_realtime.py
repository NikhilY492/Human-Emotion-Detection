# realtime_fast.py
import cv2, time, threading, numpy as np, torch, collections
from mtcnn import MTCNN
from model_multimodal import MultimodalNet
from torchvision import transforms
import librosa

# CONFIG
CAM_INDEX = 0
USE_DSHOW = True           # Windows: use CAP_DSHOW for faster capture
WIDTH, HEIGHT = 640, 480
USE_MJPG = True            # Force MJPG to reduce CPU decompression
DETECT_EVERY = 8           # run detector every N frames, use tracker otherwise
BUFF_FRAMES = 16           # how many frames to buffer for inference
PRED_INTERVAL = 0.5        # seconds between predictions
AUDIO_SR = 22050
AUDIO_WINDOW = 3.0
MFCC_N = 40
MFCC_TLEN = 80

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Device:", DEVICE)

# Load model (assume you saved checkpoint 'multimodal_best.pth')
ckpt = torch.load("multimodal_best.pth", map_location=DEVICE)
labels = ckpt['labels']
model = MultimodalNet(n_classes=len(labels)).to(DEVICE)
model.load_state_dict(ckpt['model_state'])
model.eval()

# Preproc (resize + normalize)
preproc_frame = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
])

# Simple audio buffer (shared)
from collections import deque
audio_q = deque(maxlen=int(AUDIO_SR * AUDIO_WINDOW))
audio_lock = threading.Lock()

# Background capture thread (fast)
class CameraCapture(threading.Thread):
    def __init__(self, idx=0, width=WIDTH, height=HEIGHT, use_dshow=True, use_mjpg=True):
        super().__init__(daemon=True)
        backend = cv2.CAP_DSHOW if use_dshow else 0
        self.cap = cv2.VideoCapture(idx, backend)
        if use_mjpg:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        # try to set fps
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        self.stopped = False
        self.frame = None
        self.lock = threading.Lock()

    def run(self):
        while not self.stopped:
            ret, fr = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            with self.lock:
                self.frame = fr

    def read(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()

    def stop(self):
        self.stopped = True
        try:
            self.cap.release()
        except:
            pass

# helper: get mfcc tensor from audio buffer
def make_mfcc_from_buffer():
    with audio_lock:
        arr = np.array(audio_q, dtype=np.float32)
    if arr.size == 0:
        return None
    target_len = int(AUDIO_SR * AUDIO_WINDOW)
    if arr.size < target_len:
        arr = np.pad(arr, (target_len - arr.size, 0))
    else:
        arr = arr[-target_len:]
    mfcc = librosa.feature.mfcc(y=arr, sr=AUDIO_SR, n_mfcc=MFCC_N)
    mfcc = (mfcc - mfcc.mean())/(mfcc.std()+1e-6)
    if mfcc.shape[1] < MFCC_TLEN:
        mfcc = np.pad(mfcc, ((0,0),(0,MFCC_TLEN-mfcc.shape[1])))
    else:
        mfcc = mfcc[:,:MFCC_TLEN]
    # shape to (1,1,n_mfcc,tlen)
    t = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
    return t

# Inference helper
def infer_from_buffers(frame_buffer):
    if len(frame_buffer) < BUFF_FRAMES:
        return None
    frames_np = np.stack(list(frame_buffer))  # (T,H,W,3)
    frames_t = torch.stack([preproc_frame(f) for f in frames_np]).unsqueeze(0).to(DEVICE)  # (1,T,C,H,W)
    mfcc_t = make_mfcc_from_buffer()
    if mfcc_t is None:
        return None
    with torch.no_grad():
        # use autocast for faster mixed-precision on GPU
        if DEVICE.type == 'cuda':
            with torch.cuda.amp.autocast():
                logits = model(mfcc_t, frames_t)
        else:
            logits = model(mfcc_t, frames_t)
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        idx = int(probs.argmax())
        return labels[idx], float(probs[idx])

# --- Main ---
cap_thread = CameraCapture(idx=CAM_INDEX, use_dshow=USE_DSHOW, use_mjpg=USE_MJPG)
cap_thread.start()
detector = MTCNN()           # expensive; we will call rarely
tracker = None
tracking_bbox = None

frame_buffer = deque(maxlen=BUFF_FRAMES)
last_pred_time = 0
frame_count = 0
t0 = time.time()
cur_label, cur_conf = None, 0.0

print("Starting live (press 'q' to quit)...")
try:
    while True:
        frame = cap_thread.read()
        if frame is None:
            time.sleep(0.005)
            continue

        frame_count += 1

        # run detection every DETECT_EVERY frames, otherwise use tracker
        if tracking_bbox is None or (frame_count % DETECT_EVERY) == 0:
            # run heavy detector
            res = detector.detect_faces(frame)
            if res:
                r = max(res, key=lambda x: x['confidence'])
                x,y,w,h = r['box']
                x,y,w,h = int(max(0,x)), int(max(0,y)), int(w), int(h)
                tracking_bbox = (x,y,w,h)
                # create tracker
                tracker = cv2.TrackerCSRT_create()
                tracker.init(frame, (x,y,w,h))
        else:
            # update tracker, fallback to detection if fails
            ok, bbox = tracker.update(frame)
            if ok:
                x,y,w,h = [int(v) for v in bbox]
                tracking_bbox = (x,y,w,h)
            else:
                tracking_bbox = None
                tracker = None

        if tracking_bbox is not None:
            x,y,w,h = tracking_bbox
            x2,y2 = min(frame.shape[1], x+w), min(frame.shape[0], y+h)
            face_crop = frame[y:y2, x:x2]
            if face_crop.size != 0:
                # resize once to 224x224 for buffer and draw rectangle
                face_resized = cv2.resize(face_crop, (224,224), interpolation=cv2.INTER_LINEAR)
                frame_buffer.append(face_resized)
                cv2.rectangle(frame, (x,y), (x2,y2), (0,255,0), 2)

        # prediction at fixed interval (independent of frame rate)
        now = time.time()
        if now - last_pred_time >= PRED_INTERVAL:
            pred = infer_from_buffers(frame_buffer)
            last_pred_time = now
            if pred:
                cur_label, cur_conf = pred
            else:
                cur_label, cur_conf = None, 0.0

        # overlay result top-left
        label_text = f"{cur_label} {cur_conf:.2f}" if cur_label else "Detecting..."
        cv2.rectangle(frame, (5,5), (260,45), (0,0,0), -1)
        cv2.putText(frame, label_text, (10,32), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

        # show fps
        elapsed = time.time() - t0
        fps = frame_count / elapsed if elapsed > 0 else 0.0
        cv2.putText(frame, f"FPS:{fps:.1f}", (WIDTH-120,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)

        cv2.imshow("Realtime (fast)", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    cap_thread.stop()
    cv2.destroyAllWindows()
