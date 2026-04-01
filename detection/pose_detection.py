class PoseDetector:
    def __init__(self):
        # No-op pose detector fallback (for environments without mediapipe.solutions).
        pass

    def detect(self, frame):
        return None

    def draw(self, frame, result):
        return frame
