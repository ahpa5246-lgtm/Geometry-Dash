import numpy as np

from gd_ai.vision import MotionEndDetector


def test_static_frames_eventually_end_episode():
    detector = MotionEndDetector(threshold=0.004, freeze_frames=3, warmup_frames=0)
    frame = np.zeros((84, 84, 1), dtype=np.uint8)

    ended = False
    for _ in range(6):
        ended, _ = detector.update(frame)

    assert ended is True


def test_large_frame_changes_do_not_end_episode():
    detector = MotionEndDetector(threshold=0.004, freeze_frames=3, warmup_frames=0)

    ended = False
    for i in range(10):
        value = 0 if i % 2 == 0 else 255
        frame = np.full((84, 84, 1), value, dtype=np.uint8)
        ended, _ = detector.update(frame)

    assert ended is False
