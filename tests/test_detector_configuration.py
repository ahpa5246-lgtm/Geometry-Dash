import pytest
from gd_ai.vision import MotionEndDetector
@pytest.mark.parametrize('kwargs',[{'freeze_frames':0},{'warmup_frames':-1},{'threshold':float('nan')},{'threshold':-1},{'threshold':2}])
def test_rejects_unusable_motion_detector_configuration(kwargs):
 with pytest.raises(ValueError):MotionEndDetector(**kwargs)
