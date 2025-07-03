from src.processors.preprocessing.U2Net import u2net_predict
from util.Processor import Processor


class UNetSaliency(Processor):
    def main(self, *args, **kwargs):
        self._image = u2net_predict(self._origin)
