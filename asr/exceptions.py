
class ASRError(RuntimeError):
    pass

class AudioPreprocessingError(ASRError):
    pass

class ASRModelError(ASRError):
    pass
