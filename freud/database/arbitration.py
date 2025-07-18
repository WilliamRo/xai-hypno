from roma import console



def record_conflict_court(rec_1: dict, rec_2: dict, key: str, **kwargs):
  return False



class Arbitration(object):

  Courts = {
    'record_conflict': record_conflict_court,
  }


  @staticmethod
  def handle_record_conflict(rec_1: dict, rec_2: dict, key: str, pid=None):
    return Arbitration.Courts['record_conflict'](rec_1, rec_2, key, pid=pid)
