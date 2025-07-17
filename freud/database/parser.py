import re



def parse_id(value):
  return str(value)


def parse_gender(value):
  value = str(value).strip().lower()
  if value in ('female', 'male'): return value
  if '女' in value: return 'female'
  if '男' in value: return 'male'
  if value == '': return None

  raise ValueError(f'Failed to parse gender from value: {value}')


def parse_gender_cn(value):
  return {'female': '女', 'male': '男'}[parse_gender(value)]


def parse_date(value):
  pass


class Parser(object):

  NONE_SET = ['#N/A', 'null', 'nan', '/', '']

  Library = {
    'primary_key': parse_id,
    'gender': parse_gender_cn,
    'date': parse_date,
  }

  @staticmethod
  def parse(value, attribute):
    if str(value) in Parser.NONE_SET: return None

    from freud.database.structure import Attribute
    assert isinstance(attribute, Attribute)

    if attribute.name in Parser.Library:
      parse_method = Parser.Library[attribute.name]
    elif attribute.dtype in Parser.Library:
      parse_method = Parser.Library[attribute.dtype]
    else: return str(value)

    return parse_method(value)

