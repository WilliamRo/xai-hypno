from datetime import date, datetime

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


def parse_date(value) -> date:
  """Possible formats:
    (1) 2021.6.1
    (2) 20230301
    (3) 5/26/2022
    (4) 2000-01-01
    (5) 2000/01/01
  """
  if isinstance(value, date): return value
  value = str(value)

  # Case I: Valid formats
  formats = [
    "%Y.%m.%d",  # 2021.6.1
    "%Y%m%d",  # 20230301
    "%m/%d/%Y",  # 5/26/2022
    "%Y-%m-%d",  # 2000-01-01
    "%Y/%m/%d",  # 2000/01/01
  ]

  # Try each format
  for fmt in formats:
    try:
      return datetime.strptime(value, fmt).date()
    except (ValueError, TypeError):
      continue

  # Case II: Invalid formats
  formats = [
    "%Y.%m",  # 2022.9
    # "%Y%m%d",  # 20230301
    # "%m/%d/%Y",  # 5/26/2022
    # "%Y-%m-%d",  # 2000-01-01
    # "%Y/%m/%d",  # 2000/01/01
  ]

  for fmt in formats:
    try:
      _ = datetime.strptime(value, fmt).date()
      return None
    except (ValueError, TypeError):
      continue

  # Case III:
  if value in (
      '44740',
      '44704',
  ): return None

  # If no format matched, raise an error
  raise ValueError(f"Date format not recognized: {value}")


def parse_int(value) -> int:
  return int(float(value))


def parse_float(value) -> float:
  return float(value)


class Parser(object):

  NONE_SET = ['#N/A', 'null', 'nan', '/', '']

  Library = {
    'primary_key': parse_id,
    'gender': parse_gender_cn,
    'date': parse_date,
    'int': parse_int,
    'float': parse_float,
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

