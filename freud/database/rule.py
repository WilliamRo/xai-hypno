from collections import OrderedDict
from roma import Nomear, io, console



class Rule(Nomear):

  INTERNAL_KEY_PREFIX = 'PID'
  INTERNAL_PRIMARY_KEY_DIGITS = 6

  @Nomear.property(local=True)
  def primary_key_dict(self) -> OrderedDict: return OrderedDict()


  @property
  def internal_key_to_primary_key(self) -> OrderedDict:
    """Map internal keys to primary keys"""
    return OrderedDict((v, k) for k, v in self.primary_key_dict.items())


  def register(self, primary_key: str):
    # Check if the primary key is a string
    if not isinstance(primary_key, str):
      raise TypeError('Primary key must be a string.')

    # Check if the primary key is valid
    if primary_key in self.primary_key_dict:
      return self.primary_key_dict[primary_key]
      # raise AssertionError(f'Primary key `{primary_key}` already registered.')

    # Generate an internal key
    prefix = self.INTERNAL_KEY_PREFIX
    n_digits = self.INTERNAL_PRIMARY_KEY_DIGITS
    internal_key = f'{prefix}{len(self.primary_key_dict) + 1:0{n_digits}}'

    # Check if the internal key already exists
    if internal_key in self.primary_key_dict.values():
      raise AssertionError(f'Internal key `{internal_key}` already exists.')

    # Register the primary key with the internal key
    self.primary_key_dict[primary_key] = internal_key

    # Return the internal key
    return internal_key


  def is_primary_key(self, key) -> bool:
    """Check if the given key is a primary key"""
    # Primary keys should be unique and typically have a length greater than 3 characters
    # TODO: current implementation is a placeholder, adjust as needed
    return len(str(key)) >= 3