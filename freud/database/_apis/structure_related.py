from collections import OrderedDict



class AttributeContainer(object):

  @property
  def col2attribute(self) -> OrderedDict:
    """Mapping of column names to attributes."""
    od = OrderedDict()
    for attr in self.attributes:
      assert attr.name not in od, f'Attribute name `{attr.name}` is not unique.'
      od[attr.name] = attr
      for a in attr.alias:
        assert a not in od, f'Alias `{a}` is not unique.'
        od[a] = attr
    return od
