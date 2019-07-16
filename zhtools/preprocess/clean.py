import logging
import unicodedata


LOGGER = logging.getLogger(__name__)


def to_halfwidth(text):
    result = ''
    for char in text:
        name = unicodedata.name(char, None)
        if name == 'IDEOGRAPHIC SPACE':
            result += ' '
        elif not name or name.find('FULLWIDTH') != 0:
            result += char
        elif name is not None:
            new_name = name.replace('FULLWIDTH', '').strip()
            new_char = unicodedata.lookup(new_name)
            result += new_char
        else:
            LOGGER.warning("can't get name from unicodedata: `%s`(%x)", char, ord(char))
            result += char

    return result
